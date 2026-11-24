"""Numerical implementation used by the reported experiment."""
from Models.gmmformerV2.model_components import DyGMMBlock
from easydict import EasyDict
from torch.nn import functional as F
from Models.gmmformerV2.model_components import LinearLayer
from Models.gmmformerV2.model import GMMFormerV2_Net as Original
from Models.gmmformerV2.model_components import TrainablePositionalEncoding
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch


class ClipBranch(nn.Module):
    def __init__(self,config,components=None):
        super().__init__()
        self.config=config
        if components is not None:
            self.pos_embed,self.input_proj,self.encoder=components
        else:
            self.pos_embed=TrainablePositionalEncoding(config.max_ctx_l,config.hidden_size,config.input_drop)
            self.input_proj=LinearLayer(config.visual_input_size,config.hidden_size,layer_norm=True,dropout=config.input_drop,relu=True)
            self.encoder=DyGMMBlock(EasyDict(hidden_size=config.hidden_size,intermediate_size=config.hidden_size,
                hidden_dropout_prob=config.drop,num_attention_heads=config.n_heads,attention_probs_dropout_prob=config.drop,
                frame_len=32,sft_factor=config.sft_factor))
            # Reset only this new branch, never the already aligned parent.
            Original.reset_parameters(self)

    def forward(self,features,weight_token):
        return Original.encode_input(features,None,self.input_proj,self.encoder,self.pos_embed,weight_token)


class SharedQueryClipBase(nn.Module):
    encode_input=staticmethod(Original.encode_input)
    encode_query=Original.encode_query
    get_modularized_queries=Original.get_modularized_queries
    get_clip_scale_scores=staticmethod(Original.get_clip_scale_scores)
    get_unnormalized_clip_scale_scores=staticmethod(Original.get_unnormalized_clip_scale_scores)

    def encode_context(self,clip_features):
        return {name:getattr(self,name)(clip_features,self.weight_token) for name in self.branch_names}

    def score_encoded_query(self,query,contexts,unnormalized=False):
        fn=self.get_unnormalized_clip_scale_scores if unnormalized else self.get_clip_scale_scores
        return {name:fn(query,contexts[name]) for name in self.branch_names}

    def get_pred_from_raw_query(self,query_feat,query_mask,contexts):
        return self.score_encoded_query(self.encode_query(query_feat,query_mask),contexts)

    def fuse(self,scores):
        if self.method=='DualClip':return .5*scores['clip1']+.5*scores['clip2']
        return scores['clip']

    def forward(self,batch):
        context=self.encode_context(batch['clip_video_features'])
        query=self.encode_query(batch['text_feat'],batch['text_mask'])
        scores=self.score_encoded_query(query,context)
        raw=self.score_encoded_query(query,context,unnormalized=True)
        labels=batch['text_labels'];label_dict={}
        for index,label in enumerate(labels):label_dict.setdefault(label,[]).append(index)
        # Preserve the original SECOND query encoding for QDL and OT.
        video_query=self.encode_query(batch['text_feat'],batch['text_mask'])
        first=context[self.branch_names[0]]
        total_sim=[]
        for k,v in label_dict.items():
            temp_clip_emb=first[k];temp_text_emb=video_query[v]
            if temp_text_emb.shape[0]==1:continue
            sim=-1.*torch.matmul(F.normalize(temp_clip_emb,dim=-1),F.normalize(temp_text_emb,dim=-1).t()).permute(1,0)
            q_idx,c_idx=linear_sum_assignment(sim.detach().cpu())
            for i in range(q_idx.shape[0]):total_sim.append(sim[q_idx[i],c_idx[i]])
        ot=1+torch.stack(total_sim).mean()
        return {'scores':scores,'raw_scores':raw,'label_dict':label_dict,'query':video_query,'ot':ot}


class DualClipNet(SharedQueryClipBase):
    method='DualClip'
    branch_names=('clip1','clip2')


class ClipOnlyNet(SharedQueryClipBase):
    method='ClipOnly'
    branch_names=('clip',)


class MultiClipNet(SharedQueryClipBase):
    method = 'MultiClip'

    def __init__(self, k):
        super().__init__()
        if not isinstance(k, int) or k < 2:
            raise ValueError('At least two integer views required')
        self.k = k
        self.branch_names = tuple(f'clip{i}' for i in range(1, k+1))

    def fuse(self, scores):
        # Preserve DualClip's multiply-before-add order, including K=2 bitwise equality.
        return sum((1. / self.k) * scores[name] for name in self.branch_names)


def reset_linear(layer, std):
    nn.init.normal_(layer.weight, mean=0., std=std)
    nn.init.zeros_(layer.bias)


import gc
QUERY_NAMES = ('query_pos_embed', 'query_input_proj', 'query_encoder', 'modular_vector_mapping')
CLIP_NAMES = ('clip_pos_embed', 'clip_input_proj', 'clip_encoder')

def build_triple(cfg, original_factory):
    """Initialize the tested three-branch model in the recorded draw order."""
    assert cfg["num_views"] == 3 and cfg["ffn_units"] == 191
    fresh=original_factory(cfg)
    model = MultiClipNet(cfg['num_views'])
    model.config = fresh.config
    for name in QUERY_NAMES:
        model.add_module(name, getattr(fresh, name))
    model.register_parameter('weight_token', fresh.weight_token)
    model.add_module('clip1', ClipBranch(fresh.config, tuple(getattr(fresh, name) for name in CLIP_NAMES)))
    config = fresh.config
    del fresh
    gc.collect()
    for i in range(2, cfg['num_views']+1):
        model.add_module(f'clip{i}', ClipBranch(config))
    units = 191
    h = 384
    for name in model.branch_names:
        encoder = getattr(model, name).encoder
        for block, width in [(getattr(encoder, f'attn{i}'), 4*units) for i in range(8)]+[(encoder.ca, units)]:
            block.output.layer1 = nn.Linear(h, width)
            block.output.layer2 = nn.Linear(width, h)
            reset_linear(block.output.layer1, model.config.initializer_range)
            reset_linear(block.output.layer2, model.config.initializer_range)
    return model
