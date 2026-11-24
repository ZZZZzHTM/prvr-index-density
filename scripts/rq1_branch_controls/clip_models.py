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


def widen(model,width):
 if width==1536:return
 encoder=model.clip.encoder
 with torch.random.fork_rng(devices=[]):
  for i in range(8):
   block=getattr(encoder,f'attn{i}').output
   assert block.layer1.in_features==384 and block.layer2.out_features==384
   a=nn.Linear(384,width);b=nn.Linear(width,384)
   for layer in (a,b):nn.init.normal_(layer.weight,mean=0.,std=model.config.initializer_range);nn.init.zeros_(layer.bias)
   block.layer1=a;block.layer2=b


def parameter_count(model):
 # named_parameters removes repeated parameter objects; buffers excluded.
 ps=dict(model.named_parameters());return dict(training_total=sum(p.numel() for p in ps.values()),trainable=sum(p.numel() for p in ps.values() if p.requires_grad),unique_parameter_tensors=len(ps),buffer_elements=sum(b.numel() for b in model.buffers()))


def build(cfg, method, original_factory):
 from .frame_models import RevisedFrameNet
 fresh=original_factory(cfg)
 if method=='B0':return fresh
 if method in ('ClipOnly','WideClipOnly','DualClip'):
  model=DualClipNet() if method=='DualClip' else ClipOnlyNet();model.config=fresh.config
  for name in ('query_pos_embed','query_input_proj','query_encoder','modular_vector_mapping'):model.add_module(name,getattr(fresh,name))
  model.register_parameter('weight_token',fresh.weight_token)
  model.add_module(model.branch_names[0],ClipBranch(fresh.config,tuple(getattr(fresh,n) for n in ('clip_pos_embed','clip_input_proj','clip_encoder'))))
  mc=fresh.config;del fresh
  if method=='DualClip':model.add_module('clip2',ClipBranch(mc))
  if method=='WideClipOnly':widen(model,cfg['clip_ffn_width'])
  return model
 fresh.__class__=RevisedFrameNet;fresh.fine_pool_enabled=method=='TrainedPool';fresh.sparse_frame_enabled=method=='SparseFrame';fresh.fusion_weights=cfg['score_fusion_weights']
 if method=='SparseFrame':
  with torch.random.fork_rng(devices=[]):
   layer=nn.Linear(384,32);nn.init.normal_(layer.weight,mean=0.,std=fresh.config.initializer_range);nn.init.zeros_(layer.bias)
  fresh.frame_encoder_1.layer2=layer
 return fresh
