import ast
import inspect
import textwrap
import torch
from torch.nn import functional as F

def clip_control(native):
    """Accept the official model, model_config and original loss in a module."""
    Original=native.DreamPRVR_Net
    tree=ast.parse(textwrap.dedent(inspect.getsource(Original.encode_context)))
    class FrameRemoval(ast.NodeTransformer):
     def visit_If(self,node):
      if any(isinstance(n,ast.Name) and n.id=='frame_video_feat' for n in ast.walk(node.test)):return None
      return self.generic_visit(node)
     def visit_Assign(self,node):
      if any(isinstance(n,ast.Name) and n.id=='encoded_frame_feat' for t in node.targets for n in ast.walk(t)):return None
      return self.generic_visit(node)
     def visit_Return(self,node):
      if isinstance(node.value,ast.Tuple) and isinstance(node.value.elts[0],ast.Name) and node.value.elts[0].id=='encoded_frame_feat':
       node.value=node.value.elts[1] if len(node.value.elts)==2 else ast.Tuple(elts=node.value.elts[1:],ctx=ast.Load())
      return node
    tree=FrameRemoval().visit(tree);fn=tree.body[0]
    fn.args.args=[a for a in fn.args.args if a.arg not in ('frame_video_feat','video_mask')]
    fn.args.defaults=[ast.Constant(False),ast.Constant(None)]
    scope=dict(Original.encode_context.__globals__);exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),scope)
    class ClipOnly(Original):
     encode_context=scope['encode_context']
     def get_pred_from_raw_query(self,q,mask,context):return self.get_clip_scale_scores(self.encode_query(q,mask),context)
     def forward(self,batch):
      groups={}
      for i,label in enumerate(batch['text_labels']):groups.setdefault(label,[]).append(i)
      query=self.encode_query(batch['text_feat'],batch['text_mask'])
      means=torch.stack([query[groups[k]].mean(0) for k in sorted(groups)])
      clips,mu,logvar,unused,noise,predicted,reconstructed=self.encode_context(batch['clip_video_features'],train=True,mean_video_queries=means)
      score_query=self.encode_query(batch['text_feat'],batch['text_mask'])
      return dict(clip=self.get_clip_scale_scores(score_query,clips),raw=self.get_unnormalized_clip_scale_scores(score_query,clips),groups=groups,query=query,mu=mu,logvar=logvar,noise=noise,predicted=predicted)
    class ClipLoss(native.OriginalLoss):
     def __init__(self,cfg):super().__init__(cfg);del self.video_nce_criterion
     def forward(self,out,batch):
      labels=batch['text_labels'];groups=out['groups'];f=self.cfg['loss_factor']
      clip_nce=f[0]*self.clip_nce_criterion(labels,groups,out['raw']);clip_trip=self.get_clip_triplet_loss(out['clip'],labels)
      div=f[2]*self.qdl(out['query'],groups);qsp=f[3]*self.qsp(out['query'],labels)
      pvs=f[4]*self.kld_loss(out['mu'],out['logvar']);dre=f[5]*F.mse_loss(out['noise'],out['predicted'])
      self.last=dict(clip_nce=clip_nce,clip_triplet=clip_trip,qdl=div,qsp=qsp,pvs=pvs,dre=dre)
      return (clip_nce+clip_trip)+(div+qsp)+pvs+dre,dre/(f[5]+1e-8)
    def build(cfg):
     model=Original(native.model_config(cfg));model.__class__=ClipOnly
     for name in ('frame_pos_embed','frame_input_proj','frame_encoder'):delattr(model,name)
     return model

    return build,ClipLoss
