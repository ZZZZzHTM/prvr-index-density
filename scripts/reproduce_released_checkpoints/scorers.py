"""Numerical implementation used by the reported experiment."""
from torch.nn import functional as F


def encode_query(model,method,feat,mask):
    if method=='SDM':
        clip,projected=model.encode_query(feat,mask)
        _,slots,_=model.frame_module.encode_query_by_slot(projected,mask,
                      model.frame_module.query_encoder,model.frame_module.query_pos_embed)
        return clip,slots.permute(1,0,2)
    return model.encode_query(feat,mask),None


def scores(model,method,query,slots,clip,frame,frame_mask):
    if method=='MS-SL':
        c,indices=model.get_clip_scale_scores(query,clip)
        f=ms_frame_score(model,query,clip,frame,frame_mask,indices)
        return c,f
    c=model.get_clip_scale_scores(query,clip)
    if method=='SDM':
        f=model.frame_module.get_scores_batch(slots.permute(1,0,2),frame,frame_mask).mean(0)
    else:f=model.get_clip_scale_scores(query,frame)
    return c,f


def ms_frame_score(model,query,clip,frame,frame_mask,key_indices):
    fv=model.key_clip_guided_attention_in_inference(frame,clip,frame_mask,key_indices)
    return (F.normalize(fv,dim=-1)*F.normalize(query,dim=-1).unsqueeze(0)).sum(-1).T


def raw_scores(model,method,batch,clip,frame,mask):
    if method in ('MS-SL','SDM'):
        return model.get_pred_from_raw_query(batch[0],batch[1],None,clip,frame,mask)
    return model.get_pred_from_raw_query(batch[0],batch[1],None,clip,frame)
