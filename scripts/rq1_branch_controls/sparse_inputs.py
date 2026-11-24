"""Numerical implementation used by the reported experiment."""
from Datasets.data_provider import Dataset4PRVR
from Datasets.data_provider import VisDataSet4PRVR
from Datasets.data_provider import average_to_fixed_length
from Datasets.data_provider import collate_frame_val
from Datasets.data_provider import collate_train
import h5py
from Datasets.data_provider import l2_normalize_np_array
import numpy as np
import torch


def center_indices(length):
    assert length>0
    return np.minimum(((2*np.arange(32,dtype=np.int64)+1)*length)//64,length-1)


def center_frame_sampling(features):
    assert features.ndim==2 and len(features)>0
    return features[center_indices(len(features))]


def video_features(dataset,video):
    raw=np.array([dataset.visual_feat.read_one(key) for key in dataset.video2frames[video]])
    clip=torch.from_numpy(l2_normalize_np_array(average_to_fixed_length(raw,32))).unsqueeze(0)
    frames=torch.from_numpy(l2_normalize_np_array(center_frame_sampling(raw)))
    return clip,frames


class CenterTrain(Dataset4PRVR):
    def __getitem__(self,index):
        if not self.open_file:self.text_feat=h5py.File(self.text_feat_path,'r');self.open_file=True
        video=self.video_ids[index];ids=self.vid_caps[video]
        clip,frames=video_features(self,video)
        texts=[torch.from_numpy(l2_normalize_np_array(self.text_feat[key][...]))[:self.max_desc_len] for key in ids]
        return clip,frames,texts,index,ids,video


class CenterContext(VisDataSet4PRVR):
    def __getitem__(self,index):
        video=self.video_ids[index];clip,frames=video_features(self,video)
        return clip,frames,index,video


def sparse_train_collate(examples):
    out=collate_train(examples)
    assert out['frame_video_features'].shape[1]==32
    out['videos_mask']=None
    return out


def sparse_context_collate(examples):
    clip,frames,mask,indices,ids=collate_frame_val(examples)
    assert frames.shape[1]==32
    return clip,frames,None,indices,ids
