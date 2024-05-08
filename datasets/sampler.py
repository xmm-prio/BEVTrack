import numpy as np
import torch
from nuscenes.utils import geometry_utils
from torch.utils.data import Dataset
from . import points_utils
from mmengine.registry import DATASETS


@DATASETS.register_module()
class TestSampler(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = DATASETS.build(dataset)

    def __len__(self):
        return self.dataset.get_num_tracklets()

    def __getitem__(self, index):
        tracklet_annos = self.dataset.tracklet_anno_list[index]
        frame_ids = list(range(len(tracklet_annos)))
        return self.dataset.get_frames(index, frame_ids)
