import numpy as np
import torch
from nuscenes.utils import geometry_utils
from torch.utils.data import Dataset
from . import points_utils
from mmengine.registry import DATASETS


@DATASETS.register_module()
class TrainSampler(torch.utils.data.Dataset):

    def __init__(self, dataset=None, cfg=None):
        super().__init__()
        self.config = cfg
        self.dataset = DATASETS.build(dataset)
        self.num_candidates = cfg.num_candidates
        num_frames_total = 0
        self.tracklet_start_ids = [num_frames_total]
        for i in range(self.dataset.get_num_tracklets()):
            num_frames_total += self.dataset.get_num_frames_tracklet(i)
            self.tracklet_start_ids.append(num_frames_total)

    @staticmethod
    def processing(data, config):
        prev_frame = data['prev_frame']
        this_frame = data['this_frame']
        candidate_id = data['candidate_id']
        prev_pc, prev_box = prev_frame['pc'], prev_frame['3d_bbox']
        this_pc, this_box = this_frame['pc'], this_frame['3d_bbox']

        if config.target_thr is not None:
            num_points_in_prev_box = geometry_utils.points_in_box(prev_box, prev_pc.points).sum()
            assert num_points_in_prev_box > config.target_thr, 'not enough target points'

        if candidate_id == 0:
            bbox_offset = np.zeros(4)
        else:
            bbox_offset = np.random.uniform(low=-0.3, high=0.3, size=4)
            bbox_offset[3] = bbox_offset[3] * 20.

        ref_box = points_utils.getOffsetBB(
            prev_box, bbox_offset, limit_box=False, use_z=True, degrees=True)
        prev_frame_pc = points_utils.crop_pc_in_range(prev_pc, ref_box, config.point_cloud_range)

        if candidate_id == 0:
            bbox_offset = np.zeros(4)
        else:
            bbox_offset = np.random.randn(3)
            bbox_offset[0] *= 0.3
            bbox_offset[1] *= 0.2
            bbox_offset[2] *= 0.1

        base_box = points_utils.getOffsetBB(ref_box, bbox_offset, limit_box=False, use_z=True, degrees=True)
        this_frame_pc = points_utils.crop_pc_in_range(this_pc, base_box, config.point_cloud_range)
        if config.search_thr is not None:
            assert this_frame_pc.nbr_points() > config.search_thr, 'not enough search points'

        this_box = points_utils.transform_box(this_box, base_box)

        prev_points = prev_frame_pc.points.T
        this_points = this_frame_pc.points.T

        if prev_points.shape[0] < 1:
            prev_points = np.zeros((1, 3), dtype='float32')
        if this_points.shape[0] < 1:
            this_points = np.zeros((1, 3), dtype='float32')

        if config.flip:
            prev_points, this_points, this_box = \
                points_utils.flip_augmentation(prev_points, this_points, this_box)

        box_label = this_box.center
        degrees = this_box.orientation.degrees * this_box.orientation.axis[-1]

        inputs = {'prev_points': torch.as_tensor(prev_points, dtype=torch.float32),
                  'this_points': torch.as_tensor(this_points, dtype=torch.float32),
                  'wlh': torch.as_tensor(this_box.wlh, dtype=torch.float32)}
        data_samples = {
            # 'box_label': torch.as_tensor(box_label, dtype=torch.float32),
            # 'degrees': torch.as_tensor(degrees, dtype=torch.float32),
            'box_label': torch.as_tensor([box_label[0], box_label[1], box_label[2], degrees], dtype=torch.float32),
        }
        data_dict = {
            'inputs': inputs,
            'data_samples': data_samples,
        }

        return data_dict

    def get_anno_index(self, index):
        return index // self.num_candidates

    def get_candidate_index(self, index):
        return index % self.num_candidates

    def __len__(self):
        return self.dataset.get_num_frames_total() * self.num_candidates

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:
            for i in range(0, self.dataset.get_num_tracklets()):
                if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                    tracklet_id = i
                    this_frame_id = anno_id - self.tracklet_start_ids[i]
                    prev_frame_id = max(this_frame_id - 1, 0)
                    if self.config.time_flip:
                        if np.random.rand() < 0.5:
                            frame_ids = (prev_frame_id, this_frame_id)
                        else:
                            frame_ids = (this_frame_id, prev_frame_id)
                    else:
                        frame_ids = (prev_frame_id, this_frame_id)
            prev_frame, this_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            data = {
                "prev_frame": prev_frame,
                "this_frame": this_frame,
                "candidate_id": candidate_id}
            return self.processing(data, self.config)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]


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
