import torch
import torch.nn as nn
from mmengine.model import BaseModel
from datasets.metrics import estimateOverlap, estimateAccuracy
import numpy as np
from datasets import points_utils
from mmengine.registry import MODELS


@MODELS.register_module()
class BEVTrack(BaseModel):

    def __init__(self,
                 backbone=None,
                 fuser=None,
                 head=None,
                 cfg=None):
        super().__init__()
        self.config = cfg
        self.backbone = MODELS.build(backbone)
        self.fuse = MODELS.build(fuser)
        self.head = MODELS.build(head)

    def forward(self,
                inputs,
                data_samples=None,
                mode: str = 'predict',
                **kwargs):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def get_feats(self, inputs):
        prev_points = inputs['prev_points']
        this_points = inputs['this_points']
        stack_points = prev_points + this_points

        stack_feats = self.backbone(stack_points)
        cat_feats = self.fuse(stack_feats)
        results = self.head(cat_feats)

        return results

    def inference(self, inputs):
        results = self.get_feats(inputs)
        coors = results['coors'][0]

        return coors.detach().cpu().numpy()

    def loss(self, inputs, data_samples):
        results = self.get_feats(inputs)
        losses = dict()
        losses.update(self.head.loss(results, data_samples))

        return losses

    def predict(self, inputs):
        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(inputs)):  # tracklet
            this_bb = inputs[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
            else:
                data_dict, ref_bb = self.build_input_dict(inputs, frame_id, results_bbs)
                coors = self.inference(data_dict)
                candidate_box = points_utils.getOffsetBB(
                    ref_bb, [coors[0], coors[1], coors[2], coors[3]],
                    degrees=True, use_z=True, limit_box=False)
                results_bbs.append(candidate_box)
            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
            ious.append(this_overlap)
            distances.append(this_accuracy)

        return ious, distances

    def build_input_dict(self, sequence, frame_id, results_bbs):
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        prev_frame = sequence[frame_id - 1]
        this_frame = sequence[frame_id]

        prev_pc = prev_frame['pc']
        this_pc = this_frame['pc']
        ref_box = results_bbs[-1]

        prev_frame_pc = points_utils.crop_pc_in_range(prev_pc, ref_box, self.config.point_cloud_range)
        this_frame_pc = points_utils.crop_pc_in_range(this_pc, ref_box, self.config.point_cloud_range)

        prev_points = prev_frame_pc.points.T
        this_points = this_frame_pc.points.T

        if prev_points.shape[0] < 1:
            prev_points = np.zeros((1, 3), dtype='float32')
        if this_points.shape[0] < 1:
            this_points = np.zeros((1, 3), dtype='float32')

        data_dict = {'prev_points': [torch.as_tensor(prev_points, dtype=torch.float32).cuda()],
                     'this_points': [torch.as_tensor(this_points, dtype=torch.float32).cuda()]}

        return data_dict, results_bbs[-1]
