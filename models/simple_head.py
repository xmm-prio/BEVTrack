import torch
import torch.nn as nn
from .rle_loss import RLELoss
from mmengine.registry import MODELS


@MODELS.register_module()
class SimpleHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.SyncBatchNorm(256, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Linear(256, 128, bias=False),
            nn.SyncBatchNorm(128, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Linear(128, 8)
        )
        self.criterion = RLELoss()

    def forward(self, feats):
        res = self.regression_head(feats)
        results = {
            'coors': res[:, :4],
            'sigma': res[:, 4:],
        }
        return results

    def loss(self, results, data_samples):
        losses = dict()
        pred_coors = results['coors']
        sigma = results['sigma']
        gt_coors = torch.stack(data_samples['box_label'])
        losses['regression_loss'] = self.criterion(pred_coors, sigma, gt_coors)

        return losses
