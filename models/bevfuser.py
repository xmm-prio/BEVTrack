import torch
from torch import nn
from mmengine.registry import MODELS


@MODELS.register_module()
class BEVFuser(nn.Sequential):

    def __init__(self):
        super().__init__(
            ConvReluBN(256, 256),
            ConvReluBN(256, 256),
            ConvReluBN(256, 512, 2),
            ConvReluBN(512, 512),
            ConvReluBN(512, 512),
            ConvReluBN(512, 512, 2),
            ConvReluBN(512, 512),
            ConvReluBN(512, 512),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )

    def forward(self, stack_feats):
        cat_feats = torch.cat(stack_feats.chunk(2, 0), 1)
        return super().forward(cat_feats)


class ConvReluBN(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False, groups=groups),
            nn.SyncBatchNorm(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(True)
        )

    def forward(self, x):
        return super().forward(x)
