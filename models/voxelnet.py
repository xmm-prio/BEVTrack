import torch
from torch import nn
from torch.nn import functional as F
from mmengine.registry import MODELS
from mmcv.ops import DynamicScatter
from mmcv.cnn import ConvModule
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape

from mmdet3d.models.layers import make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor


@MODELS.register_module()
class VoxelNet(nn.Module):

    def __init__(self, point_cloud_range, voxel_size, grid_size):
        super().__init__()
        self.voxel_layer = VoxelizationByGridShape(
            max_num_points=-1,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)
        self.voxel_encoder = DynamicSimpleVFE(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)
        self.encoder = SparseEncoder(sparse_shape=grid_size)

    @torch.no_grad()
    def voxelize(self, points):
        coors = []
        # dynamic voxelization only provide a coors mapping
        for i, res in enumerate(points):
            res_coors = self.voxel_layer(res)
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            coors.append(res_coors)
        voxels = torch.cat(points, dim=0)
        coors = torch.cat(coors, dim=0)

        return voxels, coors

    def forward(self, x):
        feats, coords = self.voxelize(x)
        voxel_features, coords = self.voxel_encoder(feats, coords)
        batch_size = coords[-1, 0] + 1
        encoder_features = self.encoder(voxel_features, coords, batch_size)

        return encoder_features


class SparseEncoder(nn.Module):

    def __init__(self, sparse_shape):
        super().__init__()
        self.sparse_shape = sparse_shape
        norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
        self.conv1_1 = make_sparse_convmodule(
            3, 16, 3, norm_cfg=norm_cfg, padding=1,
            indice_key='subm1', conv_type='SubMConv3d')
        self.conv1_2 = make_sparse_convmodule(
            16, 16, 3, norm_cfg=norm_cfg, padding=1,
            indice_key='subm1', conv_type='SubMConv3d')
        self.down1 = make_sparse_convmodule(
            16, 32, 3, norm_cfg=norm_cfg, stride=2,
            padding=1, indice_key=f'spconv1', conv_type='SparseConv3d')

        self.conv2_1 = make_sparse_convmodule(
            32, 32, 3, norm_cfg=norm_cfg, padding=1,
            indice_key='subm2', conv_type='SubMConv3d')
        self.conv2_2 = make_sparse_convmodule(
            32, 32, 3, norm_cfg=norm_cfg, padding=1,
            indice_key='subm2', conv_type='SubMConv3d')
        self.down2 = make_sparse_convmodule(
            32, 64, 3, norm_cfg=norm_cfg, stride=2,
            padding=1, indice_key=f'spconv2', conv_type='SparseConv3d')

        self.conv3_1 = make_sparse_convmodule(
            64, 64, 3, norm_cfg=norm_cfg, padding=1,
            indice_key='subm3', conv_type='SubMConv3d')
        self.conv3_2 = make_sparse_convmodule(
            64, 64, 3, norm_cfg=norm_cfg, padding=1,
            indice_key='subm3', conv_type='SubMConv3d')
        self.down3 = make_sparse_convmodule(
            64, 128, 3, norm_cfg=norm_cfg, stride=2,
            padding=1, indice_key=f'spconv3', conv_type='SparseConv3d')

        self.conv4_1 = make_sparse_convmodule(
            128, 128, 3, norm_cfg=norm_cfg, padding=1,
            indice_key='subm4', conv_type='SubMConv3d')
        self.conv4_2 = make_sparse_convmodule(
            128, 128, 3, norm_cfg=norm_cfg, padding=1,
            indice_key='subm4', conv_type='SubMConv3d')

        self.out = ConvModule(128 * 3, 128, 1, norm_cfg=norm_cfg)

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size)

        x = self.conv1_1(input_sp_tensor)
        x = self.conv1_2(x)
        x = self.down1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.down2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.down3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        spatial_features = x.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        spatial_features = self.out(spatial_features)
        return spatial_features


class DynamicSimpleVFE(nn.Module):
    """Simple dynamic voxel feature encoder used in DV-SECOND.

    It simply averages the values of points in a voxel.
    But the number of points in a voxel is dynamic and varies.

    Args:
        voxel_size (tupe[float]): Size of a single voxel
        point_cloud_range (tuple[float]): Range of the point cloud and voxels
    """

    def __init__(self,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1)):
        super(DynamicSimpleVFE, self).__init__()
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)
        self.fp16_enabled = False

    @torch.no_grad()
    def forward(self, features, coors, *args, **kwargs):
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, 3(4)). N is the number of points.
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (M, 3(4)).
                M is the number of voxels.
        """
        # This function is used from the start of the voxelnet
        # num_points: [concated_num_points]
        features, features_coors = self.scatter(features, coors)
        return features, features_coors
