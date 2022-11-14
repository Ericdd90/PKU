from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

#from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform

#__all__ = ["DepthLSSTransform"]


#@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        model_cfg,
        # in_channels: int,
        # out_channels: int,
        # image_size: Tuple[int, int],
        # feature_size: Tuple[int, int],
        # xbound: Tuple[float, float, float],
        # ybound: Tuple[float, float, float],
        # zbound: Tuple[float, float, float],
        # dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(model_cfg)
        self.model_cfg=model_cfg
        in_channels=self.model_cfg.IN_CHANNELS
        out_channels=self.model_cfg.OUT_CHANNELS
        image_size=self.model_cfg.IMAGE_SIZE
        feature_size=self.model_cfg.FEATURE_SIZE
        xbound=self.model_cfg.XBOUND
        ybound=self.model_cfg.YBOUND
        zbound=self.model_cfg.ZBOUND
        dbound=self.model_cfg.DBOUND
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x