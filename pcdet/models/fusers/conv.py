from typing import List

import torch
from torch import nn

#from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]


#@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, model_cfg) -> None:
        self.model_cfg = model_cfg
        in_channels = model_cfg.IN_CHANNELS
        out_channels = model_cfg.OUT_CHANNELS
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))
