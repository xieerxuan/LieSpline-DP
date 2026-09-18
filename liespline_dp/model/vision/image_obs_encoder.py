"""Image features shared by simulation and hardware policies.

Each camera has an independent ResNet-18, 32 spatial keypoints and a 64-D
projection. Low-dimensional observations pass through unchanged. These are
the original architecture and feature order, with no environment dependency.
"""

import math

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from liespline_dp.common.pytorch_util import replace_submodules
from liespline_dp.model.vision.crop_randomizer import (
    CropRandomizer,
    sample_random_image_crops,
)


class SpatialSoftmax(nn.Module):
    """Return expected (x, y) coordinates for each learned feature map."""

    def __init__(self, height, width, num_keypoints=32):
        super().__init__()
        self.keypoints = nn.Conv2d(512, num_keypoints, kernel_size=1)
        x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        self.register_buffer("x", torch.from_numpy(x.reshape(1, -1)).float())
        self.register_buffer("y", torch.from_numpy(y.reshape(1, -1)).float())

    def forward(self, feature):
        scores = self.keypoints(feature).flatten(start_dim=2)
        # Spatial probabilities retain geometric information that global average
        # pooling would discard. Softmax temperature stays fixed at 1.
        weights = torch.softmax(scores, dim=-1)
        return torch.stack(
            ((weights * self.x).sum(-1), (weights * self.y).sum(-1)), dim=-1
        ).flatten(start_dim=1)


class ImageObsEncoder(nn.Module):
    def __init__(self, shape_meta, crop_shape=(76, 76), group_norm=False, eval_fixed_crop=False):
        super().__init__()
        self.obs_shapes = {key: tuple(spec["shape"]) for key, spec in shape_meta["obs"].items()}
        self.encoders = nn.ModuleDict()
        self.crops = nn.ModuleDict()
        self.eval_fixed_crop = bool(eval_fixed_crop)
        feature_dim = 0
        for key, spec in shape_meta["obs"].items():
            shape = self.obs_shapes[key]
            modality = spec.get("type", "low_dim")
            if modality == "low_dim":
                feature_dim += math.prod(shape)
                continue
            if modality != "rgb" or len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"{key}: expected low_dim or RGB shape [3, H, W]")
            height, width = shape[1:]
            if crop_shape is not None:
                height, width = map(int, crop_shape)
                self.crops[key] = CropRandomizer(shape, height, width)
            backbone = nn.Sequential(*list(resnet18(weights=None).children())[:-2])
            if group_norm:
                replace_submodules(
                    backbone,
                    predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                    func=lambda module: nn.GroupNorm(
                        module.num_features // 16, module.num_features
                    ),
                )
            self.encoders[key] = nn.Sequential(
                backbone,
                SpatialSoftmax(math.ceil(height / 32), math.ceil(width / 32)),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            feature_dim += 64
        self.feature_dim = feature_dim

    def output_shape(self):
        return [self.feature_dim]

    def forward(self, observations):
        features = []
        for key in self.obs_shapes:
            value = observations[key]
            if key in self.crops:
                crop = self.crops[key]
                # Preserve the optional random evaluation crop. Published task
                # configurations use a deterministic center crop.
                if not self.training and not self.eval_fixed_crop:
                    value, _ = sample_random_image_crops(
                        value, crop.crop_height, crop.crop_width, num_crops=1
                    )
                    value = value.flatten(0, 1)
                else:
                    value = crop(value)
            if key in self.encoders:
                value = self.encoders[key](value)
            features.append(value.flatten(start_dim=1))
        return torch.cat(features, dim=-1)


def get_image_obs_encoder(
    shape_meta,
    crop_shape=(76, 76),
    obs_encoder_group_norm=False,
    eval_fixed_crop=False,
):
    return ImageObsEncoder(shape_meta, crop_shape, obs_encoder_group_norm, eval_fixed_crop)
