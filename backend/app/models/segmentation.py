"""
3D U-Net segmentation model using MONAI.
Used for volumetric CT/MRI segmentation (Medical Segmentation Decathlon tasks).
"""
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm


def build_segmentation_model(
    in_channels: int = 1,
    out_channels: int = 2,
    device: str = "cpu",
) -> nn.Module:
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    return model


def load_segmentation_model(weights_path: str, device: str = "cpu") -> nn.Module:
    model = build_segmentation_model(device=device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
