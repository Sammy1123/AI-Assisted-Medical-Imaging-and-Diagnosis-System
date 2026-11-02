"""
Transfer-learning classification model (DenseNet-121).
Binary output: probability of pneumonia.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights


def build_model(pretrained: bool = True) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    # Replace classifier head for binary output
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )
    return model


def load_model(weights_path: str, device: str = "cpu") -> nn.Module:
    model = build_model(pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
