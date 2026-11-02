"""
Grad-CAM implementation for DenseNet-121.
Returns a normalised heatmap overlaid on the original image.
"""
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str = "features.denseblock4"):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks(target_layer)

    def _register_hooks(self, layer_name: str):
        layer = dict(self.model.named_modules())[layer_name]

        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Return a (H, W) heatmap in [0, 1]. Also stores last_prob."""
        self.model.zero_grad()
        output = self.model(input_tensor)          # (1, 1)
        self.last_prob: float = output.item()
        output.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(original_pil: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Blend Grad-CAM heatmap onto the original PIL image."""
    w, h = original_pil.size
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig_np = np.array(original_pil.convert("RGB"))
    blended = (alpha * heatmap + (1 - alpha) * orig_np).astype(np.uint8)
    return Image.fromarray(blended)
