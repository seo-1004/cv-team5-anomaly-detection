# transforms_2.py

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

class NormalMapToTensor(nn.Module):
    def __init__(self, size=(256,256)):
        super().__init__()
        self.transform = T.Compose([
            T.ToTensor(),                         # uint8 → [0,1]
            T.Resize(size, antialias=True),
            T.Normalize(mean=(0.5,0.5,0.5),
                        std=(0.5,0.5,0.5)),       # [0,1] → [-1,1]
        ])

    def forward(self, normal_np: np.ndarray) -> torch.Tensor:
        # normal_np: float32, [-1,1], (H,W,3)
        normal_uint8 = ((normal_np + 1.0) / 2.0 * 255).astype(np.uint8)
        return self.transform(normal_uint8)
