# transforms.py

import torch
import torchvision.transforms as T

class NormalMapToTensor:
    """
    [-1,1] 범위 normal map (H,W,3) np.ndarray -> (3,H,W) tensor 로 변환 + Resize
    """
    def __init__(self, size=(256, 256)):
        # Resize는 tensor도 지원함
        self.resize = T.Resize(size, antialias=True)

    def __call__(self, normal_np: "np.ndarray[H,W,3]"):
        # np.float32 [-1,1] -> torch.float32 [-1,1], (C,H,W)
        tensor = torch.from_numpy(normal_np).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        tensor = self.resize(tensor)
        return tensor
