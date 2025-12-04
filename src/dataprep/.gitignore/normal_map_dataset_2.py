# normal_map_dataset_2.py

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from .transforms_2 import NormalMapToTensor  # 위에서 만든 거 import

class NormalMapDataset(Dataset):
    """
    prefix에 따라 서로 다른 타입의 normal map 로드:
      - prefix='N': 클린 normal map (예: N_*.npy)
      - prefix='A': 애노말리 normal map (예: A_*.npy)
    has_mask=True 이면 동일한 인덱스의 M_*.npy 마스크도 함께 로드.

    반환:
      - has_mask=False:  (normal_tensor, filename)
      - has_mask=True:   (normal_tensor, mask_tensor, filename)
    """

    def __init__(self, data_dir: str, prefix: str = 'N', has_mask: bool = False,
                 size=(256, 256)):
        self.data_dir = data_dir
        self.prefix = prefix
        self.has_mask = has_mask

        pattern = os.path.join(data_dir, f"{prefix}_*.npy")
        self.normal_files = sorted(glob.glob(pattern))

        if not self.normal_files:
            raise RuntimeError(f"'{data_dir}'에서 '{prefix}_*.npy' 파일을 찾을 수 없습니다.")

        if has_mask:
            # A_001.npy -> M_001.npy 이런 식으로 매칭
            self.mask_files = [
                f.replace(f"{prefix}_", "M_") for f in self.normal_files
            ]
            # 존재 여부 살짝 체크
            for m in self.mask_files:
                if not os.path.exists(m):
                    raise RuntimeError(f"마스크 파일이 없습니다: {m}")

        self.normal_transform = NormalMapToTensor(size=size)
        self.mask_size = size  # 마스크도 같은 크기로 리사이즈

    def __len__(self):
        return len(self.normal_files)

    def __getitem__(self, index):
        normal_path = self.normal_files[index]
        normal_np = np.load(normal_path).astype(np.float32)  # [-1,1], (H,W,3)

        normal_tensor = self.normal_transform(normal_np)  # (3,H,W), [-1,1]

        filename = os.path.basename(normal_path)

        if not self.has_mask:
            return normal_tensor, filename

        # 마스크 로드: 0/1 또는 0/255 npy라고 가정
        mask_path = self.mask_files[index]
        mask_np = np.load(mask_path).astype(np.float32)  # (H,W) 또는 (H,W,1)

        if mask_np.ndim == 2:
            mask_np = mask_np[None, ...]  # (1,H,W)
        elif mask_np.ndim == 3 and mask_np.shape[2] == 1:
            mask_np = mask_np.transpose(2,0,1)  # (1,H,W)
        else:
            raise ValueError(f"마스크 shape이 이상함: {mask_np.shape}, 파일: {mask_path}")

        mask_tensor = torch.from_numpy(mask_np)  # (1,H,W)

        # NEAREST로 리사이즈 (마스크는 interpolation 조심)
        mask_tensor = F.resize(
            mask_tensor,
            self.mask_size,
            interpolation=InterpolationMode.NEAREST
        )

        return normal_tensor, mask_tensor, filename
