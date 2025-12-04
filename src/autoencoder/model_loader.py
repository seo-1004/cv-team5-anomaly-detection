# model_loader.py

from typing import Optional, Tuple
import torch
from .model import UNet


def load_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[UNet, torch.device]:
    """
    저장된 UNet AutoEncoder 가중치를 로드하는 함수.

    Args:
        model_path: .pth 파일 경로
        device: 사용할 디바이스 (None이면 자동 선택)

    Returns:
        model: 로드된 UNet 모델 (eval 모드)
        device: 사용한 디바이스
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=3, n_classes=3).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[load_model] 모델 로드 완료: {model_path}")
    print(f"[load_model] device: {device}")

    return model, device
    
