# inference.py

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

# 단일 이미지 복원을 위한 추론 함수
def reconstruct_normal(
    model: UNet,
    input_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    입력 Normal Map(또는 이상 이미지)의 복원 결과를 얻는 함수.

    Args:
        model: 학습된 UNet AutoEncoder
        input_tensor: (B, 3, 256, 256), 값 범위 [-1, 1]
        device: 사용할 디바이스 (None이면 model이 올라가 있는 device 사용 시도)

    Returns:
        output_tensor: (B, 3, 256, 256), 값 범위 [-1, 1]
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)

    return output
