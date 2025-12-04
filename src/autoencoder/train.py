# train.py
import time
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from .model import UNet


def train_autoencoder(
    dataset,
    learning_rate: float = 5e-4,
    batch_size: int = 8,
    epochs: int = 100,
    val_split: float = 0.2,
    num_workers: int = 2,
    model_save_path: str = "best_model.pth",
    loss_fn: str = "mse",
    device: Optional[torch.device] = None,
) -> Tuple[UNet, torch.device, float]:
    """
    UNet 기반 AutoEncoder 학습 함수.

    Args:
        data_dir: 'N_*.npy'가 들어있는 폴더 경로
        learning_rate: 학습률
        batch_size: 배치 크기
        epochs: 에폭 수
        val_split: 검증 데이터 비율 (0~1)
        num_workers: DataLoader num_workers
        model_save_path: 베스트 모델 저장 경로
        loss_fn: "mse" 또는 "l1"
        device: torch.device (None이면 자동 선택)

    Returns:
        model: 학습이 끝난 UNet 모델 (best val 기준 가중치 로드됨)
        device: 사용한 디바이스
        best_val_loss: 최저 검증 손실 값
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_autoencoder] Using device: {device}")

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"[Dataset] total={total_size}, train={train_size}, val={val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Model, optimizer, criterion
    model = UNet(n_channels=3, n_classes=3).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    print("optimizer] AdamW 사용 (lr={LEARNING_RATE})")
    
    criterion = nn.MSELoss()
    print("[Loss] MSELoss 사용")

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()

        # ---- Train ----
        model.train()
        train_loss = 0.0

        for images in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            reconstructed = model(images)

            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / train_size

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item() * images.size(0)

        avg_val_loss = val_loss / val_size
        end_time = time.time()

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Time: {end_time - start_time:.1f}s | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}",
            end="",
        )

        # Best model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print("  (Val loss 개선! -> 모델 저장)")
        else:
            print("")

    print("\n[train_autoencoder] 학습 완료")
    print(f"최저 검증 손실: {best_val_loss:.6f}")
    print(f"베스트 모델 경로: {model_save_path}")

    # 베스트 모델 다시 로드해서 반환
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    return model, device, best_val_loss
