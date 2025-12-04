# evaluator.py

import os
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.utils.image_io import save_image_from_normalized, denorm_to_uint8


def evaluate_model(model, dataset, device, max_vis_samples=5, save_dir=None):
    """
    모델 평가 루프
    - pixel-level AUROC 계산
    - Error map 생성
    - 시각화 및 저장용 vis_data 반환
    - save_dir: 결과 이미지 저장 디렉토리 (None이면 저장 X)
    """

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_true = []
    all_pred = []
    vis_data = []

    # 저장 폴더 준비
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "input"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "recon"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "heatmap"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "mask"), exist_ok=True)

    model.eval()

    with torch.no_grad():
        for i, (input_tensor, mask_tensor, filename) in enumerate(loader):

            input_tensor = input_tensor.to(device)
            recon_tensor = model(input_tensor)

            # Convert tensors to numpy
            input_np = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()   # [-1,1]
            recon_np = recon_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()   # [-1,1]

            mask_np = mask_tensor.squeeze().cpu().numpy()
            mask_np = (mask_np > (mask_np.max() / 2.0)).astype(np.uint8)

            # Error map
            error_map_rgb = np.abs(input_np - recon_np)
            heatmap = error_map_rgb.mean(axis=2)    # [H,W], float

            # Flatten for AUROC
            all_true.append(mask_np.flatten())
            all_pred.append(heatmap.flatten())

            # 파일명 정규화
            fname = filename[0] if isinstance(filename[0], str) else f"sample_{i}"
            fname = os.path.splitext(fname)[0]

            # ----------- 결과 저장 ----------- #
            if save_dir is not None:

                # 1) 입력 이미지 저장
                save_image_from_normalized(
                    input_np,
                    os.path.join(save_dir, "input", f"{fname}.png")
                )

                # 2) 재구성 이미지 저장
                save_image_from_normalized(
                    recon_np,
                    os.path.join(save_dir, "recon", f"{fname}.png")
                )

                # 3) heatmap 저장 (시각화용 per-image normalization + Jet colormap)
                heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                heatmap_uint8 = heatmap_norm.astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                cv2.imwrite(
                    os.path.join(save_dir, "heatmap", f"{fname}.png"),
                    heatmap_color
                )

                # 4) mask 저장 (0/1 → 0/255)
                mask_uint8 = (mask_np * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(save_dir, "mask", f"{fname}.png"),
                    mask_uint8
                )

            # -------------------------------- #

            # Save for visualization (max N)
            if i < max_vis_samples:
                vis_data.append({
                    "filename": fname,
                    "input": input_np,
                    "recon": recon_np,
                    "heatmap": heatmap,
                    "mask": mask_np
                })
                
    if save_dir is not None:
        print("모든 결과 저장 완료!")

    # Merge flat lists
    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_pred)

    # Ensure binary mask
    y_true = (y_true > 0.5).astype(int)

    # AUROC
    auroc = roc_auc_score(y_true, y_score)

    return auroc, vis_data