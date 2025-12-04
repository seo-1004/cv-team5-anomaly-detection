# generate_normals.py

import os
import numpy as np

from .photometric_stereo.preprocess import (
    load_meta,
    load_images_gray_linear,
    adaptive_thresholds,
    make_valid_mask_per_light,
)
from .photometric_stereo.fps_solver import (
    make_subsets,
    irls_ps_subset,
)
from .photometric_stereo.diagnostics import mean_angle_deg, diag_pairwise_angles
from src.utils.image_io import save_image_from_normalized

def generate_normals_for_objects(base_dir, out_path, obj_list):
    os.makedirs(out_path, exist_ok=True)
    global_idx = 1

    for obj_name in obj_list:
        obj_dir = os.path.join(base_dir, obj_name)
        print(f"\n[RUN] {obj_name}")

        L, Iw, names, obj_mask = load_meta(obj_dir)
        imgs = load_images_gray_linear(obj_dir, names, Iw)
        N, H, W = imgs.shape

        # pot1 / pot2 처리
        if obj_name in ['pot1PNG', 'pot2PNG']:
            p95_per = np.array([
                np.percentile(imgs[i][obj_mask], 95)
                for i in range(imgs.shape[0])
            ])
            g_per = 0.40 / np.maximum(p95_per, 1e-8)
            imgs = np.clip(imgs * g_per[:, None, None], 0, 1)

        if obj_name in ['pot1PNG', 'pot2PNG']:
            low_p, high_p = 0.1, 99.9
        else:
            low_p, high_p = 1.0, 99.5

        t_shadow, t_sat = adaptive_thresholds(imgs, obj_mask, low_p, high_p)
        valid_perlight = make_valid_mask_per_light(
            imgs, obj_mask, t_shadow, t_sat, morph_ksize=3
        )
        subsets = make_subsets(L, m=18, stride=6)

        normal_maps, confidences, masks, meta = [], [], [], []

        for si, idx in enumerate(subsets):
            I_sub = imgs[idx]
            V_sub = valid_perlight[idx]
            L_sub = L[idx]
            N_k, C_k, M_k = irls_ps_subset(I_sub, L_sub, V_sub, obj_mask)
            normal_maps.append(N_k)
            confidences.append(C_k)
            masks.append(M_k)
            meta.append({"set_id": si, "light_indices": idx})

        diag_pairwise_angles(normal_maps, masks)

        keep, th_deg = [], 2.5
        for i, Ni in enumerate(normal_maps):
            dup = False
            for j in keep:
                ang = mean_angle_deg(Ni, normal_maps[j],
                                     masks[i], masks[j])
                if ang < th_deg:
                    dup = True
                    break
            if not dup:
                keep.append(i)
        print(f"[Dedup] before: {len(normal_maps)} -> after: {len(keep)}")

        for rank, i in enumerate(keep):
            np.save(os.path.join(out_path, f'N_{global_idx:03d}.npy'),
                    normal_maps[i])
            np.save(os.path.join(out_path, f'C_{global_idx:03d}.npy'),
                    confidences[i])
            save_image_from_normalized(normal_maps[i],
                            os.path.join(out_path,
                                         f'N_{global_idx:03d}.png'))
            global_idx += 1

    print(f"\n[Done] 총 {global_idx-1}개의 normal map 저장 완료.")
