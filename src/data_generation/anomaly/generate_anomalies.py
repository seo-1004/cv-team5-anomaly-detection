# generate_anomalies.py

import os
import glob
import numpy as np
import cv2
import random

from .mapping import index_ranges
from .defect_generator import generate_defects
from src.utils.image_io import save_image_from_normalized

def generate_anomalies_for_objects(normal_dir, diligent_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    normal_files = sorted(glob.glob(os.path.join(normal_dir, "N_*.npy")))
    print("총 Normal Map 개수:", len(normal_files))

    counter = 1
    random.seed(42)

    for obj_name, (start, end) in index_ranges.items():

        obj_path = os.path.join(diligent_dir, obj_name)
        mask_path = os.path.join(obj_path, "mask.png")
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask_img is None:
            print(f"[WARN] {obj_name}의 mask.png 없음")
            continue

        obj_mask = mask_img > 0

        obj_normals = normal_files[start:end]
        print(f"{obj_name}: {len(obj_normals)}개 normal 사용")

        # 이 객체에서 3장만 랜덤 선택
        target_list = random.sample(obj_normals, k=min(3, len(obj_normals)))
        print(f"  ↳ 이 중 {len(target_list)}장에 결함 생성")
        
        for npath in target_list:
            normal = np.load(npath)

            defect, mask = generate_defects(normal, obj_mask,
                                            num_scratch=2, num_dent=1)

            a_name = f"A_{counter:03d}"
            m_name = f"M_{counter:03d}"
            counter += 1

            np.save(os.path.join(out_dir, a_name + ".npy"), defect)
            np.save(os.path.join(out_dir, m_name + ".npy"), mask)

            # PNG 저장
            save_image_from_normalized(defect, os.path.join(out_dir, a_name + ".png"))
            cv2.imwrite(os.path.join(out_dir, m_name + ".png"), mask)
