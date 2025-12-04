"""
preprocess.py


Photometric Stereo 전처리용 유틸리티 모듈.
각 함수는 이미지 로딩, sRGB → Linear 변환, 메타데이터 로딩,
임계값 계산, 유효 픽셀 마스크 생성 등의 역할을 수행한다.
"""

import numpy as np
import cv2
import os

def srgb_to_linear(img01):
    """
    sRGB 이미지(0~1 범위)를 Linear 색공간으로 변환.
    sRGB 감마 커브를 적용해 비선형 값을 선형 조도값으로 변환한다.
    """
    img01 = img01.astype(np.float32)
    lin = np.where(img01 <= 0.04045,
                   img01 / 12.92,
                   ((img01 + 0.055) / 1.055) ** 2.4)
    return lin.astype(np.float32)

def load_meta(obj_dir):
    """
    photometric stereo에 필요한 메타데이터 로드.
    - light_directions.txt: 광원 방향 벡터
    - light_intensities.txt: 광원 세기
    - filenames.txt: 촬영된 이미지 파일 이름 목록
    - mask.png: 객체 마스크


    또한 광원 방향을 정규화하고, RGB 강도 입력 시 그레이스케일 강도로 변환한다.
    """
    L = np.loadtxt(os.path.join(obj_dir, 'light_directions.txt')).astype(np.float32)
    Iw = np.loadtxt(os.path.join(obj_dir, 'light_intensities.txt')).astype(np.float32)
    with open(os.path.join(obj_dir, 'filenames.txt'), 'r') as f:
        names = [ln.strip() for ln in f if ln.strip()]
    mask = cv2.imread(os.path.join(obj_dir, 'mask.png'),
                      cv2.IMREAD_GRAYSCALE) > 0

    if Iw.ndim == 2 and Iw.shape[1] == 3:
        Iw = (0.2989 * Iw[:, 0] +
              0.5870 * Iw[:, 1] +
              0.1140 * Iw[:, 2]).astype(np.float32)

    L = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-8)
    return L, Iw, names, mask

def load_images_gray_linear(obj_dir, names, intensities):
    """
    이미지 파일들을 불러와 다음을 수행:
    1) 그레이스케일 로드
    2) 0~1 정규화
    3) sRGB → Linear 변환
    4) 광원 세기로 intensity 보정


    출력 형식: (N, H, W)
    """
    imgs = []
    for i, nm in enumerate(names):
        im = cv2.imread(os.path.join(obj_dir, nm),
                        cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        im = srgb_to_linear(im)
        im = im / max(float(intensities[i]), 1e-8)
        imgs.append(im)
    return np.stack(imgs, axis=0).astype(np.float32)   # (N,H,W)

def adaptive_thresholds(imgs, obj_mask, low_p, high_p,
                        min_gap=1e-3, clip_lo=1e-6, clip_hi=0.999):
    """
    이미지 픽셀 분포 기반 자동 임계값 계산.
    - low_p 백분위수 → 그림자 임계값(t_shadow)
    - high_p 백분위수 → 포화 임계값(t_sat)
    객체 영역(obj_mask) 내 픽셀만 사용해 안정적인 임계값을 산출한다.
    """
    x = imgs
    x = imgs[:, obj_mask]
    p_low, p_high = np.percentile(x, [low_p, high_p])
    t_shadow = float(np.clip(p_low,  clip_lo, clip_hi - min_gap))
    t_sat    = float(np.clip(p_high, t_shadow + min_gap, clip_hi))
    return t_shadow, t_sat

def make_valid_mask_per_light(imgs, obj_mask,
                              t_shadow=0.008, t_sat=0.98, morph_ksize=3):
    """
    각 광원 이미지에서 유효 픽셀(valid pixel) 마스크 생성.
    - 그림자(t_shadow 이하), 포화(t_sat 이상) 제외
    - 객체 영역(obj_mask) 내부 픽셀만 사용
    - 잡음을 제거하기 위해 morphological opening 적용


    반환값: (N, H, W) Boolean mask
    """
    valid = (imgs > t_shadow) & (imgs < t_sat) & obj_mask[None, :, :]

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (morph_ksize, morph_ksize))
    out = np.empty_like(valid)
    for i in range(imgs.shape[0]):
        v = (valid[i].astype(np.uint8) * 255)
        v = cv2.morphologyEx(v, cv2.MORPH_OPEN, k)
        out[i] = (v > 0)
    return out
