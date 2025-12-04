"""
diagnostics.py

Photometric Stereo 결과 진단(diagonostics)을 위한 함수 모음.
- 노멀맵 간 평균 각도 오차 계산
- 여러 노멀맵 쌍 간 통계 출력
"""

import numpy as np
import itertools


def mean_angle_deg(Na, Nb, Ma, Mb):
    """
    두 노멀맵 Na, Nb 간의 평균 각도(도 단위)를 계산.

    Parameters
    ----------
    Na, Nb : (H,W,3) float
        비교할 두 노멀맵.
    Ma, Mb : (H,W) bool
        각 노멀맵에서 유효한 픽셀 마스크.

    Returns
    -------
    float
        유효한 공통 영역(Ma & Mb)에 대해 계산된 평균 각도.
        공통 영역이 없으면 180.0을 반환.

    Notes
    -----
    - 각 노멀은 단위 벡터로 정규화 후 내적 기반 각도 계산.
    - arccos(dot)을 이용해 각도를 구하고 deg 단위로 변환.
    """
    M = Ma & Mb
    if M.sum() == 0:
        return 180.0
    eps = 1e-8
    Na = Na / np.maximum(np.linalg.norm(Na, axis=-1, keepdims=True), eps)
    Nb = Nb / np.maximum(np.linalg.norm(Nb, axis=-1, keepdims=True), eps)
    dot = np.clip((Na * Nb).sum(-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    return float(ang[M].mean())


def diag_pairwise_angles(normal_maps, masks, sample=None):
    """
    여러 노멀맵(normal_maps) 간의 pairwise 평균 각도 통계 출력.

    Parameters
    ----------
    normal_maps : list[(H,W,3)]
        비교할 노멀맵 리스트
    masks : list[(H,W) bool]
        각 노멀맵의 유효 픽셀 마스크
    sample : int or None
        앞 sample개만 사용해 짝 조합 계산. None이면 전체 사용.

    Notes
    -----
    - itertools.combinations를 사용해 모든 쌍(i,j)에 대해 mean_angle_deg 계산.
    - 계산된 각도들에 대해 min / median / mean을 출력.
    """
    K = len(normal_maps)
    idxs = list(range(K)) if sample is None else list(range(min(sample, K)))
    vals = []
    for i, j in itertools.combinations(idxs, 2):
        vals.append(mean_angle_deg(normal_maps[i], normal_maps[j],
                                   masks[i], masks[j]))
    if not vals:
        return
    arr = np.array(vals)
    print(f"[Diag] angle deg -> min: {arr.min():.3f}, "
          f"median: {np.median(arr):.3f}, mean: {arr.mean():.3f}")
    
