"""
fps_solver.py

Photometric Stereo(FPS 기반 서브셋 선택 + IRLS-Huber PS) 관련 함수 모음.
각 함수는 조명 방향 기반 샘플링, 서브셋 생성, 유효 마스크 계산,
그리고 IRLS(Iteratively Reweighted Least Squares)를 이용한
픽셀별 노멀 복원 작업을 수행한다.
"""

import numpy as np
from .preprocess import make_valid_mask_per_light, adaptive_thresholds
from .diagnostics import mean_angle_deg   # 아래에서 정의 (필요 시 분리)


def farthest_point_order(L):
    """
    조명 벡터 L(m,3)에 대해 farthest point sampling 순서를 생성.

    - 첫 점은 z축 성분이 가장 큰 조명 선택.
    - 이후 이미 선택된 조명과의 코사인 거리를 기준으로 가장 먼 조명순으로 추가.
    - 출력: 조명 index 리스트
    """
    Ln = L / np.linalg.norm(L, axis=1, keepdims=True)
    idx = [int(np.argmax(Ln[:, 2]))]
    used = set(idx)
    for _ in range(1, L.shape[0]):
        cos = np.clip(Ln @ Ln[idx].T, -1, 1)
        dist = 1 - cos.max(axis=1)
        for u in used:
            dist[u] = -1
        j = int(dist.argmax())
        idx.append(j)
        used.add(j)
    return idx


def make_subsets(L, m=18, stride=6):
    """
    farthest point order 기반으로 조명 서브셋(subset) 생성.

    Parameters
    ----------
    L : (N,3) 조명 벡터
    m : subset 크기
    stride : 시작 지점 간격

    Returns
    -------
    subsets : list of list
        FPS 순서에서 일정 간격으로 잘라 만든 조명 index 서브셋 리스트
    """
    order = farthest_point_order(L)
    subsets = []
    for s in range(0, len(order) - m + 1, stride):
        subsets.append(order[s:s + m])
    return subsets


def subset_valid_mask(valid_perlight, subset_idx, min_ratio=0.6):
    """
    주어진 조명 subset에 대해 유효(valid) 픽셀 비율을 계산하고
    충분히 유효한 픽셀만 True로 갖는 마스크 생성.

    valid_perlight : (N,H,W)
    subset_idx : list[int]
    min_ratio : 한 픽셀의 유효 비율이 이 값 이상이면 사용

    Returns
    -------
    mask : (H,W) boolean
    ratio : (H,W) float
    """
    V = valid_perlight[subset_idx]      # (m,H,W)
    ratio = V.mean(axis=0)
    return (ratio >= min_ratio), ratio


# ---------- 5) IRLS(Huber) 포토메트릭 스테레오 (픽셀별 가중 WLS) ----------
def irls_ps_subset(imgs_sub, L_sub, valid_sub, base_mask,
                   lam=1e-3, iters=5, delta=0.02, min_ratio=0.6):
    """
    IRLS(Huber) 기반 포토메트릭 스테레오 수행.

    Parameters
    ----------
    imgs_sub : (m,H,W) float
        linear + intensity 보정된 입력 이미지 집합
    L_sub : (m,3)
        선택된 서브셋의 조명 벡터
    valid_sub : (m,H,W) bool
        그림자/포화 제외된 per-light 유효 픽셀
    base_mask : (H,W) bool
        객체 전체 마스크
    lam : float
        ridge 안정화 항
    iters : int
        IRLS 반복 횟수
    delta : float
        Huber 손실의 전환 지점
    min_ratio : float
        subset 유효 비율 임계값

    Returns
    -------
    n : (H,W,3) float
        추정된 단위 노멀 벡터
    conf : (H,W) float
        confidence 값(오차의 역수 기반)
    use_mask : (H,W) bool
        해당 subset에서 충분히 유효한 픽셀들
    """
    m, H, W = imgs_sub.shape
    subset_mask, ratio = subset_valid_mask(valid_sub, list(range(m)), min_ratio=min_ratio)
    use_mask = subset_mask & base_mask

    # 초기 가중치
    Wv = valid_sub.astype(np.float32)
    Ls = L_sub.astype(np.float32)
    outers = np.einsum('mi,mj->mij', Ls, Ls)
    I = imgs_sub
    eps = 1e-8

    def solve_wls(Wv):
        """
        픽셀별 WLS + ridge 로 법선 추정.
        A = Σ w_k (l_k l_k^T) + lam I
        b = Σ w_k l_k I_k
        """
        A = np.zeros((H, W, 3, 3), dtype=np.float32)
        b = np.zeros((H, W, 3), dtype=np.float32)
        for k in range(m):
            wk = Wv[k]
            wk3 = wk[..., None, None]
            A += wk3 * outers[k]
            b += (wk[..., None]) * (Ls[k] * I[k][..., None])

        A += lam * np.eye(3, dtype=np.float32)[None, None]
        A2 = A.reshape(-1,3,3)
        b2 = b.reshape(-1,3)
        invA = np.linalg.inv(A2)
        n2 = (invA @ b2[..., None]).squeeze(-1)
        n = n2.reshape(H, W, 3)
        n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
        n = n / np.clip(n_norm, eps, None)
        n[~base_mask] = 0.0
        return n

    n = solve_wls(Wv)

    for _ in range(iters):
        pred = np.tensordot(Ls, n, axes=([1],[2]))  # (m,H,W)
        res = (imgs_sub - pred) * Wv
        abs_r = np.abs(res)
        w = np.where(abs_r <= delta, 1.0, delta/np.maximum(abs_r, eps))
        Wv = w * valid_sub
        n = solve_wls(Wv)

    resid = np.mean(np.abs(res), axis=0) + 1e-6
    conf = 1.0 / resid
    conf[~use_mask] = 0

    return n, conf, use_mask
