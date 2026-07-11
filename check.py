import torch
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

from attribution.explainers_pna import compute_trend_kalman
from datasets.PAM import PAM


# 기존 pykalman 버전
def _ref(inputs, oc=1.0, tc=0.01):
    B, T, D = inputs.shape

    x = inputs.detach().cpu().numpy()
    out = np.zeros((B, T, D), dtype=np.float32)
    idx = pd.RangeIndex(T)

    for b in range(B):
        for d in range(D):
            s = pd.Series(x[b, :, d], index=idx)

            kf = KalmanFilter(
                initial_state_mean=s.iloc[0],
                n_dim_obs=1,
                transition_matrices=[1],
                observation_matrices=[1],
                observation_covariance=oc,
                transition_covariance=tc,
            )

            m, _ = kf.smooth(s.values)
            out[b, :, d] = m.flatten()

    return torch.from_numpy(out)


# 같은 PAM 인스턴스를 사용해야 함
dataset = PAM(fold=1, seed=42)

# train 데이터로 mean/std 초기화
dataset.preprocess(split="train")

# 초기화된 mean/std로 validation 데이터 정규화
x = dataset.preprocess(split="val")["x"][:8]

gpu = compute_trend_kalman(
    x.cuda(),
    observation_covariance=1.0,
    transition_covariance=0.01,
).cpu()

ref = _ref(x, oc=1.0, tc=0.01)

diff = (gpu - ref).abs()

print("max abs diff:", diff.max().item())
print("mean abs diff:", diff.mean().item())
print("passed:", diff.max().item() <= 1e-6)