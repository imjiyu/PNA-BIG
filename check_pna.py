import numpy as np, glob

D, M, F, S = "epilepsy", "state", 0, 42
def load(key, lam): 
    return np.load(f"results_pna/{D}_{M}_{key}_val_lam{lam}_result_{F}_{S}.npy")

# ① anchor가 0 아닌지 (=PNA 실제로 먹었나): trend/resid가 전부 0이면 이상
t = load("timing_td_trend_signed_kalman_seg0_min1_max48", "1.0x1.0")
r = load("timing_td_residual_signed_kalman_seg0_min1_max48", "1.0x1.0")
print("① |T| max:", np.abs(t).max(), "| |R| max:", np.abs(r).max(), "→ 0 아니면 OK")

# ② completeness: sum(T+R) ≈ fxc
f = load("timing_td_fxc_kalman_seg0_min1_max48", "1.0x1.0")
lhs = (t + r).reshape(t.shape[0], -1).sum(1)
print("② sum(T+R) vs fxc 평균오차:", np.abs(lhs - f).mean(), "→ 작으면 OK")

# ③ λ가 결과를 바꾸나
a = load("timing_td_trend_signed_kalman_seg0_min1_max48", "0.1x0.1")
b = load("timing_td_trend_signed_kalman_seg0_min1_max48", "10.0x10.0")
print("③ lam차이 max:", np.abs(a - b).max(), "→ 0 아니면 λ 먹는 것")