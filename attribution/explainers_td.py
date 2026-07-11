"""
Trend-Residual decomposition extension of TIMING.
Trend는 원본 x에서 Kalman(pykalman)으로 global하게 한 번만 추출, Residual = x - Trend.
(모든 mask sample이 그 trend를 공유하도록 최종 수정 / mask마다 trend를 다시 뽑지 않음!)
"""

import torch
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

def apply_kalman_smoother(series, observation_covariance, transition_covariance):
    kf = KalmanFilter(
        initial_state_mean=series.iloc[0],
        n_dim_obs=1,
        transition_matrices=[1],
        observation_matrices=[1],
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
    )
    filtered_state_means, _ = kf.smooth(series.values)
    return pd.Series(filtered_state_means.flatten(), index=series.index)

"""
# Kalman-Filter ver. (칼만 필터 코드 / + 아래에도 apply_kalman_filter로 바꾸기)
def apply_kalman_filter(series, observation_covariance, transition_covariance):
    kf = KalmanFilter(
        initial_state_mean=series.iloc[0],
        n_dim_obs=1,
        transition_matrices=[1],
        observation_matrices=[1],
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
    )
    filtered_state_means, _ = kf.filter(series.values)
    return pd.Series(filtered_state_means.flatten(), index=series.index)
"""

def compute_trend_kalman(inputs, observation_covariance=1.0, transition_covariance=0.01):
    """원본 시계열 x에서 Kalman smoother로 trend 추출"""
    B, T, D = inputs.shape
    x_np = inputs.detach().cpu().numpy()
    trend_np = np.zeros((B, T, D), dtype=np.float32)
    idx = pd.RangeIndex(T)
    for b in range(B):
        for d in range(D):
            s = pd.Series(x_np[b, :, d], index=idx)
            trend_np[b, :, d] = apply_kalman_smoother(
                s, observation_covariance, transition_covariance
            ).values
    return torch.from_numpy(trend_np).to(inputs.device).float()


class OUR_TD:
    """TIMING + Trend-Residual decomposition."""

    def __init__(self, model):
        self.model = model

    def _ig_phase(self, start, end, time_mask, fixed_inputs,
                  alphas, targets, return_all,
                  n_samples, n_alphas, B, T, D, alpha_chunk):
        direction = end - start
        start4 = start.unsqueeze(1)
        dir4   = direction.unsqueeze(1)
        tm4    = time_mask.unsqueeze(1)
        fix4   = fixed_inputs.unsqueeze(1)

        attr_sum = torch.zeros(n_samples, B, T, D, device=start.device)

        for a0 in range(0, n_alphas, alpha_chunk):
            a1 = min(a0 + alpha_chunk, n_alphas)
            a_chunk = alphas[a0:a1].view(1, -1, 1, 1, 1)
            ck = a1 - a0

            interp = start4 + a_chunk * dir4
            path = tm4 * interp + (1 - tm4) * fix4
            path.requires_grad_(True)

            pred = self.model(
                path.reshape(-1, T, D),
                mask=None, timesteps=None, return_all=return_all,
            )
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)
            pred = pred.view(n_samples, ck, B, -1)
            tgt = targets.view(1, 1, B, 1).expand(n_samples, ck, B, 1)
            g = pred.gather(3, tgt).squeeze(-1)
            grad = torch.autograd.grad(g.sum(), path, retain_graph=False)[0]
            grad = grad * tm4

            attr_sum += (grad * dir4).sum(dim=1)

            del path, pred, g, grad, interp
            torch.cuda.empty_cache()

        #attr = attr_sum / n_alphas ### completeness 실험 시 이부분 주석 풀기
        #attr = attr.mean(dim=0)    ### completeness 실험 시 이부분 주석 풀기 (대신 아래 N_free 3줄 주석 처리)

        N_free = time_mask.sum(dim=0)
        attr = attr_sum.sum(dim=0) / (n_alphas * N_free.clamp_min(1))
        attr = torch.where(N_free > 0, attr, torch.zeros_like(attr))

        return attr

    def attribute_trend_residual_segments(
        self,
        inputs, baselines, targets, additional_forward_args,
        n_samples=50, num_segments=3,
        max_seg_len=None, min_seg_len=None,
        kalman_obs_cov=1.0, kalman_trans_cov=0.01,
        n_alphas=None, alpha_chunk=10,
    ):
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device
        return_all = additional_forward_args[2]

        if max_seg_len is None:
            max_seg_len = T
        else:
            max_seg_len = min(max_seg_len, T)

        if min_seg_len is None:
            min_seg_len = 1
        else:
            min_seg_len = max(1, min(min_seg_len, T))
        
        if min_seg_len > max_seg_len:
            raise ValueError("min_seg_len must be <= max_seg_len.")
        
        if n_alphas is None:
            n_alphas = n_samples

        alphas = torch.linspace(0, 1 - 1/n_alphas, n_alphas, device=device)

        # segment mask 생성 (mask==1: free, IG path 흐름 / mask==0: 원본 x로 고정)
        time_mask = torch.ones(n_samples, B, T, D, device=device)
        dims     = torch.randint(0, D, (n_samples, B, num_segments), device=device)
        seg_lens = torch.randint(min_seg_len, max_seg_len+1, (n_samples, B, num_segments), device=device)
        t_starts = (torch.rand(n_samples, B, num_segments, device=device) * (T - seg_lens + 1)).long()

        batch_indices  = torch.arange(B, device=device)
        sample_indices = torch.arange(n_samples, device=device)
        for s in range(num_segments):
            mlen = seg_lens[:, :, s].max()
            base_range = torch.arange(mlen, device=device).unsqueeze(0).unsqueeze(0)
            indices = t_starts[:, :, s].unsqueeze(-1) + base_range
            end_points = (t_starts[:, :, s] + seg_lens[:, :, s]).unsqueeze(-1)
            valid = (indices < end_points) & (indices < T)

            # valid=False인 위치는 아예 indexing에 참여하지 않아서 t=0 오염이 없어지도록,,
            si = sample_indices.view(-1, 1, 1).expand_as(indices)
            bi = batch_indices.view(1, -1, 1).expand_as(indices)
            di = dims[:, :, s].unsqueeze(-1).expand_as(indices)
            
            time_mask[
                si[valid],
                bi[valid],
                indices[valid],
                di[valid]
            ] = 0

        # 원본 x에서 trend / residual 추출 (mask 무관, 한 번만)
        trend    = compute_trend_kalman(inputs,    kalman_obs_cov, kalman_trans_cov)  # Tx
        trend_c  = compute_trend_kalman(baselines, kalman_obs_cov, kalman_trans_cov)  # Tc

        baselines_s = baselines.unsqueeze(0).expand(n_samples, B, T, D).contiguous()   # c = Tc+Rc
        inputs_s    = inputs.unsqueeze(0).expand(n_samples, B, T, D).contiguous()      # x = Tx+Rx
        trend_s     = trend.unsqueeze(0).expand(n_samples, B, T, D).contiguous()       # Tx
        trend_c_s   = trend_c.unsqueeze(0).expand(n_samples, B, T, D).contiguous()     # Tc
        resid_c_s   = baselines_s - trend_c_s                                          # Rc
        fixed_inputs = inputs_s.detach()

        waypoint = trend_s + resid_c_s   # = Tx + Rc  (c=0이면 waypoint == trend_s → 기존과 동일)

        # Phase 1: Tc+Rc -> Tx+Rc  (trend 교체, Rc 고정)
        trend_attr = self._ig_phase(
            baselines_s, waypoint, time_mask, fixed_inputs,
            alphas, targets, return_all,
            n_samples, n_alphas, B, T, D, alpha_chunk)

        # Phase 2: Tx+Rc -> Tx+Rx = x  (residual 교체, Tx 고정)
        resid_attr = self._ig_phase(
            waypoint, inputs_s, time_mask, fixed_inputs,
            alphas, targets, return_all,
            n_samples, n_alphas, B, T, D, alpha_chunk)

        with torch.no_grad():
            c_masked = time_mask * baselines_s + (1 - time_mask) * fixed_inputs
            fc = self.model(c_masked.reshape(-1, T, D),
                            mask=None, timesteps=None, return_all=return_all)
            fx = self.model(inputs,
                            mask=None, timesteps=None, return_all=return_all)
            if fc.dim() == 1: fc = fc.unsqueeze(-1)
            if fx.dim() == 1: fx = fx.unsqueeze(-1)
            fc = fc.view(n_samples, B, -1)
            fc = fc.gather(2, targets.view(1, B, 1).expand(n_samples, B, 1)).squeeze(-1)
            fx = fx.gather(1, targets.view(B, 1)).squeeze(-1)
            fxc = fx - fc.mean(dim=0)

        return trend_attr, resid_attr, fxc