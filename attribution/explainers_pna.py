"""
Trend는 원본 x와 baseline c에서 Kalman smoother로 global하게 한 번만 추출.
Residual = input - Trend.

Order-averaged PNA 방식:
1) trend-first:
   c = Tc + Rc  ->  Tx + Rc  ->  x = Tx + Rx
2) residual-first:
   c = Tc + Rc  ->  Tc + Rx  ->  x = Tx + Rx

최종:
   trend_attr = 0.5 * (A_T^empty + A_T^R)
   resid_attr = 0.5 * (A_R^empty + A_R^T)
"""

import torch
import numpy as np
import pandas as pd
from pykalman import KalmanFilter


@torch.no_grad()
def compute_trend_kalman(inputs, observation_covariance=1.0,
                         transition_covariance=0.01, P0=1.0):
    dev, in_dtype = inputs.device, inputs.dtype
    B, T, D = inputs.shape
    Q, R = float(transition_covariance), float(observation_covariance)
    y = inputs.to(torch.float64).permute(0, 2, 1).reshape(B * D, T)   # [N,T]

    Kf = [0.0]*T; Pf = [0.0]*T
    pc = P0; Kf[0] = pc/(pc+R); Pf[0] = (1-Kf[0])*pc
    for t in range(1, T):
        pc = Pf[t-1] + Q; Kf[t] = pc/(pc+R); Pf[t] = (1-Kf[t])*pc
    C = [0.0]*T
    for t in range(T-2, -1, -1):
        C[t] = Pf[t]/(Pf[t]+Q)
    Kf = torch.tensor(Kf, dtype=torch.float64, device=dev)
    C  = torch.tensor(C,  dtype=torch.float64, device=dev)

    fm = torch.empty_like(y); fm[:, 0] = y[:, 0]
    for t in range(1, T):
        fm[:, t] = fm[:, t-1] + Kf[t]*(y[:, t] - fm[:, t-1])
    sm = torch.empty_like(y); sm[:, T-1] = fm[:, T-1]
    for t in range(T-2, -1, -1):
        sm[:, t] = fm[:, t] + C[t]*(sm[:, t+1] - fm[:, t])
    return sm.reshape(B, D, T).permute(0, 2, 1).to(in_dtype).contiguous()


class OUR_PNA:
    def __init__(self, model):
        self.model = model

    def _select_target_output(self, out, targets):
        """
        모델 출력 out에서 target class score만 뽑는 함수.

        out:
            [B, C]이면 각 sample의 target class score 선택
            [B]이면 binary/single score로 보고 그대로 사용

        targets:
            [B] class index
        """
        if isinstance(out, (tuple, list)):
            out = out[0]

        if out.dim() == 1:
            return out

        if out.shape[-1] == 1:
            return out.squeeze(-1)

        return out.gather(1, targets.reshape(-1, 1)).squeeze(1)


    def _repeat_forward_arg(self, arg, C):
        """
        alpha chunk 개수 C만큼 data_mask/timesteps를 반복.

        원래:
            arg = [B, ...]
        변환:
            [C*B, ...]
        """
        if arg is None:
            return None

        return arg.unsqueeze(0).expand(C, *arg.shape).reshape(
            C * arg.shape[0], *arg.shape[1:]
        )


    def _ig_phase_plain(
        self,
        start,
        end,
        alphas,
        targets,
        return_all,
        n_alphas,
        B,
        T,
        D,
        alpha_chunk,
        data_mask=None,
        timesteps=None,
    ):
        """
        PNA-BIG용 plain IG phase.

        start:
            phase 시작점. 예: c, Tx+Rc, Tc+Rx

        end:
            phase 끝점. 예: Tx+Rc, x

        계산:
            IG(start -> end)
            = (end - start) * 평균 gradient

        기존 TIMING의 random time_mask, N_free normalization이 없음!!!
        """
        direction = end - start
        grad_sum = torch.zeros_like(start)

        for a0 in range(0, n_alphas, alpha_chunk):
            a1 = min(a0 + alpha_chunk, n_alphas)
            a_chunk = alphas[a0:a1].to(device=start.device, dtype=start.dtype)
            C = a_chunk.numel()

            # path: start + alpha * (end - start)
            # shape: [C, B, T, D]
            path = start.unsqueeze(0) + a_chunk.view(C, 1, 1, 1) * direction.unsqueeze(0)
            path = path.detach().requires_grad_(True)

            # model forward를 위해 [C*B, T, D]로 펼침
            path_flat = path.reshape(C * B, T, D)

            # data_mask/timesteps도 path_flat과 같은 batch 크기로 반복
            mask_flat = self._repeat_forward_arg(data_mask, C)
            time_flat = self._repeat_forward_arg(timesteps, C)

            pred = self.model(
                path_flat,
                mask=mask_flat,
                timesteps=time_flat,
                return_all=return_all,
            )

            target_rep = targets.repeat(C)
            score = self._select_target_output(pred, target_rep).sum()

            grad = torch.autograd.grad(
                score,
                path,
                retain_graph=False,
                create_graph=False,
            )[0]

            grad_sum += grad.sum(dim=0)

            del path, path_flat, pred, score, grad

        return direction * (grad_sum / float(n_alphas))


    def attribute_order_averaged(
        self, inputs, baselines, targets, additional_forward_args,
        n_samples=1, num_segments=0, max_seg_len=None, min_seg_len=None,
        kalman_obs_cov=1.0, kalman_trans_cov=0.01, n_alphas=50, alpha_chunk=10,
    ):
        """
        Order-averaged PNA-BIG attribution.

        x = inputs    = Tx + Rx
        c = baselines = Tc + Rc

        Trend-first:
            c = Tc+Rc -> Tx+Rc -> x = Tx+Rx

        Residual-first:
            c = Tc+Rc -> Tc+Rx -> x = Tx+Rx

        최종:
            trend_attr = 0.5 * (A_T_empty + A_T_R)
            resid_attr = 0.5 * (A_R_empty + A_R_T)

        주의:
            기존 TIMING random time_mask / N_free normalization은 사용하지 않음.
            n_samples, num_segments, min_seg_len, max_seg_len은 main_td.py 호출 호환용 인자.
        """
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device

        # additional_forward_args = (data_mask, timesteps, return_all)
        data_mask, timesteps, return_all = None, None, False
        if additional_forward_args is not None:
            if len(additional_forward_args) > 0:
                data_mask = additional_forward_args[0]
            if len(additional_forward_args) > 1:
                timesteps = additional_forward_args[1]
            if len(additional_forward_args) > 2:
                return_all = additional_forward_args[2]

        if data_mask is not None:
            data_mask = data_mask.to(device)
        if timesteps is not None:
            timesteps = timesteps.to(device)

        # IG alpha grid
        alphas = torch.linspace(0, 1 - 1 / n_alphas, n_alphas, device=device)

        # ------------------------------------------------------------
        # 1. x와 c를 각각 trend/residual로 분해
        # ------------------------------------------------------------
        trend   = compute_trend_kalman(inputs,    kalman_obs_cov, kalman_trans_cov)  # Tx
        trend_c = compute_trend_kalman(baselines, kalman_obs_cov, kalman_trans_cov)  # Tc

        resid_x = (inputs - trend).contiguous()       # Rx = x - Tx
        resid_c = (baselines - trend_c).contiguous()  # Rc = c - Tc

        # ------------------------------------------------------------
        # 2. Trend-first path
        #    c = Tc+Rc -> Tx+Rc -> x = Tx+Rx
        # ------------------------------------------------------------
        waypoint_tf = (trend + resid_c).contiguous()  # Tx + Rc

        A_T_empty = self._ig_phase_plain(
            baselines, waypoint_tf, alphas, targets, return_all,
            n_alphas, B, T, D, alpha_chunk,
            data_mask=data_mask, timesteps=timesteps,
        )  # A_T^empty: Rc 고정, Tc -> Tx

        A_R_T = self._ig_phase_plain(
            waypoint_tf, inputs, alphas, targets, return_all,
            n_alphas, B, T, D, alpha_chunk,
            data_mask=data_mask, timesteps=timesteps,
        )  # A_R^T: Tx 고정, Rc -> Rx

        # ------------------------------------------------------------
        # 3. Residual-first path
        #    c = Tc+Rc -> Tc+Rx -> x = Tx+Rx
        # ------------------------------------------------------------
        waypoint_rf = (trend_c + resid_x).contiguous()  # Tc + Rx

        A_R_empty = self._ig_phase_plain(
            baselines, waypoint_rf, alphas, targets, return_all,
            n_alphas, B, T, D, alpha_chunk,
            data_mask=data_mask, timesteps=timesteps,
        )  # A_R^empty: Tc 고정, Rc -> Rx

        A_T_R = self._ig_phase_plain(
            waypoint_rf, inputs, alphas, targets, return_all,
            n_alphas, B, T, D, alpha_chunk,
            data_mask=data_mask, timesteps=timesteps,
        )  # A_T^R: Rx 고정, Tc -> Tx

        # ------------------------------------------------------------
        # 4. Order-average
        # ------------------------------------------------------------
        trend_attr = 0.5 * (A_T_empty + A_T_R)
        resid_attr = 0.5 * (A_R_empty + A_R_T)

        # ------------------------------------------------------------
        # 5. Completeness 확인용
        #    fxc = F_y(x) - F_y(c)
        #    Top-K PNA에서는 main_td.py에서 anchor별 fxc를 평균냄.
        # ------------------------------------------------------------
        with torch.no_grad():
            fx = self.model(inputs,    mask=data_mask, timesteps=timesteps, return_all=return_all)
            fc = self.model(baselines, mask=data_mask, timesteps=timesteps, return_all=return_all)

            fx_score = self._select_target_output(fx, targets)
            fc_score = self._select_target_output(fc, targets)
            fxc = fx_score - fc_score

        return trend_attr, resid_attr, fxc