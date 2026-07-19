import torch
import torch.nn as nn
from utils.tensor_manipulation import normalize as normal


class MAGIG:
    def __init__(self, classifier, vae, device="cuda"):
        self.classifier = classifier
        self.vae = vae
        self.device = device
        self.vae.eval()

    @torch.no_grad()
    def _encode_mean(self, x):
        mu, _ = self.vae.encode(x)
        return mu  # posterior mean [B, T, L]
    
    """
    [TODO] GP-bridge latent path sampling을 VAE latent space에서 Guided IG 하는 형태로 변경
    """
    def _gp_bridge_samples(
        self,
        z_b,            # [B, T, L]
        z_x,            # [B, T, L]
        alphas,         # [n_steps]  in [0, 1]
        n_paths=1,
        ls=0.3,             # GP lengthscale over alpha (scalar)
        amp=0.4,            # GP amplitude (std of the bridge bump)
        time_ls=0.0,
        jitter=1e-5,
    ):
       
        device = z_b.device
        n_steps = alphas.shape[0]

        # straight-line mean path: [n_steps, B, T, L]
        a = alphas.view(-1, 1, 1, 1)
        mean_path = z_b.unsqueeze(0) + a * (z_x.unsqueeze(0) - z_b.unsqueeze(0))

        if n_paths == 1 and amp == 0.0:
            # deterministic straight line (no GP) — useful for ablation
            return mean_path.unsqueeze(0)  # [1, n_steps, B, T, L]

        # --- RBF kernel over alpha grid, conditioned to be zero at {0,1} ---
        # full grid including the two anchor endpoints 0 and 1
        anchors = torch.tensor([0.0, 1.0], device=device)
        a_all = torch.cat([alphas, anchors])               # [n_steps+2]

        def rbf(x1, x2):
            d2 = (x1.view(-1, 1) - x2.view(1, -1)) ** 2
            return (amp ** 2) * torch.exp(-0.5 * d2 / (ls ** 2))

        K = rbf(a_all, a_all)
        K = K + jitter * torch.eye(K.shape[0], device=device)

        n_obs = 2  # the two endpoints we condition on (value 0)
        idx_f = torch.arange(n_steps, device=device)               # free pts
        idx_o = torch.arange(n_steps, n_steps + n_obs, device=device)  # anchors

        K_ff = K[idx_f][:, idx_f]      # [n_steps, n_steps]
        K_fo = K[idx_f][:, idx_o]      # [n_steps, 2]
        K_oo = K[idx_o][:, idx_o]      # [2, 2]

        K_oo_inv = torch.linalg.inv(K_oo)
        # conditional mean is 0 (anchors are 0); conditional cov:
        cond_cov = K_ff - K_fo @ K_oo_inv @ K_fo.transpose(-1, -2)
        cond_cov = cond_cov + jitter * torch.eye(n_steps, device=device)
        L_chol = torch.linalg.cholesky(cond_cov)           # [n_steps, n_steps]

        # sample bridge perturbations independently per latent coordinate
        # shape target: [n_paths, n_steps, B, T, L]
        B, T, Ld = z_b.shape
        eps = torch.randn(n_paths, n_steps, B, T, Ld, device=device)
        # (1) alpha-wise perturbation: [n_steps, n_steps] @ [n_steps, B, T, L] -> [n_steps, B, T, L]
        perturb = torch.einsum("ij,pjbtl->pibtl", L_chol, eps)
        # (2) only if time_ls > 0 -------------------------
        if time_ls is not None and time_ls > 0.0:
            t_grid = torch.arange(T, device=device).float()
            d2_t = (t_grid.view(-1, 1) - t_grid.view(1, -1)) ** 2
            K_t = torch.exp(-0.5 * d2_t / (time_ls ** 2))
            K_t = K_t + jitter * torch.eye(T, device=device)
            L_t = torch.linalg.cholesky(K_t)

            perturb_before = perturb
            perturb = torch.einsum("ts,pibsl->pibtl", L_t, perturb)

            # --- 분산 재정규화: 시간축 색칠 전후의 RMS를 맞춤 ---
            std_before = perturb_before.std()
            std_after = perturb.std() + 1e-8
            perturb = perturb * (std_before / std_after)
        # ----------------------------------------------------------------

        paths = mean_path.unsqueeze(0) + perturb           # [n_paths,n_steps,B,T,L]
        return paths

    def attribute(
        self,
        inputs,                       # [B, T, D]
        baselines,                    # [B, T, D]
        targets,                      # [B]
        additional_forward_args=None,
        n_steps=50,
        n_paths=10,                   # GP path samples (>=2 to get uncertainty)
        gp_lengthscale=0.3,           # smoothness of bridge over alpha
        gp_amplitude=0.25,            # magnitude of bridge perturbation
        gp_time_lengthscale=0.0,
        jitter=1e-5,
        return_uncertainty=False,     # also return per-cell std across paths
        normalize=True,
    ):
        if inputs.shape != baselines.shape:
            raise ValueError("inputs and baselines must have same shape")

        B, T, D = inputs.shape
        device = inputs.device

        # 1) encode endpoints
        with torch.no_grad():
            z_x = self._encode_mean(inputs)        # [B, T, L]
            z_b = self._encode_mean(baselines)     # [B, T, L]

        Ld = z_x.shape[-1]

        if additional_forward_args is not None:
            return_all = additional_forward_args[2]
        else:
            return_all = False

        alphas = torch.linspace(0, 1, n_steps, device=device)  # include 1.0

        # 2) sample GP-bridge latent paths: [n_paths, n_steps, B, T, L]
        paths = self._gp_bridge_samples(
            z_b, z_x, alphas, n_paths,
            ls=gp_lengthscale, amp=gp_amplitude, jitter=jitter,
            time_ls=gp_time_lengthscale, 
        )
        P = paths.shape[0]

        attr_paths_input = torch.zeros(P, B, T, D, device=device)

        # --- roughness 진단 누적용 ---
        rough_num_sum = 0.0   # 분자: path 차분 노름 누적
        rough_den_sum = 0.0   # 분모: gradient 노름 누적
        
        for p in range(P):
            z_path = paths[p]                            # [n_steps, B, T, L]

            # ---- input 공간에서 직접 적분 ----
            z_path = z_path.detach()
            z_flat = z_path.reshape(n_steps * B, T, Ld)
            # decode 후 input 공간에서 leaf로 만든다
            x_tilde = self.vae.decode(z_flat).view(n_steps, B, T, D)

            # End point correction
            # 양 끝점을 재구성값 대신 원본 baseline/input으로 치환하여, completeness가 f(x) - f(x') 를 target하도록 한다.
            x_tilde = x_tilde.clone()
            x_tilde[0]  = baselines          # alpha = 0  -> 원본 baseline x'
            x_tilde[-1] = inputs             # alpha = 1  -> 원본 input x

            x_tilde = x_tilde.detach().requires_grad_(True)
            logits = self.classifier(
                x_tilde.view(n_steps * B, T, D),
                mask=None, timesteps=None, return_all=return_all,
            )

            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            logits = logits.view(n_steps, B, -1)
            tgt = targets.view(1, B, 1).expand(n_steps, B, 1)
            target_logits = logits.gather(2, tgt).squeeze(-1)
            # dF/dx_tilde : input 공간 gradient
            grads_x = torch.autograd.grad(
                outputs=target_logits.sum(),
                inputs=x_tilde,
                retain_graph=False, create_graph=False,
            )[0]                                     # [n_steps, B, T, D]

            # --- path roughness 측정 (이 path p에 대해) ---
            with torch.no_grad():
                g = grads_x                                  # [n_steps, B, T, D]
                # 인접 step 차분: [n_steps-1, B, T, D]
                dg = g[1:] - g[:-1]
                # 각 (step, sample)별 (T,D) 평탄화 후 L2 노름 -> [n_steps-1, B]
                dg_norm = dg.reshape(n_steps - 1, B, -1).norm(dim=-1)
                g_norm  = g.reshape(n_steps,     B, -1).norm(dim=-1)  # [n_steps, B]
                # 분자: step 평균한 차분 노름을 sample 합산
                rough_num_sum += dg_norm.mean(dim=0).sum().item()    # sum over B
                # 분모: step 평균한 gradient 노름을 sample 합산
                rough_den_sum += g_norm.mean(dim=0).sum().item()
            # ------------------------------------------------

            # input 공간 trapezoidal path integral
            xt = x_tilde.detach()
            g = grads_x
            dx = xt[1:] - xt[:-1]                    # [n_steps-1, B, T, D]
            g_avg = 0.5 * (g[1:] + g[:-1])
            A_x = (g_avg * dx).sum(dim=0)            # [B, T, D]
            attr_paths_input[p] = A_x

        attr = attr_paths_input.mean(dim=0)         # [B, T, D]
        attr_unc = attr_paths_input.std(dim=0)

        # --- roughness 결과 저장 ---
        self.last_rough_num = rough_num_sum        # 분자 누적 (paths×B 합산)
        self.last_rough_den = rough_den_sum + 1e-8 # 분모 누적
        self.last_rough_count = P * B              # 정규화용 (path×sample 수)
        # ---------------------------
        
        if normalize:
            attr = normal(torch.abs(attr))

        if return_uncertainty:
            return attr, attr_unc
        return attr
