"""
Projected Neutral Anchor (PNA) baseline selection for BIG.

zero baseline 대신, 각 입력 x 에 대해 training pool 에서
  (1) representation space 에서 x 와 가깝고        -> D_phi (식 3)
  (2) normalized input space 에서 zero 에 가깝고    -> L0    (식 4 좌)
  (3) decision-neutral (softmax ~ uniform)          -> Lf    (식 4 우)
  (4) 실제 training sample (=on-manifold)
한 anchor 를 고른다. 모든 forward 는 eval mode 에서 수행(dropout/BN 고정).

[사용법]
  # 루프 밖 1회: pool 통계(phi_c, L0, Lf) 캐싱
  #   - pool 이 raw 데이터면 학습때 train_mean/std 를 input_mu/input_sd 로 전달.
  #   - pool 이 이미 normalized 면 그대로(=None) 두면 됨. (preprocess 확인 필요)
  cache = build_pna_cache(x_train, classifier, feature="hidden")  # binary 데이터는 hidden 권장

  # 루프 안: anchors [B,Ka,T,D]
  anchors = select_pna_baselines(x_batch, cache, classifier, Ka=5)
  # Ka개 anchor 각각으로 attribution 을 따로 계산한 뒤 "attribution 을 평균"할 것 (식 9).
  # raw input 평균(c_bar) 아님: F(mean c) != mean F(c) 이고 c_bar 는 실제 sample 이 아니라
  # on-manifold 성질이 깨짐.
"""
import torch

@torch.no_grad()
def _extract(classifier, z, kind, chunk):
    def _fwd(t): return classifier(t, mask=None, timesteps=None, return_all=False)
    was = classifier.training; classifier.eval()
    try:
        if kind == "logits":
            return torch.cat([_fwd(z[i:i+chunk]) for i in range(0, len(z), chunk)], 0)
        holder = {}
        def _hook(mod, inp): holder["h"] = inp[0].detach()
        handle = classifier.net.regressor.register_forward_pre_hook(_hook)
        try:
            outs = []
            for i in range(0, len(z), chunk):
                holder.clear(); _fwd(z[i:i+chunk])
                if "h" not in holder:
                    raise RuntimeError("hidden hook not fired; check classifier.net.regressor path")
                outs.append(holder["h"])
            return torch.cat(outs, 0)
        finally:
            handle.remove()
    finally:
        classifier.train(was)

@torch.no_grad()
def _neutrality(logits):
    if logits.dim() == 1: logits = logits.unsqueeze(-1)
    if logits.shape[1] == 1:
        p1 = torch.sigmoid(logits); probs = torch.cat([1.0 - p1, p1], dim=1)
    else:
        probs = logits.softmax(-1)
    K = probs.shape[1]
    return (K * probs.pow(2).sum(1) - 1.0) / (K - 1)

@torch.no_grad()
def build_pna_cache(pool, classifier, feature="hidden", lam0=1.0, lamf=1.0,
                    input_mu=None, input_sd=None, chunk=256):
    param = next(classifier.parameters())
    device, dtype = param.device, param.dtype
    pool = pool.to(device=device, dtype=dtype); N = pool.shape[0]
    phi_c = _extract(classifier, pool, feature, chunk).reshape(N, -1)
    mu = phi_c.mean(0, keepdim=True); sd = phi_c.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
    phi_c = (phi_c - mu) / sd
    if input_mu is not None and input_sd is not None:
        pool_l0 = (pool - input_mu.to(device)) / input_sd.to(device).clamp_min(1e-6)
    else:
        pool_l0 = pool
    L0 = pool_l0.pow(2).reshape(N, -1).mean(1)
    Lf = _neutrality(_extract(classifier, pool, "logits", chunk))
    base = lam0 * L0 + lamf * Lf
    return {"pool": pool, "phi_c": phi_c, "mu": mu, "sd": sd,
            "m": phi_c.shape[1], "base": base, "feature": feature,
            "device": device, "dtype": dtype}

@torch.no_grad()
def select_pna_baselines(inputs, cache, classifier, Ka=1, chunk=256):
    inputs = inputs.to(device=cache["device"], dtype=cache["dtype"]); B = inputs.shape[0]
    phi_x = _extract(classifier, inputs, cache["feature"], chunk).reshape(B, -1)
    phi_x = (phi_x - cache["mu"]) / cache["sd"]
    phi_c = cache["phi_c"]
    Dphi = (phi_x.pow(2).sum(1,keepdim=True) - 2*phi_x@phi_c.t()
            + phi_c.pow(2).sum(1)[None,:]) / cache["m"]
    J = Dphi + cache["base"][None,:]
    Ka = max(1, min(Ka, cache["pool"].shape[0]))
    idx = torch.topk(-J, Ka, dim=1).indices
    return cache["pool"][idx].contiguous()

@torch.no_grad()
def select_global_neutral_anchors(cache, Ka=5):
    """
    입력 x 와 무관하게, base(=λ0·L0 + λf·Lf) 가 가장 작은 top-Ka anchor 를
    pool 에서 '한 번' 고른다. 모든 샘플이 이 anchor 를 공유.

    return: anchors [Ka, T, D]  (batch 차원 없음. 호출부에서 expand)
    """
    base = cache["base"]                       # [N]  (x 의존 없음)
    Ka = max(1, min(Ka, cache["pool"].shape[0]))
    idx = torch.topk(-base, Ka, dim=0).indices  # 가장 neutral+low-energy
    return cache["pool"][idx].contiguous()      # [Ka, T, D]


# select_pna_baselines 와 '수학적으로 동일'하며, 최종 pool[idx] 대신 idx 만 반환함
# (anchors = pool[idx] 이므로 재현·검증·저장에 idx 만 있으면 충분)
@torch.no_grad()
def select_pna_indices(inputs, cache, classifier, Ka=1, chunk=256):
    """
    select_pna_baselines 와 동일한 J 계산 후, pool-relative anchor 인덱스 [B, Ka] 반환.
    저장/검증용: anchors = cache["pool"][idx] 로 언제든 복원 가능 (RNG 의존 0).
    """
    inputs = inputs.to(device=cache["device"], dtype=cache["dtype"]); B = inputs.shape[0]
    phi_x = _extract(classifier, inputs, cache["feature"], chunk).reshape(B, -1)
    phi_x = (phi_x - cache["mu"]) / cache["sd"]
    phi_c = cache["phi_c"]
    Dphi = (phi_x.pow(2).sum(1, keepdim=True) - 2 * phi_x @ phi_c.t()
            + phi_c.pow(2).sum(1)[None, :]) / cache["m"]
    J = Dphi + cache["base"][None, :]
    Ka = max(1, min(Ka, cache["pool"].shape[0]))
    return torch.topk(-J, Ka, dim=1).indices  # [B, Ka]
