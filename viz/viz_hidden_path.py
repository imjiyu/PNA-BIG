"""
viz_hidden_path.py

목적: baseline -> input path 를 따라가며 hidden state trajectory 를 뽑아,
      zero baseline vs PNA anchor 중 어느 쪽이 training manifold 안에 머무는지를 보인다.
      (= supervisor 요청 "visualize OOD mitigation")

설계:
  - hidden = classifier.net.regressor 직전 representation (pna.py hook 과 동일 지점)
  - manifold 거리 = per-dim standardized(z-score) full-hidden 공간에서
    training hidden 까지의 k-NN 거리.
    * 주의: 이 표준화는 per-dimension z-score 이지 covariance whitening 이 아니다
      (off-diagonal 상관은 제거하지 않음). hidden dim=200 에서 full whitening 은
      공분산 추정/역행렬이 불안정하므로 per-dim standardize 로 충분.
    * center 거리(평균까지 거리)를 쓰지 않는 이유: 클래스 cluster 가 분리돼 있으면
      '데이터 없는 중앙'을 최적으로 오판하기 때문.
  - PCA 2D 는 그림 전용, 정량값은 원 hidden dim 에서 계산
  - endpoint(baseline, input) 와 interior(0<a<1) 분리.
    PNA anchor 는 training sample 이라 endpoint 가 in-manifold 인 것은 정의상 당연 →
    OOD 완화 주장은 interior 거리로.
  - line / trend-first / residual-first interior 거리 모두 정량화
  - PNA 는 top-Ka(기본5): 그림엔 5 path 겹치기(첫 anchor 진하게),
    정량은 5 anchor 평균 (실제 attribution 이 Ka=5 평균이므로 일관).
    anchor waveform 자체는 평균하지 않음.
  - 대표 그림 샘플은 seed 고정 무작위 추출(라벨 정렬 편향 방지)
  - 조건(ka/knn_k/lambda/n_alphas) 을 파일명 tag 로 → 조합별 결과 덮어쓰기 방지

사용 예:
  python viz_hidden_path.py --data epilepsy --fold 0 --device cuda:0 \
      --pna_lam0 10.0 --pna_lamf 10.0 --pna_ka 5 \
      --n_alphas 50 --n_samples 3 --summary_n 100 \
      --out_dir ./viz_hidden/epilepsy
"""
import os
import sys
import csv
import argparse

import numpy as np
import torch as th
import matplotlib.pyplot as plt

# (viz/ 의 부모 = 프로젝트 루트)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real.classifier import MimicClassifierNet
from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer

from attribution.pna import build_pna_cache, select_pna_baselines
from attribution.explainers_pna import compute_trend_kalman


# freezer 제외 (연구에서 미사용)
CFG = {
    "PAM":      (PAM,      17, 8, 600, False),
    "boiler":   (Boiler,   20, 2, 36,  False),
    "epilepsy": (Epilepsy, 1,  2, 178, False),
    "wafer":    (Wafer,    1,  2, 152, True),
}


def build_datamodule(data, fold, seed):
    DM, _, _, _, needs_folds = CFG[data]
    if needs_folds:
        return DM(n_folds=5, fold=fold, seed=seed)
    return DM(fold=fold, seed=seed)


@th.no_grad()
def extract_hidden(classifier, z, chunk=256):
    """classifier.net.regressor 직전 representation. z:[N,T,D] -> [N,m]"""
    was = classifier.training
    classifier.eval()
    holder = {}

    def _hook(mod, inp):
        holder["h"] = inp[0].detach()

    handle = classifier.net.regressor.register_forward_pre_hook(_hook)
    try:
        outs = []
        for i in range(0, len(z), chunk):
            holder.clear()
            classifier(z[i:i + chunk], mask=None, timesteps=None, return_all=False)
            if "h" not in holder:
                raise RuntimeError(
                    "hidden hook not fired; check classifier.net.regressor path"
                )
            outs.append(holder["h"])
        return th.cat(outs, 0).reshape(len(z), -1)
    finally:
        handle.remove()
        classifier.train(was)


# ----------------------------------------------------------------------
# per-dim standardize (z-score) + k-NN manifold 거리
#  (covariance whitening 아님. off-diagonal 상관 미제거)
# ----------------------------------------------------------------------
def fit_standardizer(pool_hidden):
    mu = pool_hidden.mean(0, keepdims=True)
    sd = pool_hidden.std(0, keepdims=True) + 1e-8
    return mu, 1.0 / sd


def standardize(h, mu, inv_sd):
    return (h - mu) * inv_sd


def knn_dist_to_pool(query_s, pool_t, k=1, chunk=512):
    """
    query_s: numpy [Q, m]  (standardized)
    pool_t : torch [P, m], 이미 device 에 있음 (1회만 올림)
    return : numpy [Q] (k-NN 평균 L2, standardized 공간)
    """
    query_t = th.as_tensor(query_s, dtype=pool_t.dtype, device=pool_t.device)
    kk = min(k, pool_t.shape[0])
    outs = []
    for i in range(0, query_t.shape[0], chunk):
        d = th.cdist(query_t[i:i + chunk], pool_t)
        vals = th.topk(d, kk, dim=1, largest=False).values
        outs.append(vals.mean(dim=1))
    return th.cat(outs).cpu().numpy()


# ----------------------------------------------------------------------
# path 생성 (경계 중복 제거)
# ----------------------------------------------------------------------
def line_path(c, x, alphas):
    a = alphas.view(-1, 1, 1)
    return c.unsqueeze(0) + a * (x - c).unsqueeze(0)


def two_phase_path(c, x, alphas, obs_cov, trans_cov, order="trend"):
    """
    order='trend': c=Tc+Rc -> Tx+Rc -> x
    order='resid': c=Tc+Rc -> Tc+Rx -> x
    phase2 는 alphas[1:] 만 사용(waypoint 중복 제거). return [2A-1,T,D]
    """
    Tx = compute_trend_kalman(x.unsqueeze(0), obs_cov, trans_cov)[0]
    Tc = compute_trend_kalman(c.unsqueeze(0), obs_cov, trans_cov)[0]
    Rx = x - Tx
    Rc = c - Tc
    waypoint = (Tx + Rc) if order == "trend" else (Tc + Rx)
    a = alphas.view(-1, 1, 1)
    p1 = c.unsqueeze(0) + a * (waypoint - c).unsqueeze(0)
    a2 = alphas[1:].view(-1, 1, 1)
    p2 = waypoint.unsqueeze(0) + a2 * (x - waypoint).unsqueeze(0)
    return th.cat([p1, p2], 0)


def interior_mask(n_points):
    m = np.ones(n_points, dtype=bool)
    m[0] = False
    m[-1] = False
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, choices=list(CFG))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_type", default="state")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pna_feature", default="hidden", choices=["hidden", "logits"])
    p.add_argument("--pna_lam0", type=float, default=10.0)
    p.add_argument("--pna_lamf", type=float, default=10.0)
    p.add_argument("--pna_ka", type=int, default=5,
                   help="PNA anchor 수(그림 겹치기 + 정량 평균). 실제 attribution과 동일하게 5.")
    p.add_argument("--pool_size", type=int, default=1000)
    p.add_argument("--n_alphas", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=3, help="그림 그릴 test 샘플 수")
    p.add_argument("--summary_n", type=int, default=100,
                   help="정량요약 test 샘플 수(0이면 스킵)")
    p.add_argument("--knn_k", type=int, default=1)
    p.add_argument("--kalman_obs_cov", type=float, default=1.0)
    p.add_argument("--kalman_trans_cov", type=float, default=0.01)
    p.add_argument("--out_dir", default="./viz_hidden")
    args = p.parse_args()

    device = th.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    _, feat, n_state, n_t, _ = CFG[args.data]

    # 조건 tag: 조합별 결과가 덮어써지지 않도록 파일명에 포함
    #   lam / ka / knn_k / n_alphas 를 담는다.
    lam_str = f"{args.pna_lam0}x{args.pna_lamf}"
    cond_tag = (f"lam{lam_str}_ka{args.pna_ka}_k{args.knn_k}_a{args.n_alphas}")

    # --- classifier ---
    classifier = MimicClassifierNet(
        feature_size=feat, n_state=n_state, n_timesteps=n_t,
        hidden_size=200, regres=True, loss="cross_entropy",
        lr=1e-4, l2=1e-3, model_type=args.model_type,
    )
    ckpt = (f"./model/{args.data}/{args.model_type}_classifier_"
            f"{args.fold}_{args.seed}_no_imputation")
    classifier.load_state_dict(th.load(ckpt, map_location=device))
    classifier.eval().to(device)
    # NOTE: 이 스크립트는 전부 no_grad. cudnn.enabled=False 를 두면 GRU forward 가
    #       느린 fallback 이 되므로 두지 않는다.

    # --- data ---
    dm = build_datamodule(args.data, args.fold, args.seed)
    x_train = dm.preprocess(split="train")["x"].to(device)
    x_test = dm.preprocess(split="test")["x"].to(device)

    # --- PNA pool & cache (main_td.py와 동일 규칙: 시드 고정 subsample) ---
    g = th.Generator().manual_seed(args.seed)
    idx = th.randperm(x_train.shape[0], generator=g)[:args.pool_size]
    pool = x_train[idx]
    cache = build_pna_cache(pool, classifier, feature=args.pna_feature,
                            lam0=args.pna_lam0, lamf=args.pna_lamf)

    # --- manifold 기준: pool hidden ---
    pool_hidden = extract_hidden(classifier, pool).cpu().numpy()
    mu, inv_sd = fit_standardizer(pool_hidden)
    pool_s = standardize(pool_hidden, mu, inv_sd)

    # pool 을 GPU 텐서로 1회만 올림 (반복 변환 제거)
    pool_s_t = th.as_tensor(pool_s, dtype=th.float32, device=device)

    # --- PCA 2D (그림 전용): pool standardized 로 fit ---
    ps_center = pool_s.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(pool_s - ps_center, full_matrices=False)
    comp2 = Vt[:2]

    def project2d(hidden_np):
        s = standardize(hidden_np, mu, inv_sd)
        return (s - ps_center) @ comp2.T
    pool_2d = project2d(pool_hidden)

    alphas = th.linspace(0, 1, args.n_alphas, device=device)

    def path_dist(path_zTD):
        """path:[N,T,D] -> (2d[N,2], interior_dist, endpoint dict)"""
        h = extract_hidden(classifier, path_zTD).cpu().numpy()
        hs = standardize(h, mu, inv_sd)
        d_all = knn_dist_to_pool(hs, pool_s_t, k=args.knn_k)
        im = interior_mask(len(d_all))
        return project2d(h), float(d_all[im].mean()), \
            {"start": float(d_all[0]), "end": float(d_all[-1])}

    # ==================================================================
    # 1) 대표 샘플 그림 (seed 고정 무작위 추출 → 라벨 정렬 편향 방지)
    # ==================================================================
    n_show = min(args.n_samples, x_test.shape[0])
    g_show = th.Generator().manual_seed(args.seed + 7)
    show_idx = th.randperm(x_test.shape[0], generator=g_show)[:n_show].tolist()
    print(f"[repr samples] randomly picked idx = {show_idx}")

    for si in show_idx:
        x = x_test[si]
        c_zero = th.zeros_like(x)
        anchors = select_pna_baselines(x.unsqueeze(0), cache, classifier,
                                       Ka=max(1, args.pna_ka))  # [1,Ka,T,D]

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.scatter(pool_2d[:, 0], pool_2d[:, 1], s=8, c="lightgray",
                   alpha=0.5, label="training pool (manifold)", zorder=1)

        # zero
        lz2d, lz_int, lz_ep = path_dist(line_path(c_zero, x, alphas))
        tz2d, tz_int, _ = path_dist(two_phase_path(
            c_zero, x, alphas, args.kalman_obs_cov, args.kalman_trans_cov, "trend"))
        ax.plot(lz2d[:, 0], lz2d[:, 1], "-", color="tab:red", lw=2,
                label="zero (line)", zorder=3)
        ax.plot(tz2d[:, 0], tz2d[:, 1], "--", color="tab:red", lw=1.4,
                alpha=0.8, label="zero (trend-first)", zorder=3)
        ax.scatter(lz2d[0, 0], lz2d[0, 1], s=140, c="tab:red", marker="*",
                   edgecolors="black", linewidths=0.8, zorder=5)

        # PNA top-Ka: 5 path 겹치기 + interior 거리 5개 평균
        pna_line_ints, pna_tf_ints, pna_starts = [], [], []
        for k in range(anchors.shape[1]):
            c_pna = anchors[0, k]
            lp2d, lp_int, lp_ep = path_dist(line_path(c_pna, x, alphas))
            tp2d, tp_int, _ = path_dist(two_phase_path(
                c_pna, x, alphas, args.kalman_obs_cov, args.kalman_trans_cov, "trend"))
            first = (k == 0)
            ax.plot(lp2d[:, 0], lp2d[:, 1], "-", color="tab:blue",
                    lw=2 if first else 1, alpha=1.0 if first else 0.30,
                    label="PNA (line)" if first else None, zorder=3)
            ax.plot(tp2d[:, 0], tp2d[:, 1], "--", color="tab:blue",
                    lw=1.4 if first else 0.9, alpha=0.8 if first else 0.25,
                    label="PNA (trend-first)" if first else None, zorder=3)
            ax.scatter(lp2d[0, 0], lp2d[0, 1], s=140 if first else 55,
                       c="tab:blue", marker="*", edgecolors="black",
                       linewidths=0.8, alpha=1.0 if first else 0.4, zorder=5)
            pna_line_ints.append(lp_int)
            pna_tf_ints.append(tp_int)
            pna_starts.append(lp_ep["start"])

        # input endpoint
        x2d = project2d(extract_hidden(classifier, x.unsqueeze(0)).cpu().numpy())
        ax.scatter(x2d[:, 0], x2d[:, 1], s=140, c="black", marker="X",
                   label="input x", zorder=6)

        pna_line_m = float(np.mean(pna_line_ints))
        pna_tf_m = float(np.mean(pna_tf_ints))
        pna_start_m = float(np.mean(pna_starts))
        subtitle = (
            f"INTERIOR knn-dist (line)  zero={lz_int:.2f}  "
            f"PNA_top{anchors.shape[1]}={pna_line_m:.2f}\n"
            f"INTERIOR knn-dist (tf)    zero={tz_int:.2f}  "
            f"PNA_top{anchors.shape[1]}={pna_tf_m:.2f}\n"
            f"endpoint(start)           zero={lz_ep['start']:.2f}  "
            f"PNA={pna_start_m:.2f}  (PNA start low by definition)"
        )
        ax.set_title(f"{args.data} fold{args.fold} sample{si}  [{cond_tag}]\n{subtitle}",
                     fontsize=9)
        ax.set_xlabel("PC1 (standardized, fit on training hidden)")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        out_path = os.path.join(
            args.out_dir,
            f"{args.data}_fold{args.fold}_sample{si}_{cond_tag}_hidden.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[saved] {out_path}")
        print(f"    zero interior: line={lz_int:.3f} tf={tz_int:.3f} | "
              f"start={lz_ep['start']:.3f}")
        print(f"    PNA_top{anchors.shape[1]} interior: line={pna_line_m:.3f} "
              f"tf={pna_tf_m:.3f} | start={pna_start_m:.3f}")

    # ==================================================================
    # 2) 전체 test 정량 요약 (seed 고정 무작위 subset)
    #    PNA 는 sample별 top-Ka 평균 사용
    # ==================================================================
    if args.summary_n > 0:
        N = min(args.summary_n, x_test.shape[0])
        gg = th.Generator().manual_seed(args.seed + 1)
        sub = th.randperm(x_test.shape[0], generator=gg)[:N].tolist()
        agg = {("zero", "line"): [], ("zero", "tf"): [], ("zero", "rf"): [],
               ("pna", "line"): [], ("pna", "tf"): [], ("pna", "rf"): []}

        for j, si in enumerate(sub):
            x = x_test[si]
            c_zero = th.zeros_like(x)
            _, zl, _ = path_dist(line_path(c_zero, x, alphas))
            _, zt, _ = path_dist(two_phase_path(
                c_zero, x, alphas, args.kalman_obs_cov, args.kalman_trans_cov, "trend"))
            _, zr, _ = path_dist(two_phase_path(
                c_zero, x, alphas, args.kalman_obs_cov, args.kalman_trans_cov, "resid"))
            agg[("zero", "line")].append(zl)
            agg[("zero", "tf")].append(zt)
            agg[("zero", "rf")].append(zr)

            anc = select_pna_baselines(x.unsqueeze(0), cache, classifier,
                                       Ka=max(1, args.pna_ka))
            pl, pt, pr = [], [], []
            for k in range(anc.shape[1]):
                c_pna = anc[0, k]
                _, a_l, _ = path_dist(line_path(c_pna, x, alphas))
                _, a_t, _ = path_dist(two_phase_path(
                    c_pna, x, alphas, args.kalman_obs_cov, args.kalman_trans_cov, "trend"))
                _, a_r, _ = path_dist(two_phase_path(
                    c_pna, x, alphas, args.kalman_obs_cov, args.kalman_trans_cov, "resid"))
                pl.append(a_l); pt.append(a_t); pr.append(a_r)
            agg[("pna", "line")].append(float(np.mean(pl)))
            agg[("pna", "tf")].append(float(np.mean(pt)))
            agg[("pna", "rf")].append(float(np.mean(pr)))

            if (j + 1) % 25 == 0:
                print(f"  summary {j+1}/{N} ...")

        print(f"\n==== INTERIOR knn-dist summary (N={N}, k={args.knn_k}, "
              f"PNA=top{args.pna_ka} mean) : lower = more in-manifold ====")
        for path_tag in ["line", "tf", "rf"]:
            z = np.array(agg[("zero", path_tag)])
            pn = np.array(agg[("pna", path_tag)])
            win = float((pn < z).mean()) * 100
            print(f"[{path_tag:4s}] zero {z.mean():.3f}±{z.std():.3f} | "
                  f"PNA {pn.mean():.3f}±{pn.std():.3f} | "
                  f"PNA<zero in {win:.1f}% of samples")

        csv_path = os.path.join(
            args.out_dir,
            f"{args.data}_fold{args.fold}_{cond_tag}_interior_summary.csv")
        with open(csv_path, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["# metric: k-NN dist in per-dim standardized (z-score) "
                        "hidden space (NOT covariance whitening)"])
            w.writerow(["# aggregation: PNA = per-sample top-Ka anchor mean, "
                        f"Ka={args.pna_ka}; interior excludes both endpoints; "
                        f"knn_k={args.knn_k}; n_alphas={args.n_alphas}; N={N}; "
                        f"lam={lam_str}"])
            w.writerow(["path", "baseline", "agg", "mean", "std", "N",
                        "pct_PNA_below_zero"])
            for path_tag in ["line", "tf", "rf"]:
                z = np.array(agg[("zero", path_tag)])
                pn = np.array(agg[("pna", path_tag)])
                win = float((pn < z).mean()) * 100
                w.writerow([path_tag, "zero", "single", z.mean(), z.std(), len(z), ""])
                w.writerow([path_tag, "pna", f"top{args.pna_ka}_mean",
                            pn.mean(), pn.std(), len(pn), f"{win:.1f}"])
        print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
