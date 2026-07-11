"""
probe_redundancy.py

경로 순서 뒤집힘(PAM dominant flip 등)의 원인을 직접 측정하는 진단 스크립트.

아이디어 (component sufficiency probe):
  baseline c=0 에서, 각 성분을 "단독으로" 모델에 넣었을 때 예측 클래스 확률을
  얼마나 회복하는지를 잰다. TIMING 마스크 없이 통짜로 T, R 를 넣는다.

    dfull = F_y(x) - F_y(c)          # 전체 효과
    dT    = F_y(T) - F_y(c)          # trend 단독
    dR    = F_y(R) - F_y(c)          # residual 단독
    (F_y = predict 의 softmax 확률,  y = argmax F(x) = 파이프라인과 동일한 예측 클래스)

  성분별 회복률:
    r_T = dT / dfull ,  r_R = dR / dfull
  중복도(redundancy) 지표:
    r_sum = r_T + r_R = (dT + dR) / dfull

해석:
  r_sum >> 1  : 두 성분이 겹치는(redundant) 증거를 담음 → 어느 쪽을 먼저 적분하느냐로
                공유 credit 이 크게 재분배됨 → path-order 스윙 큼 (PAM 예상)
  r_sum ≈ 1   : 상보적(complementary)/가법적 → 스윙 작음
  r_T≈1, r_R≈0: trend 단독으로 충분, residual 은 거의 노이즈 (Wafer/Freezer 예상)

사용:
  python probe_redundancy.py --data PAM --device cuda:0
  python probe_redundancy.py --data PAM boiler freezer epilepsy wafer --device cuda:0 --max_samples 500
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

from real.classifier import MimicClassifierNet
from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer
from datasets.freezer import Freezer
from attribution.explainers_td import compute_trend_kalman  # 두 explainer 공통 함수


# ── main_td.py 와 동일한 데이터셋별 설정 ─────────────────────────────
CFG = {
    "PAM":      dict(feature_size=17, n_state=8, n_timesteps=600,
                     make=lambda fold, seed: PAM(fold=fold, seed=seed)),
    "boiler":   dict(feature_size=20, n_state=2, n_timesteps=36,
                     make=lambda fold, seed: Boiler(fold=fold, seed=seed)),
    "epilepsy": dict(feature_size=1,  n_state=2, n_timesteps=178,
                     make=lambda fold, seed: Epilepsy(fold=fold, seed=seed)),
    "freezer":  dict(feature_size=1,  n_state=2, n_timesteps=301,
                     make=lambda fold, seed: Freezer(n_folds=5, fold=fold, seed=seed)),
    "wafer":    dict(feature_size=1,  n_state=2, n_timesteps=152,
                     make=lambda fold, seed: Wafer(n_folds=5, fold=fold, seed=seed)),
}


def build_classifier(cfg, model_type):
    return MimicClassifierNet(
        feature_size=cfg["feature_size"],
        n_state=cfg["n_state"],
        n_timesteps=cfg["n_timesteps"],
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
        model_type=model_type,
    )


@th.no_grad()
def predict_probs(clf, x, bs, device, desc=None):
    """x: [N,T,D] → softmax 확률 [N,C] (파이프라인과 동일하게 predict 사용)."""
    outs = []
    n_batches = (x.shape[0] + bs - 1) // bs
    iterator = range(0, x.shape[0], bs)
    if desc is not None:
        iterator = tqdm(iterator, total=n_batches, desc=desc, leave=False)
    for i in iterator:
        xb = x[i:i + bs].to(device).float()
        p = clf.predict(xb, mask=None, timesteps=None, return_all=False)
        outs.append(p.detach().cpu())
    return th.cat(outs, dim=0)


def recovery_stats(pf, pc, pt, pr, thr):
    """
    pf,pc,pt,pr: [N] 예측클래스 확률 (full / baseline / trend / residual).
    반환: dict(trend_recovery, residual_recovery,
               redundancy(=median-sum), redundancy_sample_median, redundancy_pooled,
               n_valid, n_total, valid_rate, pooled 성분별 회복률)
    """
    dfull = pf - pc
    dT = pt - pc
    dR = pr - pc

    n_total = int(dfull.numel())
    valid = dfull > thr  # dfull 이 너무 작으면 비율이 폭주 → 제외
    n_valid = int(valid.sum())
    valid_rate = n_valid / n_total if n_total > 0 else 0.0

    if n_valid == 0:
        return dict(trend_recovery=np.nan, residual_recovery=np.nan,
                    redundancy=np.nan,
                    redundancy_sample_median=np.nan, redundancy_pooled=np.nan,
                    n_valid=0, n_total=n_total, valid_rate=valid_rate,
                    trend_rec_pooled=np.nan, residual_rec_pooled=np.nan)

    rT = (dT[valid] / dfull[valid]).numpy()
    rR = (dR[valid] / dfull[valid]).numpy()

    trend_rec = float(np.median(rT))
    resid_rec = float(np.median(rR))

    # pooled(합 기준) 교차검증: median 비율과 크게 다르면 소수 이상치 영향
    trend_pooled = float(dT[valid].sum() / dfull[valid].sum())
    resid_pooled = float(dR[valid].sum() / dfull[valid].sum())

    # 세 가지 redundancy 를 함께 저장 (세 값이 일치할수록 해석이 강해짐)
    #  1) median-sum        : median(rT) + median(rR)   -- 성분별 대표 회복률의 합
    #  2) sample-wise median: median(rT + rR)           -- 샘플별 중복도의 대표값
    #  3) pooled            : sum(dT+dR) / sum(dfull)    -- 소수 이상치에 강건
    redundancy_median_sum = trend_rec + resid_rec
    redundancy_sample_median = float(np.median(rT + rR))
    redundancy_pooled = trend_pooled + resid_pooled

    return dict(
        trend_recovery=trend_rec,
        residual_recovery=resid_rec,
        redundancy=redundancy_median_sum,
        redundancy_sample_median=redundancy_sample_median,
        redundancy_pooled=redundancy_pooled,
        n_valid=n_valid,
        n_total=n_total,
        valid_rate=valid_rate,
        trend_rec_pooled=trend_pooled,
        residual_rec_pooled=resid_pooled,
    )


def run_dataset(data, args):
    cfg = CFG[data]
    device = th.device(args.device if th.cuda.is_available() or "cpu" in args.device else "cpu")
    rows = []

    for fold in args.folds:
        ckpt = f"./model/{data}/{args.model_type}_classifier_{fold}_{args.seed}_no_imputation"
        if not os.path.exists(ckpt):
            print(f"[warn] {data} fold {fold}: 체크포인트 없음 → 스킵 ({ckpt})")
            continue

        t0 = time.time()

        clf = build_classifier(cfg, args.model_type)
        clf.load_state_dict(th.load(ckpt, map_location="cpu"))
        clf.eval().to(device)

        dm = cfg["make"](fold, args.seed)

        # PAM 등 일부 dataset은 train preprocess에서 mean/std를 먼저 세팅해야 test normalize가 가능함
        dm.preprocess(split="train")

        x_test = dm.preprocess(split="test")["x"].float()  # [N,T,D]

        # 랜덤 서브샘플링 (앞부분만 자르면 데이터셋 정렬/클래스 순서에 따라 편향 가능)
        if args.max_samples and args.max_samples > 0 and x_test.shape[0] > args.max_samples:
            g = th.Generator().manual_seed(args.seed + fold)  # fold별로 다르지만 재현 가능하게
            perm = th.randperm(x_test.shape[0], generator=g)[:args.max_samples]
            x_test = x_test[perm]

        n_samples = x_test.shape[0]
        print(f"  {data} fold {fold}: n_samples={n_samples}  (device={device})")

        # 예측 클래스 (= argmax F(x), 파이프라인과 동일)
        p_full = predict_probs(clf, x_test, args.bs, device, desc=f"{data} f{fold} predict(x)")   # [N,C]
        y = p_full.argmax(dim=-1)                              # [N]

        # trend / residual (통짜, TIMING 마스크 없음). c = 0.
        # Kalman은 배치 단위로 돌면서 진행상황 표시
        T_list = []
        for i in tqdm(range(0, n_samples, args.bs), desc=f"{data} f{fold} kalman",
                      total=(n_samples + args.bs - 1) // args.bs, leave=False):
            xb = x_test[i:i + args.bs].to(device)
            Tb = compute_trend_kalman(xb, args.kalman_obs_cov, args.kalman_trans_cov)
            T_list.append(Tb.detach().cpu())
        T = th.cat(T_list, dim=0)
        R = x_test - T
        c = th.zeros_like(x_test)

        p_c = predict_probs(clf, c, args.bs, device, desc=f"{data} f{fold} predict(c)")
        p_T = predict_probs(clf, T, args.bs, device, desc=f"{data} f{fold} predict(T)")
        p_R = predict_probs(clf, R, args.bs, device, desc=f"{data} f{fold} predict(R)")

        idx = y.view(-1, 1)
        pf = p_full.gather(1, idx).squeeze(1)
        pc = p_c.gather(1, idx).squeeze(1)
        pt = p_T.gather(1, idx).squeeze(1)
        pr = p_R.gather(1, idx).squeeze(1)

        st = recovery_stats(pf, pc, pt, pr, args.dfull_thr)
        st.update(dict(data=data, fold=fold))
        rows.append(st)

        elapsed = time.time() - t0
        print(f"  {data} fold {fold}: "
              f"trend_rec={st['trend_recovery']:.3f}  "
              f"resid_rec={st['residual_recovery']:.3f}  |  "
              f"redun(med-sum={st['redundancy']:.3f}, "
              f"samp-med={st['redundancy_sample_median']:.3f}, "
              f"pooled={st['redundancy_pooled']:.3f})  "
              f"valid={st['n_valid']}/{st['n_total']} ({st['valid_rate']:.1%})  "
              f"[{elapsed:.1f}s]")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["PAM"],
                    choices=list(CFG.keys()))
    ap.add_argument("--model_type", default="state")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=500,
                    help="fold당 test 샘플 상한 (0=전체). Kalman이 느려서 기본 500. "
                         "상한을 넘으면 앞부분이 아니라 랜덤 서브샘플링함(재현 가능, seed+fold 고정).")
    ap.add_argument("--dfull_thr", type=float, default=0.05,
                    help="dfull 이 값보다 커야 비율 집계에 포함 (비율 폭주 방지).")
    ap.add_argument("--kalman_obs_cov", type=float, default=1.0)
    ap.add_argument("--kalman_trans_cov", type=float, default=0.01)  # 논문 ρ=0.01
    ap.add_argument("--out", default="./results_table/redundancy_probe.csv")
    args = ap.parse_args()

    print("[probe] component sufficiency: dT=F(T)-F(c), dR=F(R)-F(c), dfull=F(x)-F(c)")
    print("        redundancy = median(dT/dfull) + median(dR/dfull)")
    print("        >1 겹침(스윙 큼) / ≈1 상보 / trend_rec≈1&resid_rec≈0 = trend 단독 충분\n")

    all_rows = []
    for data in args.data:
        print(f"=== {data} ===")
        all_rows.extend(run_dataset(data, args))

    if not all_rows:
        print("결과 없음 (체크포인트 확인).")
        return

    df = pd.DataFrame(all_rows)

    # fold 평균 요약 (valid_rate, 세 redundancy 함께)
    summary = (
        df.groupby("data")[["trend_recovery", "residual_recovery",
                            "redundancy", "redundancy_sample_median",
                            "redundancy_pooled", "valid_rate"]]
        .agg(["mean", "std"])
        .round(3)
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.round(4).to_csv(args.out, index=False)
    summary.to_csv(args.out.replace(".csv", "_summary.csv"))

    print("\n===== fold 평균 요약 =====")
    print(summary.to_string())
    print(f"\n[saved] {args.out}")
    print(f"[saved] {args.out.replace('.csv', '_summary.csv')}")


if __name__ == "__main__":
    main()