"""
TR_table.py

TR_all.py + TR_all_fold_summary.py + positional_analysis.py(trend_share) 를 하나로 합침!

핵심:
  - trend_share = mean|A_T| / (mean|A_T| + mean|A_R|),  fold 별로 계산 후 평균)
  - CPD 는 fold 평균 ± std, 나머지(AUCC, Comp, Suff, Acc, CE, Log-odds)는 fold 평균.
  - 최종 출력은 논문 Table 3 형태 (데이터셋당 Trend/Residual 2행).

산출물 (results_table/):
  - result_TR_raw.csv   : (data, component) 별 mean/std 전체 (full precision)
  - result_TR.csv       : 소수점 3자리 반올림
  - result_table3.csv   : 논문 표 형태 (Ratio %, Dominant, CPD=mean±std, 나머지=mean)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PNA_HP = {
    "PAM":      dict(ka=10, lam0=0.5,  lamf=3.0),
    "boiler":   dict(ka=5,  lam0=10.0, lamf=3.0),
    "epilepsy": dict(ka=10, lam0=10.0, lamf=1.0),
    "wafer":    dict(ka=1,  lam0=0.5,  lamf=0.5),
}
def _lam_tag(data):
    h = PNA_HP[data]
    return f"_lam{h['lam0']}x{h['lamf']}"

# ────────────────────────────────────────────────────────────────────
# 데이터셋별 segment 설정 (npy 파일명 SEG 키 구성용) — README 값과 동일
# ────────────────────────────────────────────────────────────────────
SEG_CFG = {
    "epilepsy": dict(num_segments=50, min_seg_len=1, max_seg_len=48),
    "wafer":  dict(num_segments=50, min_seg_len=1, max_seg_len=48),
    "boiler":   dict(num_segments=50, min_seg_len=1, max_seg_len=48),
    "PAM":      dict(num_segments=50, min_seg_len=1, max_seg_len=48),
}

# full_eval.csv 컬럼 → 논문 표 metric 매핑
# CPD = cum_diff (metric=="CPD" 행)
METRIC_COLS = {
    "AUCC":     "AUCC",
    "Comp.":    "comprehensiveness",
    "Suff.":    "sufficiency",
    "Acc.":     "accuracy",
    "CE":       "cross_entropy",
    "Log-odds": "log_odds",
}
METRIC_ORDER = ["AUCC", "Comp.", "Suff.", "Acc.", "CE", "Log-odds"]


def seg_key(cfg):
    return f"kalman_seg{cfg['num_segments']}_min{cfg['min_seg_len']}_max{cfg['max_seg_len']}"


def npy_path(results_dir, data, kind, fold, cfg, model_type, seed):
    key = f"timing_td_{kind}_{seg_key(cfg)}{_lam_tag(data)}"   # ← _lam tag 추가!
    return Path(results_dir) / f"{data}_{model_type}_{key}_result_{fold}_{seed}.npy"


# ────────────────────────────────────────────────────────────────────
# 1. 실제 attribution magnitude Ratio / Dominant (fold 별)
# ────────────────────────────────────────────────────────────────────
def compute_ratio(results_dir, data, cfg, folds, model_type, seed):
    """
    fold 별 trend_share = mean|A_T| / (mean|A_T| + mean|A_R|) 를 계산.
    반환: dict(trend_ratio_mean, trend_ratio_std, dominant_label, n_folds, ok)
    """
    shares = []
    trend_wins = 0
    resid_wins = 0
    per_fold = []   # fold별 상세

    for f in folds:
        p_t = npy_path(results_dir, data, "trend", f, cfg, model_type, seed)
        p_r = npy_path(results_dir, data, "residual", f, cfg, model_type, seed)
        if not (p_t.exists() and p_r.exists()):
            print(f"  [warn] {data} fold {f}: npy 없음 → 스킵 ({p_t.name})")
            continue

        at = np.abs(np.load(p_t))   # [B,T,D], 이미 |T| 이지만 안전하게 abs
        ar = np.abs(np.load(p_r))

        mt = float(at.mean())
        mr = float(ar.mean())
        share = mt / (mt + mr + 1e-12)   # trend share
        shares.append(share)

        if mt > mr:
            trend_wins += 1
            direction = "Trend"
        elif mr > mt:
            resid_wins += 1
            direction = "Residual"
        else:
            direction = "Tie"

        per_fold.append(dict(
            data=data, fold=f,
            mean_abs_A_T=mt, mean_abs_A_R=mr,
            trend_ratio=share, residual_ratio=1.0 - share,
            direction=direction,
        ))

    if len(shares) == 0:
        return dict(ok=False, per_fold=[])

    shares = np.array(shares)
    n = len(shares)
    trend_mean = float(shares.mean())
    trend_std = float(shares.std(ddof=1)) if n > 1 else 0.0

    if trend_wins >= resid_wins:
        dominant = f"Trend ({trend_wins}/{n})"
    else:
        dominant = f"Residual ({resid_wins}/{n})"

    return dict(
        ok=True,
        n_folds=n,
        trend_ratio_mean=trend_mean,
        trend_ratio_std=trend_std,
        resid_ratio_mean=1.0 - trend_mean,
        resid_ratio_std=trend_std,   # residual share = 1 - trend share → std 동일
        dominant=dominant,
        per_fold=per_fold,
    )


# ────────────────────────────────────────────────────────────────────
# 2. faithfulness metric 집계 (full_eval.csv)
# ────────────────────────────────────────────────────────────────────
def extract_tr(method_str):
    if "trend" in method_str:
        return "Trend"
    if "residual" in method_str:
        return "Residual"
    return "Unknown"


def aggregate_metrics(eval_csv, mask_ref=None):
    """
    (data, TR) 별로 CPD(cum_diff) 및 나머지 metric 의 fold 평균/표준편차 반환.
    반환: dict[(data, TR)] -> dict(metric -> (mean, std))
    """
    df = pd.read_csv(eval_csv)
    if mask_ref is not None:
        df = df[df["mask_ref"] == mask_ref].copy()
        if len(df) == 0:
            raise ValueError(f"mask_ref={mask_ref} 행 없음")
    df["TR"] = df["method"].apply(extract_tr)
    cpd = df[df["metric"] == "CPD"].copy()

    # CPD = cum_diff, 나머지는 METRIC_COLS 매핑
    value_cols = {"CPD": "cum_diff", **METRIC_COLS}

    out = {}
    for (data, tr), g in cpd.groupby(["data", "TR"]):
        d = {}
        for disp, col in value_cols.items():
            vals = pd.to_numeric(g[col], errors="coerce").dropna().values
            if len(vals) == 0:
                d[disp] = (np.nan, np.nan)
            else:
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                d[disp] = (m, s)
        out[(data, tr)] = d

    dataset_order = df["data"].drop_duplicates().tolist()
    return out, dataset_order


# ────────────────────────────────────────────────────────────────────
# 3. 조립
# ────────────────────────────────────────────────────────────────────
def build(args):
    metrics, dataset_order = aggregate_metrics(args.eval_csv, args.mask_ref)
    folds = [int(x) for x in args.folds]

    raw_rows = []     # 전체 mean/std 롱포맷
    paper_rows = []   # 논문 표 포맷
    perfold_rows = [] # fold별 방향/비율 상세

    for data in dataset_order:
        cfg = SEG_CFG.get(data)
        if cfg is None:
            print(f"[warn] {data}: SEG_CFG 에 설정 없음 → Ratio 계산 스킵")
            ratio = dict(ok=False, per_fold=[])
        else:
            ratio = compute_ratio(
                args.results_dir, data, cfg, folds, args.model_type, args.seed
            )
        perfold_rows.extend(ratio.get("per_fold", []))

        for tr in ["Trend", "Residual"]:
            m = metrics.get((data, tr))
            if m is None:
                print(f"[warn] full_eval.csv 에 ({data}, {tr}) 없음 → 스킵")
                continue

            # Ratio (component 별)
            if ratio.get("ok"):
                if tr == "Trend":
                    r_mean, r_std = ratio["trend_ratio_mean"], ratio["trend_ratio_std"]
                else:
                    r_mean, r_std = ratio["resid_ratio_mean"], ratio["resid_ratio_std"]
                dominant = ratio["dominant"]
            else:
                r_mean = r_std = np.nan
                dominant = "N/A"

            cpd_m, cpd_s = m["CPD"]

            # ---- raw 롱포맷 (full precision) ----
            raw = dict(
                data=data, component=tr,
                Ratio_mean=r_mean, Ratio_std=r_std, Dominant=dominant,
                CPD_mean=cpd_m, CPD_std=cpd_s,
            )
            for disp in METRIC_ORDER:
                mm, ss = m[disp]
                raw[f"{disp}_mean"] = mm
                raw[f"{disp}_std"] = ss
            raw_rows.append(raw)

            # ---- 논문 표 포맷 ----
            paper = dict(
                Dataset=data,
                Component=tr,
                Ratio=(f"{100*r_mean:.1f}%" if np.isfinite(r_mean) else "N/A"),
                Dominant=dominant,
                CPD=(f"{cpd_m:.3f}±{cpd_s:.3f}" if np.isfinite(cpd_m) else "N/A"),
            )
            for disp in METRIC_ORDER:
                mm, _ = m[disp]
                paper[disp] = (f"{mm:.3f}" if np.isfinite(mm) else "N/A")
            paper_rows.append(paper)

    raw_df = pd.DataFrame(raw_rows)
    paper_df = pd.DataFrame(paper_rows)

    # 정렬: 데이터셋 원본 순서, Trend→Residual
    for d in (raw_df, paper_df):
        col = "data" if "data" in d.columns else "Dataset"
        comp = "component" if "component" in d.columns else "Component"
        d[col] = pd.Categorical(d[col], categories=dataset_order, ordered=True)
        d[comp] = pd.Categorical(d[comp], categories=["Trend", "Residual"], ordered=True)
        d.sort_values([col, comp], inplace=True)
        d.reset_index(drop=True, inplace=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "result_TR_raw.csv"
    round_path = out_dir / "result_TR.csv"
    paper_path = out_dir / "result_table3.csv"

    raw_df.to_csv(raw_path, index=False)
    num_cols = [c for c in raw_df.columns if c not in ("data", "component", "Dominant")]
    rounded = raw_df.copy()
    rounded[num_cols] = rounded[num_cols].round(3)
    rounded.to_csv(round_path, index=False)
    paper_df.to_csv(paper_path, index=False)

    # ---- fold별 방향/비율 상세 ----
    perfold_path = out_dir / "result_TR_perfold.csv"
    perfold_df = pd.DataFrame(perfold_rows)
    if len(perfold_df):
        perfold_df["data"] = pd.Categorical(perfold_df["data"], categories=dataset_order, ordered=True)
        perfold_df = perfold_df.sort_values(["data", "fold"]).reset_index(drop=True)
        num = ["mean_abs_A_T", "mean_abs_A_R", "trend_ratio", "residual_ratio"]
        perfold_df[num] = perfold_df[num].round(4)
        perfold_df.to_csv(perfold_path, index=False)

    print(f"\n[saved] {raw_path}   (full precision mean/std)")
    print(f"[saved] {round_path}   (3자리 반올림 mean/std)")
    print(f"[saved] {paper_path}   (논문 표 형태)")
    if len(perfold_df):
        print(f"[saved] {perfold_path}   (fold별 방향/비율)")

    # ---- 콘솔 프리뷰: 논문 표 형태 (중복 Dataset/Dominant 는 빈칸 처리) ----
    pretty = paper_df.copy()
    pretty["Dataset"] = pretty["Dataset"].astype(str)
    pretty["Component"] = pretty["Component"].astype(str)
    dup = pretty.duplicated(subset=["Dataset"])
    pretty.loc[dup, ["Dataset", "Dominant"]] = ""
    print("\n=== Table 3 (Ratio = 실제 |A_T|/|A_R| 질량비) ===")
    print(pretty.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results_our",
                   help="npy 저장 폴더 (Ratio 계산용)")
    p.add_argument("--eval_csv", default="./results_our/full_eval.csv",
                   help="eval_cpd_cpp.py 가 만든 metric csv")
    p.add_argument("--out_dir", default="./results_table")
    p.add_argument("--model_type", default="state")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    p.add_argument("--mask_ref", default="pna", choices=["zero","average","pna","na"])
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
