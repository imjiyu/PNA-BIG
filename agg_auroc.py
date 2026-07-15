#!/usr/bin/env python
"""
AUROC-masking 결과 집계.
입력: results_pna/eval_auroc/{data}.csv           (PNA-BIG, 헤더 1개)
      results_pna/eval_auroc/{data}_baselines.csv (7 baseline, cat 로 헤더 반복 가능)
출력(out_dir):
  - auroc_summary_long.csv : (data,method,mask_ref,k) fold mean/std/count
  - auroc_k{K}_table_{data}.csv / .tex : k=K 지점 drop mean±std (행=method, 열=mask_ref)
  - auroc_aopc{K}_table_{data}.csv      : AOPC@K mean±std
  - fig_auroc_{data}.png                : k vs AUROC-drop 곡선 (ref 별 subplot)
스코어 의미: auroc_drop ↑ = 중요한 걸 지웠을 때 성능이 더 떨어짐 = 더 faithful.
"""
import argparse, os, re
import numpy as np
import pandas as pd

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy 1.x/2.x 호환
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 저장 키 → 표시 이름
def method_label(m):
    m = str(m)
    if m.startswith("timing_td_combined"): return "PNA-BIG"
    if m.startswith("timing_sample"):      return "TIMING"
    return {
        "augmented_occlusion": "AFO",
        "gate_mask": "GateMask",
        "gradientshap_abs": "GradSHAP",
        "timex": "TimeX",
        "timex++": "TimeX++",
        "integrated_gradients_base_abs": "IG",
    }.get(m, m)

METHOD_ORDER = ["AFO","GateMask","GradSHAP","TimeX","TimeX++","IG","TIMING","PNA-BIG"]
REF_ORDER    = ["zero","average","pna","na"]
EXPECTED_FOLDS = {0,1,2,3,4}

def load_clean(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # cat 로 섞여 들어간 반복 헤더 제거 (data 컬럼 값이 'data' 인 행)
    df = df[df["data"] != "data"].copy()
    for c in ["fold","k","auroc","auroc_drop","base_auroc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["fold","k","auroc","auroc_drop","base_auroc"])
    df["fold"] = df["fold"].astype(int)
    return df

def load_dataset(in_dir, data):
    parts = []
    for suffix in ("", "_baselines"):
        p = os.path.join(in_dir, f"{data}{suffix}.csv")
        d = load_clean(p)
        if d is not None:
            parts.append(d)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["method"] = df["method"].map(method_label)
    # 혹시 모를 중복(재실행 double-append) 제거: 최신 결과 유지
    df = df.drop_duplicates(subset=["data","fold","method","mask_ref","k"], keep="last")
    return df

def order_methods(idx):
    present = [m for m in METHOD_ORDER if m in idx]
    present += [m for m in idx if m not in METHOD_ORDER]
    return present

def fmt(mean, std):
    return f"{mean:.3f}±{std:.3f}"

def warn_missing_folds(df, data):
    for (method, ref, k), grp in df.groupby(["method","mask_ref","k"]):
        found = set(grp["fold"].unique())
        if found != EXPECTED_FOLDS:
            print(f"[warn] {data} / {method} / {ref} / k={k}: "
                  f"fold={sorted(found)}, missing={sorted(EXPECTED_FOLDS-found)}")

def kpoint_table(df, k_point):
    sub = df[np.isclose(df["k"], k_point)].copy()
    if sub.empty:
        raise ValueError(f"k={k_point} 결과가 없습니다. 사용 가능한 k={sorted(df['k'].unique())}")
    g = sub.groupby(["method","mask_ref"])["auroc_drop"].agg(["mean","std","count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    for _, row in g[g["count"] != 5].iterrows():
        print(f"[warn] k={k_point} fold 부족: {row['method']} / {row['mask_ref']} / count={int(row['count'])}")
    piv_mean = g.pivot(index="method", columns="mask_ref", values="mean")
    piv_std  = g.pivot(index="method", columns="mask_ref", values="std")
    rows = order_methods(list(piv_mean.index))
    cols = [c for c in REF_ORDER if c in piv_mean.columns]
    cell = pd.DataFrame(index=rows, columns=cols, dtype=object)
    for r in rows:
        for c in cols:
            cell.loc[r, c] = fmt(piv_mean.loc[r, c], piv_std.loc[r, c])
    return cell, piv_mean.loc[rows, cols]

def to_latex(cell, mean_df, caption):
    # 각 열(ref)에서 drop 최대(=best)를 bold
    best = {c: mean_df[c].idxmax() for c in mean_df.columns}
    lines = [r"\begin{table}[t]\centering", f"\\caption{{{caption}}}",
             r"\begin{tabular}{l" + "c"*len(cell.columns) + "}", r"\toprule",
             "Method & " + " & ".join(cell.columns) + r" \\", r"\midrule"]
    for r in cell.index:
        cells = []
        for c in cell.columns:
            v = str(cell.loc[r, c]).replace("±", r"$\pm$")
            cells.append(f"\\textbf{{{v}}}" if best[c] == r else v)
        lines.append(f"{r} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

def aopc_table(df, max_k=0.2):
    """
    fold별로 k=0부터 max_k까지의 정규화된 AUROC-drop 면적을 계산하고,
    method × mask_ref별 fold 평균±표준편차를 반환한다.

    예:
      max_k=0.2 → AOPC@20%
    """
    recs = []
    for (m, ref, fold), grp in df.groupby(["method","mask_ref","fold"]):
        grp = grp[grp["k"] <= max_k + 1e-12].sort_values("k")
        if len(grp) < 2:
            print(f"[warn] AOPC 계산 불가: {m} / {ref} / fold={fold}, points={len(grp)}")
            continue

        ks = grp["k"].to_numpy(dtype=float)
        dr = grp["auroc_drop"].to_numpy(dtype=float)

        if not np.any(np.isclose(ks, 0.0)):
            print(f"[warn] {m} / {ref} / fold={fold}: k=0.0 결과가 없어 AOPC 계산에서 제외")
            continue
        if not np.any(np.isclose(ks, max_k)):
            print(f"[warn] {m} / {ref} / fold={fold}: k={max_k} 결과가 없어 AOPC 계산에서 제외")
            continue
        if not np.isfinite(dr).all():
            print(f"[warn] {m} / {ref} / fold={fold}: AUROC drop에 NaN/Inf가 있어 제외")
            continue

        span = ks.max() - ks.min()
        if span <= 0:
            print(f"[warn] {m} / {ref} / fold={fold}: 유효하지 않은 k 범위")
            continue

        area = _trapz(dr, ks) / span  # 정규화 AOPC = 해당 구간의 평균 AUROC drop
        recs.append(dict(method=m, mask_ref=ref, fold=fold, aopc=area))

    a = pd.DataFrame(recs)
    if a.empty:
        return pd.DataFrame(), pd.DataFrame()

    g = a.groupby(["method","mask_ref"])["aopc"].agg(["mean","std","count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    for _, row in g[g["count"] != 5].iterrows():
        print(f"[warn] AOPC fold 부족: {row['method']} / {row['mask_ref']} / count={int(row['count'])}")

    pm = g.pivot(index="method", columns="mask_ref", values="mean")
    ps = g.pivot(index="method", columns="mask_ref", values="std")
    rows = order_methods(list(pm.index))
    cols = [c for c in REF_ORDER if c in pm.columns]
    cell = pd.DataFrame(index=rows, columns=cols, dtype=object)
    for r in rows:
        for c in cols:
            cell.loc[r, c] = fmt(pm.loc[r, c], ps.loc[r, c])
    return cell, pm.loc[rows, cols]

def plot_curves(df, data, out_png):
    refs = [c for c in REF_ORDER if c in df["mask_ref"].unique()]
    fig, axes = plt.subplots(1, len(refs), figsize=(5*len(refs), 4), sharey=True)
    if len(refs) == 1:
        axes = [axes]
    methods = order_methods(list(df["method"].unique()))
    cmap = plt.get_cmap("tab10")
    for ax, ref in zip(axes, refs):
        sub = df[df["mask_ref"] == ref]
        for i, m in enumerate(methods):
            s = sub[sub["method"] == m]
            if s.empty: continue
            g = s.groupby("k")["auroc_drop"].agg(["mean","std"]).reset_index().sort_values("k")
            g["std"] = g["std"].fillna(0.0)
            style = dict(lw=2.4, marker="o", ms=4) if m == "PNA-BIG" else dict(lw=1.4, marker=".", ms=3, alpha=0.85)
            ax.plot(g["k"], g["mean"], color=cmap(i % 10), label=m, **style)
            ax.fill_between(g["k"], g["mean"]-g["std"], g["mean"]+g["std"], color=cmap(i % 10), alpha=0.12)
        ax.set_title(f"{data}  (mask ref = {ref})")
        ax.set_xlabel("masking fraction k")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC drop  (↑ = more faithful)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results_pna/eval_auroc")
    ap.add_argument("--datasets", nargs="+", default=["wafer","boiler","epilepsy","PAM"])
    ap.add_argument("--out_dir", default="results_pna/eval_auroc/agg")
    ap.add_argument("--k_point", type=float, default=0.1)
    ap.add_argument("--aopc_max_k", type=float, default=0.2, help="AOPC 계산 최대 masking 비율. 기본값 0.2 = AOPC@20%%")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    long_all = []
    for data in args.datasets:
        df = load_dataset(args.in_dir, data)
        if df is None:
            print(f"[skip] {data}: no csv"); continue

        warn_missing_folds(df, data)

        # long summary
        summ = (df.groupby(["data","method","mask_ref","k"])["auroc_drop"]
                  .agg(["mean","std","count"]).reset_index())
        long_all.append(summ)

        # k-point table
        cell, mean_df = kpoint_table(df, args.k_point)
        cell.to_csv(os.path.join(args.out_dir, f"auroc_k{args.k_point}_table_{data}.csv"))
        with open(os.path.join(args.out_dir, f"auroc_k{args.k_point}_table_{data}.tex"), "w") as f:
            f.write(to_latex(cell, mean_df,
                    f"{data}: AUROC drop at top-{int(args.k_point*100)}\\% masking (mean$\\pm$std, higher=better)"))

        # AOPC@max_k
        acell, _ = aopc_table(df, args.aopc_max_k)
        aopc_pct = int(round(args.aopc_max_k * 100))
        if not acell.empty:
            acell.to_csv(os.path.join(args.out_dir, f"auroc_aopc{aopc_pct}_table_{data}.csv"))
        else:
            print(f"[warn] {data}: AOPC@{aopc_pct}% 결과가 비어 있어 CSV를 저장하지 않음")

        # plot
        plot_curves(df, data, os.path.join(args.out_dir, f"fig_auroc_{data}.png"))

        print(f"\n===== {data} : AUROC drop @ k={args.k_point} (mean±std, ↑ better) =====")
        print(cell.to_string())

    if long_all:
        pd.concat(long_all, ignore_index=True).to_csv(
            os.path.join(args.out_dir, "auroc_summary_long.csv"), index=False)
    print(f"\n[done] -> {args.out_dir}")

if __name__ == "__main__":
    main()