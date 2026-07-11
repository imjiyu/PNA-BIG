import pandas as pd
import numpy as np
from pathlib import Path
import glob

# 경로 설정 -> 실험 바꾸면 '### 경로 수정' 부분 바꿔서 따로 저장
### timing100_dir = Path("results_100_jiyu")
abs_eval_dir  = Path("results_our") ### 경로 수정

raw_out_path   = Path("results_table/result_compare_raw.csv") ### 경로 수정
round_out_path = Path("results_table/result_compare.csv") ### 경로 수정
raw_out_path.parent.mkdir(parents=True, exist_ok=True)

# 공통 metric 순서
# 최종 출력 컬럼 순서 (모든 소스에서 이 순서로 맞춤)
metric_cols = [
    "cum_diff", "AUCC", "cum_50_diff",
    "accuracy", "comprehensiveness", "cross_entropy", "log_odds", "sufficiency",
]

datasets = ["boiler", "PAM", "epilepsy", "wafer", "freezer"]

"""
# 1. TIMING-100 파싱 (headerless CSV, Zeros baseline만!)
# 열 순서: seed,fold,baseline,topk,method,l1,l2,l3,
#          cum_50_diff,cum_diff,AUCC,accuracy,comp,ce,lodds,suff

timing100_col_names = [
    "seed", "fold", "baseline", "topk", "method",
    "lambda1", "lambda2", "lambda3",
    "cum_50_diff", "cum_diff", "AUCC",
    "accuracy", "comprehensiveness", "cross_entropy", "log_odds", "sufficiency",
]

rows_100 = []
for ds in datasets:
    for fold in range(5):
        fpath = timing100_dir / f"state_{ds}_{fold}_0_results_ht.csv"
        if not fpath.exists():
            print(f"[WARN] Missing: {fpath}")
            continue
        tmp = pd.read_csv(fpath, header=None, names=timing100_col_names)
        # Zeros baseline만
        tmp = tmp[tmp["baseline"] == "Zeros"]
        if tmp.empty:
            print(f"[WARN] No Zeros row in {fpath}")
            continue
        row = tmp.iloc[0]
        rows_100.append({
            "data": ds, "fold": int(fold), "method_label": "TIMING-100",
            **{c: float(row[c]) for c in metric_cols},
        })

df_100 = pd.DataFrame(rows_100)
"""

# 2. |T+R| (combined) 및 |T|+|R| (T_plus_R) 파싱
#    abs_full_eval_{dataset}.csv → CPD 행만

rows_abs = []
for ds in datasets:
    fpath = abs_eval_dir / f"abs_full_eval_{ds}.csv"
    if not fpath.exists():
        print(f"[WARN] Missing: {fpath}")
        continue
    tmp = pd.read_csv(fpath)
    tmp = tmp[tmp["metric"] == "CPD"]

    for _, r in tmp.iterrows():
        if "combined" in r["method"]:
            label = "|T+R|"
        elif "T_plus_R" in r["method"]:
            label = "|T|+|R|"
        else:
            continue
        rows_abs.append({
            "data": r["data"], "fold": int(r["fold"]), "method_label": label,
            **{c: float(r[c]) for c in metric_cols},
        })

df_abs = pd.DataFrame(rows_abs)


# 3. 합치고 데이터셋 × method별 mean/std 집계
### df_all = pd.concat([df_100, df_abs], ignore_index=True) 얘 대신 아래
df_all = df_abs.copy()

### method_order = ["TIMING-100", "|T+R|", "|T|+|R|"] 얘 대신 아래
method_order = ["|T+R|", "|T|+|R|"]

grp = df_all.groupby(["data", "method_label"])[metric_cols]
mean_df = grp.mean();          mean_df["stat"] = "mean"
std_df  = grp.std(ddof=1);     std_df["stat"]  = "std"

combined = pd.concat([mean_df, std_df]).reset_index()

# 정렬
combined["data"]         = pd.Categorical(combined["data"],         categories=datasets, ordered=True)
combined["method_label"] = pd.Categorical(combined["method_label"], categories=method_order, ordered=True)
combined["stat"]         = pd.Categorical(combined["stat"],         categories=["mean", "std"], ordered=True)

# 순서: 데이터셋 → stat(mean 먼저) → method
combined = combined.sort_values(["data", "stat", "method_label"]).reset_index(drop=True)

out_cols = ["data", "method_label", "stat"] + metric_cols
combined = combined[out_cols]

# 저장~~~
combined.to_csv(raw_out_path, index=False)
print(f"Saved raw → {raw_out_path}")

numeric = [c for c in combined.columns if c not in ("data", "method_label", "stat")]
rounded = combined.copy()
rounded[numeric] = rounded[numeric].round(3)
rounded.to_csv(round_out_path, index=False)
print(f"Saved rounded → {round_out_path}")

print("\n=== Preview (rounded) ===")
print(rounded.to_string(index=False))
