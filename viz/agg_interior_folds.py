"""
agg_interior_folds.py

viz_hidden_path.py 가 fold 별로 저장한
  {data}_fold{F}_{cond_tag}_interior_summary.csv
들을 모아서 데이터셋별로 fold 평균 ± fold 표준편차를 낸다.

중요:
  각 fold CSV 의 'std' 는 '그 fold 안 100샘플의 표준편차'다.
  여기서 내는 fold-std 는 5개 fold 의 'mean' 값들에 대한 표준편차 (fold 간 변동).
  둘은 다른 것이므로 섞지 않는다.

사용:
  python agg_interior_folds.py --root viz_hidden
  # 또는 데이터셋 하나만:
  python agg_interior_folds.py --root viz_hidden --only epilepsy
"""
import os
import re
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np

DATASETS = ["epilepsy", "wafer", "PAM", "boiler"]
PATHS = ["line", "tf", "rf"]


def parse_one_csv(path):
    """
    한 fold CSV 를 읽어 {(path, baseline): (mean, pct)} 반환.
    주석(#) 행은 건너뛴다.
    """
    rows = {}
    with open(path, newline="") as fp:
        for r in csv.reader(fp):
            if not r or r[0].startswith("#") or r[0] == "path":
                continue
            # path, baseline, agg, mean, std, N, pct_PNA_below_zero
            p, base, agg, mean, std, N = r[0], r[1], r[2], r[3], r[4], r[5]
            pct = r[6] if len(r) > 6 else ""
            rows[(p, base)] = (float(mean), pct)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="viz_hidden",
                    help="viz_hidden 루트 (데이터셋별 하위폴더 포함)")
    ap.add_argument("--only", default=None, help="특정 데이터셋만")
    ap.add_argument("--out", default="viz_hidden/fold_aggregated.csv")
    args = ap.parse_args()

    datasets = [args.only] if args.only else DATASETS

    out_rows = []
    for data in datasets:
        # 데이터셋 폴더 안 모든 fold summary csv
        pattern = os.path.join(args.root, data,
                               f"{data}_fold*_interior_summary.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[skip] {data}: no csv found ({pattern})")
            continue

        # (path, baseline) -> fold별 mean 리스트
        means = defaultdict(list)
        pcts = defaultdict(list)  # PNA<zero 비율 (pna 행에만)
        folds_found = []

        for f in files:
            m = re.search(r"_fold(\d+)_", os.path.basename(f))
            fold = int(m.group(1)) if m else -1
            folds_found.append(fold)
            rows = parse_one_csv(f)
            for (p, base), (mean, pct) in rows.items():
                means[(p, base)].append(mean)
                if base == "pna" and pct not in ("", None):
                    try:
                        pcts[p].append(float(pct))
                    except ValueError:
                        pass

        print(f"\n==== {data}  (folds: {sorted(folds_found)}) ====")
        print(f"{'path':4s} {'baseline':8s} "
              f"{'fold-mean':>10s} {'fold-std':>9s} {'n_folds':>7s} "
              f"{'PNA<zero%':>9s}")
        for p in PATHS:
            for base in ["zero", "pna"]:
                vals = np.array(means.get((p, base), []))
                if len(vals) == 0:
                    continue
                fm, fs = vals.mean(), vals.std()
                pct_str = ""
                if base == "pna" and p in pcts and len(pcts[p]) > 0:
                    pct_str = f"{np.mean(pcts[p]):.1f}"
                print(f"{p:4s} {base:8s} {fm:10.3f} {fs:9.3f} "
                      f"{len(vals):7d} {pct_str:>9s}")
                out_rows.append([data, p, base, fm, fs, len(vals), pct_str])

    # 통합 csv 저장
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["# fold-mean/std = mean & std across fold-level means "
                    "(NOT within-fold sample std)"])
        w.writerow(["data", "path", "baseline",
                    "fold_mean", "fold_std", "n_folds", "pct_PNA_below_zero"])
        w.writerows(out_rows)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()