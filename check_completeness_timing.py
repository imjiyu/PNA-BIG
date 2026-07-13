"""
TIMING completeness 검산.

PNA-BIG는 check_completeness.py 로 sum(T_signed + R_signed) / (F(x)-F(c)) 를 잰다.
이 스크립트는 '같은 방식'으로 TIMING을 잰다:

    sum(A_TIMING_signed) / (F(x) - E_M[F(cM)])

TIMING은 단일 map이라 T/R 분해가 없고, main_td.py의 `timing_comp` 블록이
global-normalized signed attribution 과 fx(=F(x)-E_M[F(cM)]) 를 저장한다.

필요 npy (results_pna/ 기본):
    {data}_{model}_timing_comp_signed_{SEG}_result_{fold}_{seed}.npy
    {data}_{model}_timing_comp_fxc_{SEG}_result_{fold}_{seed}.npy
  SEG = seg{num_segments}_min{min_seg_len}_max{max_seg_len}

실행:
    python check_completeness_timing.py --results-dir ./results_pna
"""

import os
import argparse
import numpy as np


# check_completeness.py 와 동일한 세그먼트 설정 (run_td_all.sh 기준)
SEG_CONFIG = {
    "boiler":   {"num_segments": 50, "min_seg_len": 1,  "max_seg_len": 36},
    "epilepsy": {"num_segments": 10, "min_seg_len": 10, "max_seg_len": 10},
    "wafer":    {"num_segments": 5,  "min_seg_len": 10, "max_seg_len": 152},
    "PAM":      {"num_segments": 10, "min_seg_len": 10, "max_seg_len": 600},
}

DATASETS = ["boiler", "epilepsy", "wafer", "PAM"]


def make_seg(data):
    cfg = SEG_CONFIG[data]
    # 주의: TIMING comp key에는 'kalman' 접두어가 없다 (kalman smoother 미사용)
    return f"seg{cfg['num_segments']}_min{cfg['min_seg_len']}_max{cfg['max_seg_len']}"


def npy_path(results_dir, data, model, key, fold, seed):
    return os.path.join(
        results_dir,
        f"{data}_{model}_{key}_result_{fold}_{seed}.npy"
    )


def load_npy(results_dir, data, model, key, fold, seed):
    path = npy_path(results_dir, data, model, key, fold, seed)
    if not os.path.exists(path):
        return None
    return np.load(path)


def safe_ratio(sum_attr, fxc):
    eps = 1e-8
    denom = np.where(fxc >= 0, fxc + eps, fxc - eps)
    return sum_attr / denom


def subset_ratio_stats(ratio, fxc, keep_ratio):
    if keep_ratio >= 1.0:
        valid = np.ones_like(fxc, dtype=bool)
    else:
        threshold = np.percentile(np.abs(fxc), 100 * (1.0 - keep_ratio))
        valid = np.abs(fxc) > threshold

    if not np.any(valid):
        return None, None

    return (
        float(np.median(ratio[valid])),
        float(np.mean(ratio[valid] < 0)),
    )


def completeness_stats(attr, fxc):
    eps = 1e-8

    axes = tuple(range(1, attr.ndim))
    sum_attr = attr.sum(axis=axes)
    fxc = fxc.reshape(-1)

    if sum_attr.shape[0] != fxc.shape[0]:
        return None

    abs_err = np.abs(sum_attr - fxc)
    norm_err = abs_err / (np.abs(fxc).mean() + eps)

    ratio = safe_ratio(sum_attr, fxc)

    all_med, all_neg = subset_ratio_stats(ratio, fxc, keep_ratio=1.0)
    top75_med, top75_neg = subset_ratio_stats(ratio, fxc, keep_ratio=0.75)
    top50_med, top50_neg = subset_ratio_stats(ratio, fxc, keep_ratio=0.50)

    return {
        "all_med": all_med,
        "top75_med": top75_med,
        "top50_med": top50_med,
        "all_neg": all_neg,
        "top75_neg": top75_neg,
        "top50_neg": top50_neg,
        "norm_err_med": float(np.median(norm_err)),
    }


def fmt_float(x):
    return "N/A" if x is None else f"{x:.4f}"


def fmt_pct(x):
    return "N/A" if x is None else f"{x:.1%}"


def mean_std(rows, key, as_pct=False):
    vals = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
    if len(vals) == 0:
        return "N/A"
    if as_pct:
        return f"{vals.mean():.1%} ± {vals.std():.1%}"
    return f"{vals.mean():.4f} ± {vals.std():.4f}"


def check_dataset(data, model, seed, folds, results_dir):
    if data not in SEG_CONFIG:
        print(f"\n=== {data} ===")
        print(f"등록되지 않은 데이터셋입니다. 사용 가능: {DATASETS}")
        return

    seg = make_seg(data)

    attr_key = f"timing_comp_signed_{seg}"
    fxc_key = f"timing_comp_fxc_{seg}"

    print(f"\n=== TIMING / {data} / {seg} ===")
    print(
        f"{'fold':<5}"
        f"{'all_med':>10}"
        f"{'top75_med':>12}"
        f"{'top50_med':>12}"
        f"{'all_neg':>10}"
        f"{'top75_neg':>11}"
        f"{'top50_neg':>11}"
        f"{'norm_err':>10}"
    )
    print("-" * 81)

    rows = []
    for fold in folds:
        A = load_npy(results_dir, data, model, attr_key, fold, seed)
        fxc = load_npy(results_dir, data, model, fxc_key, fold, seed)

        if A is None or fxc is None:
            print(f"{fold:<5}  (npy 없음, 스킵)  기대: {npy_path(results_dir, data, model, attr_key, fold, seed)}")
            continue

        stats = completeness_stats(A, fxc)
        if stats is None:
            print(f"{fold:<5}  (표본 수 불일치 A={A.shape} fxc={fxc.shape}, 스킵)")
            continue

        rows.append(stats)
        print(
            f"{fold:<5}"
            f"{fmt_float(stats['all_med']):>10}"
            f"{fmt_float(stats['top75_med']):>12}"
            f"{fmt_float(stats['top50_med']):>12}"
            f"{fmt_pct(stats['all_neg']):>10}"
            f"{fmt_pct(stats['top75_neg']):>11}"
            f"{fmt_pct(stats['top50_neg']):>11}"
            f"{fmt_float(stats['norm_err_med']):>10}"
        )

    if not rows:
        print("-" * 81)
        print("유효한 fold 결과가 없습니다.")
        return

    print("-" * 81)
    print(f"{'avg all_med':<18}: {mean_std(rows, 'all_med')}")
    print(f"{'avg top75_med':<18}: {mean_std(rows, 'top75_med')}")
    print(f"{'avg top50_med':<18}: {mean_std(rows, 'top50_med')}")
    print(f"{'avg all_neg':<18}: {mean_std(rows, 'all_neg', as_pct=True)}")
    print(f"{'avg top75_neg':<18}: {mean_std(rows, 'top75_neg', as_pct=True)}")
    print(f"{'avg top50_neg':<18}: {mean_std(rows, 'top50_neg', as_pct=True)}")
    print(f"{'avg norm_err':<18}: {mean_std(rows, 'norm_err_med')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=DATASETS,
                    help="데이터셋 하나 또는 여러 개. 생략하면 전체")
    ap.add_argument("--model", default="state")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--results-dir", default="./results_pna",
                    help="main_td.py 저장 폴더 (기본 results_pna)")
    args = ap.parse_args()

    print("[TIMING 검산] sum(A_signed) / (F(x)-E_M[F(cM)])")
    print(f"       model={args.model}, seed={args.seed}, results_dir={args.results_dir}")
    print("       all=전체 샘플, top75/top50=|fxc| 상위 75/50%")
    print("       ratio는 1에 가까울수록 좋고, neg/norm_err는 0에 가까울수록 좋음")

    for data in args.data:
        check_dataset(
            data=data, model=args.model, seed=args.seed,
            folds=args.folds, results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
