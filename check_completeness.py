import os
import argparse
import numpy as np

PNA_HP = {
    "PAM":      dict(ka=10, lam0=0.5,  lamf=3.0),
    "boiler":   dict(ka=5,  lam0=10.0, lamf=3.0),
    "epilepsy": dict(ka=10, lam0=10.0, lamf=1.0),
    "wafer":    dict(ka=1,  lam0=0.5,  lamf=0.5),
}
def _lam_tag(data):
    h = PNA_HP[data]
    return f"_lam{h['lam0']}x{h['lamf']}"

SEG_CONFIG = {
    "boiler":   {"num_segments": 50, "min_seg_len": 1,  "max_seg_len": 48},
    "epilepsy": {"num_segments": 50, "min_seg_len": 1,  "max_seg_len": 48},
    "wafer":    {"num_segments": 50, "min_seg_len": 1,  "max_seg_len": 48},
    "PAM":      {"num_segments": 50, "min_seg_len": 1,  "max_seg_len": 48},
}

DATASETS = ["boiler", "epilepsy", "wafer", "PAM"]


def make_seg(data):
    cfg = SEG_CONFIG[data]
    return (
        f"kalman_seg{cfg['num_segments']}"
        f"_min{cfg['min_seg_len']}"
        f"_max{cfg['max_seg_len']}"
    )


def npy_path(results_dir, data, model, key, fold, seed):
    lam_tag = _lam_tag(data)

    return os.path.join(
        results_dir,
        f"{data}_{model}_{key}{lam_tag}_result_{fold}_{seed}.npy"
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

    # sample별 sum(T+R)이 F(x)-F(c)와 맞는지 확인
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

    trend_key = f"timing_td_trend_signed_{seg}"
    resid_key = f"timing_td_residual_signed_{seg}"
    fxc_key = f"timing_td_fxc_{seg}"

    print(f"\n=== {data} / {seg} ===")
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
        T = load_npy(results_dir, data, model, trend_key, fold, seed)
        R = load_npy(results_dir, data, model, resid_key, fold, seed)
        fxc = load_npy(results_dir, data, model, fxc_key, fold, seed)

        if T is None or R is None or fxc is None:
            print(f"{fold:<5}  (npy 없음, 스킵)")
            continue

        if T.shape != R.shape:
            print(f"{fold:<5}  (shape 불일치 T={T.shape} R={R.shape}, 스킵)")
            continue

        # completeness 검산 대상은 signed T + signed R
        stats = completeness_stats(T + R, fxc)

        if stats is None:
            print(f"{fold:<5}  (표본 수 불일치, 스킵)")
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
    ap.add_argument(
        "--data",
        nargs="+",
        default=DATASETS,
        help="데이터셋 하나 또는 여러 개. 생략하면 전체 데이터셋 실행",
    )
    ap.add_argument("--model", default="state")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--results-dir", default="./results_pna_10x10", help="PNA-BIG attribution 결과 폴더",)
    args = ap.parse_args()

    print("[검산] sum(T_signed + R_signed) / (F(x)-F(c))")
    print(f"       model={args.model}, seed={args.seed}, results_dir={args.results_dir}")
    print("       all=전체 샘플, top75=|fxc| 상위 75%, top50=|fxc| 상위 50%")
    print("       ratio는 1에 가까울수록 좋고, neg/norm_err는 0에 가까울수록 좋음")

    for data in args.data:
        check_dataset(
            data=data,
            model=args.model,
            seed=args.seed,
            folds=args.folds,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
