import os
import argparse
import numpy as np

# 색/cmap 정의와 plot_heatmap 은 positional_analysis 와 공유
from positional_analysis import plot_heatmap, DOMINANCE_CMAP, TOTAL_CMAP


def load_attr(results_dir, data, model_type, key, fold, seed):
    path = os.path.join(
        results_dir,
        f"{data}_{model_type}_{key}_result_{fold}_{seed}.npy",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)  # [B, T, D]


def parse_channel_names(channel_names):
    if channel_names is None or channel_names.strip() == "":
        return None
    return [x.strip() for x in channel_names.split(",")]


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--results_dir", type=str, default="./results_our")
    p.add_argument("--out_dir", type=str, default="./figs_positional")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model_type", type=str, default="state")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=str, default="0,1,2,3,4")

    p.add_argument("--num_segments", type=int, default=50)
    p.add_argument("--min_seg_len", type=int, default=1)
    p.add_argument("--max_seg_len", type=int, default=48)

    p.add_argument("--channel_names", type=str, default=None)

    args = p.parse_args()

    folds = [int(x.strip()) for x in args.folds.split(",")]
    channel_names = parse_channel_names(args.channel_names)

    SEG = f"kalman_seg{args.num_segments}_min{args.min_seg_len}_max{args.max_seg_len}"

    trend_list = []
    residual_list = []

    for fold in folds:
        at = load_attr(
            args.results_dir,
            args.data,
            args.model_type,
            f"timing_td_trend_{SEG}",
            fold,
            args.seed,
        )

        ar = load_attr(
            args.results_dir,
            args.data,
            args.model_type,
            f"timing_td_residual_{SEG}",
            fold,
            args.seed,
        )

        assert at.shape == ar.shape, f"shape mismatch fold {fold}: {at.shape} vs {ar.shape}"

        trend_list.append(np.abs(at))
        residual_list.append(np.abs(ar))

        print(f"[loaded] fold={fold}, shape={at.shape}")

    # fold별 sample 수가 달라도 안전하게 전체 sample concat
    trend_all = np.concatenate(trend_list, axis=0)      # [B_all, T, D]
    residual_all = np.concatenate(residual_list, axis=0)

    trend_td = trend_all.mean(axis=0)                   # [T, D]
    residual_td = residual_all.mean(axis=0)             # [T, D]
    total_td = trend_td + residual_td                   # [T, D]

    dominance_td = (trend_td - residual_td) / (total_td + 1e-12)

    B_all, T, D = trend_all.shape

    if channel_names is not None and len(channel_names) != D:
        raise ValueError(
            f"--channel_names length mismatch: got {len(channel_names)}, expected D={D}"
        )

    print("========== 5-fold heatmap summary ==========")
    print(f"all samples shape = {trend_all.shape}  (B_all, T, D)")
    print(f"mean total attribution = {total_td.mean():.6g}")
    print(f"mean trend attribution = {trend_td.mean():.6g}")
    print(f"mean residual attribution = {residual_td.mean():.6g}")
    print("============================================")

    os.makedirs(args.out_dir, exist_ok=True)

    tag = f"{args.data}_{args.model_type}_5fold_seed{args.seed}"

    plot_heatmap(
        total_td,
        save_path=os.path.join(args.out_dir, f"total_heatmap_{tag}.png"),
        title=f"{args.data} / {args.model_type} — 5-fold mean total attribution",
        cmap=TOTAL_CMAP,
        channel_names=channel_names,
    )

    plot_heatmap(
        dominance_td,
        save_path=os.path.join(args.out_dir, f"dominance_heatmap_{tag}.png"),
        title=f"{args.data} / {args.model_type} — 5-fold dominance (+trend, -residual)",
        cmap=DOMINANCE_CMAP,
        vmin=-1,
        vmax=1,
        channel_names=channel_names,
    )

    np.save(os.path.join(args.out_dir, f"trend_td_{tag}.npy"), trend_td)
    np.save(os.path.join(args.out_dir, f"residual_td_{tag}.npy"), residual_td)
    np.save(os.path.join(args.out_dir, f"total_td_{tag}.npy"), total_td)
    np.save(os.path.join(args.out_dir, f"dominance_td_{tag}.npy"), dominance_td)

    print(f"[saved] npy matrices to {args.out_dir}")


if __name__ == "__main__":
    main()
