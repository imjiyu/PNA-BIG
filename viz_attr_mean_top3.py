import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pytorch_lightning import seed_everything

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer
from datasets.freezer import Freezer

CFG = {
    "PAM":      (PAM,      17, 8, 600, False),
    "boiler":   (Boiler,   20, 2,  36, False),
    "epilepsy": (Epilepsy,  1, 2, 178, False),
    "wafer":    (Wafer,     1, 2, 152, True),
    "freezer":  (Freezer,   1, 2, 301, True),
}

NPY_PATTERNS = {
    "PAM": (
        "PAM_state_timing_td_trend_kalman_seg10_min10_max600_result_{fold}_42.npy",
        "PAM_state_timing_td_residual_kalman_seg10_min10_max600_result_{fold}_42.npy",
        "PAM_state_timing_td_trend_signed_kalman_seg10_min10_max600_result_{fold}_42.npy",
        "PAM_state_timing_td_residual_signed_kalman_seg10_min10_max600_result_{fold}_42.npy",
    ),
    "boiler": (
        "boiler_state_timing_td_trend_kalman_seg50_min1_max36_result_{fold}_42.npy",
        "boiler_state_timing_td_residual_kalman_seg50_min1_max36_result_{fold}_42.npy",
        "boiler_state_timing_td_trend_signed_kalman_seg50_min1_max36_result_{fold}_42.npy",
        "boiler_state_timing_td_residual_signed_kalman_seg50_min1_max36_result_{fold}_42.npy",
    ),
    "epilepsy": (
        "epilepsy_state_timing_td_trend_kalman_seg10_min10_max10_result_{fold}_42.npy",
        "epilepsy_state_timing_td_residual_kalman_seg10_min10_max10_result_{fold}_42.npy",
        "epilepsy_state_timing_td_trend_signed_kalman_seg10_min10_max10_result_{fold}_42.npy",
        "epilepsy_state_timing_td_residual_signed_kalman_seg10_min10_max10_result_{fold}_42.npy",
    ),
    "wafer": (
        "wafer_state_timing_td_trend_kalman_seg5_min10_max152_result_{fold}_42.npy",
        "wafer_state_timing_td_residual_kalman_seg5_min10_max152_result_{fold}_42.npy",
        "wafer_state_timing_td_trend_signed_kalman_seg5_min10_max152_result_{fold}_42.npy",
        "wafer_state_timing_td_residual_signed_kalman_seg5_min10_max152_result_{fold}_42.npy",
    ),
    "freezer": (
        "freezer_state_timing_td_trend_kalman_seg5_min10_max100_result_{fold}_42.npy",
        "freezer_state_timing_td_residual_kalman_seg5_min10_max100_result_{fold}_42.npy",
        "freezer_state_timing_td_trend_signed_kalman_seg5_min10_max100_result_{fold}_42.npy",
        "freezer_state_timing_td_residual_signed_kalman_seg5_min10_max100_result_{fold}_42.npy",
    ),
}

def make_time_ticks(T, max_ticks=8):
    """
    너무 촘촘하지 않게 time step tick 생성!
    """
    if T <= max_ticks:
        return np.arange(T)

    step = int(np.ceil((T - 1) / (max_ticks - 1)))
    ticks = np.arange(0, T, step)

    if ticks[-1] != T - 1:
        ticks = np.append(ticks, T - 1)

    return ticks

def build_datamodule(data, fold, seed):
    DM, _, _, _, needs_folds = CFG[data]
    if needs_folds:
        return DM(n_folds=5, fold=fold, seed=seed)
    return DM(fold=fold, seed=seed)

def select_top_channels(mean_abs_trend, mean_abs_resid, topk=3):
    """
    mean_abs_trend / mean_abs_resid : (T, C)
    topk <= 0 이면 전체 채널을 원래 순서(0..C-1)로 반환,
    그 외에는 channel_score = mean_t(|Trend|) + mean_t(|Residual|) 상위 topk 채널 반환!

    중요도 기준:
    channel_score = mean_t(|Trend|) + mean_t(|Residual|)
    """
    C = mean_abs_trend.shape[1]
    if topk <= 0:
        return np.arange(C)   # 전채널 모드

    ch_score = mean_abs_trend.mean(axis=0) + mean_abs_resid.mean(axis=0)
    topk = min(topk, C)
    top_channels = np.argsort(ch_score)[::-1][:topk]
    return top_channels

def plot_mean_heatmap(mean_trend, mean_resid, channel_indices,
                      save_path, tag, signed):
    """
    mean_trend / mean_resid : (T, C) - 샘플 평균
    채널별로 row 1개씩, Trend / Residual 나란히 표시
    """
    T, C = mean_trend.shape
    channel_indices = np.asarray(channel_indices, dtype=int)
    n_channels = len(channel_indices)
    suffix = "signed" if signed else "abs"
    cmap   = "RdBu_r" if signed else "Greens"

    fig_w = max(14, T // 20)
    fig_h = max(3.4, n_channels * 0.8 + 2.2)

    fig, axes = plt.subplots(
        n_channels, 3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [6, 6, 0.3], "wspace": 0.05, "hspace": 0.3},
    )
    if n_channels == 1:
        axes = [axes]   # 채널이 1개일 때 shape 통일

    fig.suptitle(f"{tag} - Mean Attribution ({suffix}) | length={T}", fontsize=13, y=0.95)

    if signed:
        vmax_global = max(
            np.abs(mean_trend[:, channel_indices]).max(),
            np.abs(mean_resid[:, channel_indices]).max(),
            1e-9,
        )
        vmin_global = -vmax_global
    else:
        vmin_global = 0.0
        vmax_global = max(
            mean_trend[:, channel_indices].max(),
            mean_resid[:, channel_indices].max(),
            1e-9,
        )

    for row_i, c in enumerate(channel_indices):
        ax_t, ax_r, ax_cb = axes[row_i]
        
        t_row = mean_trend[:, c]
        r_row = mean_resid[:, c]

        im = ax_t.imshow(t_row[np.newaxis, :], aspect="auto",
                        vmin=vmin_global, vmax=vmax_global, cmap=cmap,
                        extent=(-0.5, T - 0.5, -0.5, 0.5))

        ax_r.imshow(r_row[np.newaxis, :], aspect="auto",
                    vmin=vmin_global, vmax=vmax_global, cmap=cmap,
                    extent=(-0.5, T - 0.5, -0.5, 0.5))
        plt.colorbar(im, cax=ax_cb)

        for ax in (ax_t, ax_r):
            ax.set_yticks([])
            ax.set_xlim(-0.5, T - 0.5)

            if row_i == n_channels - 1:
                ticks = make_time_ticks(T, max_ticks=8)
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=8)
                ax.set_xlabel("time step t", fontsize=9)
            else:
                ax.set_xticks([])

        ax_t.set_ylabel(f"ch{c}", fontsize=8, rotation=0, labelpad=18)

        if row_i == 0:
            ax_t.set_title("Trend",    fontsize=9, pad=6)
            ax_r.set_title("Residual", fontsize=9, pad=6)

    top_margin = 0.72 if n_channels == 1 else 0.86

    fig.subplots_adjust(
        top=top_margin, bottom=0.22, left=0.04, right=0.96, wspace=0.05, hspace=0.45,)

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"  saved → {save_path}")
    plt.close()


def run_dataset(data, args):
    seed_everything(args.seed, workers=True)

    pat_at, pat_ar, pat_st, pat_sr = NPY_PATTERNS[data]
    abs_trend_list, abs_resid_list = [], []
    sgn_trend_list, sgn_resid_list = [], []

    for fold in range(5):
        paths = {
            "abs_trend":    os.path.join(args.npy_dir, pat_at.format(fold=fold)),
            "abs_resid":    os.path.join(args.npy_dir, pat_ar.format(fold=fold)),
            "signed_trend": os.path.join(args.npy_dir, pat_st.format(fold=fold)),
            "signed_resid": os.path.join(args.npy_dir, pat_sr.format(fold=fold)),
        }
        missing = [k for k, v in paths.items() if not os.path.exists(v)]
        if missing:
            print(f"  [skip fold {fold}] missing: {missing}")
            continue

        abs_trend_list.append(np.load(paths["abs_trend"]))
        abs_resid_list.append(np.load(paths["abs_resid"]))
        sgn_trend_list.append(np.load(paths["signed_trend"]))
        sgn_resid_list.append(np.load(paths["signed_resid"]))

    if not abs_trend_list:
        print(f"  [skip {data}] no files found")
        return

    # fold concat → 샘플 축 평균 → (T, C)
    mean_abs_trend    = np.concatenate(abs_trend_list, axis=0).mean(axis=0)
    mean_abs_resid    = np.concatenate(abs_resid_list, axis=0).mean(axis=0)
    mean_signed_trend = np.concatenate(sgn_trend_list, axis=0).mean(axis=0)
    mean_signed_resid = np.concatenate(sgn_resid_list, axis=0).mean(axis=0)

    top_channels = select_top_channels(
        mean_abs_trend,
        mean_abs_resid,
        topk=args.topk_channels,
    )
    
    print(f"  selected top channels: {top_channels.tolist()}")
    actual_topk = len(top_channels)

    viz_dir = os.path.join(args.viz_dir, data)
    os.makedirs(viz_dir, exist_ok=True)

    plot_mean_heatmap(mean_abs_trend, mean_abs_resid,
                      channel_indices=top_channels,
                      save_path=os.path.join(viz_dir, f"heatmap_mean_abs_top{actual_topk}.png"),
                      tag=f"{data} (all folds, top-{actual_topk} channels)",
                      signed=False)

    plot_mean_heatmap(mean_signed_trend, mean_signed_resid,
                  channel_indices=top_channels,
                  save_path=os.path.join(viz_dir, f"heatmap_mean_signed_top{actual_topk}.png"),
                  tag=f"{data} (all folds, top-{actual_topk} channels)",
                  signed=True)

    print(f"[done] {data} → {viz_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="all",
                   choices=list(CFG) + ["all"])
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--npy_dir",    default="results_our")
    p.add_argument("--viz_dir",    default="viz_out_top3")
    p.add_argument("--topk_channels", type=int, default=3,
                   help="중요도 상위 채널 수. 0 이하면 전채널(원래 순서)로 표시")
    args = p.parse_args()

    datasets = list(CFG) if args.data == "all" else [args.data]
    for data in datasets:
        print(f"\n── {data} ──")
        run_dataset(data, args)


if __name__ == "__main__":
    main()
