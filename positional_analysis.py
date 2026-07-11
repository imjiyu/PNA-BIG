"""
Positional analysis: time-axis / position-level magnitude of |A_T| and |A_R|.
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

TREND_COLOR = "#009E73"       # green
RESIDUAL_COLOR = "#7B3294"    # purple
NEUTRAL_COLOR = "#F7F7F7"

TOTAL_CMAP = "Greys"

DOMINANCE_CMAP = LinearSegmentedColormap.from_list(
    "residual_white_trend",
    [RESIDUAL_COLOR, NEUTRAL_COLOR, TREND_COLOR],
)

# ---------- loading ----------

def load_attr(results_dir, data, model_type, key, fold, seed):
    path = os.path.join(
        results_dir,
        f"{data}_{model_type}_{key}_result_{fold}_{seed}.npy",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)  # [B, T, D]


# ---------- utility ----------

def aggregate(arr, axis=0, agg="mean"):
    """
    Shared aggregator used by figures and CSV summaries.
    """
    if agg == "mean":
        return arr.mean(axis=axis)
    elif agg == "median":
        return np.median(arr, axis=axis)
    else:
        raise ValueError(f"Unknown agg: {agg}")


def winner_from_values(t_val, r_val):
    if t_val > r_val:
        return "trend"
    elif r_val > t_val:
        return "residual"
    else:
        return "tie"


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[saved] {path}")


def parse_channel_names(channel_names):
    if channel_names is None or channel_names.strip() == "":
        return None
    return [x.strip() for x in channel_names.split(",")]


# ---------- statistics ----------

def positional_stats(attr_abs, band="sem", agg="mean"):
    """
    attr_abs: [B, T, D], already abs-applied.

    Feature/channel dimension D is averaged first:
      [B, T, D] -> [B, T]

    Then sample dimension B is summarized using agg.
    This is synchronized with compute_time_curves().
    """
    per_sample = attr_abs.mean(axis=2)  # [B, T]
    center = aggregate(per_sample, axis=0, agg=agg)

    if band == "std":
        sd = per_sample.std(axis=0)
        lower = center - sd
        upper = center + sd
        label = f"{agg} ± std"

    elif band == "sem":
        if agg != "mean":
            raise ValueError(
                "--band sem은 --agg mean에서만 사용하세요. "
                "median 기반이면 --band iqr를 권장합니다."
            )
        sem = per_sample.std(axis=0) / max(np.sqrt(per_sample.shape[0]), 1.0)
        lower = center - 1.96 * sem
        upper = center + 1.96 * sem
        label = "mean ± 95% CI"

    elif band == "iqr":
        q1 = np.quantile(per_sample, 0.25, axis=0)
        q3 = np.quantile(per_sample, 0.75, axis=0)
        lower = q1
        upper = q3

        if agg == "median":
            label = "median + IQR"
        else:
            label = "mean + IQR band"

    else:
        raise ValueError(f"Unknown band: {band}")

    return dict(
        center=center,
        lower=lower,
        upper=upper,
        label=label,
    )


def compute_time_curves(at_abs, ar_abs, agg="mean", eps=1e-12):
    """
    at_abs, ar_abs: [B, T, D], already abs-applied.

    Returns time-level curves.
    This uses the same aggregation rule as plot_overall().
    """
    at_bt = at_abs.mean(axis=2)  # [B, T]
    ar_bt = ar_abs.mean(axis=2)  # [B, T]

    trend_t = aggregate(at_bt, axis=0, agg=agg)
    residual_t = aggregate(ar_bt, axis=0, agg=agg)
    total_t = trend_t + residual_t

    trend_share = trend_t / (total_t + eps)
    dominance = (trend_t - residual_t) / (total_t + eps)

    return dict(
        trend=trend_t,
        residual=residual_t,
        total=total_t,
        trend_share=trend_share,
        dominance=dominance,
    )


def compute_position_matrices(at_abs, ar_abs, agg="mean", eps=1e-12):
    """
    at_abs, ar_abs: [B, T, D], already abs-applied.

    Returns position-level matrices with shape [T, D].
    """
    trend_td = aggregate(at_abs, axis=0, agg=agg)       # [T, D]
    residual_td = aggregate(ar_abs, axis=0, agg=agg)    # [T, D]
    total_td = trend_td + residual_td

    trend_share_td = trend_td / (total_td + eps)
    dominance_td = (trend_td - residual_td) / (total_td + eps)

    return dict(
        trend=trend_td,
        residual=residual_td,
        total=total_td,
        trend_share=trend_share_td,
        dominance=dominance_td,
    )


# ---------- summaries ----------

def save_time_summary(curves, save_path):
    rows = []
    T = len(curves["trend"])

    for t in range(T):
        trend = float(curves["trend"][t])
        residual = float(curves["residual"][t])
        total = float(curves["total"][t])
        share = float(curves["trend_share"][t])
        dominance = float(curves["dominance"][t])

        rows.append(dict(
            t=t,
            trend_abs=trend,
            residual_abs=residual,
            total_abs=total,
            trend_share=share,
            dominance_signed=dominance,
            winner=winner_from_values(trend, residual),
            abs_gap=abs(trend - residual),
        ))

    save_csv(
        save_path,
        fieldnames=[
            "t",
            "trend_abs",
            "residual_abs",
            "total_abs",
            "trend_share",
            "dominance_signed",
            "winner",
            "abs_gap",
        ],
        rows=rows,
    )


def top_indices_1d(values, topk):
    values = np.asarray(values)
    idx = np.argsort(values)[::-1]
    idx = idx[np.isfinite(values[idx])]
    return idx[:topk]


def save_top_time_summary(curves, save_path, topk=10):
    rows = []

    targets = [
        ("trend_magnitude", curves["trend"]),
        ("residual_magnitude", curves["residual"]),
        ("total_magnitude", curves["total"]),
        ("trend_dominance", curves["dominance"]),
        ("residual_dominance", -curves["dominance"]),
    ]

    for kind, score_arr in targets:
        idxs = top_indices_1d(score_arr, topk)

        for rank, t in enumerate(idxs, start=1):
            trend = float(curves["trend"][t])
            residual = float(curves["residual"][t])
            total = float(curves["total"][t])
            share = float(curves["trend_share"][t])
            dominance = float(curves["dominance"][t])

            rows.append(dict(
                rank=rank,
                type=kind,
                t=int(t),
                score=float(score_arr[t]),
                trend_abs=trend,
                residual_abs=residual,
                total_abs=total,
                trend_share=share,
                dominance_signed=dominance,
                winner=winner_from_values(trend, residual),
            ))

    save_csv(
        save_path,
        fieldnames=[
            "rank",
            "type",
            "t",
            "score",
            "trend_abs",
            "residual_abs",
            "total_abs",
            "trend_share",
            "dominance_signed",
            "winner",
        ],
        rows=rows,
    )


def top_indices_2d(score_mat, topk, mask=None):
    score = np.array(score_mat, dtype=float, copy=True)

    if mask is not None:
        score[~mask] = -np.inf

    flat = score.reshape(-1)
    idx = np.argsort(flat)[::-1]
    idx = idx[np.isfinite(flat[idx])]
    return idx[:topk]


def save_top_position_summary(
    pos,
    save_path,
    topk=20,
    active_percentile=75.0,
    channel_names=None,
):
    """
    Saves top (time, channel) positions.

    Dominance rankings are filtered by total attribution magnitude to avoid
    meaningless high ratios in near-zero regions.
    """
    trend = pos["trend"]          # [T, D]
    residual = pos["residual"]    # [T, D]
    total = pos["total"]          # [T, D]
    share = pos["trend_share"]    # [T, D]
    dominance = pos["dominance"]  # [T, D]

    T, D = trend.shape

    active_thr = np.percentile(total, active_percentile)
    active_mask = total >= active_thr

    rows = []

    targets = [
        ("trend_position_magnitude", trend, None),
        ("residual_position_magnitude", residual, None),
        ("total_position_magnitude", total, None),
        ("trend_dominant_position", dominance, active_mask),
        ("residual_dominant_position", -dominance, active_mask),
    ]

    for kind, score_mat, mask in targets:
        flat_idxs = top_indices_2d(score_mat, topk, mask=mask)

        for rank, flat_idx in enumerate(flat_idxs, start=1):
            t, d = np.unravel_index(flat_idx, (T, D))

            t_val = float(trend[t, d])
            r_val = float(residual[t, d])
            total_val = float(total[t, d])
            share_val = float(share[t, d])
            dom_val = float(dominance[t, d])

            if channel_names is not None and d < len(channel_names):
                ch_name = channel_names[d]
            else:
                ch_name = ""

            rows.append(dict(
                rank=rank,
                type=kind,
                t=int(t),
                channel=int(d),
                channel_name=ch_name,
                score=float(score_mat[t, d]),
                trend_abs=t_val,
                residual_abs=r_val,
                total_abs=total_val,
                trend_share=share_val,
                dominance_signed=dom_val,
                winner=winner_from_values(t_val, r_val),
                active_threshold=active_thr,
            ))

    save_csv(
        save_path,
        fieldnames=[
            "rank",
            "type",
            "t",
            "channel",
            "channel_name",
            "score",
            "trend_abs",
            "residual_abs",
            "total_abs",
            "trend_share",
            "dominance_signed",
            "winner",
            "active_threshold",
        ],
        rows=rows,
    )


def contiguous_regions(mask):
    regions = []
    start = None

    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            regions.append((start, i - 1))
            start = None

    if start is not None:
        regions.append((start, len(mask) - 1))

    return regions


def save_region_summary(
    curves,
    save_path,
    dominance_threshold=0.55,
    active_percentile=60.0,
):
    """
    Trend-dominant region:
      trend_share >= dominance_threshold

    Residual-dominant region:
      trend_share <= 1 - dominance_threshold

    Low-attribution regions are filtered by total magnitude percentile.
    """
    trend = curves["trend"]
    residual = curves["residual"]
    total = curves["total"]
    share = curves["trend_share"]
    dominance = curves["dominance"]

    active_thr = np.percentile(total, active_percentile)
    active = total >= active_thr

    trend_mask = active & (share >= dominance_threshold)
    residual_mask = active & (share <= (1.0 - dominance_threshold))

    rows = []

    for role, mask in [("trend", trend_mask), ("residual", residual_mask)]:
        regs = contiguous_regions(mask)

        for start, end in regs:
            sl = slice(start, end + 1)
            length = end - start + 1

            mean_trend = float(trend[sl].mean())
            mean_residual = float(residual[sl].mean())
            mean_total = float(total[sl].mean())
            mean_share = float(share[sl].mean())
            mean_dom = float(dominance[sl].mean())
            peak_total = float(total[sl].max())

            region_score = float(length * mean_total * abs(mean_dom))

            rows.append(dict(
                role=role,
                start_t=int(start),
                end_t=int(end),
                length=int(length),
                region_score=region_score,
                mean_trend_abs=mean_trend,
                mean_residual_abs=mean_residual,
                mean_total_abs=mean_total,
                peak_total_abs=peak_total,
                mean_trend_share=mean_share,
                mean_dominance_signed=mean_dom,
                active_threshold=active_thr,
                dominance_threshold=dominance_threshold,
            ))

    rows = sorted(rows, key=lambda r: r["region_score"], reverse=True)

    save_csv(
        save_path,
        fieldnames=[
            "role",
            "start_t",
            "end_t",
            "length",
            "region_score",
            "mean_trend_abs",
            "mean_residual_abs",
            "mean_total_abs",
            "peak_total_abs",
            "mean_trend_share",
            "mean_dominance_signed",
            "active_threshold",
            "dominance_threshold",
        ],
        rows=rows,
    )


def save_global_summary(
    save_path,
    data,
    model_type,
    fold,
    seed,
    agg,
    band,
    curves,
    at_abs,
    ar_abs,
    dominance_threshold,
):
    """
    Single-row CSV for easy 5-fold / multi-dataset aggregation.

    mean_abs_A_T and mean_abs_A_R are global means over [B, T, D].
    frac_time_* is computed from the same time curves used in plots/CSV.
    """
    mean_abs_A_T = float(at_abs.mean())
    mean_abs_A_R = float(ar_abs.mean())

    global_share = mean_abs_A_T / (mean_abs_A_T + mean_abs_A_R + 1e-12)
    winner = winner_from_values(mean_abs_A_T, mean_abs_A_R)

    trend_dom = curves["trend_share"] >= dominance_threshold
    residual_dom = curves["trend_share"] <= (1.0 - dominance_threshold)

    row = dict(
        data=data,
        model_type=model_type,
        fold=fold,
        seed=seed,
        agg=agg,
        band=band,
        mean_abs_A_T=mean_abs_A_T,
        mean_abs_A_R=mean_abs_A_R,
        global_trend_share=global_share,
        winner=winner,
        frac_time_trend_dominant=float(trend_dom.mean()),
        frac_time_residual_dominant=float(residual_dom.mean()),
        dominance_threshold=dominance_threshold,
    )

    save_csv(
        save_path,
        fieldnames=[
            "data",
            "model_type",
            "fold",
            "seed",
            "agg",
            "band",
            "mean_abs_A_T",
            "mean_abs_A_R",
            "global_trend_share",
            "winner",
            "frac_time_trend_dominant",
            "frac_time_residual_dominant",
            "dominance_threshold",
        ],
        rows=[row],
    )


# ---------- plots ----------

def plot_overall(
    at_abs,
    ar_abs,
    save_path,
    band="sem",
    agg="mean",
    title="",
    paper_style=False,
):
    sT = positional_stats(at_abs, band=band, agg=agg)
    sR = positional_stats(ar_abs, band=band, agg=agg)

    T = len(sT["center"])
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, sT["center"], label="|A_T| (trend)", color=TREND_COLOR, lw=2)
    ax.fill_between(x, sT["lower"], sT["upper"], color=TREND_COLOR, alpha=0.20)

    ax.plot(x, sR["center"], label="|A_R| (residual)", color=RESIDUAL_COLOR, lw=2)
    ax.fill_between(x, sR["lower"], sR["upper"], color=RESIDUAL_COLOR, alpha=0.20)

    ax.set_xlabel("time step t")

    if paper_style:
        ax.set_ylabel("attribution magnitude")
    else:
        ax.set_ylabel(f"attribution magnitude\n({sT['label']})")

    ax.set_title(title or "Positional magnitude of trend vs residual attribution")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[saved] {save_path}")


def plot_ratio(curves, save_path, title="", paper_style=False):
    mean = curves["trend_share"]
    x = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.plot(
        x,
        mean,
        color="#2ca02c",
        lw=2,
        label="|A_T| / (|A_T| + |A_R|)",
    )

    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5)

    ax.set_ylim(0, 1)
    ax.set_xlabel("time step t")
    ax.set_ylabel("trend share")

    if not paper_style:
        ax.set_title(title or "Trend share over time")
        ax.legend(loc="best", frameon=False)

    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[saved] {save_path}")


def plot_dominance(
    curves,
    save_path,
    dominance_threshold=0.55,
    title="",
    paper_style=False,
):
    dom = curves["dominance"]
    x = np.arange(len(dom))

    dom_thr = 2.0 * dominance_threshold - 1.0

    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.plot(
        x,
        dom,
        color="#444444",
        lw=2,
        label="(|A_T| - |A_R|) / (|A_T| + |A_R|)",
    )

    ax.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(dom_thr, color=TREND_COLOR, ls=":", lw=1.0, alpha=0.8)
    ax.axhline(-dom_thr, color=RESIDUAL_COLOR, ls=":", lw=1.0, alpha=0.8)

    ax.set_ylim(-1, 1)
    ax.set_xlabel("time step t")
    ax.set_ylabel("signed dominance")

    if not paper_style:
        ax.set_title(title or "Signed dominance over time")
        ax.legend(loc="best", frameon=False)

    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[saved] {save_path}")


def plot_heatmap(
    mat,
    save_path,
    title="",
    ylabel="channel",
    cmap="viridis",
    vmin=None,
    vmax=None,
    channel_names=None,
    paper_style=False,
    integer_channels=True,
):
    """
    mat shape: [T, D]
    Display as channel x time.

    y-axis is always forced to actual channel indices:
      D=1  -> ch0 only
      D=20 -> ch0 ... ch19
    """
    T, D = mat.shape

    fig_h = max(3.0, min(10.0, 0.35 * D + 2.0))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    im = ax.imshow(
        mat.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(-0.5, T - 0.5, -0.5, D - 0.5),
    )

    ax.set_xlabel("time step t")
    ax.set_ylabel(ylabel)

    # 핵심 수정: channel_names가 없어도 실제 channel index로 y축 고정
    if integer_channels:
        yticks = np.arange(D)

        if channel_names is not None:
            ylabels = channel_names
        else:
            ylabels = [f"ch{i}" for i in range(D)]

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        ax.set_ylim(-0.5, D - 0.5)

    if not paper_style:
        ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[saved] {save_path}")


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--results_dir", type=str, default="./results_our")
    p.add_argument("--out_dir", type=str, default="./figs_positional")

    p.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["mimic3", "PAM", "boiler", "epilepsy", "freezer", "wafer"],
    )

    p.add_argument("--model_type", type=str, default="state")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_segments", type=int, default=50)
    p.add_argument("--min_seg_len", type=int, default=1)
    p.add_argument("--max_seg_len", type=int, default=48)

    p.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation over samples. Used consistently for plots and CSV.",
    )

    p.add_argument(
        "--band",
        type=str,
        default="sem",
        choices=["iqr", "std", "sem"],
        help="Uncertainty band for overall plot. Paper default: sem with agg=mean.",
    )

    p.add_argument("--topk_time", type=int, default=10)
    p.add_argument("--topk_position", type=int, default=20)

    p.add_argument(
        "--dominance_threshold",
        type=float,
        default=0.55,
        help=(
            "trend_share >= threshold => trend-dominant, "
            "trend_share <= 1-threshold => residual-dominant"
        ),
    )

    p.add_argument(
        "--active_percentile",
        type=float,
        default=60.0,
        help="Ignore low-total-attribution regions for dominance summaries.",
    )

    p.add_argument(
        "--paper_style",
        action="store_true",
        help="Make figures cleaner by removing some descriptive labels/titles.",
    )

    p.add_argument(
        "--channel_names",
        type=str,
        default=None,
        help="Optional comma-separated channel names for heatmap/top-position CSV.",
    )

    args = p.parse_args()

    if args.band == "sem" and args.agg != "mean":
        raise ValueError(
            "--band sem은 --agg mean에서만 사용하세요. "
            "median 기반 분석은 --band iqr를 사용하세요."
        )

    channel_names = parse_channel_names(args.channel_names)

    SEG = f"kalman_seg{args.num_segments}_min{args.min_seg_len}_max{args.max_seg_len}"

    at = load_attr(
        args.results_dir,
        args.data,
        args.model_type,
        f"timing_td_trend_{SEG}",
        args.fold,
        args.seed,
    )

    ar = load_attr(
        args.results_dir,
        args.data,
        args.model_type,
        f"timing_td_residual_{SEG}",
        args.fold,
        args.seed,
    )

    assert at.shape == ar.shape, f"shape mismatch: {at.shape} vs {ar.shape}"

    # abs is applied only once here.
    at = np.abs(at)
    ar = np.abs(ar)

    B, T, D = at.shape

    if channel_names is not None and len(channel_names) != D:
        raise ValueError(
            f"--channel_names length mismatch: got {len(channel_names)}, expected D={D}"
        )

    print(f"loaded |A_T|, |A_R| shape = {at.shape}  (B, T, D)")
    print(f"aggregation = {args.agg}, band = {args.band}")

    os.makedirs(args.out_dir, exist_ok=True)

    tag = f"{args.data}_{args.model_type}_fold{args.fold}_seed{args.seed}"

    curves = compute_time_curves(at, ar, agg=args.agg)
    pos = compute_position_matrices(at, ar, agg=args.agg)

    mean_abs_A_T = float(at.mean())
    mean_abs_A_R = float(ar.mean())
    global_share = mean_abs_A_T / (mean_abs_A_T + mean_abs_A_R + 1e-12)

    print("========== global summary ==========")
    print(f"mean |A_T|          = {mean_abs_A_T:.6g}")
    print(f"mean |A_R|          = {mean_abs_A_R:.6g}")
    print(f"global trend share  = {global_share:.4f}")
    print(f"winner              = {winner_from_values(mean_abs_A_T, mean_abs_A_R)}")
    print("====================================")

    # ---------- figures ----------

    plot_overall(
        at,
        ar,
        save_path=os.path.join(args.out_dir, f"overall_{tag}.png"),
        band=args.band,
        agg=args.agg,
        title=f"{args.data} / {args.model_type} - positional |A(T)| vs |A(R)|",
        paper_style=args.paper_style,
    )

    plot_ratio(
        curves,
        save_path=os.path.join(args.out_dir, f"ratio_{tag}.png"),
        title=f"{args.data} / {args.model_type} - trend share over time",
        paper_style=args.paper_style,
    )

    plot_dominance(
        curves,
        save_path=os.path.join(args.out_dir, f"dominance_{tag}.png"),
        dominance_threshold=args.dominance_threshold,
        title=f"{args.data} / {args.model_type} - signed dominance over time",
        paper_style=args.paper_style,
    )

    plot_heatmap(
        pos["dominance"],
        save_path=os.path.join(args.out_dir, f"dominance_heatmap_{tag}.png"),
        title=f"{args.data} / {args.model_type} - dominance heatmap (+trend, -residual)",
        cmap=DOMINANCE_CMAP,
        vmin=-1,
        vmax=1,
        channel_names=channel_names,
        paper_style=args.paper_style,
    )

    plot_heatmap(
        pos["total"],
        save_path=os.path.join(args.out_dir, f"total_heatmap_{tag}.png"),
        title=f"{args.data} / {args.model_type} - total attribution magnitude heatmap",
        cmap=TOTAL_CMAP,
        channel_names=channel_names,
        paper_style=args.paper_style,
    )

    # ---------- CSV summaries ----------

    save_time_summary(
        curves,
        save_path=os.path.join(args.out_dir, f"time_summary_{tag}.csv"),
    )

    save_top_time_summary(
        curves,
        save_path=os.path.join(args.out_dir, f"top_times_{tag}.csv"),
        topk=args.topk_time,
    )

    save_top_position_summary(
        pos,
        save_path=os.path.join(args.out_dir, f"top_positions_{tag}.csv"),
        topk=args.topk_position,
        active_percentile=args.active_percentile,
        channel_names=channel_names,
    )

    save_region_summary(
        curves,
        save_path=os.path.join(args.out_dir, f"region_summary_{tag}.csv"),
        dominance_threshold=args.dominance_threshold,
        active_percentile=args.active_percentile,
    )

    save_global_summary(
        save_path=os.path.join(args.out_dir, f"global_{tag}.csv"),
        data=args.data,
        model_type=args.model_type,
        fold=args.fold,
        seed=args.seed,
        agg=args.agg,
        band=args.band,
        curves=curves,
        at_abs=at,
        ar_abs=ar,
        dominance_threshold=args.dominance_threshold,
    )


if __name__ == "__main__":
    main()
