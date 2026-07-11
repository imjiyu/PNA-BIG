import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


TREND_COLOR = "#009E73"       # green
RESIDUAL_COLOR = "#7B3294"    # purple
BASELINE_COLOR = "#666666"    # gray

def load_global_files(out_dir, data, model_type, seed, folds):
    dfs = []

    for fold in folds:
        path = os.path.join(
            out_dir,
            f"global_{data}_{model_type}_fold{fold}_seed{seed}.csv",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_time_summary_files(out_dir, data, model_type, seed, folds):
    dfs = []

    for fold in folds:
        path = os.path.join(
            out_dir,
            f"time_summary_{data}_{model_type}_fold{fold}_seed{seed}.csv",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        df["fold"] = fold
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def summarize_global(df):
    rows = []

    metrics = [
        "mean_abs_A_T",
        "mean_abs_A_R",
        "global_trend_share",
        "frac_time_trend_dominant",
        "frac_time_residual_dominant",
    ]

    row = {}

    row["data"] = df["data"].iloc[0]
    row["model_type"] = df["model_type"].iloc[0]
    row["seed"] = df["seed"].iloc[0]
    row["n_folds"] = len(df)

    for m in metrics:
        row[f"{m}_mean"] = df[m].mean()
        row[f"{m}_std"] = df[m].std(ddof=1)

    row["trend_wins"] = int((df["winner"] == "trend").sum())
    row["residual_wins"] = int((df["winner"] == "residual").sum())
    row["ties"] = int((df["winner"] == "tie").sum())

    rows.append(row)
    return pd.DataFrame(rows)


def summarize_time_curves(time_df):
    """
    Fold별 time_summary를 평균내서 dataset-level positional curve 생성.

    각 fold에서 이미 time별 trend_abs/residual_abs가 계산되어 있으므로,
    여기서는 fold를 반복 단위로 보고 mean ± std를 계산한다.
    """
    grouped = time_df.groupby("t")

    summary = grouped.agg(
        trend_abs_mean=("trend_abs", "mean"),
        trend_abs_std=("trend_abs", "std"),
        residual_abs_mean=("residual_abs", "mean"),
        residual_abs_std=("residual_abs", "std"),
        total_abs_mean=("total_abs", "mean"),
        total_abs_std=("total_abs", "std"),
        trend_share_mean=("trend_share", "mean"),
        trend_share_std=("trend_share", "std"),
        dominance_signed_mean=("dominance_signed", "mean"),
        dominance_signed_std=("dominance_signed", "std"),
    ).reset_index()

    return summary


def plot_5fold_overall(summary, save_path, title=""):
    x = summary["t"].values

    trend_mean = summary["trend_abs_mean"].values
    trend_std = summary["trend_abs_std"].fillna(0).values

    residual_mean = summary["residual_abs_mean"].values
    residual_std = summary["residual_abs_std"].fillna(0).values

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        x,
        trend_mean,
        label="|A(T)| (trend)",
        lw=2,
        color=TREND_COLOR,
    )
    ax.fill_between(
        x,
        trend_mean - trend_std,
        trend_mean + trend_std,
        alpha=0.20,
        color=TREND_COLOR,
    )

    ax.plot(
        x,
        residual_mean,
        label="|A(R)| (residual)",
        lw=2,
        color=RESIDUAL_COLOR,
    )
    ax.fill_between(
        x,
        residual_mean - residual_std,
        residual_mean + residual_std,
        alpha=0.20,
        color=RESIDUAL_COLOR,
    )

    ax.set_xlabel("time step t")
    ax.set_ylabel("attribution magnitude")
    ax.set_title(title or "5-fold positional attribution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[saved] {save_path}")


def plot_5fold_trend_share(summary, save_path, title=""):
    x = summary["t"].values

    mean = summary["trend_share_mean"].values
    std = summary["trend_share_std"].fillna(0).values

    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.plot(
        x,
        mean,
        lw=2,
        label="trend share",
        color=TREND_COLOR,
    )
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.20,
        color=TREND_COLOR,
    )

    ax.axhline(
        0.5,
        color=BASELINE_COLOR,
        ls="--",
        lw=0.8,
        alpha=0.7,
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("time step t")
    ax.set_ylabel("trend share")
    ax.set_title(title or "5-fold trend share")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[saved] {save_path}")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--out_dir", type=str, default="./figs_positional")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model_type", type=str, default="state")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=str, default="0,1,2,3,4")

    args = p.parse_args()

    folds = [int(x.strip()) for x in args.folds.split(",")]

    global_df = load_global_files(
        args.out_dir,
        args.data,
        args.model_type,
        args.seed,
        folds,
    )

    time_df = load_time_summary_files(
        args.out_dir,
        args.data,
        args.model_type,
        args.seed,
        folds,
    )

    global_summary = summarize_global(global_df)
    time_summary = summarize_time_curves(time_df)

    global_save = os.path.join(
        args.out_dir,
        f"global_5fold_summary_{args.data}_{args.model_type}_seed{args.seed}.csv",
    )

    time_save = os.path.join(
        args.out_dir,
        f"time_5fold_summary_{args.data}_{args.model_type}_seed{args.seed}.csv",
    )

    global_summary.to_csv(global_save, index=False)
    time_summary.to_csv(time_save, index=False)

    print(f"[saved] {global_save}")
    print(f"[saved] {time_save}")

    plot_5fold_overall(
        time_summary,
        save_path=os.path.join(
            args.out_dir,
            f"overall_5fold_{args.data}_{args.model_type}_seed{args.seed}.png",
        ),
        title=f"{args.data} / {args.model_type} [ 5-fold positional |A(T)| vs |A(R)| ]",
    )

    plot_5fold_trend_share(
        time_summary,
        save_path=os.path.join(
            args.out_dir,
            f"ratio_5fold_{args.data}_{args.model_type}_seed{args.seed}.png",
        ),
        title=f"{args.data} / {args.model_type} - 5-fold trend share",
    )


if __name__ == "__main__":
    main()
