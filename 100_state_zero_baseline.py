import os
import re
import argparse
import pandas as pd


COLS = [
    "seed",
    "fold",
    "baseline",
    "topk",
    "explainer",
    "lambda_1",
    "lambda_2",
    "lambda_3",
    "cum_50_diff",
    "cum_diff",
    "AUCC",
    "mean_acc",
    "mean_comp",
    "mean_ce",
    "mean_lodds",
    "mean_suff",
]

METRICS = [
    "cum_50_diff",
    "cum_diff",
    "AUCC",
    "mean_acc",
    "mean_comp",
    "mean_ce",
    "mean_lodds",
    "mean_suff",
]


def parse_filename(filename):
    """
    Expected:
      state_boiler_0_results_baseline.csv
      state_PAM_4_results_baseline.csv
    """
    m = re.match(
        r"(?P<model_type>[^_]+)_(?P<data>.+)_(?P<fold>\d+)_results_baseline\.csv$",
        filename,
    )

    if m is None:
        return None

    model_type = m.group("model_type")
    data = m.group("data")
    fold = int(m.group("fold"))

    return model_type, data, fold


def read_one_csv(path):
    df = pd.read_csv(path, header=None, names=COLS)

    numeric_cols = [
        "seed",
        "fold",
        "topk",
        "lambda_1",
        "lambda_2",
        "lambda_3",
    ] + METRICS

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["baseline"] = df["baseline"].astype(str)
    df["explainer"] = df["explainer"].astype(str)

    return df


def load_all(input_dir):
    dfs = []

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith("_results_baseline.csv"):
            continue

        parsed = parse_filename(filename)

        if parsed is None:
            print(f"[skip] filename pattern mismatch: {filename}")
            continue

        model_type, data, fold_from_filename = parsed
        path = os.path.join(input_dir, filename)

        df = read_one_csv(path)

        df["model_type"] = model_type
        df["data"] = data
        df["fold"] = fold_from_filename
        df["source_file"] = filename

        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No *_results_baseline.csv files found in {input_dir}")

    return pd.concat(dfs, ignore_index=True)


def filter_zero_baseline(df):
    return df[df["baseline"].str.lower().str.contains("zero")].copy()


def make_fold_values(df_zero):
    """
    같은 data / model / explainer / topk / fold 안에 중복 row가 있으면 평균.
    lambda_1, lambda_2, lambda_3는 grouping에서 제외.
    """
    group_cols = [
        "model_type",
        "data",
        "seed",
        "fold",
        "explainer",
        "topk",
    ]

    fold_values = (
        df_zero
        .groupby(group_cols, dropna=False)[METRICS]
        .mean()
        .reset_index()
    )

    return fold_values


def make_5fold_mean_std(fold_values):
    """
    fold별 값들을 모아서 mean/std 계산.
    """
    group_cols = [
        "model_type",
        "data",
        "seed",
        "explainer",
        "topk",
    ]

    agg_dict = {}

    for m in METRICS:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")

    summary = (
        fold_values
        .groupby(group_cols, dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            **agg_dict,
        )
        .reset_index()
    )

    return summary


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input_dir",
        type=str,
        default="./100_state/100_state",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="./baseline_zero_summary",
    )

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_all = load_all(args.input_dir)
    df_zero = filter_zero_baseline(df_all)

    if df_zero.empty:
        raise RuntimeError("No Zero/Zeros baseline rows found.")

    print("========== loaded ==========")
    print(f"all rows  : {len(df_all)}")
    print(f"zero rows : {len(df_zero)}")
    print("datasets  :", sorted(df_zero["data"].unique().tolist()))
    print("explainers:", sorted(df_zero["explainer"].unique().tolist()))
    print("============================")

    fold_values = make_fold_values(df_zero)
    summary = make_5fold_mean_std(fold_values)

    fold_save = os.path.join(args.out_dir, "zero_fold_values.csv")
    summary_save = os.path.join(args.out_dir, "zero_5fold_mean_std.csv")

    fold_values.to_csv(fold_save, index=False)
    
    # zero_5fold_mean_std.csv만 소수점 3자리로 저장
    summary.to_csv(summary_save, index=False, float_format="%.3f")

    print(f"[saved] {fold_save}")
    print(f"[saved] {summary_save}")


if __name__ == "__main__":
    main()