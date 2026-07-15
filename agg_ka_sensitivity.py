#!/usr/bin/env python
# =============================================================================
# agg_ka_sensitivity.py
#
# run_ka_sensitivity.sh가 생성한 Ka별 5-fold validation CPD를 집계한다.
# lambda는 chosen_lambdas.csv에서 선택된 값으로 고정되어 있다고 가정한다.
#
# 출력:
#   ka_sensitivity_summary.csv
#   ka_sensitivity_table_<ref>.csv
#   ka_sensitivity_std_<ref>.csv
#   ka_sensitivity_count_<ref>.csv
#   ka_sensitivity_delta_vs_ka5.csv
#   ka_sensitivity_best_<select_ref>.csv   (진단용)
#
# 주의:
#   이는 "Ka=5에서 선택된 lambda를 고정한 국소 민감도 분석"이다.
#   Ka별로 lambda를 재튜닝한 joint optimization 결과가 아니다.
# =============================================================================
import argparse
import csv
import glob
import os
import re

import pandas as pd

FILE_RE = re.compile(
    r"cmb_(.+?)_f(\d+)_ka(\d+)_lam"
    r"([0-9]+(?:\.[0-9]+)?)x([0-9]+(?:\.[0-9]+)?)\.csv"
)
EXPECTED_FOLDS = {0, 1, 2, 3, 4}


def load_rows(root):
    rows = []
    pattern = os.path.join(root, "ka*", "eval", "cmb_*_f*_ka*_lam*.csv")

    for path in sorted(glob.glob(pattern)):
        m = FILE_RE.fullmatch(os.path.basename(path))
        if not m:
            print(f"[warn] 파일명 규칙 불일치로 건너뜀: {path}")
            continue

        data = m.group(1)
        fold = int(m.group(2))
        ka = int(m.group(3))
        lam0 = float(m.group(4))
        lamf = float(m.group(5))

        try:
            with open(path, newline="") as fp:
                for r in csv.DictReader(fp):
                    if r.get("metric") != "CPD":
                        continue
                    try:
                        cpd = float(r["cum_diff"])
                    except (KeyError, TypeError, ValueError):
                        print(f"[warn] CPD 값 파싱 실패: {path}")
                        continue

                    rows.append({
                        "data": data,
                        "fold": fold,
                        "Ka": ka,
                        "lam0": lam0,
                        "lamf": lamf,
                        "mask_ref": r.get("mask_ref", "?"),
                        "CPD": cpd,
                    })
        except (OSError, csv.Error) as e:
            print(f"[warn] CSV 읽기 실패: {path} ({e})")

    if not rows:
        raise RuntimeError(f"{root}에서 Ka sensitivity CPD 결과를 찾지 못했습니다.")

    df = pd.DataFrame(rows)

    # 재실행으로 동일 행이 중복된 경우 fold 단위 평균
    key = ["data", "fold", "Ka", "lam0", "lamf", "mask_ref"]
    before = len(df)
    df = df.groupby(key, as_index=False)["CPD"].mean()
    if len(df) != before:
        print(f"[warn] 중복 행 {before-len(df)}개를 fold 단위 평균으로 정리했습니다.")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_pna/ka_sensitivity")
    ap.add_argument("--select_ref", default="average")
    ap.add_argument("--out_dir", default="results_pna/ka_sensitivity")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_rows(args.root)

    summary = (
        df.groupby(["data", "Ka", "lam0", "lamf", "mask_ref"])
        .agg(
            CPD_mean=("CPD", "mean"),
            CPD_std=("CPD", "std"),
            n_folds=("fold", "nunique"),
            folds=("fold", lambda s: ",".join(map(str, sorted(set(s))))),
        )
        .reset_index()
    )
    summary["CPD_std"] = summary["CPD_std"].fillna(0.0)
    summary["complete"] = summary["folds"].apply(
        lambda x: set(map(int, x.split(","))) == EXPECTED_FOLDS
    )

    summary_path = os.path.join(args.out_dir, "ka_sensitivity_summary.csv")
    summary.to_csv(summary_path, index=False)

    incomplete = summary[~summary["complete"]]
    if not incomplete.empty:
        print("\n[warn] 5-fold 미완성 결과가 있습니다.")
        print(
            incomplete[
                ["data", "Ka", "mask_ref", "folds", "n_folds"]
            ].to_string(index=False)
        )
        incomplete.to_csv(
            os.path.join(args.out_dir, "ka_sensitivity_incomplete.csv"),
            index=False,
        )

    valid = summary[summary["complete"]].copy()
    if valid.empty:
        raise RuntimeError("0~4 fold가 모두 존재하는 Ka 결과가 없습니다.")

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    for ref in sorted(valid["mask_ref"].unique()):
        sub = valid[valid["mask_ref"] == ref]

        mean = sub.pivot(index="Ka", columns="data", values="CPD_mean")
        std = sub.pivot(index="Ka", columns="data", values="CPD_std")
        count = sub.pivot(index="Ka", columns="data", values="n_folds")

        col_order = [
            c for c in ["boiler", "PAM", "epilepsy", "wafer"]
            if c in mean.columns
        ]
        col_order += sorted(c for c in mean.columns if c not in col_order)

        mean = mean.sort_index().reindex(columns=col_order)
        std = std.sort_index().reindex(columns=col_order)
        count = count.sort_index().reindex(columns=col_order)

        print(f"\n==== Ka sensitivity: 5-fold val CPD 평균 (mask_ref={ref}) ====")
        print(mean.to_string())

        mean.to_csv(
            os.path.join(args.out_dir, f"ka_sensitivity_table_{ref}.csv")
        )
        std.to_csv(
            os.path.join(args.out_dir, f"ka_sensitivity_std_{ref}.csv")
        )
        count.to_csv(
            os.path.join(args.out_dir, f"ka_sensitivity_count_{ref}.csv")
        )

    # Ka=5 대비 차이: 양수면 해당 Ka가 Ka=5보다 CPD가 큼
    ka5 = valid[valid["Ka"] == 5][
        ["data", "mask_ref", "CPD_mean"]
    ].rename(columns={"CPD_mean": "CPD_Ka5"})

    delta = valid.merge(ka5, on=["data", "mask_ref"], how="left")
    delta["delta_vs_Ka5"] = delta["CPD_mean"] - delta["CPD_Ka5"]
    delta.to_csv(
        os.path.join(args.out_dir, "ka_sensitivity_delta_vs_ka5.csv"),
        index=False,
    )

    # 진단용 best Ka. 최종 채택은 표와 안정성을 함께 보고 결정한다.
    selected = valid[valid["mask_ref"] == args.select_ref].copy()
    if selected.empty:
        raise RuntimeError(
            f"select_ref={args.select_ref}의 완성된 결과가 없습니다."
        )

    selected = selected.sort_values(
        ["data", "CPD_mean", "CPD_std", "Ka"],
        ascending=[True, False, True, True],
    )
    best = selected.groupby("data", as_index=False).head(1).copy()
    best.to_csv(
        os.path.join(
            args.out_dir, f"ka_sensitivity_best_{args.select_ref}.csv"
        ),
        index=False,
    )

    print(f"\n==== 진단용 best Ka (mask_ref={args.select_ref}, CPD ↑) ====")
    for _, r in best.sort_values("data").iterrows():
        print(
            f"  {r['data']:9s} Ka={int(r['Ka']):2d} "
            f"CPD={r['CPD_mean']:.4f}±{r['CPD_std']:.4f} "
            f"(lam0={r['lam0']:g}, lamf={r['lamf']:g})"
        )

    # 모든 데이터셋의 macro 평균도 참고용 출력
    macro = (
        selected.groupby("Ka")
        .agg(
            macro_CPD=("CPD_mean", "mean"),
            dataset_std=("CPD_mean", "std"),
            n_datasets=("data", "nunique"),
        )
        .reset_index()
        .sort_values("Ka")
    )
    macro.to_csv(
        os.path.join(
            args.out_dir, f"ka_sensitivity_macro_{args.select_ref}.csv"
        ),
        index=False,
    )

    print(f"\n==== 데이터셋 macro 평균 (mask_ref={args.select_ref}) ====")
    print(macro.to_string(index=False))

    print(f"\n[saved] {summary_path}")
    print(f"[saved] {args.out_dir}/ka_sensitivity_table_<ref>.csv")
    print(f"[saved] {args.out_dir}/ka_sensitivity_std_<ref>.csv")
    print(f"[saved] {args.out_dir}/ka_sensitivity_delta_vs_ka5.csv")
    print(
        "\n※ 이 결과는 Ka=5에서 선택된 lambda를 고정한 민감도 분석입니다. "
        "Ka별 최적 lambda를 다시 찾은 결과로 해석하면 안 됩니다."
    )


if __name__ == "__main__":
    main()
