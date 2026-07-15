#!/usr/bin/env python
# =============================================================================
# pick_lambda.py
#   run_sweep_5fold.sh 가 만든 per-fold val CSV 들을 읽어서
#   (dataset, combo) 별 5-fold 평균 CPD 를 구하고,
#   데이터셋별로 select_ref(기본 average) 기준 최고 (lam0,lamf) 를 고른다.
#
#   출력:
#     - 콘솔: mask_ref 별 (combo x dataset) 평균 CPD 표
#     - sweep5_table_<ref>.csv       : ref 별 5-fold 평균 CPD
#     - sweep5_std_<ref>.csv         : ref 별 fold 간 표준편차
#     - sweep5_count_<ref>.csv       : ref 별 수집된 fold 수
#     - sweep5_incomplete_<ref>.csv  : 5-fold 미완성 조합(있을 때만)
#     - chosen_lambdas.csv           : 데이터셋별 선택된 lam0,lamf
#
#   선택은 average 로만 (lambda 무관 fill). zero 는 참고용으로 같이 출력.
#   0~4 fold가 모두 존재하는 조합만 lambda 선택 후보로 사용한다.
# =============================================================================
import argparse
import csv
import glob
import os
import re

import pandas as pd

COMBO_RE = re.compile(
    r"cmb_(.+?)_f(\d+)_lam([0-9]+(?:\.[0-9]+)?)x([0-9]+(?:\.[0-9]+)?)\.csv"
)
EXPECTED_FOLDS = {0, 1, 2, 3, 4}


def load_rows(eval_dir):
    rows = []
    paths = sorted(glob.glob(os.path.join(eval_dir, "cmb_*_f*_lam*.csv")))

    for path in paths:
        m = COMBO_RE.fullmatch(os.path.basename(path))
        if not m:
            print(f"[warn] 파일명 규칙 불일치로 건너뜀: {path}")
            continue

        data = m.group(1)
        fold = int(m.group(2))
        l0, lf = m.group(3), m.group(4)

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
                        "lam0": float(l0),
                        "lamf": float(lf),
                        "combo": f"{l0}x{lf}",
                        "mask_ref": r.get("mask_ref", "?"),
                        "CPD": cpd,
                    })
        except (OSError, csv.Error) as e:
            print(f"[warn] CSV 읽기 실패: {path} ({e})")

    if not rows:
        raise RuntimeError(
            f"{eval_dir} 에서 CPD 결과를 못 찾았습니다. eval 로그와 CSV를 확인하세요."
        )

    df = pd.DataFrame(rows)

    # 동일한 data/fold/combo/ref 행이 중복된 경우 한 fold당 한 값으로 정리
    key = ["data", "fold", "lam0", "lamf", "combo", "mask_ref"]
    n_before = len(df)
    df = df.groupby(key, as_index=False)["CPD"].mean()
    if len(df) != n_before:
        print(f"[warn] 중복 CPD 행 {n_before-len(df)}개를 fold 단위 평균으로 정리했습니다.")

    return df


def combo_order(combos):
    return sorted(
        combos,
        key=lambda c: (float(c.split("x")[0]), float(c.split("x")[1])),
    )


def save_ref_tables(df, ref, out_dir):
    sub = df[df.mask_ref == ref].copy()
    order = combo_order(sub["combo"].unique())

    mean = sub.groupby(["combo", "data"])["CPD"].mean().unstack("data")
    std = sub.groupby(["combo", "data"])["CPD"].std().unstack("data")
    count = sub.groupby(["combo", "data"])["fold"].nunique().unstack("data")

    col_order = [c for c in ["boiler", "PAM", "epilepsy", "wafer"] if c in mean.columns]
    extras = [c for c in mean.columns if c not in col_order]
    col_order += sorted(extras)

    mean = mean.reindex(index=order, columns=col_order)
    std = std.reindex(index=order, columns=col_order)
    count = count.reindex(index=order, columns=col_order)

    print(f"\n==== 5-fold val CPD 평균  (mask_ref = {ref}) ====")
    print(mean.to_string())

    mean.to_csv(os.path.join(out_dir, f"sweep5_table_{ref}.csv"))
    std.to_csv(os.path.join(out_dir, f"sweep5_std_{ref}.csv"))
    count.to_csv(os.path.join(out_dir, f"sweep5_count_{ref}.csv"))

    incomplete = []
    for (data, combo), grp in sub.groupby(["data", "combo"]):
        folds = sorted(set(grp["fold"]))
        if set(folds) != EXPECTED_FOLDS:
            incomplete.append({
                "data": data,
                "combo": combo,
                "folds": ",".join(map(str, folds)),
                "n_folds": len(folds),
                "missing_folds": ",".join(map(str, sorted(EXPECTED_FOLDS-set(folds)))),
            })

    if incomplete:
        pd.DataFrame(incomplete).to_csv(
            os.path.join(out_dir, f"sweep5_incomplete_{ref}.csv"), index=False
        )


def select_lambdas(df, select_ref):
    if select_ref not in set(df["mask_ref"]):
        raise RuntimeError(
            f"선택 기준 mask_ref='{select_ref}' 결과가 없습니다. "
            f"사용 가능한 ref={sorted(df['mask_ref'].unique())}"
        )

    sel = df[df.mask_ref == select_ref].copy()

    agg = (
        sel.groupby(["data", "lam0", "lamf", "combo"])
        .agg(
            CPD=("CPD", "mean"),
            CPD_std=("CPD", "std"),
            n_folds=("fold", "nunique"),
            folds=("fold", lambda s: ",".join(map(str, sorted(set(s))))),
        )
        .reset_index()
    )
    agg["CPD_std"] = agg["CPD_std"].fillna(0.0)
    agg["complete"] = agg["folds"].apply(
        lambda x: set(map(int, x.split(","))) == EXPECTED_FOLDS
    )

    incomplete = agg[~agg["complete"]].copy()
    if not incomplete.empty:
        print("\n[warn] 0~4 fold가 모두 없는 조합은 lambda 선택에서 제외합니다.")
        print(incomplete[["data", "combo", "folds", "n_folds"]].to_string(index=False))

    valid = agg[agg["complete"]].copy()
    if valid.empty:
        raise RuntimeError("0~4 fold가 모두 존재하는 lambda 조합이 없습니다.")

    all_data = set(sel["data"].unique())
    valid_data = set(valid["data"].unique())
    missing_data = sorted(all_data-valid_data)
    if missing_data:
        raise RuntimeError(
            f"다음 데이터셋에는 완성된 5-fold 조합이 없습니다: {missing_data}"
        )

    # CPD 최대 → 동률이면 fold std가 작은 조합 → 그래도 같으면 작은 lambda 우선
    valid = valid.sort_values(
        ["data", "CPD", "CPD_std", "lam0", "lamf"],
        ascending=[True, False, True, True, True],
    )
    chosen = valid.groupby("data", as_index=False).head(1).copy()
    return chosen.sort_values("data"), incomplete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="results_pna/sweep5_eval")
    ap.add_argument("--select_ref", default="average",
                    help="lambda 선택 기준 fill (기본 average)")
    ap.add_argument("--out_dir", default=".",
                    help="집계 결과 저장 폴더 (기본 현재 폴더)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_rows(args.eval_dir)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    # ---- ref 별 (combo x data) 평균/std/count 표 ----
    for ref in sorted(df.mask_ref.unique()):
        save_ref_tables(df, ref, args.out_dir)

    # ---- 선택: select_ref 기준, 완성된 5-fold 조합 중 데이터셋별 argmax ----
    chosen, incomplete = select_lambdas(df, args.select_ref)

    if not incomplete.empty:
        incomplete.to_csv(
            os.path.join(args.out_dir, f"sweep5_incomplete_{args.select_ref}.csv"),
            index=False,
        )

    print(f"\n==== 선택된 lambda  (기준: {args.select_ref} fill, 5-fold 평균 CPD ↑) ====")
    for _, r in chosen.iterrows():
        print(
            f"  {r['data']:9s}  lam0={r['lam0']:<5g} lamf={r['lamf']:<5g}"
            f"  (combo {r['combo']}, CPD={r['CPD']:.4f}±{r['CPD_std']:.4f},"
            f" folds={int(r['n_folds'])}/5)"
        )

    chosen[[
        "data", "lam0", "lamf", "combo", "CPD", "CPD_std", "n_folds"
    ]].to_csv(
        os.path.join(args.out_dir, "chosen_lambdas.csv"), index=False
    )

    print(f"\n[saved] {os.path.join(args.out_dir, 'chosen_lambdas.csv')}")
    print(f"[saved] {args.out_dir}/sweep5_table_<ref>.csv")
    print(f"[saved] {args.out_dir}/sweep5_std_<ref>.csv")
    print(f"[saved] {args.out_dir}/sweep5_count_<ref>.csv")
    print("\n※ epilepsy/PAM 표의 combo 간 편차가 fold std 안에 묻히면(=flat), "
          "단순 argmax보다 안정적인 중앙 영역 조합을 별도로 검토하는 것이 좋습니다.")


if __name__ == "__main__":
    main()