#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


DATASET_ORDER = [
    "boiler",
    "PAM",
    "epilepsy",
    "wafer",
]

DISPLAY_NAME = {
    "boiler": "Boiler",
    "pam": "PAM",
    "epilepsy": "Epilepsy",
    "wafer": "Wafer",
}

COLUMNS = [
    "Dataset",
    "CR Dev All",
    "CR Dev Top-50",
    "Neg Rate All",
    "Neg Rate Top-50",
    "Norm Error",
]


def normalize_dataset_name(name):
    return name.strip().lower()


def parse_result_file(file_path):
    """
    다음 평균 ± 표준편차 행을 읽는다.

    예:
      avg all_med       : 1.0257 ± 0.0148
      avg top50_med     : 1.0219 ± 0.0147
      avg all_neg       : 0.0% ± 0.0%
      avg top50_neg     : 0.0% ± 0.0%
      avg norm_err      : 0.0271 ± 0.0138
    """
    results = {}
    current_dataset = None

    # 지원 형식:
    # === boiler / kalman_seg... ===
    # === TIMING / boiler / seg... ===
    section_pattern = re.compile(
        r"^===\s+(?:TIMING\s*/\s*)?([^/\s]+)\s*/"
    )

    avg_pattern = re.compile(
        r"^avg\s+"
        r"(all_med|top50_med|all_neg|top50_neg|norm_err)"
        r"\s*:\s*"
        r"([-+]?\d+(?:\.\d+)?)"       # mean
        r"\s*(%?)"
        r"\s*±\s*"
        r"([-+]?\d+(?:\.\d+)?)"       # std
        r"\s*(%?)"
    )

    with open(file_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            section_match = section_pattern.match(line)
            if section_match:
                current_dataset = normalize_dataset_name(
                    section_match.group(1)
                )
                results.setdefault(current_dataset, {})
                continue

            if current_dataset is None:
                continue

            avg_match = avg_pattern.match(line)
            if avg_match is None:
                continue

            key = avg_match.group(1)
            mean = float(avg_match.group(2))
            std = float(avg_match.group(4))

            results[current_dataset][key] = {
                "mean": mean,
                "std": std,
            }

    return results


def make_rows(parsed_results):
    mean_rows = []
    mean_std_rows = []

    required_keys = {
        "all_med",
        "top50_med",
        "all_neg",
        "top50_neg",
        "norm_err",
    }

    for dataset in DATASET_ORDER:
        normalized = normalize_dataset_name(dataset)

        if normalized not in parsed_results:
            continue

        values = parsed_results[normalized]
        missing = required_keys - set(values)

        if missing:
            print(
                f"[WARN] {dataset}: 다음 항목이 없어 제외합니다: "
                f"{sorted(missing)}"
            )
            continue

        all_med_mean = values["all_med"]["mean"]
        all_med_std = values["all_med"]["std"]

        top50_med_mean = values["top50_med"]["mean"]
        top50_med_std = values["top50_med"]["std"]

        all_neg_mean = values["all_neg"]["mean"]
        all_neg_std = values["all_neg"]["std"]

        top50_neg_mean = values["top50_neg"]["mean"]
        top50_neg_std = values["top50_neg"]["std"]

        norm_err_mean = values["norm_err"]["mean"]
        norm_err_std = values["norm_err"]["std"]

        # 기존 표와 동일한 계산
        cr_dev_all_mean = abs(all_med_mean - 1.0)
        cr_dev_top50_mean = abs(top50_med_mean - 1.0)

        # 기존 mean 값을 유지하기 위해 summary의 std를 그대로 사용
        cr_dev_all_std = all_med_std
        cr_dev_top50_std = top50_med_std

        mean_rows.append({
            "Dataset": DISPLAY_NAME[normalized],
            "CR Dev All": f"{cr_dev_all_mean:.3f}",
            "CR Dev Top-50": f"{cr_dev_top50_mean:.3f}",
            "Neg Rate All": f"{all_neg_mean:.1f}%",
            "Neg Rate Top-50": f"{top50_neg_mean:.1f}%",
            "Norm Error": f"{norm_err_mean:.3f}",
        })

        mean_std_rows.append({
            "Dataset": DISPLAY_NAME[normalized],
            "CR Dev All": (
                f"{cr_dev_all_mean:.3f} ± {cr_dev_all_std:.3f}"
            ),
            "CR Dev Top-50": (
                f"{cr_dev_top50_mean:.3f} ± {cr_dev_top50_std:.3f}"
            ),
            "Neg Rate All": (
                f"{all_neg_mean:.1f}% ± {all_neg_std:.1f}%"
            ),
            "Neg Rate Top-50": (
                f"{top50_neg_mean:.1f}% ± {top50_neg_std:.1f}%"
            ),
            "Norm Error": (
                f"{norm_err_mean:.3f} ± {norm_err_std:.3f}"
            ),
        })

    return mean_rows, mean_std_rows


def save_csv(rows, output_path):
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVE] {output_path}")


def convert_file(input_path, output_prefix, output_dir):
    parsed_results = parse_result_file(input_path)
    mean_rows, mean_std_rows = make_rows(parsed_results)

    if not mean_rows:
        raise RuntimeError(
            f"{input_path}에서 유효한 평균 결과를 찾지 못했습니다."
        )

    save_csv(
        mean_rows,
        output_dir / f"{output_prefix}_mean.csv",
    )

    save_csv(
        mean_std_rows,
        output_dir / f"{output_prefix}_mean_std.csv",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Completeness txt 결과를 mean 및 mean±std CSV로 변환"
        )
    )

    parser.add_argument(
        "--pna-file",
        required=True,
        help="PNA-BIG completeness 결과 txt",
    )
    parser.add_argument(
        "--timing-file",
        required=True,
        help="TIMING completeness 결과 txt",
    )
    parser.add_argument(
        "--output-dir",
        default="./completeness_tables",
        help="CSV 저장 폴더",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_file(
        input_path=args.pna_file,
        output_prefix="pna_big_completeness",
        output_dir=output_dir,
    )

    convert_file(
        input_path=args.timing_file,
        output_prefix="timing_completeness",
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()