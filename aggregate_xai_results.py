#!/usr/bin/env python3
"""
XAI 결과 통합 및 5-fold 평균±표준편차 집계 스크립트.

기본 입력 구조
--------------
<root>/
  state_boiler_0_0_results_baseline.csv
  state_boiler_1_0_results_baseline.csv
  ...
  results_pna_hpt/eval_anchor/
    boiler_baselines_pna.csv
    epilepsy_baselines_pna.csv
    wafer_baselines_pna.csv
    PAM_baselines_pna.csv
    boiler.csv
    epilepsy.csv
    wafer.csv
    PAM.csv

실행 예시
---------
python aggregate_xai_results.py --root .

출력
----
<root>/aggregated_results/
  normalized_fold_results.csv       # 출처가 다른 CSV를 공통 형식으로 통합한 fold 단위 값
  summary_all_metrics_long.csv      # 모든 지표의 수치형 mean/std
  summary_all_metrics_wide.csv      # 방법별 모든 지표를 "mean ± std"로 정리
  cpd_fold_values.csv               # CPD의 fold별 값
  cpd_summary_numeric.csv           # CPD mean/std 수치형 long table
  cpd_table.csv                     # 요청한 블록형 CPD 표
  cpd_table.html                    # 사진과 유사한 스타일의 CPD 표
  warnings.txt                      # 누락/중복/형식 문제

집계 규칙
---------
1. 동일 dataset × mask_ref × method × fold에서 여러 행이 발견되면 먼저 그 fold 안에서 평균.
2. 그 뒤 fold 0~4의 fold-level 값으로 평균과 표본표준편차(ddof=1)를 계산.
3. cum_diff를 CPD로 사용.
4. area/topk=0.1, top=0 결과만 사용.

외부 패키지가 필요하지 않으며 Python 표준 라이브러리만 사용합니다.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


DATASET_ORDER = ["boiler", "epilepsy", "wafer", "PAM"]
DATASET_DISPLAY = {
    "boiler": "Boiler",
    "epilepsy": "Epilepsy",
    "wafer": "Wafer",
    "PAM": "PAM",
}

MASK_ORDER = ["average", "zero", "pna"]
MASK_DISPLAY = {
    "average": "Average",
    "zero": "Zero",
    "pna": "PNA",
}

METHOD_ORDER = [
    "AFO",
    "GateMask",
    "GradSHAP",
    "TimeX",
    "TimeX++",
    "IG",
    "TIMING",
    "PNA-BIG",
]

# 공통 지표 순서. CPD는 원본의 cum_diff입니다.
METRICS = [
    "CPD",
    "AUCC",
    "cum_50_diff",
    "accuracy",
    "comprehensiveness",
    "cross_entropy",
    "log_odds",
    "sufficiency",
]

RAW_BASELINE_COLUMNS = [
    "seed",
    "fold",
    "baseline",
    "area",
    "explainer",
    "lambda_1",
    "lambda_2",
    "lambda_3",
    "cum_50_diff",
    "cum_diff",
    "AUCC",
    "accuracy",
    "comprehensiveness",
    "cross_entropy",
    "log_odds",
    "sufficiency",
]

EXPECTED_FOLDS = set(range(5))


def canonical_dataset(value: str) -> str:
    token = value.strip().casefold()
    mapping = {
        "boiler": "boiler",
        "epilepsy": "epilepsy",
        "wafer": "wafer",
        "pam": "PAM",
    }
    if token not in mapping:
        raise ValueError(f"알 수 없는 데이터셋 이름: {value!r}")
    return mapping[token]


def canonical_mask(value: str) -> str:
    token = value.strip().casefold().replace("_", "").replace("-", "")
    mapping = {
        "average": "average",
        "avg": "average",
        "mean": "average",
        "zero": "zero",
        "zeros": "zero",
        "pna": "pna",
    }
    if token not in mapping:
        raise ValueError(f"알 수 없는 mask_ref/baseline 이름: {value!r}")
    return mapping[token]


def canonical_method(value: str) -> str:
    token = value.strip().casefold()
    exact = {
        "augmented_occlusion": "AFO",
        "afo": "AFO",
        "gate_mask": "GateMask",
        "gatemask": "GateMask",
        "gradientshap_abs": "GradSHAP",
        "gradient_shap_abs": "GradSHAP",
        "gradshap": "GradSHAP",
        "timex": "TimeX",
        "timex++": "TimeX++",
        "integrated_gradients_base_abs": "IG",
        "integrated_gradients": "IG",
        "ig": "IG",
        "timing_sample100_seg10_min10_max600": "TIMING",
        "timing": "TIMING",
    }
    if token in exact:
        return exact[token]
    if token.startswith("timing_td_combined"):
        return "PNA-BIG"
    if token.startswith("timing_sample"):
        return "TIMING"
    raise ValueError(f"알 수 없는 method/explainer 이름: {value!r}")


def safe_float(value: object, *, field: str, path: Path, row_number: int) -> float:
    text = "" if value is None else str(value).strip()
    if text == "":
        raise ValueError(f"{path}:{row_number} - {field} 값이 비어 있습니다.")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"{path}:{row_number} - {field}={text!r}를 숫자로 변환할 수 없습니다."
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"{path}:{row_number} - {field}={text!r}가 유한수가 아닙니다.")
    return number


def safe_int(value: object, *, field: str, path: Path, row_number: int) -> int:
    number = safe_float(value, field=field, path=path, row_number=row_number)
    rounded = int(round(number))
    if not math.isclose(number, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{path}:{row_number} - {field}={number}가 정수가 아닙니다.")
    return rounded


def close_to(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def normalize_metric_row(row: Mapping[str, object], path: Path, row_number: int) -> Dict[str, float]:
    return {
        "CPD": safe_float(row["cum_diff"], field="cum_diff(CPD)", path=path, row_number=row_number),
        "AUCC": safe_float(row["AUCC"], field="AUCC", path=path, row_number=row_number),
        "cum_50_diff": safe_float(row["cum_50_diff"], field="cum_50_diff", path=path, row_number=row_number),
        "accuracy": safe_float(row["accuracy"], field="accuracy", path=path, row_number=row_number),
        "comprehensiveness": safe_float(
            row["comprehensiveness"], field="comprehensiveness", path=path, row_number=row_number
        ),
        "cross_entropy": safe_float(
            row["cross_entropy"], field="cross_entropy", path=path, row_number=row_number
        ),
        "log_odds": safe_float(row["log_odds"], field="log_odds", path=path, row_number=row_number),
        "sufficiency": safe_float(
            row["sufficiency"], field="sufficiency", path=path, row_number=row_number
        ),
    }


def make_record(
    *,
    dataset: str,
    fold: int,
    seed: int,
    mask_ref: str,
    method: str,
    raw_method: str,
    source: str,
    metrics: Mapping[str, float],
    source_file: Path,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "mask_ref": mask_ref,
        "method": method,
        "raw_method": raw_method,
        "source": source,
        "source_file": str(source_file),
    }
    result.update(metrics)
    return result


def read_headerless_baseline_file(
    path: Path,
    *,
    dataset_from_name: str,
    fold_from_name: int,
    area: float,
    warnings: List[str],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]

    if not rows:
        warnings.append(f"빈 파일: {path}")
        return records

    first = [cell.strip() for cell in rows[0]]
    has_header = first and first[0].casefold() == "seed"
    data_rows = rows[1:] if has_header else rows

    for index, values in enumerate(data_rows, start=2 if has_header else 1):
        if len(values) != len(RAW_BASELINE_COLUMNS):
            warnings.append(
                f"열 개수가 {len(RAW_BASELINE_COLUMNS)}개가 아니어서 건너뜀: "
                f"{path}:{index} (실제 {len(values)}개)"
            )
            continue
        row = dict(zip(RAW_BASELINE_COLUMNS, values))
        try:
            row_area = safe_float(row["area"], field="area", path=path, row_number=index)
            if not close_to(row_area, area):
                continue
            row_fold = safe_int(row["fold"], field="fold", path=path, row_number=index)
            if row_fold != fold_from_name:
                warnings.append(
                    f"파일명 fold={fold_from_name}, 행 fold={row_fold} 불일치: {path}:{index}; 행 값을 사용"
                )
            mask_ref = canonical_mask(str(row["baseline"]))
            if mask_ref not in {"average", "zero"}:
                continue
            raw_method = str(row["explainer"]).strip()
            method = canonical_method(raw_method)
            if method == "PNA-BIG":
                warnings.append(f"일반 baseline 파일에서 PNA-BIG 행은 제외: {path}:{index}")
                continue
            metrics = normalize_metric_row(row, path, index)
            records.append(
                make_record(
                    dataset=dataset_from_name,
                    fold=row_fold,
                    seed=safe_int(row["seed"], field="seed", path=path, row_number=index),
                    mask_ref=mask_ref,
                    method=method,
                    raw_method=raw_method,
                    source="baseline_zero_average",
                    metrics=metrics,
                    source_file=path,
                )
            )
        except (KeyError, ValueError) as exc:
            warnings.append(str(exc))
    return records


def read_dict_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return [], []
        fieldnames = [name.strip() if name is not None else "" for name in reader.fieldnames]
        rows: List[Dict[str, str]] = []
        for raw in reader:
            cleaned = {
                (key.strip() if key else ""): (value.strip() if value is not None else "")
                for key, value in raw.items()
            }
            if any(cleaned.values()):
                rows.append(cleaned)
        return fieldnames, rows


def validate_pna_columns(path: Path, fieldnames: Sequence[str]) -> None:
    required = {
        "data",
        "fold",
        "seed",
        "method",
        "mask_ref",
        "metric",
        "topk",
        "top",
        "cum_diff",
        "AUCC",
        "cum_50_diff",
        "accuracy",
        "comprehensiveness",
        "cross_entropy",
        "log_odds",
        "sufficiency",
    }
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{path}에 필요한 열이 없습니다: {', '.join(missing)}")


def read_pna_baselines(
    path: Path,
    *,
    dataset_from_name: str,
    topk: float,
    top: int,
    warnings: List[str],
) -> List[Dict[str, object]]:
    fieldnames, rows = read_dict_csv(path)
    if not fieldnames:
        warnings.append(f"빈 파일 또는 헤더 없음: {path}")
        return []
    validate_pna_columns(path, fieldnames)

    records: List[Dict[str, object]] = []
    for index, row in enumerate(rows, start=2):
        try:
            if str(row["metric"]).strip().casefold() != "cpd":
                continue
            if not close_to(safe_float(row["topk"], field="topk", path=path, row_number=index), topk):
                continue
            if safe_int(row["top"], field="top", path=path, row_number=index) != top:
                continue
            dataset = canonical_dataset(row["data"])
            if dataset != dataset_from_name:
                warnings.append(
                    f"파일명 데이터셋={dataset_from_name}, 행 데이터셋={dataset} 불일치: {path}:{index}; 행 값을 사용"
                )
            mask_ref = canonical_mask(row["mask_ref"])
            if mask_ref != "pna":
                continue
            raw_method = row["method"].strip()
            method = canonical_method(raw_method)
            if method == "PNA-BIG":
                warnings.append(f"PNA baseline 파일에서 PNA-BIG 행은 제외: {path}:{index}")
                continue
            records.append(
                make_record(
                    dataset=dataset,
                    fold=safe_int(row["fold"], field="fold", path=path, row_number=index),
                    seed=safe_int(row["seed"], field="seed", path=path, row_number=index),
                    mask_ref=mask_ref,
                    method=method,
                    raw_method=raw_method,
                    source="baseline_pna",
                    metrics=normalize_metric_row(row, path, index),
                    source_file=path,
                )
            )
        except (KeyError, ValueError) as exc:
            warnings.append(str(exc))
    return records


def read_pna_big(
    path: Path,
    *,
    dataset_from_name: str,
    topk: float,
    top: int,
    warnings: List[str],
) -> List[Dict[str, object]]:
    fieldnames, rows = read_dict_csv(path)
    if not fieldnames:
        warnings.append(f"빈 파일 또는 헤더 없음: {path}")
        return []
    validate_pna_columns(path, fieldnames)

    filtered: List[Tuple[int, Dict[str, str]]] = []
    for index, row in enumerate(rows, start=2):
        try:
            if str(row["metric"]).strip().casefold() != "cpd":
                continue
            if not close_to(safe_float(row["topk"], field="topk", path=path, row_number=index), topk):
                continue
            if safe_int(row["top"], field="top", path=path, row_number=index) != top:
                continue
            method = canonical_method(row["method"])
            if method != "PNA-BIG":
                warnings.append(f"PNA-BIG 파일의 비-PNA-BIG 행은 제외: {path}:{index}")
                continue
            filtered.append((index, row))
        except (KeyError, ValueError) as exc:
            warnings.append(str(exc))

    unique_raw_methods = sorted({row["method"].strip() for _, row in filtered})
    if len(unique_raw_methods) > 1:
        raise ValueError(
            f"{path}에 PNA-BIG 하이퍼파라미터 method가 여러 개 있습니다. "
            "서로 다른 설정을 임의로 평균내지 않도록 파일을 하나의 최종 method만 남겨 주세요.\n"
            + "\n".join(f"  - {name}" for name in unique_raw_methods)
        )

    records: List[Dict[str, object]] = []
    for index, row in filtered:
        try:
            dataset = canonical_dataset(row["data"])
            if dataset != dataset_from_name:
                warnings.append(
                    f"파일명 데이터셋={dataset_from_name}, 행 데이터셋={dataset} 불일치: {path}:{index}; 행 값을 사용"
                )
            mask_ref = canonical_mask(row["mask_ref"])
            records.append(
                make_record(
                    dataset=dataset,
                    fold=safe_int(row["fold"], field="fold", path=path, row_number=index),
                    seed=safe_int(row["seed"], field="seed", path=path, row_number=index),
                    mask_ref=mask_ref,
                    method="PNA-BIG",
                    raw_method=row["method"].strip(),
                    source="pna_big",
                    metrics=normalize_metric_row(row, path, index),
                    source_file=path,
                )
            )
        except (KeyError, ValueError) as exc:
            warnings.append(str(exc))
    return records


def find_aggregate_file(directory: Path, expected_stem: str) -> Path | None:
    matches = [
        path
        for path in directory.glob("*.csv")
        if path.stem.casefold() == expected_stem.casefold()
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"동일한 aggregate 파일 후보가 여러 개입니다 ({expected_stem}): "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def load_all_records(
    *,
    root: Path,
    pna_dir: Path,
    out_dir: Path,
    area: float,
    top: int,
    warnings: List[str],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    raw_pattern = re.compile(
        r"^state_(boiler|epilepsy|wafer|PAM)_(\d+)_(\d+)_results_baseline\.csv$",
        re.IGNORECASE,
    )
    raw_files: List[Path] = []
    for path in root.rglob("state_*_*_*_results_baseline.csv"):
        try:
            path.resolve().relative_to(out_dir.resolve())
            continue
        except ValueError:
            pass
        match = raw_pattern.match(path.name)
        if not match:
            continue
        if int(match.group(3)) != top:
            continue
        raw_files.append(path)

    raw_files.sort(key=lambda p: str(p).casefold())
    if not raw_files:
        warnings.append(
            f"일반 Average/Zero baseline 파일을 찾지 못했습니다: {root}/**/state_*_*_{top}_results_baseline.csv"
        )

    seen_raw_keys: set[Tuple[str, int]] = set()
    for path in raw_files:
        match = raw_pattern.match(path.name)
        assert match is not None
        dataset = canonical_dataset(match.group(1))
        fold = int(match.group(2))
        seen_raw_keys.add((dataset, fold))
        records.extend(
            read_headerless_baseline_file(
                path,
                dataset_from_name=dataset,
                fold_from_name=fold,
                area=area,
                warnings=warnings,
            )
        )

    for dataset in DATASET_ORDER:
        for fold in sorted(EXPECTED_FOLDS):
            if (dataset, fold) not in seen_raw_keys:
                warnings.append(
                    f"일반 Average/Zero baseline 파일 누락: dataset={dataset}, fold={fold}, top={top}"
                )

    if not pna_dir.exists():
        warnings.append(f"PNA 결과 폴더가 없습니다: {pna_dir}")
        return records

    for dataset in DATASET_ORDER:
        pna_baseline_path = find_aggregate_file(pna_dir, f"{dataset}_baselines_pna")
        if pna_baseline_path is None:
            warnings.append(f"PNA baseline aggregate 파일 누락: {pna_dir}/{dataset}_baselines_pna.csv")
        else:
            records.extend(
                read_pna_baselines(
                    pna_baseline_path,
                    dataset_from_name=dataset,
                    topk=area,
                    top=top,
                    warnings=warnings,
                )
            )

        pna_big_path = find_aggregate_file(pna_dir, dataset)
        if pna_big_path is None:
            warnings.append(f"PNA-BIG aggregate 파일 누락: {pna_dir}/{dataset}.csv")
        else:
            records.extend(
                read_pna_big(
                    pna_big_path,
                    dataset_from_name=dataset,
                    topk=area,
                    top=top,
                    warnings=warnings,
                )
            )

    return records


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: Sequence[float]) -> float:
    # 5-fold 표준편차는 표본표준편차(ddof=1)로 계산합니다.
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def collapse_to_fold_level(
    records: Sequence[Mapping[str, object]], warnings: List[str]
) -> List[Dict[str, object]]:
    grouped: MutableMapping[
        Tuple[str, int, str, str], List[Mapping[str, object]]
    ] = defaultdict(list)
    for record in records:
        key = (
            str(record["dataset"]),
            int(record["fold"]),
            str(record["mask_ref"]),
            str(record["method"]),
        )
        grouped[key].append(record)

    fold_rows: List[Dict[str, object]] = []
    for (dataset, fold, mask_ref, method), items in grouped.items():
        if len(items) > 1:
            warnings.append(
                "동일 dataset/mask/method/fold 행이 여러 개라 fold 내부 평균을 사용: "
                f"dataset={dataset}, mask_ref={mask_ref}, method={method}, fold={fold}, n={len(items)}"
            )
        row: Dict[str, object] = {
            "dataset": dataset,
            "dataset_display": DATASET_DISPLAY[dataset],
            "fold": fold,
            "mask_ref": mask_ref,
            "mask_display": MASK_DISPLAY[mask_ref],
            "method": method,
            "n_rows_averaged": len(items),
            "seeds": ";".join(sorted({str(item["seed"]) for item in items})),
            "raw_methods": ";".join(sorted({str(item["raw_method"]) for item in items})),
            "sources": ";".join(sorted({str(item["source"]) for item in items})),
            "source_files": ";".join(sorted({str(item["source_file"]) for item in items})),
        }
        for metric in METRICS:
            row[metric] = mean([float(item[metric]) for item in items])
        fold_rows.append(row)

    dataset_rank = {value: index for index, value in enumerate(DATASET_ORDER)}
    mask_rank = {value: index for index, value in enumerate(MASK_ORDER)}
    method_rank = {value: index for index, value in enumerate(METHOD_ORDER)}
    fold_rows.sort(
        key=lambda row: (
            dataset_rank[str(row["dataset"])],
            mask_rank[str(row["mask_ref"])],
            method_rank.get(str(row["method"]), 999),
            int(row["fold"]),
        )
    )
    return fold_rows


def summarize_fold_rows(
    fold_rows: Sequence[Mapping[str, object]], warnings: List[str]
) -> List[Dict[str, object]]:
    grouped: MutableMapping[
        Tuple[str, str, str], List[Mapping[str, object]]
    ] = defaultdict(list)
    for row in fold_rows:
        key = (str(row["dataset"]), str(row["mask_ref"]), str(row["method"]))
        grouped[key].append(row)

    summaries: List[Dict[str, object]] = []
    for (dataset, mask_ref, method), items in grouped.items():
        items = sorted(items, key=lambda row: int(row["fold"]))
        folds = [int(row["fold"]) for row in items]
        missing = sorted(EXPECTED_FOLDS - set(folds))
        extra = sorted(set(folds) - EXPECTED_FOLDS)
        if missing or extra or len(folds) != len(set(folds)):
            warnings.append(
                f"fold 구성 확인 필요: dataset={dataset}, mask_ref={mask_ref}, method={method}, "
                f"folds={folds}, missing={missing}, extra={extra}"
            )
        summary: Dict[str, object] = {
            "dataset": dataset,
            "dataset_display": DATASET_DISPLAY[dataset],
            "mask_ref": mask_ref,
            "mask_display": MASK_DISPLAY[mask_ref],
            "method": method,
            "n_folds": len(items),
            "folds": ";".join(str(value) for value in folds),
            "missing_folds": ";".join(str(value) for value in missing),
            "complete_5fold": not missing and not extra and len(items) == 5,
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in items]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = sample_std(values)
        summaries.append(summary)

    # 완전히 누락된 조합도 경고합니다.
    present = {
        (str(row["dataset"]), str(row["mask_ref"]), str(row["method"]))
        for row in summaries
    }
    for dataset in DATASET_ORDER:
        for mask_ref in MASK_ORDER:
            for method in METHOD_ORDER:
                if (dataset, mask_ref, method) not in present:
                    warnings.append(
                        f"결과 조합 완전 누락: dataset={dataset}, mask_ref={mask_ref}, method={method}"
                    )

    dataset_rank = {value: index for index, value in enumerate(DATASET_ORDER)}
    mask_rank = {value: index for index, value in enumerate(MASK_ORDER)}
    method_rank = {value: index for index, value in enumerate(METHOD_ORDER)}
    summaries.sort(
        key=lambda row: (
            dataset_rank[str(row["dataset"])],
            mask_rank[str(row["mask_ref"])],
            method_rank.get(str(row["method"]), 999),
        )
    )
    return summaries


def fmt_number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def fmt_mean_std(mean_value: float, std_value: float, digits: int) -> str:
    return f"{mean_value:.{digits}f} ± {std_value:.{digits}f}"


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(
    *,
    out_dir: Path,
    fold_rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    warnings: Sequence[str],
    digits: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_fields = [
        "dataset",
        "dataset_display",
        "fold",
        "mask_ref",
        "mask_display",
        "method",
        "n_rows_averaged",
        "seeds",
        "raw_methods",
        "sources",
        "source_files",
        *METRICS,
    ]
    write_csv(out_dir / "normalized_fold_results.csv", normalized_fields, fold_rows)

    long_rows: List[Dict[str, object]] = []
    for row in summaries:
        for metric in METRICS:
            mean_value = float(row[f"{metric}_mean"])
            std_value = float(row[f"{metric}_std"])
            long_rows.append(
                {
                    "dataset": row["dataset"],
                    "dataset_display": row["dataset_display"],
                    "mask_ref": row["mask_ref"],
                    "mask_display": row["mask_display"],
                    "method": row["method"],
                    "metric": metric,
                    "n_folds": row["n_folds"],
                    "folds": row["folds"],
                    "missing_folds": row["missing_folds"],
                    "complete_5fold": row["complete_5fold"],
                    "mean": mean_value,
                    "std": std_value,
                    "mean_std": fmt_mean_std(mean_value, std_value, digits),
                }
            )
    write_csv(
        out_dir / "summary_all_metrics_long.csv",
        [
            "dataset",
            "dataset_display",
            "mask_ref",
            "mask_display",
            "method",
            "metric",
            "n_folds",
            "folds",
            "missing_folds",
            "complete_5fold",
            "mean",
            "std",
            "mean_std",
        ],
        long_rows,
    )

    wide_rows: List[Dict[str, object]] = []
    for row in summaries:
        wide: Dict[str, object] = {
            "dataset": row["dataset"],
            "dataset_display": row["dataset_display"],
            "mask_ref": row["mask_ref"],
            "mask_display": row["mask_display"],
            "method": row["method"],
            "n_folds": row["n_folds"],
            "folds": row["folds"],
            "missing_folds": row["missing_folds"],
            "complete_5fold": row["complete_5fold"],
        }
        for metric in METRICS:
            wide[metric] = fmt_mean_std(
                float(row[f"{metric}_mean"]), float(row[f"{metric}_std"]), digits
            )
        wide_rows.append(wide)
    write_csv(
        out_dir / "summary_all_metrics_wide.csv",
        [
            "dataset",
            "dataset_display",
            "mask_ref",
            "mask_display",
            "method",
            "n_folds",
            "folds",
            "missing_folds",
            "complete_5fold",
            *METRICS,
        ],
        wide_rows,
    )

    cpd_fold_rows = [
        {
            "dataset": row["dataset"],
            "dataset_display": row["dataset_display"],
            "mask_ref": row["mask_ref"],
            "mask_display": row["mask_display"],
            "method": row["method"],
            "fold": row["fold"],
            "CPD": row["CPD"],
        }
        for row in fold_rows
    ]
    write_csv(
        out_dir / "cpd_fold_values.csv",
        ["dataset", "dataset_display", "mask_ref", "mask_display", "method", "fold", "CPD"],
        cpd_fold_rows,
    )

    cpd_numeric_rows: List[Dict[str, object]] = []
    summary_lookup: Dict[Tuple[str, str, str], Mapping[str, object]] = {}
    for row in summaries:
        key = (str(row["dataset"]), str(row["mask_ref"]), str(row["method"]))
        summary_lookup[key] = row
        cpd_numeric_rows.append(
            {
                "dataset": row["dataset"],
                "dataset_display": row["dataset_display"],
                "mask_ref": row["mask_ref"],
                "mask_display": row["mask_display"],
                "method": row["method"],
                "n_folds": row["n_folds"],
                "folds": row["folds"],
                "missing_folds": row["missing_folds"],
                "mean": row["CPD_mean"],
                "std": row["CPD_std"],
                "mean_std": fmt_mean_std(
                    float(row["CPD_mean"]), float(row["CPD_std"]), digits
                ),
            }
        )
    write_csv(
        out_dir / "cpd_summary_numeric.csv",
        [
            "dataset",
            "dataset_display",
            "mask_ref",
            "mask_display",
            "method",
            "n_folds",
            "folds",
            "missing_folds",
            "mean",
            "std",
            "mean_std",
        ],
        cpd_numeric_rows,
    )

    cpd_matrix_rows: List[Dict[str, object]] = []
    for mask_ref in MASK_ORDER:
        for method_index, method in enumerate(METHOD_ORDER):
            matrix_row: Dict[str, object] = {
                "[CPD] Mask Ref.": MASK_DISPLAY[mask_ref] if method_index == 0 else "",
                "Method": method,
            }
            for dataset in DATASET_ORDER:
                summary = summary_lookup.get((dataset, mask_ref, method))
                if summary is None:
                    matrix_row[DATASET_DISPLAY[dataset]] = ""
                else:
                    matrix_row[DATASET_DISPLAY[dataset]] = fmt_mean_std(
                        float(summary["CPD_mean"]), float(summary["CPD_std"]), digits
                    )
            cpd_matrix_rows.append(matrix_row)
    cpd_matrix_fields = ["[CPD] Mask Ref.", "Method", *[DATASET_DISPLAY[d] for d in DATASET_ORDER]]
    write_csv(out_dir / "cpd_table.csv", cpd_matrix_fields, cpd_matrix_rows)

    write_cpd_html(
        out_dir / "cpd_table.html",
        cpd_matrix_rows,
        dataset_headers=[DATASET_DISPLAY[d] for d in DATASET_ORDER],
    )

    unique_warnings = list(dict.fromkeys(warnings))
    with (out_dir / "warnings.txt").open("w", encoding="utf-8") as handle:
        if unique_warnings:
            handle.write("\n".join(f"- {message}" for message in unique_warnings))
            handle.write("\n")
        else:
            handle.write("경고 없음\n")


def write_cpd_html(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    dataset_headers: Sequence[str],
) -> None:
    table_rows: List[str] = []
    for index, row in enumerate(rows):
        method = str(row["Method"])
        mask_value = str(row["[CPD] Mask Ref."])
        classes: List[str] = []
        if method == "PNA-BIG":
            classes.append("pna-big")
        if mask_value:
            classes.append("block-start")
        class_attr = f' class="{" ".join(classes)}"' if classes else ""
        cells = [
            f"<td class=\"mask\">{html.escape(mask_value)}</td>",
            f"<td class=\"method\">{html.escape(method)}</td>",
        ]
        cells.extend(
            f"<td class=\"value\">{html.escape(str(row[header]))}</td>"
            for header in dataset_headers
        )
        table_rows.append(f"<tr{class_attr}>{''.join(cells)}</tr>")

    headers_html = "".join(
        ["<th>[CPD]<br>Mask Ref.</th>", "<th>Method</th>"]
        + [f"<th>{html.escape(header)}</th>" for header in dataset_headers]
    )
    document = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>CPD 5-fold Mean ± Std</title>
<style>
  body {{ font-family: Arial, "Noto Sans KR", sans-serif; margin: 24px; color: #111827; }}
  h1 {{ font-size: 18px; margin: 0 0 6px; }}
  p.note {{ margin: 0 0 16px; color: #4b5563; font-size: 13px; }}
  table {{ border-collapse: collapse; min-width: 900px; table-layout: fixed; }}
  th, td {{ border: 1px solid #d1d5db; padding: 9px 10px; font-size: 13px; }}
  th {{ background: #f3f4f6; text-align: center; font-weight: 700; }}
  td.mask {{ width: 90px; font-weight: 600; vertical-align: top; }}
  td.method {{ width: 140px; }}
  td.value {{ width: 150px; text-align: center; font-variant-numeric: tabular-nums; }}
  tr.block-start td {{ border-top: 2px solid #9ca3af; }}
  tr.pna-big td {{ background: #dbeef9; font-weight: 700; }}
</style>
</head>
<body>
<h1>CPD 결과: 5-fold 평균 ± 표본표준편차</h1>
<p class="note">cum_diff를 CPD로 사용하며, 각 셀은 fold 0~4의 mean ± std(ddof=1)입니다.</p>
<table>
<thead><tr>{headers_html}</tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average/Zero/PNA XAI 결과를 통합하고 5-fold 평균±표준편차를 계산합니다."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="state_*_results_baseline.csv 파일이 있는 프로젝트 루트 (기본: 현재 폴더)",
    )
    parser.add_argument(
        "--pna-dir",
        type=Path,
        default=None,
        help="eval_anchor 폴더. 생략 시 <root>/results_pna_hpt/eval_anchor",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="출력 폴더. 생략 시 <root>/aggregated_results",
    )
    parser.add_argument("--area", type=float, default=0.1, help="사용할 area/topk (기본: 0.1)")
    parser.add_argument("--top", type=int, default=0, help="사용할 top 값 (기본: 0)")
    parser.add_argument("--digits", type=int, default=4, help="mean ± std 표시 소수점 자리수 (기본: 4)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.expanduser().resolve()
    pna_dir = (
        args.pna_dir.expanduser().resolve()
        if args.pna_dir is not None
        else root / "results_pna_hpt" / "eval_anchor"
    )
    out_dir = (
        args.out.expanduser().resolve()
        if args.out is not None
        else root / "aggregated_results"
    )

    warnings: List[str] = []
    try:
        records = load_all_records(
            root=root,
            pna_dir=pna_dir,
            out_dir=out_dir,
            area=args.area,
            top=args.top,
            warnings=warnings,
        )
        if not records:
            raise RuntimeError("조건에 맞는 결과 행을 하나도 읽지 못했습니다. warnings.txt 내용을 확인하세요.")
        fold_rows = collapse_to_fold_level(records, warnings)
        summaries = summarize_fold_rows(fold_rows, warnings)
        write_outputs(
            out_dir=out_dir,
            fold_rows=fold_rows,
            summaries=summaries,
            warnings=warnings,
            digits=args.digits,
        )
    except Exception as exc:  # 최상위에서 명확한 오류 메시지를 출력
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[OK] 입력 행 수: {len(records)}")
    print(f"[OK] fold-level 행 수: {len(fold_rows)}")
    print(f"[OK] 요약 조합 수: {len(summaries)}")
    print(f"[OK] 출력 폴더: {out_dir}")
    print(f"[INFO] 경고 수: {len(list(dict.fromkeys(warnings)))} -> {out_dir / 'warnings.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
