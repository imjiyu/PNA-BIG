# Bifurcated Integrated Gradients (BIG)
**Trend-Residual Path Decomposition for Time Series Attribution**  
KCC 2026 XAI Workshop  
Jiyu Lim, Jisu Yeo, Haksoo Lim, Jaesik Choi (KAIST)

---

## Overview

BIG는 시계열 분류기의 attribution을 **trend / residual** 두 성분으로 분해한다.
입력 `x`를 Kalman(RTS) smoother로 trend `T_x`와 residual `R_x = x − T_x`로 나눈 뒤,
하나의 baseline→input 직선 경로가 아니라 **두 단계 경로 `c → T_x → x`** 를 따라 Integrated Gradients를 적분한다.
IG의 path-additivity 덕분에 두 성분 기여도는 *근사 분해가 아니라* 정확한 분해가 된다:

```
A_T + A_R = F(x) − F(c_M)
```

temporal context 유지를 위해 두 phase 모두 [TIMING (Jang et al., 2025)](https://arxiv.org/abs/2506.05035)의 random temporal segment mask를 공유한다. 본 코드는 TIMING 코드베이스를 확장한 것이다.

---

## Requirements

```bash
pip install -r requirement.txt
# 핵심: torch==1.13.1, pytorch-lightning==2.0.7, captum==0.6.0, pykalman, time_interpret(tint)
```

---

## Repository Structure (핵심 스크립트)

| 파일 | 역할 | 산출물 |
|---|---|---|
| `real/main_td.py` | BIG attribution 생성 (`--explainers our_td`) | `results_our/*.npy` |
| `attribution/explainers_td.py` | BIG 핵심 구현 (Kalman 분해 + 2-phase IG) | — |
| `run_td_all.sh` | 5 dataset × 5 fold 일괄 attribution 생성 | `results_our/*.npy` |
| `check_completeness.py` | 경로 완전성 진단 | Table 2 |
| `eval_cpd_cpp.py` | 저장된 npy로 CPD/AUCC + 5개 metric 재계산 | `*/full_eval.csv` |
| `TR_table.py` | Trend vs Residual metric 집계 | Table 3 (metric) `results_table/*.csv` |
| `TR100_compare.py` | `|T+R|` vs `|T|+|R|` 집계 | Table 4 |
| `100_state_zero_baseline.py` | TIMING-100 baseline 집계 | Table 4·6 (TIMING 행) |
| `positional_analysis.py` | fold별 positional 통계·그림 | Figure 2 (per-fold) |
| `aggregate_positional_folds.py` | 5-fold positional 집계·플롯 | Figure 2 |
| `viz_attr_mean_top3.py` | 채널별 mean signed attribution heatmap | Figure 3 |
| `aggregate_heatmap_folds.py` | 5-fold total/dominance heatmap (보조) | 탐색용 |
| `real/main_td_viz.py` | per-sample 2-phase path 시각화 | Figure 4 (Appendix) |

---

## Output Directory Convention

> **모든 메인(=Kalman smoother) 결과는 단일 폴더 `results_our/`로 통일되어 있다.**
> 평가 스크립트 기본값이 전부 `results_our`를 가리키므로, 메인 재현은 별도 경로 지정 없이 동작한다.

| 폴더 | 용도 | 비고 |
|---|---|---|
| `results_our/` | 메인(smoother) attribution npy + eval csv | `main_td.py` 기본 저장 위치 |
| `results_filter/` | Kalman **filter** ablation (Table 5) | 수동 전환 필요 (아래 ⚠️A) |
| `results_comp/` | completeness 진단 (global norm, Table 2) | 수동 전환 필요 (아래 ⚠️B) |
| `results_table/` | 집계표 csv (Table 3·4) | `TR_*` / `TR100_compare` 출력 |
| `figs_positional/` | positional 그림·csv (Figure 2·보조 heatmap) | — |

---

## ⚠️ 공통 주의사항 (실행 전 반드시 확인)

**A. `main_td.py` 저장 폴더는 실험 종류에 따라 수동 전환한다.**
`real/main_td.py` 하단(약 1154줄)에 저장 경로가 하드코딩되어 있다. 주석에 표기된 규칙:
`results_our` = Kalman smoother(메인) / `results_filter` = Kalman filter(Table 5) / `results_comp` = completeness(Table 2).
**메인 재현만 할 경우 이 줄은 손대지 않는다.** filter·completeness 실험을 할 때만 이 경로를 바꾼다.

**B. completeness(Table 2)는 normalization을 바꿔야 한다.**
메인 평가는 **per-position normalization**을 쓰지만, completeness ratio 계산에는 **global normalization**이 필요하다.
`attribution/explainers_td.py`의 `_ig_phase`에서:
- 활성화: `attr = attr_sum / n_alphas` / `attr = attr.mean(dim=0)` (현재 주석 상태인 97–98줄)
- 주석 처리: `N_free = ...` 3줄 (현재 활성 상태인 100–102줄)
그리고 `main_td.py` 저장 경로를 `results_comp/`로 바꾼 뒤 attribution을 재생성한다. **메인 평가로 돌아갈 땐 반드시 원복**한다.

**C. `eval_cpd_cpp.py`는 append 모드로 csv를 쓴다.**
같은 `--output_file`에 행을 *누적*하므로, 새로 돌리기 전 기존 csv를 삭제한다 (`rm -f results_our/full_eval.csv`). 안 지우면 fold가 중복 집계된다.

**D. `--methods` 키는 npy 파일명과 정확히 일치해야 한다.**
키 형식: `timing_td_{kind}_kalman_seg{N}_min{m}_max{M}` (`kind` ∈ `trend`, `residual`, `combined`, `T_plus_R`).
seg/min/max는 **데이터셋마다 다르다**(아래 표). 안 맞으면 `FileNotFoundError`.

**E. 모든 실험은 5-fold CV, `seed=42` 고정**이며 결과는 fold 평균이다.
`run_td_all.sh`의 GPU 할당(`CUDA_VISIBLE_DEVICES`)은 실행 환경에 맞게 조정한다.

**데이터셋별 segment 설정 (모든 실험 공통):**

| dataset | num_segments | min_seg_len | max_seg_len | (testbs: GPU에 따라 조정 가능) |
|---|---|---|---|---|
| epilepsy | 10 | 10 | 10 | 5 |
| freezer | 5 | 10 | 100 | 5 |
| boiler | 50 | 1 | 36 | 30 |
| wafer | 5 | 10 | 152 | 10 |
| PAM | 10 | 10 | 600 | 3 |

(`testbs`는 GPU 메모리에 맞춘 배치 크기일 뿐 결과 수치에 영향 없음.)

**F. 모든 faithfulness 평가는 10% masking 기준이다: `area(topk) 0.1` + `top 0` + baseline `0.0` (전 실험 공통).**
`cumulative_difference`는 `top != 0`이면 area 비율을 **무시하고 고정 `top` 개수**로 마스킹한다.
- `eval_cpd_cpp.py` 는 기본값이 `top=0`, `topk=0.1` 이라 별도 지정 없이 10% 비율 마스킹으로 동작한다 (정식 재현 경로).
- 반면 TIMING 원본 코드의 `main_preserve*`·`main.py` 는 `--top` 기본값이 **50**이므로, 10% 비율 마스킹을 쓰려면 반드시 **`--top 0`** 을 명시해야 한다. (안 주면 area 0.1이 무시되고 고정 50칸만 마스킹)

---

## Reproduction

### Step 1 — Attribution 생성 (모든 Table·Figure의 선행 단계)

`main_td.py`가 trend/residual/combined/signed 6종 npy를 `results_our/`에 저장한다.

```bash
bash run_td_all.sh        # 5 dataset × 5 fold → results_our/
```

> 저장 키: `{data}_state_timing_td_{kind}_kalman_seg{N}_min{m}_max{M}_result_{fold}_42.npy`
> `kind` ∈ `trend`(|T|), `residual`(|R|), `trend_signed`(T), `residual_signed`(R), `combined`(|T+R|), `T_plus_R`(|T|+|R|)

---

### Table 2 — Path Completeness Diagnostic

> ⚠️ 먼저 **공통 주의사항 B**(global normalization 전환 + `main_td.py` 저장 경로 `results_comp/`)를 적용하고 Step 1을 재실행해 `results_comp/`에 npy를 만든다.

```bash
python check_completeness.py \
  --model state --seed 42 --folds 0 1 2 3 4 \
  --results-dir ./results_comp \
  > completeness_check.txt
```

`completeness_check.txt`에서 `all_med`가 1.0에 가깝고 `norm_err`가 0에 가까우면 정상.
**진단이 끝나면 explainers_td.py normalization과 main_td.py 경로를 메인(per-position, results_our)으로 원복한다.**

---

### Table 3 — Component Comparison (Trend vs Residual)

`|T|`, `|R|` npy를 평가해 `results_our/full_eval.csv`에 모은 뒤 집계한다.

```bash
rm -f results_our/full_eval.csv      # ⚠️ append 모드 — 재실행 전 삭제

# 데이터셋별 (seg 값은 위 표 참고). fold 0–4 반복.
for f in 0 1 2 3 4; do
  python eval_cpd_cpp.py --data epilepsy --fold $f --device cuda:0 \
    --methods timing_td_trend_kalman_seg10_min10_max10 \
              timing_td_residual_kalman_seg10_min10_max10
done
# freezer  : ..._kalman_seg5_min10_max100
# boiler   : ..._kalman_seg50_min1_max36
# wafer    : ..._kalman_seg5_min10_max152
# PAM      : ..._kalman_seg10_min10_max600
```

집계:

```bash
python TR_table.py          # Table 3 생성 및 results_table 폴더 저장
```

> 참고: `eval_cpd_cpp.py`는 현재 CPD만 출력한다(코드 내 CPP 루프 주석 처리됨). `TR_all.py`의 CPP 컬럼은 비어 나오므로 무시.

---

### Tables 4 & 6 — Aggregation (|T+R| vs |T|+|R| vs TIMING)

같은 `eval_cpd_cpp.py`로 `combined`(=|T+R|), `T_plus_R`(=|T|+|R|)를 평가하고, fold csv를 합쳐 `results_our/abs_full_eval_{ds}.csv`를 만든다.

```bash
# 예: PAM (다른 데이터셋은 seg 값만 교체)
mkdir -p results_our/tmp_PAM
for f in 0 1 2 3 4; do
  python eval_cpd_cpp.py --data PAM --fold $f --device cuda:0 \
    --output_file results_our/tmp_PAM/PAM_fold${f}.csv \
    --methods timing_td_combined_kalman_seg10_min10_max600 \
              timing_td_T_plus_R_kalman_seg10_min10_max600
done
awk 'FNR==1 && NR!=1 {next} {print}' results_our/tmp_PAM/PAM_fold*.csv \
  > results_our/abs_full_eval_PAM.csv

python TR100_compare.py        # → results_table/result_compare.csv
```

TIMING baseline 및 7개 XAI baseline(AFO/GateMask/GradSHAP/TimeX/TimeX++/IG/TIMING) 수치는 TIMING 원본 평가 파이프라인 산출물을 사용한다. Zero baseline 집계는:

```bash
python 100_state_zero_baseline.py \
  --input_dir ./100_state/100_state --out_dir ./baseline_zero_summary
```

---

### Table 5 — Ablation: Kalman Filter (causal)

> ⚠️ `main_td.py`의 explainer import를 **filter 변형**으로 바꾸고, 저장 경로를 `results_filter/`로 바꾼다(공통 주의사항 A).
> 그 뒤 Step 1 → Table 3 → Table 4 절차를 `--npy_dir results_filter`로 동일 반복한다. (실험 후 원복.)

---

### Figure 2 — Positional |A_T| vs |A_R|

fold별 통계 생성 → 5-fold 집계 플롯. `--paper_style` 플래그를 켠다.

```bash
declare -A NSEG=([boiler]=50 [PAM]=10 [freezer]=5 [epilepsy]=10 [wafer]=5)
declare -A MINL=([boiler]=1  [PAM]=10 [freezer]=10 [epilepsy]=10 [wafer]=10)
declare -A MAXL=([boiler]=36 [PAM]=600 [freezer]=100 [epilepsy]=10 [wafer]=152)

for data in boiler PAM freezer epilepsy wafer; do
  for fold in 0 1 2 3 4; do
    python positional_analysis.py --data $data --fold $fold --model_type state --seed 42 \
      --num_segments ${NSEG[$data]} --min_seg_len ${MINL[$data]} --max_seg_len ${MAXL[$data]} \
      --agg mean --band sem --paper_style
  done
  python aggregate_positional_folds.py --data $data --model_type state --seed 42 --folds 0,1,2,3,4
done
```

(선택) 5-fold total/dominance heatmap — 논문 그림은 아니며 탐색용:

```bash
for data in boiler PAM freezer epilepsy wafer; do
  python aggregate_heatmap_folds.py --data $data --model_type state --seed 42 --folds 0,1,2,3,4 \
    --num_segments ${NSEG[$data]} --min_seg_len ${MINL[$data]} --max_seg_len ${MAXL[$data]}
done
```

---

### Figure 3 — Channel-wise Mean Signed Attribution Heatmap

```bash
python viz_attr_mean_top3.py --data all                    # 상위 3채널 (Boiler 등 멀티변량 — Fig 3)
python viz_attr_mean_top3.py --data all --topk_channels 0  # 전채널 평균 (univariate는 동일 결과)
```

---

### Figure 4 (Appendix) — Two-Phase Attribution Path (per-sample)

```bash
# 예: boiler (다른 데이터셋은 seg/채널만 교체)
python real/main_td_viz.py --explainers our_td --data boiler --fold 0 --testbs 50 \
  --num_segments 50 --min_seg_len 1 --max_seg_len 36 \
  --viz --viz_dir ./viz_td/boiler --viz_n_samples 5 --viz_channels 6 --device cuda:0
```

| dataset | num/min/max | viz_channel number |
|---|---|---|
| boiler | 50/1/36 | 6 |
| wafer | 5/10/152 | 0 |
| PAM | 10/10/600 | 9 |
| epilepsy | 10/10/10 | 0 |
| freezer | 5/10/100 | 0 |

---

## 원본 TIMING 호환 평가 스크립트 (참고)

`real/main_preserve.py`는 TIMING 원본에서 이어받은 평가 스크립트로,
`results_our/`의 npy를 읽어 Table 3 / Table 4 수치를 한 번에 산출하는 **대체 경로**다.
본 repo의 정식 재현 경로는 위 `eval_cpd_cpp.py` 체인이며, 이 스크립트는 호환성을 위해 보존만 한다.
(`pred_diff`를 `results_TRC/`·`results_pred/`에 저장하므로, 직접 실행하려면 해당 폴더를 미리 생성해야 한다.)

---

Built on TIMING (Jang, Kim, Yang, 2025): https://arxiv.org/abs/2506.05035
