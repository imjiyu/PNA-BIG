# PNA-BIG

**Projected Neutral Anchor Baseline for Bifurcated Integrated Gradients**
*In-Manifold, Decision-Neutral Baselines for Time-Series Attribution*

---

## 1. Overview

PNA-BIG는 BIG의 attribution 경로에서 **baseline 설계**를 개선한 확장이다.

- **기존 BIG / TIMING**: baseline `c = 0`. zero 시퀀스는 데이터 manifold 밖이라 RNN/GRU의 hidden-state 경로가 학습 분포 밖(OOD)으로 벗어난다.
- **PNA-BIG**: 각 입력 `x`에 대해 training pool에서 (1) representation space에서 가깝고 (2) input space에서 low-energy이며 (3) decision-neutral하고 (4) 실제 training sample인 anchor를 골라 baseline으로 쓴다.

```
c*(x) = argmin_c  Dφ(x,c) + λ0·L0(c) + λf·Lf(c)
        Dφ : representation(hidden) 거리      (x 의존)
        L0 : zero-proximity (input energy)
        Lf : output neutrality (softmax ~ uniform)
```

baseline이 non-zero이므로 경로도 확장된다. 입력과 baseline을 **모두 trend/residual로 분해**해 component-replacement path를 만들고, trend-first / residual-first 두 순서를 평균한다:

```
trend-first    :  Tc+Rc → Tx+Rc → Tx+Rx
residual-first :  Tc+Rc → Tc+Rx → Tx+Rx
Ā_T = ½(A_T^∅ + A_T^R),   Ā_R = ½(A_R^∅ + A_R^T)
```

completeness는 zero가 아니라 **선택된 neutral anchor 기준**으로 성립한다:

```
Σ A_i(x) = F_y(x) − (1/Ka) Σ_c F_y(c)      (Top-Ka anchor 평균)
```

---

## 2. Requirements

```bash
pip install -r requirement.txt
# 핵심: torch==1.13.1, pytorch-lightning==2.0.7, captum==0.6.0, pykalman, time_interpret(tint)
```

**사전 조건**

1. 체크포인트: `model/{data}/state_classifier_{fold}_42_no_imputation` (4 dataset × 5 fold = 20개).
   BIG과 동일한 GRU(200 hidden) 분류기. 없으면 `scripts/real/train.sh`로 먼저 생성.
2. 데이터셋: `epilepsy`, `wafer`, `PAM`, `boiler` (freezer는 이번 실험에서 제외).

---

## 3. 하이퍼파라미터 — 이미 결정되어 있다

**튜닝은 다시 할 필요 없다.** 데이터셋별 최적값이 `hp_pna.sh`에 들어 있고, 모든 bash 스크립트가 이 파일만 읽는다.
(튜닝 과정을 재현하고 싶으면 → [부록 A](#부록-a-하이퍼파라미터-튜닝-재현-선택))

| dataset  | λ0   | λf  | Ka | TIMING npy 이름 |
| -------- | ---- | --- | -- | --------------- |
| epilepsy | 10.0 | 1.0 | 10 | `timing_sample100_seg10_min10_max10` |
| wafer    | 0.5  | 0.5 | 1  | `timing_sample100_seg5_min10_max152` |
| PAM      | 0.5  | 3.0 | 10 | `timing_sample100_seg10_min10_max600` |
| boiler   | 10.0 | 3.0 | 5  | `timing_sample100_seg50_min1_max36` |

**고정 설정 (전 데이터셋 공통)**

| 항목 | 값 |
| --- | --- |
| baseline | `pna` (per-sample projection) |
| pool size | 1000 (train subsample, seed 42) |
| n_alphas | 50 / phase |
| segment tag | `seg50 / min1 / max48` — **파일명 태그일 뿐, PNA-BIG 결과에 영향 없음** (주의사항 ⑤) |
| CV | 5-fold, seed 42, fold 평균 |
| masking | top 10% (`--topk 0.1 --top 0`) |
| 평가 mask_ref | `zero`, `average`, `pna` |

> ⚠️ 하이퍼파라미터가 **세 군데에 중복**되어 있다: `hp_pna.sh`(bash), `check_completeness.py`의 `PNA_HP`, `TR_table.py`의 `PNA_HP`.
> 값을 바꾸려면 세 곳을 모두 고쳐야 한다.

---

## 4. Quickstart

```bash
conda activate timing
cd ~/TIMING-main/PNA-BIG-XAI

tmux new -s pna
GPUS="1 2 3 4 5" bash run_all.sh all 2>&1 | tee logs/run_all.log
# Ctrl+b, d 로 detach → tmux attach -t pna
```

`run_all.sh`가 아래 파이프라인 전체를 순서대로 돌린다. 단계별로 따로 돌리려면:

```bash
bash run_all.sh attr        # Step 1    : PNA-BIG attribution
bash run_all.sh comp        # Step 2    : completeness (+ TIMING attribution 생성)
bash run_all.sh baseattr    # Step 3b-1 : baseline 6종 attribution + zero/avg CPD
bash run_all.sh cpd         # Step 3a   : PNA-BIG CPD (zero/average/pna)
bash run_all.sh baselines   # Step 3b-2 : baseline 7종 CPD (pna fill)
bash run_all.sh dominance   # Step 3c   : Trend vs Residual
bash run_all.sh domtables   # Step 4a   : Trend vs Residual 표만 (mask_ref 3종)
bash run_all.sh tables      # Step 4    : 표 전체 (4a + 8-method 통합)
```

**환경변수**

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `GPUS` | `1 2 3 4 5` | fold 0~4 를 매핑할 물리 GPU. GPU가 4개면 `GPUS="0 1 2 3"` |
| `DATASETS` | `wafer boiler epilepsy PAM` | |
| `FOLDS` | `0 1 2 3 4` | |
| `ATTR_DIR` | `results_pna_hpt` | PNA-BIG attribution 저장 폴더 |
| `BASE_DIR` | `results_our` | baseline 7종 + TIMING attribution 폴더 |

**예상 소요**: Step 1 ≈ 1~2h. Step 3b는 PAM 혼자 ≈ 1.5h (동시에 데이터셋 2개까지는 가능).

---

## 5. 파이프라인 상세

```
Step 1    attribution        → results_pna_hpt/*.npy  +  results_pna/*anchoridx*.npy
Step 2    completeness       → completeness_tables/   (+ results_our/ 에 TIMING npy)
Step 3b-1 baseline attr      → results_our/*.npy  +  state_{data}_{fold}_0_results_baseline.csv
Step 3a   PNA-BIG CPD        → results_pna_hpt/eval_anchor/{data}.csv
Step 3b-2 baseline CPD       → results_pna_hpt/eval_anchor/{data}_baselines_pna.csv
Step 3c   Trend vs Residual  → results_pna_hpt/eval_dominance/full_eval.csv
Step 4    표                 → aggregated_results/ , results_pna_hpt/eval_dominance/{zero,average,pna}/
```

### Step 1 — Attribution 생성

`real/main_td.py --baseline pna`가 npy 7종을 `$ATTR_DIR`에 저장한다.

저장 키: `{data}_state_timing_td_{kind}_kalman_seg50_min1_max48_lam{λ0}x{λf}_result_{fold}_42.npy`
`kind ∈ trend(|T|), residual(|R|), trend_signed(T), residual_signed(R), combined(|T+R|), T_plus_R(|T|+|R|), fxc(F(x)−mean F(c))`

동시에 `SAVE_ANCHOR_IDX=1`이 켜져 있어 **anchor 재현 증명용** 파일이 `results_pna/`에 저장된다:

```
results_pna/{data}_state_pool_{fold}_42.npy                        # pool [1000,T,D]
results_pna/{data}_state_anchoridx_lam{λ0}x{λf}_{fold}_42.npy      # idx  [N,Ka]
```

검증 (`run_all.sh attr`가 자동 출력):

```bash
ls results_pna_hpt/*combined* | wc -l                   # → 20
grep -L "SAVE_ANCHOR_IDX. saved" logs/pna_hpt/*.log     # → 출력 없음
grep -il "error\|traceback" logs/pna_hpt/*.log          # → 출력 없음
```

### Step 2 — Completeness 진단 (+ TIMING attribution)

PNA-BIG의 signed 합이 `F(x) − mean F(c)`를 얼마나 잘 닫는지 검산하고 TIMING과 비교한다.

> **중요**: `run_timing_comp.sh`는 `--explainers our timing_comp --baseline zero`라서
> **TIMING attribution(`timing_sample100_seg*`)과 completeness 버전을 한 번에** 뽑아 `results_our/`에 저장한다.
> Step 3b-2에서 쓰는 TIMING npy도 여기서 나온다. (그래서 별도의 TIMING attribution 스크립트는 필요 없다.)

산출: `completeness_pna_hpt.txt`, `completeness_timing.txt`, `completeness_tables/*.csv`

### Step 3b-1 — Baseline 6종 attribution + zero/average CPD

```bash
bash scripts/real/run_10perc_masking_6xai.sh    # 100 step
```

npy는 `results_our/`에, zero/average fill CPD는 루트의 `state_{data}_{fold}_0_results_baseline.csv`에 쓴다.
Step 4의 `aggregate_xai_results.py`가 이 CSV를 읽어 Average/Zero 행을 만든다.

CSV 헤더 순서:
```
seed,fold,baseline,area,explainer,lambda_1,lambda_2,lambda_3,cum_50_diff,cum_diff,AUCC,accuracy,comprehensiveness,cross_entropy,log_odds,sufficiency
```

### Step 3a — PNA-BIG anchor-masking CPD

`--mask_refs zero average pna`로 **같은 attribution을 세 가지 masking reference에서** 평가한다.
`--verify_anchors`가 Step 1에서 저장한 anchor와 평가 시 재도출한 anchor가 bit-identical인지 assert한다.

```bash
grep -h "max|loaded" logs/eval_anchor/*.log | sort -u    # → 전부 0.0 이어야 정상
```

### Step 3b-2 — Baseline 7종 CPD (pna fill)

`augmented_occlusion`, `gate_mask`, `gradientshap_abs`, `timex`, `timex++`,
`integrated_gradients_base_abs`, `TIMING(=timing_sample100_seg*)` 을 **PNA-BIG와 같은 anchor**로 평가한다.
zero/average는 Step 3b-1에 이미 있으므로 여기선 `--mask_refs pna`만 돌린다.

병합 파일명은 반드시 `{data}_baselines_pna.csv` 여야 한다 — `aggregate_xai_results.py`가 이 이름으로 찾는다.
이름이 다르면 에러 없이 `aggregated_results/warnings.txt`에 "PNA baseline aggregate 파일 누락"만 남고 표에서 7줄이 조용히 빠진다.

### Step 3c — Trend vs Residual

`timing_td_trend_*` vs `timing_td_residual_*` 을 Step 3a와 동일하게
`--mask_refs zero average pna` 로 평가한다 (dominance 표용).
λ0/λf/Ka 는 `hp_pna.sh` 값이 그대로 들어가며, `--verify_anchors` 로 Step 1 anchor와 일치를 검증한다.

```bash
grep -h "max|loaded" logs/eval_dom/*.log | sort -u    # → 전부 0.0 이어야 정상
```

> `full_eval.csv` 에는 mask_ref 3종 × 5 fold 행이 함께 들어 있다.
> `TR_table.py` 는 반드시 `--mask_ref` 로 **하나만 골라서** 집계해야 한다 (안 그러면 3종이 한 평균으로 뭉개짐).

### Step 4 — 표 생성

`tables` 는 4a(Trend vs Residual 표) + 4b/4c(8-method CPD 통합)를 전부 돈다.
TR 표만 필요하면 `domtables` 로 4a만 돌린다 — `eval_anchor/` CSV가 없어도 동작한다.

```
results_pna_hpt/eval_dominance/{zero,average,pna}/
  result_table3.csv       # 논문 Table 3 형태 (Ratio / Dominant / CPD±std / 나머지)
  result_TR_raw.csv, result_TR.csv, result_TR_perfold.csv
results_pna_hpt/eval_anchor/all_8methods.csv
aggregated_results/
  cpd_table.csv / cpd_table.html    # mask_ref × 8 method × 4 dataset CPD 표
  summary_all_metrics_wide.csv
  warnings.txt                      # ← 반드시 확인
```

---

## 6. ⚠️ 실행 전 주의사항

**① `--train` 플래그는 절대 붙이지 않는다.**
Python에서 `bool("False") == True`라, `--train False`는 학습을 실행해버린다. load 모드는 플래그를 **생략**한다.
특히 Wafer는 절대 재학습하지 말 것 (체크포인트 로드만).

**② eval CSV는 append 모드다.**
`eval_cpd_cpp.py`는 같은 `--output_file`에 행을 누적한다. 재실행 전 반드시 삭제해야 fold가 중복 집계되지 않는다.
`run_all.sh`는 각 stage 시작 시 자동으로 `rm -f` 한다.

**③ anchoridx 파일명에는 Ka가 없다.**
`{data}_state_anchoridx_lam{λ0}x{λf}_{fold}_42.npy` — λ만 태그되고 Ka는 안 들어간다. 하지만 내용은 `[N, Ka]`로 Ka 의존이다.
**Ka를 바꾸면 이 파일을 지우고 Step 1부터 다시** 돌려야 한다. 안 그러면 `--verify_anchors`에서 shape/값 불일치로 죽는다.

**④ `--methods` 키는 npy 파일명과 정확히 일치해야 한다.**
`timing_td_{kind}_kalman_seg50_min1_max48_lam{λ0}x{λf}` — 안 맞으면 `FileNotFoundError`.
λ 태그는 float 문자열이다 (`10` → `10.0`, `0.5` → `0.5`).

**⑤ `--num_segments / --min_seg_len / --max_seg_len`는 PNA-BIG 결과에 영향이 없다.**
PNA-BIG는 order-averaged component-replacement path를 쓰고 TIMING의 random temporal mask를 쓰지 않는다.
`seg50_min1_max48`은 파일명 고정 태그일 뿐이다. (baseline 7종·TIMING은 seg 값이 실제로 의미를 가지며 데이터셋별로 다르다 — `hp_pna.sh`의 `TIMING_NAME` 참고.)

**⑥ 저장 폴더는 `SAVE_DIR` 환경변수로 정한다.**
`real/main_td.py`의 기본값은 `./results_our`다. PNA-BIG 결과는 `SAVE_DIR=./results_pna_hpt`로 분리한다 (`run_all.sh`가 처리).
단 pool/anchoridx/val_idx는 `SAVE_DIR`과 무관하게 `./results_pna/`에 하드코딩되어 저장된다.

**⑦ 결정성.** `cudnn.enabled=False`가 생성·평가 양쪽에 걸려 있다. anchor 재현(`--verify_anchors` == 0.0)의 전제이므로 **제거하지 말 것**.

---

## 7. 결과 (5-fold 평균 ± 표준편차)

### 7.1 Completeness (모든 값 낮을수록 좋음)

**PNA-BIG**

| Dataset | CR Dev All | CR Dev Top-50 | Neg Rate All | Neg Rate Top-50 | Norm Error |
| --- | --- | --- | --- | --- | --- |
| Boiler | 0.025 | 0.021 | 0.0% | 0.0% | 0.026 |
| PAM | 0.018 | 0.012 | 0.1% | 0.1% | 0.024 |
| Epilepsy | 0.027 | 0.028 | 0.3% | 0.0% | 0.001 |
| Wafer | 0.003 | 0.006 | 0.1% | 0.0% | 0.000 |
| **평균** | **0.018** | **0.017** | **0.13%** | **0.03%** | **0.013** |

**TIMING (비교)**

| Dataset | CR Dev All | CR Dev Top-50 | Neg Rate All | Neg Rate Top-50 | Norm Error |
| --- | --- | --- | --- | --- | --- |
| Boiler | 0.028 | 0.014 | 6.3% | 1.6% | 0.063 |
| PAM | 0.036 | 0.036 | 2.5% | 1.6% | 0.136 |
| Epilepsy | 0.014 | 0.019 | 0.9% | 0.7% | 0.001 |
| Wafer | 0.002 | 0.004 | 0.9% | 0.4% | 0.124 |
| **평균** | **0.020** | **0.018** | **2.65%** | **1.08%** | **0.081** |

### 7.2 CPD (10% masking, ↑ 높을수록 좋음)

| Mask Ref | Method | Boiler | Epilepsy | Wafer | PAM |
| --- | --- | --- | --- | --- | --- |
| **Average** | AFO | 0.2354 ± 0.0984 | 0.0285 ± 0.0074 | 0.0178 ± 0.0065 | 0.1398 ± 0.0193 |
| | GateMask | 0.3582 ± 0.0907 | 0.0159 ± 0.0026 | 0.1260 ± 0.0780 | 0.0440 ± 0.0104 |
| | GradSHAP | 0.6244 ± 0.1789 | 0.0518 ± 0.0085 | 0.4884 ± 0.0338 | 0.4318 ± 0.0310 |
| | TimeX | 0.2456 ± 0.0571 | 0.0366 ± 0.0091 | 0.0000 ± 0.0000 | 0.0685 ± 0.0114 |
| | TimeX++ | 0.0775 ± 0.0482 | 0.0339 ± 0.0113 | 0.0000 ± 0.0000 | 0.0582 ± 0.0281 |
| | IG | 0.6280 ± 0.1930 | 0.0526 ± 0.0080 | 0.5003 ± 0.0383 | 0.4364 ± 0.0369 |
| | **TIMING** | **1.1346 ± 0.3929** | **0.0576 ± 0.0102** | **0.7002 ± 0.0369** | **0.4766 ± 0.0442** |
| | PNA-BIG | 0.9359 ± 0.3480 | 0.0517 ± 0.0088 | 0.4846 ± 0.0622 | 0.3262 ± 0.0535 |
| **Zero** | AFO | 0.3285 ± 0.1124 | 0.0308 ± 0.0084 | 0.0178 ± 0.0065 | 0.1998 ± 0.0288 |
| | GateMask | 0.4567 ± 0.0989 | 0.0160 ± 0.0029 | 0.1260 ± 0.0780 | 0.0537 ± 0.0080 |
| | GradSHAP | 0.6114 ± 0.1169 | 0.0538 ± 0.0093 | 0.4884 ± 0.0338 | 0.5222 ± 0.0357 |
| | TimeX | 0.3526 ± 0.1491 | 0.0390 ± 0.0096 | 0.0000 ± 0.0000 | 0.0910 ± 0.0167 |
| | TimeX++ | 0.1544 ± 0.0710 | 0.0357 ± 0.0118 | 0.0000 ± 0.0000 | 0.0610 ± 0.0268 |
| | IG | 0.5928 ± 0.1273 | 0.0545 ± 0.0088 | 0.5003 ± 0.0383 | 0.5649 ± 0.0482 |
| | **TIMING** | **1.4650 ± 0.3705** | **0.0607 ± 0.0113** | **0.7002 ± 0.0369** | **0.6019 ± 0.0660** |
| | PNA-BIG | 1.3057 ± 0.4433 | 0.0534 ± 0.0095 | 0.4846 ± 0.0622 | 0.4322 ± 0.0153 |
| **PNA** | AFO | 0.2171 ± 0.0354 | 0.0304 ± 0.0081 | 0.0051 ± 0.0028 | 0.2081 ± 0.0190 |
| | GateMask | 0.2071 ± 0.0616 | 0.0167 ± 0.0038 | 0.0075 ± 0.0028 | 0.0745 ± 0.0084 |
| | GradSHAP | 0.4611 ± 0.0543 | 0.0525 ± 0.0091 | 0.0193 ± 0.0085 | 0.5416 ± 0.0364 |
| | TimeX | 0.1408 ± 0.0689 | 0.0381 ± 0.0094 | 0.0000 ± 0.0000 | 0.1247 ± 0.0179 |
| | TimeX++ | 0.1335 ± 0.0704 | 0.0350 ± 0.0110 | 0.0000 ± 0.0000 | 0.0763 ± 0.0260 |
| | IG | 0.4416 ± 0.0758 | 0.0532 ± 0.0085 | 0.0193 ± 0.0081 | 0.5557 ± 0.0485 |
| | TIMING | 0.7725 ± 0.0517 | **0.0591 ± 0.0107** | 0.0198 ± 0.0076 | **0.5970 ± 0.0617** |
| | **PNA-BIG** | **0.9839 ± 0.1268** | 0.0526 ± 0.0093 | **0.0381 ± 0.0075** | 0.5260 ± 0.0244 |

> **읽는 법**: CPD는 masking reference에 강하게 coupled되어 있다. zero/average fill은 zero-baseline 방법(TIMING 등)에 구조적으로 유리하고, in-manifold anchor로 fill하면 Boiler·Wafer에서 PNA-BIG가 역전한다. 즉 이 표의 요점은 "PNA-BIG가 이긴다"가 아니라 **CPD 자체가 reference-dependent** 하다는 것.

---

## 8. Repository Structure (핵심)

| 파일 | 역할 |
| --- | --- |
| `hp_pna.sh` | **데이터셋별 λ0/λf/Ka + TIMING npy 이름 (single source of truth)** |
| `run_all.sh` | **전체 파이프라인 (Step 1~4)** |
| `real/main_td.py` | attribution 생성. `--baseline pna` 분기, anchor 캐싱·선택·Ka 평균 |
| `attribution/pna.py` | PNA anchor 선택 (`build_pna_cache`, `select_pna_indices`, `select_global_neutral_anchors`) |
| `attribution/explainers_pna.py` | order-averaged PNA-BIG attribution (baseline도 trend/residual 분해) |
| `eval_cpd_cpp.py` | 저장된 npy로 CPD + faithfulness 재계산. `--mask_refs`, `--verify_anchors` |
| `run_timing_comp.sh` | TIMING attribution + completeness 버전 (한 번에) |
| `scripts/real/run_10perc_masking_6xai.sh` | baseline 6종 attribution + zero/avg CPD |
| `check_completeness.py` / `check_completeness_timing.py` | completeness 진단 |
| `make_comp_table.py` | completeness txt → CSV 표 |
| `TR_table.py` | Trend vs Residual dominance 표 |
| `aggregate_xai_results.py` | 흩어진 CSV → 최종 8-method CPD 표 |
| `sweep_hpo.sh` / `pick_lambda.py` / `run_ka_sensitivity.sh` / `agg_ka_sensitivity.py` | HPO (부록 A) |
| `viz/viz_hidden_path.py` / `viz/run_viz.sh` | hidden trajectory OOD 진단 (부록 B) |

**출력 폴더**

| 폴더 | 내용 |
| --- | --- |
| `results_pna_hpt/` | PNA-BIG attribution npy + eval CSV |
| `results_pna/` | pool / anchoridx / val_idx (재현 증명용, 경로 하드코딩) |
| `results_our/` | baseline 6종 + TIMING attribution npy |
| `aggregated_results/` | 최종 CPD 표 |
| `completeness_tables/` | completeness 표 |

---

## 부록 A. 하이퍼파라미터 튜닝 재현 (선택)

> **이 절은 안 돌려도 된다.** 결과는 이미 `hp_pna.sh`에 반영되어 있다. 35 조합 × 4 dataset × 5 fold라 매우 오래 걸린다.

### 설계 원칙

- **선택 기준 = `average` fill CPD의 5-fold validation 평균.**
  `average`/`zero`는 λ와 무관한 fill이라 측정자가 움직이지 않는다(공정).
  `pna` fill은 fill 자체가 λ에 의존 → 선택에 쓰면 순환 논리(anchor를 x에서 멀리 미는 λ가 이겨버림). 최종 test 표에서만 본다.
- `average` fill = 각 샘플의 시간 평균을 모든 timestep에 반복한 상수 시계열.
- **Wafer만 val 분할이 없다.** `PNA_TUNE_VAL=1`일 때만 `datasets/wafer.py`가 TRAIN을 fold-aware stratified로 `train_tune(≈80%) / val_tune(≈20%)`으로 나눈다.
  anchor pool과 mean/std를 `train_tune`으로만 fit → val leakage 0.
  이 환경변수는 sweep 스크립트 안에서만 export되며, **최종 test·다른 attribution 생성에는 영향이 없다** (미설정 시 100% TRAIN + 진짜 TEST + 원본 normalization).

### 절차

```bash
# 1) λ sweep : (λ0 ∈ {0.5,1,3,5,10}) × (λf ∈ {0.5,1,3,5,10,15,20}) = 35 조합, Ka=5 고정
#    ★ main_td.py 기본 저장 폴더가 results_our 이므로 SAVE_DIR 을 맞춰줘야 한다
SAVE_DIR=./results_pna bash sweep_hpo.sh
#    → results_pna/sweep5_eval/*.csv, chosen_lambdas.csv

# (attribution 을 이미 뽑아둔 경우 평가+선택만)
bash sweep_hpt_pick.sh
python pick_lambda.py --eval_dir results_pna/sweep5_eval --select_ref average

# 2) λ 고정 후 Ka ∈ {1,3,5,10} 민감도
REUSE_KA5=1 bash run_ka_sensitivity.sh 2>&1 | tee logs/ka_sensitivity_console.log
#    → results_pna/ka_sensitivity/ka_sensitivity_*.csv
```

전체 흐름:
```
λ sweep attribution (val) → average-fill validation CPD → 데이터셋별 최적 λ 선택
→ λ 고정 → Ka={1,3,5,10} validation sensitivity → 최종 Ka 결정 → hp_pna.sh 갱신
```

### 선택 결과

**λ (Ka=5 고정, average fill 5-fold val CPD ↑)**

| dataset | λ0 | λf | val CPD |
| --- | --- | --- | --- |
| PAM | 0.5 | 3 | 0.3117 ± 0.0518 |
| boiler | 10 | 3 | 0.9524 ± 0.3593 |
| epilepsy | 10 | 1 | 0.0455 ± 0.0116 |
| wafer | 0.5 | 0.5 | 0.4538 ± 0.0615 |

**Ka (λ 고정)**

| dataset | Ka | λ0 | λf | val CPD |
| --- | --- | --- | --- | --- |
| PAM | 10 | 0.5 | 3 | 0.3351 ± 0.0354 |
| boiler | 5 | 10 | 3 | 0.9524 ± 0.3593 |
| epilepsy | 10 | 10 | 1 | 0.0483 ± 0.0117 |
| wafer | 1 | 0.5 | 0.5 | 0.5022 ± 0.0615 |

> sweep은 파일명 태그가 `kalman_seg0_min1_max48_val_lam{λ0}x{λf}`로, 최종 test(`kalman_seg50_min1_max48_lam{λ0}x{λf}`)와 다르다. `--methods` 키를 헷갈리지 말 것.

---

## 부록 B. OOD 완화 시각화 (선택)

baseline→input 경로를 따라 GRU hidden state를 뽑아, z-score된 hidden 공간에서 training pool까지의 k-NN 거리로 "경로가 학습 manifold 안에 머무는 정도"를 잰다. zero vs PNA 비교.
endpoint를 빼고 **interior(0<α<1)** 거리로 판단한다 (PNA anchor는 training sample이라 시작점 in-manifold는 자명).

```bash
bash viz/run_viz.sh
python viz/agg_interior_folds.py --root viz_hidden
```

> ⚠️ `viz/run_viz.sh`는 아직 λ=10.0×10.0, Ka=5가 하드코딩되어 있다. HPO 결과로 돌리려면 `hp_pna.sh`를 source해서 `--pna_lam0/--pna_lamf/--pna_ka`를 바꿔야 한다.

---

## 부록 C. Legacy 스크립트

아래는 HPO 이전(λ=10×10, Ka=5 하드코딩) 버전이며 `run_all.sh`로 대체되었다. 참고용으로만 남긴다.

`run_pna_all.sh`, `run_pna_eval.sh`, `run_PNA_anchor_cpd.sh`, `run_Other_anchor_cpd.sh`, `run_td_all.sh`(원본 BIG), `eval_wafer_lf01_only.sh`

---

Built on TIMING (Jang, Kim, Yang, 2025) — <https://arxiv.org/abs/2506.05035>
