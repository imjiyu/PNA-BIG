# PNA-BIG: Projected Neutral Anchor Baseline for BIG

**Trend-Residual Path Decomposition with In-Manifold Neutral Anchor Baselines**
Built on BIG / TIMING (Jang et al., 2025).

---

## Overview

PNA-BIG는 BIG의 attribution 경로에서 **baseline 설계**를 개선한 확장이다.

- **기존 BIG**: baseline `c = 0` (zero). zero 시퀀스는 데이터 manifold 밖이라, RNN/GRU의 hidden-state 경로가 학습 분포 밖(OOD)으로 벗어난다.
- **PNA-BIG**: baseline을 zero 대신 **Projected Neutral Anchor(PNA)** — 각 입력 `x`에 대해 training pool에서 (1) representation space에서 가깝고 (2) input space에서 zero에 가까우며 (3) decision-neutral하고 (4) 실제 training sample인 anchor를 골라 baseline으로 사용한다.

anchor 선택 목적함수:

```
c*(x) = argmin_c  Dφ(x,c) + λ0·L0(c) + λf·Lf(c)
        Dφ : representation(hidden) 거리        (x 의존)
        L0 : zero-proximity (input energy)
        Lf : output neutrality (softmax ~ uniform)
```

baseline이 non-zero이므로 경로도 확장된다. 입력과 baseline을 **모두 trend/residual로 분해**하여 component-replacement path를 구성하고, trend-first / residual-first 두 순서를 평균한다:

```
trend-first :  Tc+Rc → Tx+Rc → Tx+Rx
residual-first: Tc+Rc → Tc+Rx → Tx+Rx
Ā_T = ½(A_T^∅ + A_T^R),   Ā_R = ½(A_R^∅ + A_R^T)
```

completeness는 zero가 아니라 선택된 neutral anchor 기준으로 성립한다:

```
Σ A_i(x) = F_y(x) − (1/Ka) Σ_c F_y(c)      (Top-Ka anchor 평균)
```

---

## 이번 실험의 고정 설정

| 항목 | 값 |
| --- | --- |
| baseline | `pna` (per-sample projection) — 기본 실험 |
| λ0 × λf | **10.0 × 10.0** (고정) |
| Ka (anchor 수) | 5 |
| pool size | 1000 (train subsample, seed 고정) |
| n_alphas (적분 스텝) | 50 / phase |
| segment (num/min/max) | **50 / 1 / 48** (전 데이터셋 공통) |
| datasets | epilepsy, wafer, PAM, boiler (**freezer 제외**) |
| CV | 5-fold, seed 42, fold 평균 |
| faithfulness masking | top 10% (`--topk 0.1 --top 0`, baseline `0.0`) |

> **freezer는 이 연구에서 사용하지 않는다.** 스크립트/명령어에서 wafer와 혼동하지 말 것.
> 주의: PNA-BIG는 order-averaged component-replacement path를 사용하며 TIMING의 random temporal mask를 쓰지 않는다. 따라서 --num_segments / --min_seg_len / --max_seg_len는 attribution 결과에 영향을 주지 않는 호환용 인자이며, 파일명의 seg50_min1_max48는 고정 태그일 뿐이다. (다음 rerun에서 SEG="kalman"으로 정리 예정)
---

## 변경/추가된 파일 (기존 BIG 대비)

| 파일 | 상태 | 역할 |
| --- | --- | --- |
| `attribution/explainers_pna.py` | **신규 (explainers_td.py 대체)** | order-averaged PNA-BIG attribution. baseline도 trend/residual 분해 |
| `attribution/pna.py` | **신규** | PNA anchor 선택 (`build_pna_cache`, `select_pna_baselines`) |
| `real/main_td.py` | **수정** | `--baseline pna` 분기, anchor 캐싱·선택·Ka 평균, `_lam{λ0}x{λf}` 파일 태그 |
| `eval_cpd_cpp.py` | 기존 | 저장된 npy로 CPD + faithfulness 재계산 |
| `TR_table.py` | 기존 | Trend vs Residual dominance 집계표 |
| `viz/viz_hidden_path.py` | **신규** | hidden-state trajectory 시각화 (OOD 완화 진단) |
| `viz/agg_interior_folds.py` | **신규** | fold별 interior 거리 CSV 집계 |
| `viz/run_viz.sh` | **신규** | 시각화 일괄 실행 (GPU 4개 병렬) |

---

## Requirements

```bash
pip install -r requirement.txt
# 핵심: torch==1.13.1, pytorch-lightning==2.0.7, captum==0.6.0, pykalman, time_interpret(tint)
```

사전 조건: `model/{data}/state_classifier_{fold}_42_no_imputation` 체크포인트가 있어야 한다
(BIG과 동일한 GRU 분류기; 없으면 기존 학습 절차로 먼저 생성).

---

## 실행 전 주의사항

- **A. append 모드 CSV**: `eval_cpd_cpp.py`는 같은 `--output_file`에 행을 누적한다. 재실행 전 반드시 기존 csv를 삭제하거나(`rm -f`), fold별 다른 파일에 쓴 뒤 병합한다. 안 지우면 fold 중복 집계.
- **B. `--methods` 키는 npy 파일명과 정확히 일치**해야 한다. PNA 결과는 파일명에 `_lam10.0x10.0` 태그가 붙는다. 형식:
  `timing_td_{kind}_kalman_seg50_min1_max48_lam10.0x10.0`
  (`kind` ∈ `trend`, `residual`, `combined`, `T_plus_R`). 안 맞으면 `FileNotFoundError`.
- **C. `--train` 플래그는 절대 붙이지 않는다.** load 모드는 플래그를 생략해야 한다 (`bool("False") == True` 파이썬 버그로 `--train False`는 학습을 실행해버림).
- **D. 저장 위치**: PNA 결과는 `results_pna/`에 저장된다.
- **E. GPU 할당**: 모든 스크립트의 `CUDA_VISIBLE_DEVICES`는 실행 환경(GPU 개수)에 맞게 조정.

---

## 전체 파이프라인 (한눈에)

```
Step 1. attribution 생성      : run_pna_all.sh          → results_pna/*.npy
Step 2. CPD/faithfulness 평가 : run_pna_eval.sh         → 표 2종 CSV
Step 3. OOD 시각화            : viz/run_viz.sh           → viz_hidden/ 그림·CSV
```

**가장 빠른 재현: 아래 세 스크립트를 순서대로 실행하면 끝난다.**

```bash
bash run_pna_all.sh     # (1) attribution — 4 dataset × 5 fold
bash run_pna_eval.sh    # (2) 평가 — |T+R| vs |T|+|R| + Trend vs Residual
bash viz/run_viz.sh     # (3) 시각화 — hidden trajectory OOD 진단
```

각 단계 세부는 아래.

---

## Step 1 — Attribution 생성

`real/main_td.py`가 `--baseline pna`로 PNA-BIG attribution 6종 npy를 `results_pna/`에 저장한다.

**저장 키**:
`{data}_state_timing_td_{kind}_kalman_seg50_min1_max48_lam10.0x10.0_result_{fold}_42.npy`
`kind` ∈ `trend`(|T|), `residual`(|R|), `trend_signed`(T), `residual_signed`(R), `combined`(|T+R|), `T_plus_R`(|T|+|R|)
추가로 `timing_td_fxc_...` (completeness 검증용 F(x)−mean F(c)).

### 일괄 실행 스크립트: `run_pna_all.sh`

```bash
#!/usr/bin/env bash
set -u
mkdir -p logs/pna_lam10 results_pna

idx=0; pids=(); names=(); failed=0
wait_batch() {
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "[OK] ${names[$i]}"
    else echo "[FAIL] ${names[$i]} — check log"; failed=$((failed+1)); fi
  done
  pids=(); names=()
}

for data in epilepsy wafer PAM boiler; do
  for fold in 0 1 2 3 4; do
    gpu=$((idx % 4))                       # GPU 개수에 맞게 조정
    name="${data}_f${fold}"
    echo "[START] gpu=${gpu} ${name}"
    CUDA_VISIBLE_DEVICES=${gpu} \
      python real/main_td.py \
        --explainers our_td \
        --data "${data}" --fold "${fold}" --seed 42 \
        --baseline pna --model_type state --eval_split test \
        --pna_lam0 10.0 --pna_lamf 10.0 --pna_ka 5 \
        --num_segments 50 --min_seg_len 1 --max_seg_len 48 \
        --device cuda:0 --testbs 200 \
        > "logs/pna_lam10/${name}.log" 2>&1 &
    pids+=("$!"); names+=("${name}"); idx=$((idx+1))
    if (( ${#pids[@]} == 4 )); then wait_batch; fi   # GPU 개수만큼 배치
  done
done
(( ${#pids[@]} > 0 )) && wait_batch
echo "[DONE] attribution 생성 완료, 실패=${failed}"
```

```bash
bash run_pna_all.sh
```

### (선택) Completeness 빠른 확인

수식이 잘 닫히는지 `fxc`로 검증할 수 있다 (median CR ≈ 1, norm error ≈ 0이면 정상).
이번 실험 기준 median CR 1.006 / mean 0.9996으로 확인됨.

---

## Step 2 — CPD / Faithfulness 평가

두 종류 표를 만든다:
1. **Aggregation 표** — `|T+R|` vs `|T|+|R|` (combined vs T_plus_R)
2. **Dominance 표** — Trend vs Residual (`trend` vs `residual`, `TR_table.py`)

### 일괄 실행 스크립트: `run_pna_eval.sh`

```bash
#!/usr/bin/env bash
set -u
SEG="kalman_seg50_min1_max48"
LAM="_lam10.0x10.0"
ROOT="results_pna"
mkdir -p ${ROOT}/eval_combined ${ROOT}/eval_dominance logs/pna_eval

# --- 재실행 대비: 이전 결과 삭제 (주의사항 A) ---
rm -f ${ROOT}/eval_combined/*.csv ${ROOT}/eval_dominance/*.csv

idx=0
run_eval() {   # $1=subdir $2=prefix  $3,$4=methods
  local sub="$1" pref="$2" m1="$3" m2="$4"
  for data in epilepsy wafer PAM boiler; do
    for fold in 0 1 2 3 4; do
      gpu=$((idx % 4))
      CUDA_VISIBLE_DEVICES=$gpu nohup python eval_cpd_cpp.py \
        --data "$data" --fold "$fold" --device cuda:0 \
        --npy_dir ${ROOT} \
        --output_file "${ROOT}/${sub}/${pref}_${data}_f${fold}.csv" \
        --methods "${m1}" "${m2}" \
        > "logs/pna_eval/${pref}_${data}_f${fold}.log" 2>&1 &
      idx=$((idx+1)); (( idx % 4 == 0 )) && wait
    done
  done
  wait
}

# (1) Aggregation: |T+R| vs |T|+|R|
run_eval eval_combined combined \
  "timing_td_combined_${SEG}${LAM}" \
  "timing_td_T_plus_R_${SEG}${LAM}"

# (2) Dominance: Trend vs Residual
run_eval eval_dominance full \
  "timing_td_trend_${SEG}${LAM}" \
  "timing_td_residual_${SEG}${LAM}"

# --- fold별 CSV 병합 ---
awk 'FNR==1 && NR!=1 {next} {print}' ${ROOT}/eval_combined/combined_*.csv \
  > ${ROOT}/eval_combined/combined_eval.csv
awk 'FNR==1 && NR!=1 {next} {print}' ${ROOT}/eval_dominance/full_*.csv \
  > ${ROOT}/eval_dominance/full_eval.csv
echo "[DONE] 평가 CSV 병합 완료"
```

```bash
bash run_pna_eval.sh
```

### 표 조립

**① Aggregation 표 (|T+R| vs |T|+|R|)** — CPD 평균±표준편차:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("results_pna/eval_combined/combined_eval.csv")
mmap = {
  "timing_td_combined_kalman_seg50_min1_max48_lam10.0x10.0": "|T + R|",
  "timing_td_T_plus_R_kalman_seg50_min1_max48_lam10.0x10.0": "|T| + |R|",
}
df = df[df["metric"] == "CPD"].copy()
df["Method"] = df["method"].map(mmap)
df = df.dropna(subset=["Method"])
s = df.groupby(["Method","data"])["cum_diff"].agg(["mean","std"]).reset_index()
s["value"] = s.apply(lambda r: f'{r["mean"]:.3f} ± {r["std"]:.3f}', axis=1)
t = s.pivot(index="Method", columns="data", values="value")
t = t.reindex(index=["|T + R|","|T| + |R|"],
              columns=["boiler","PAM","epilepsy","wafer"])
print(t.to_string())
t.to_csv("results_pna/eval_combined/cpd_mean_std.csv")
print("\nsaved: results_pna/eval_combined/cpd_mean_std.csv")
PY
```

**② Dominance 표 (Trend vs Residual)**:

```bash
python TR_table.py \
  --results_dir results_pna \
  --eval_csv results_pna/eval_dominance/full_eval.csv \
  --out_dir results_pna/eval_dominance \
  --folds 0 1 2 3 4
```

> `eval_cpd_cpp.py`는 현재 CPD만 출력한다(CPP 루프 주석 처리). 다른 faithfulness 컬럼(comp/ce/log-odds/suff/acc)은 함께 계산되어 CSV에 저장된다.

---

## Step 3 — OOD 완화 시각화 (hidden trajectory)

baseline→input 경로를 따라 GRU hidden state를 뽑아, per-dim 표준화(z-score) hidden 공간에서 **training pool까지의 k-NN 거리**로 "경로가 학습 manifold 안에 머무는 정도"를 측정한다. zero vs PNA 비교.

- endpoint(baseline·input)는 제외하고 **interior(0<α<1)** 거리로 OOD 완화를 판단한다 (PNA anchor는 training sample이라 시작점 in-manifold는 자명).
- line(직선 c→x) / trend-first / residual-first 세 경로 모두 정량화.
- fold 0: 대표 그림 3장(무작위 샘플) + summary, fold 1~4: summary만.

### 일괄 실행: `viz/run_viz.sh`

```bash
bash viz/run_viz.sh
# 완료 후 5-fold 집계:
python viz/agg_interior_folds.py --root viz_hidden
```

> `viz/run_viz.sh`는 데이터셋 하나당 GPU 하나(0~3)를 배정해 4개를 병렬로 돌린다.
> 각 fold는 `viz_hidden/{data}/{data}_fold{F}_lam10.0x10.0_ka5_k5_a50_interior_summary.csv`를 남기고,
> `agg_interior_folds.py`가 이를 모아 **fold-평균 ± fold-표준편차 + PNA<zero 비율(%)**을 출력한다.
> 결과 해석: `PNA<zero %`가 높을수록(→100%) 해당 데이터셋에서 OOD 완화가 강함.

**tmux로 백그라운드 실행 (권장):**

```bash
tmux new -s viz
bash viz/run_viz.sh
# Ctrl+b, d 로 detach
# 나중에:  tmux attach -t viz
python viz/agg_interior_folds.py --root viz_hidden
```

---

## (참고) Baseline "global 1회" 변형 — NA

사수님 제안 검증용. anchor를 매 샘플마다 뽑지 않고 projection(Dφ)을 제거,
`base = λ0·L0 + λf·Lf`만으로 pool에서 top-Ka anchor를 **전체 1회** 골라 모든 샘플이 공유하는 변형.
`attribution/pna.py`의 `select_global_neutral_anchors`와 `main_td.py`의 `--baseline na`로 구현.
per-sample PNA와 CPD 비교용이며, Step 1에서 `--baseline na`로 바꾸면 `_na_lam10.0x10.0` 태그로 저장된다.

```bash
# 예: NA 변형 attribution
python real/main_td.py --explainers our_td \
  --data epilepsy --fold 0 --seed 42 \
  --baseline na --model_type state \
  --pna_lam0 10.0 --pna_lamf 10.0 --pna_ka 5 \
  --num_segments 50 --min_seg_len 1 --max_seg_len 48 \
  --device cuda:0 --testbs 200
# 평가 시 --methods 태그를 _na_lam10.0x10.0 로 교체
```

---

## 결과 요약 (λ=10×10, 5-fold 평균)

**CPD (|T+R|)** — per-sample PNA:

| Method | boiler | PAM | epilepsy | wafer |
| --- | --- | --- | --- | --- |
| \|T + R\| | 1.311 ± 0.408 | 0.401 ± 0.032 | 0.050 ± 0.010 | 0.154 ± 0.058 |
| \|T\| + \|R\| | 1.350 ± 0.472 | 0.366 ± 0.025 | 0.050 ± 0.009 | 0.183 ± 0.064 |

**OOD 완화 (interior k-NN dist, PNA<zero 비율)**:

| dataset | line | tf | rf | 판정 |
| --- | --- | --- | --- | --- |
| boiler | 100% | 100% | 100% | 완화 명확 |
| wafer | 67.6% | 38.0% | 65.0% | 대체로 완화 |
| PAM | 46.8% | 35.8% | 66.8% | 혼재 |
| epilepsy | 0.0% | 0.2% | 0.0% | 역전 |

→ OOD 완화 효과는 데이터셋(hidden manifold 구조)에 의존적.

---

Built on TIMING (Jang, Kim, Yang, 2025): https://arxiv.org/abs/2506.05035
