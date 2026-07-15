"""
Re-evaluate CPD (+ acc/comp/ce/lodds/suff) from saved attribution .npy files,
under one or more MASKING REFERENCES (perturbation baselines).

reference: zero / average(TIMING x_avg) / pna / na
  - 한 비교 안에선 모든 method 를 같은 reference 로 평가 (공정성).
  - TIMING 원 코드도 masking 값을 [x_avg, 0.0] 로 이미 두 개 돌림 → reference 는 설계 축.

ANCHOR 재현 (핵심):
  main_td.py 는 독립 generator 로 pool 을 뽑으므로(전역 RNG 아님) 재도출이 결정적이다.
  그래도 '증명'을 위해 --anchor_idx_dir 를 주면 생성 때 저장한
     {data}_{mt}_pool_{fold}_{seed}.npy          (pool [P,T,D], seed 만 의존)
     {data}_{mt}_anchoridx_lam{l0}x{lf}_{fold}_{seed}.npy  (idx [N,Ka], lam 의존)
  을 로드해 anchors = pool[idx] 로 복원한다 (RNG·cudnn·tie 의존 0).
  --verify_anchors 를 함께 주면 로드본과 재도출본이 bit-identical 인지 assert.

Table 3 default: --topk 0.1 --top 0 (10% masking).

Usage (증명 경로, 권장):
    CUDA_VISIBLE_DEVICES=0 python real/eval_cpd_cpp.py \
        --data wafer --fold 0 --device cuda:0 \
        --mask_refs zero average pna --pna_lam0 10.0 --pna_lamf 10.0 --pna_ka 5 \
        --anchor_idx_dir results_pna --verify_anchors \
        --npy_dir results_pna --output_file results_pna/eval_anchor/wafer.csv \
        --methods timing_td_combined_kalman_seg50_min1_max48_lam10.0x10.0

Usage (재도출 fallback, --anchor_idx_dir 생략):
    ... --mask_refs zero average pna --pna_lam0 10.0 --pna_lamf 10.0 --pna_ka 5 ...

주의:
    - --pna_lam0/lamf 는 PNA-BIG npy 태그의 lam 과 반드시 일치.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch as th
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real.classifier import MimicClassifierNet
from real.cumulative_difference import cumulative_difference
from tint.metrics import (
    accuracy, comprehensiveness, cross_entropy, log_odds, sufficiency,
)
from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer


CFG = {
    "PAM":      (PAM,      17, 8, 600, False),
    "boiler":   (Boiler,   20, 2, 36,  False),
    "epilepsy": (Epilepsy, 1,  2, 178, False),
    "wafer":    (Wafer,    1,  2, 152, True),
}


def build_datamodule(data, fold, seed):
    DM, _, _, _, needs_folds = CFG[data]
    if needs_folds:
        return DM(n_folds=5, fold=fold, seed=seed)
    return DM(fold=fold, seed=seed)


def compute_extra_metrics(classifier, x_test, mask_test, timesteps, attr,
                          baselines, topk, testbs, device):
    test_loader = DataLoader(
        TensorDataset(x_test, mask_test), batch_size=testbs, shuffle=False
    )
    tot = {"acc": 0.0, "comp": 0.0, "ce": 0.0, "lodds": 0.0, "suff": 0.0}
    n_total = 0
    for batch_idx, (x_batch, m_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device); m_batch = m_batch.to(device)
        bs = x_batch.shape[0]
        start = batch_idx * testbs; end = start + bs
        ts_batch = timesteps[start:end]
        if isinstance(baselines, th.Tensor):
            b_batch = baselines[start:end].to(device)
        else:
            b_batch = baselines
        v_batch = attr[start:end].to(device)
        fwd_args = (m_batch, ts_batch, False)
        kw = dict(attributions=v_batch, baselines=b_batch,
                  topk=topk, additional_forward_args=fwd_args)
        tot["acc"]   += float(accuracy(classifier, x_batch, **kw))          * bs
        tot["comp"]  += float(comprehensiveness(classifier, x_batch, **kw)) * bs
        tot["ce"]    += float(cross_entropy(classifier, x_batch, **kw))     * bs
        tot["lodds"] += float(log_odds(classifier, x_batch, **kw))          * bs
        tot["suff"]  += float(sufficiency(classifier, x_batch, **kw))       * bs
        n_total += bs
    return {k: v / n_total for k, v in tot.items()}


def build_cache(x_train, classifier, args):
    """main_td.py 와 동일한 pool 구성 + cache. (독립 generator → 결정적)"""
    from attribution.pna import build_pna_cache
    g = th.Generator().manual_seed(args.seed)
    n_train = x_train.shape[0]
    pool_n = min(args.pool_size, n_train)
    idx = th.randperm(n_train, generator=g)[:pool_n]
    pool = x_train[idx]
    cache = build_pna_cache(pool, classifier, feature=args.pna_feature,
                            lam0=args.pna_lam0, lamf=args.pna_lamf)
    return cache


def derive_anchors(which, x_test, cache, classifier, args):
    """anchors [N,Ka,T,D] (CPU). chunk 선택으로 GPU 메모리 안전."""
    from attribution.pna import select_pna_baselines, select_global_neutral_anchors
    if which == "na":
        g_anchors = select_global_neutral_anchors(cache, Ka=args.pna_ka).cpu()  # [Ka,T,D]
        N = x_test.shape[0]
        return g_anchors.unsqueeze(0).expand(N, *g_anchors.shape).contiguous()
    outs = []
    for s in range(0, x_test.shape[0], args.anchor_chunk):
        xb = x_test[s:s + args.anchor_chunk]
        outs.append(select_pna_baselines(xb, cache, classifier, Ka=args.pna_ka).cpu())
    return th.cat(outs, 0)  # [N,Ka,T,D]


def load_anchor_idx(which, x_test, args):
    """생성 때 저장한 pool + anchoridx 로드 → anchors [N,Ka,T,D] (CPU). 없으면 None."""
    if not args.anchor_idx_dir or which != "pna":
        return None
    d, mt, data, fold, seed = (args.anchor_idx_dir, args.model_type,
                               args.data, args.fold, args.seed)
    pool_p = f"{d}/{data}_{mt}_pool_{fold}_{seed}.npy"
    idx_p  = f"{d}/{data}_{mt}_anchoridx_lam{args.pna_lam0}x{args.pna_lamf}_{fold}_{seed}.npy"
    if not (os.path.exists(pool_p) and os.path.exists(idx_p)):
        return None
    pool = th.from_numpy(np.load(pool_p)).float()          # [P,T,D]
    idx  = th.from_numpy(np.load(idx_p)).long()            # [N,Ka]
    if idx.shape[0] != x_test.shape[0]:
        raise ValueError(f"anchoridx N={idx.shape[0]} != eval N={x_test.shape[0]}")
    anchors = pool[idx].contiguous()                       # [N,Ka,T,D]
    print(f"[anchors:pna] LOADED pool{tuple(pool.shape)} idx{tuple(idx.shape)} "
          f"-> {tuple(anchors.shape)}  ({idx_p})")
    return anchors


def fills_for_ref(ref, x_test, anchors_dict, args):
    if ref == "zero":
        return [0.0], "-"
    if ref == "average":
        x_avg = x_test.mean(1, keepdim=True).repeat(1, x_test.shape[1], 1)
        return [x_avg], "-"
    if ref in ("pna", "na"):
        anchors = anchors_dict[ref]                        # [N,Ka,T,D] CPU
        if args.anchor_reduce == "mean":
            return [anchors.mean(dim=1)], "mean"
        return [anchors[:, k] for k in range(anchors.shape[1])], "peranchor"
    raise ValueError(ref)


def eval_setting(
    classifier,
    x_test,
    attr,
    fills,
    args,
    mask_test,
    timesteps,
    device,
):
    cds, auccs, c50s = [], [], []
    ex = {
        "acc": [],
        "comp": [],
        "ce": [],
        "lodds": [],
        "suff": [],
    }

    for b in fills:
        b_dev = b.to(device) if isinstance(b, th.Tensor) else b

        cum_diff, AUCC, cum_50_diff, _ = cumulative_difference(
            classifier,
            x_test,
            attributions=attr,
            baselines=b_dev,
            topk=args.topk,
            top=args.top,
            testbs=args.testbs,
            largest=True,
            additional_forward_args=(mask_test, timesteps, False),
        )

        cds.append(float(cum_diff))
        auccs.append(float(AUCC))
        c50s.append(float(cum_50_diff))

        if not args.cpd_only:
            e = compute_extra_metrics(
                classifier,
                x_test,
                mask_test,
                timesteps,
                attr,
                b_dev,
                args.topk,
                args.testbs,
                device,
            )

            for k in ex:
                ex[k].append(e[k])

        del b_dev

    mean_value = lambda values: float(np.mean(values))

    if args.cpd_only:
        extras = {k: float("nan") for k in ex}
    else:
        extras = {k: mean_value(v) for k, v in ex.items()}

    return (
        mean_value(cds),
        mean_value(auccs),
        mean_value(c50s),
        extras,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, choices=list(CFG))
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_type", default="state")
    p.add_argument("--eval_split", default="test", choices=["test", "val"])
    p.add_argument("--testbs", type=int, default=30)
    p.add_argument("--topk", type=float, default=0.1)
    p.add_argument("--top", type=int, default=0)
    p.add_argument("--npy_dir", default="results_pna")
    p.add_argument("--output_file", default="results_pna/full_eval.csv")
    p.add_argument("--device", default="cuda:0")

    p.add_argument("--mask_refs", nargs="+", default=["zero"],
                   choices=["zero", "average", "pna", "na"])
    p.add_argument("--anchor_reduce", default="peranchor",
                   choices=["peranchor", "mean"])
    p.add_argument("--pna_feature", default="hidden", choices=["hidden", "logits"])
    p.add_argument("--pna_ka", type=int, default=5)
    p.add_argument("--pna_lam0", type=float, default=10.0)
    p.add_argument("--pna_lamf", type=float, default=10.0)
    p.add_argument("--pool_size", type=int, default=1000)
    p.add_argument("--anchor_chunk", type=int, default=512,
                   help="anchor 재도출 시 GPU 메모리 안전용 chunk")
    p.add_argument("--anchor_idx_dir", default=None,
                   help="생성 때 저장한 pool/anchoridx 디렉토리. 주면 로드 경로 우선.")
    p.add_argument("--verify_anchors", action="store_true",
                   help="로드본 vs 재도출본 bit-identical assert (증명용)")

    p.add_argument(
        "--cpd_only",
        action="store_true",
        help="CPD 계열만 계산하고 acc/comp/ce/log-odds/sufficiency는 생략",
        )

    p.add_argument("--methods", nargs="+", required=True)
    args = p.parse_args()

    seed_everything(args.seed, workers=True)
    device = th.device(args.device)
    _, feat, n_state, n_t, _ = CFG[args.data]

    classifier = MimicClassifierNet(
        feature_size=feat, n_state=n_state, n_timesteps=n_t,
        hidden_size=200, regres=True, loss="cross_entropy",
        lr=1e-4, l2=1e-3, model_type=args.model_type,
    )
    ckpt = (f"./model/{args.data}/{args.model_type}_classifier_"
            f"{args.fold}_{args.seed}_no_imputation")
    classifier.load_state_dict(th.load(ckpt, map_location=device))
    classifier.eval().to(device)
    th.backends.cudnn.enabled = False  # main_td 와 동일 (forward 수치 일치 → anchor 재현 보장)

    dm = build_datamodule(args.data, args.fold, args.seed)
    tr = dm.preprocess(split="train")            # mean/std init + pool 소스 (1회)
    x_train = tr["x"].to(device)
    ev = dm.preprocess(split=args.eval_split)    # split 당 1회
    x_test = ev["x"].to(device)
    mask_test = ev["mask"].to(device)

    if args.eval_split == "val":
        idx_path = f"{args.npy_dir}/{args.data}_{args.model_type}_val_idx_{args.fold}_{args.seed}.npy"
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Missing val index file: {idx_path}")
        idx = th.from_numpy(np.load(idx_path)).long().to(device)
        x_test = x_test[idx]; mask_test = mask_test[idx]

    data_len, t_len, _ = x_test.shape
    timesteps = (th.linspace(0, 1, t_len, device=device)
                 .unsqueeze(0).repeat(data_len, 1))

    # --- anchor 준비 (필요 ref 만) : 로드 우선, 없으면 재도출 ---
    anchors_dict = {}
    cache = None
    need_derive = any((r in ("pna", "na")) for r in args.mask_refs)
    for which in ("pna", "na"):
        if which not in args.mask_refs:
            continue
        loaded = load_anchor_idx(which, x_test, args)   # pna 만, dir 있을 때
        if loaded is not None and not args.verify_anchors:
            anchors_dict[which] = loaded
            continue
        if cache is None:
            cache = build_cache(x_train, classifier, args)
        derived = derive_anchors(which, x_test, cache, classifier, args)
        if loaded is not None and args.verify_anchors:
            max_abs = (loaded - derived).abs().max().item()
            print(f"[verify:{which}] max|loaded - derived| = {max_abs:.3e} "
                  f"({'OK bit-identical' if max_abs == 0.0 else 'MISMATCH!!'})")
            assert max_abs == 0.0, "anchor mismatch: 생성/평가 재현 불일치"
            anchors_dict[which] = loaded
        else:
            anchors_dict[which] = derived
        print(f"[anchors:{which}] shape={tuple(anchors_dict[which].shape)} "
              f"lam={args.pna_lam0}x{args.pna_lamf} Ka={args.pna_ka} "
              f"reduce={args.anchor_reduce}")

    ref_fills = {ref: fills_for_ref(ref, x_test, anchors_dict, args)
                 for ref in args.mask_refs}

    files = []
    for meth in args.methods:
        f = (f"{args.npy_dir}/{args.data}_{args.model_type}_{meth}"
             f"_result_{args.fold}_{args.seed}.npy")
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing attribution file: {f}")
        files.append((meth, f))

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output_file)
    fp = open(args.output_file, "a", newline="")
    w = csv.writer(fp)
    if write_header:
        w.writerow([
            "data", "fold", "seed", "method", "mask_ref", "reduce", "metric",
            "topk", "top", "cum_diff", "AUCC", "cum_50_diff",
            "accuracy", "comprehensiveness", "cross_entropy", "log_odds", "sufficiency",
        ])

    for method, f in files:
        attr_np = np.load(f)
        if attr_np.shape[0] != x_test.shape[0]:
            raise ValueError(f"attr N={attr_np.shape[0]} != eval N={x_test.shape[0]} ({f})")
        if not np.isfinite(attr_np).all():
            n_bad = int((~np.isfinite(attr_np)).sum())
            raise ValueError(f"attr has {n_bad} NaN/Inf: {f}")
        attr = th.from_numpy(attr_np).float().cpu()

        for ref in args.mask_refs:
            fills, reduce_tag = ref_fills[ref]
            cum_diff, AUCC, cum_50_diff, extras = eval_setting(
                classifier, x_test, attr, fills, args, mask_test, timesteps, device)
            w.writerow([
                args.data, args.fold, args.seed, method, ref, reduce_tag,
                "CPD", args.topk, args.top, cum_diff, AUCC, cum_50_diff,
                extras["acc"], extras["comp"], extras["ce"],
                extras["lodds"], extras["suff"],
            ])
            print(f"[{args.data} f{args.fold}] {method:52s} "
                  f"mask={ref:7s}({reduce_tag}) CPD={cum_diff:.4f} "
                  f"AUCC={AUCC:.4f} comp={extras['comp']:.4f} suff={extras['suff']:.4f}")
    fp.close()


if __name__ == "__main__":
    main()
