# eval_auroc.py  (eval_cpd_cpp.py 와 비슷,,,)
"""
Faithfulness = classifier AUROC drop under top-k% attribution masking.
- attribution 상위 k% (t,d) cell 을 baseline(reference)으로 치환 → classifier AUROC 재계산.
- reference/anchor 로직은 eval_cpd_cpp.py 를 그대로 재사용 (zero/average/pna/na 동일).
- k 를 sweep 해서 AUROC 하락 곡선을 뽑음. k=0 이 baseline AUROC.

Usage (CPD 와 동일한 인자 + --topks):
  CUDA_VISIBLE_DEVICES=1 python eval_auroc.py \
    --data wafer --fold 0 --device cuda:0 \
    --mask_refs zero average pna --pna_lam0 10.0 --pna_lamf 10.0 --pna_ka 5 \
    --anchor_idx_dir results_pna --verify_anchors \
    --npy_dir results_our --output_file results_pna/eval_auroc/wafer_baselines.csv \
    --topks 0.0 0.05 0.1 0.15 0.2 0.3 0.5 \
    --methods augmented_occlusion gate_mask ... timing_sample100_seg5_min10_max152
"""
import argparse, csv, os, sys
import numpy as np
import torch as th
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# eval_cpd_cpp.py 의 setup/anchor 로직 재사용 (동일 reference 보장)
from eval_cpd_cpp import (
    CFG, build_datamodule, build_cache, derive_anchors,
    load_anchor_idx, fills_for_ref,
)
from pytorch_lightning import seed_everything
from real.classifier import MimicClassifierNet


def get_labels(ev):
    """preprocess dict 에서 label 추출. 키가 다르면 여기서 잡아줌."""
    for key in ("y", "label", "labels", "target"):
        if key in ev:
            y = ev[key]
            break
    else:
        raise KeyError(f"label key 못 찾음. ev.keys()={list(ev.keys())} "
                       f"→ get_labels() 의 후보 키에 추가하세요.")
    y = th.as_tensor(y)
    if y.dim() == 2 and y.shape[1] > 1:   # one-hot 이면 argmax
        y = y.argmax(1)
    return y.long().view(-1)


@th.no_grad()
def predict_proba(classifier, x, mask, testbs, device):
    probs = []
    for s in range(0, x.shape[0], testbs):
        xb = x[s:s + testbs].to(device)
        mb = mask[s:s + testbs].to(device)
        out = classifier.predict(xb, mask=mb)   # net(x, mask=mask).softmax(-1)
        if out.dim() == 3:
            out = out[:, -1, :]
        probs.append(out.cpu())                 # 이미 softmax 됨 → 외부 softmax 제거
    return th.cat(probs, 0)


def importance_from_attr(attr, valid):
    """절댓값 기반 중요도. padding/missing cell 은 후보에서 제외(-1)."""
    imp = attr.abs()                       # imp >= 0
    imp = imp.masked_fill(~valid, -1.0)    # -1 이면 topk(largest) 에 절대 안 뽑힘
    return imp

def mask_topk(x, imp, fill, k):
    """importance 상위 k%(largest) cell 을 fill 로 치환. 예산은 전체 T*D 기준(CPD 와 동일)."""
    if k <= 0:
        return x
    N, T, D = x.shape
    n_mask = int(np.ceil(k * T * D))
    flat = imp.reshape(N, -1)
    idx = flat.topk(n_mask, dim=1, largest=True).indices
    m = th.zeros_like(flat, dtype=th.bool).scatter_(1, idx, True).reshape(N, T, D)
    fill_t = th.full_like(x, fill) if isinstance(fill, float) else fill
    return th.where(m, fill_t, x)


def auroc_score(probs, y, n_state):
    y_np, p_np = y.numpy(), probs.numpy()
    try:
        if n_state == 2:
            return float(roc_auc_score(y_np, p_np[:, 1]))
        return float(roc_auc_score(y_np, p_np, multi_class="ovr",
                                   average="macro", labels=list(range(n_state))))
    except ValueError:      # 어떤 fold 에서 class 하나만 있으면
        return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, choices=list(CFG))
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_type", default="state")
    p.add_argument("--eval_split", default="test", choices=["test", "val"])
    p.add_argument("--testbs", type=int, default=30)
    p.add_argument("--topks", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
    p.add_argument("--npy_dir", default="results_our")
    p.add_argument("--output_file", default="results_pna/eval_auroc/full.csv")
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
    p.add_argument("--anchor_chunk", type=int, default=512)
    p.add_argument("--anchor_idx_dir", default=None)
    p.add_argument("--verify_anchors", action="store_true")
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
    th.backends.cudnn.enabled = False        # CPD/생성과 동일 (수치 일치)

    dm = build_datamodule(args.data, args.fold, args.seed)
    tr = dm.preprocess(split="train")
    x_train = tr["x"].to(device)
    ev = dm.preprocess(split=args.eval_split)
    x_test = ev["x"].to(device)
    mask_test = ev["mask"].to(device)
    y_test = get_labels(ev)

    if args.eval_split == "val":
        idx_path = f"{args.npy_dir}/{args.data}_{args.model_type}_val_idx_{args.fold}_{args.seed}.npy"
        idx = th.from_numpy(np.load(idx_path)).long()
        x_test = x_test[idx.to(device)]
        mask_test = mask_test[idx.to(device)]
        y_test = y_test[idx]

    data_len, t_len, _ = x_test.shape
    timesteps = (th.linspace(0, 1, t_len, device=device)
                 .unsqueeze(0).repeat(data_len, 1))

    # top-k 범위 검증
    for k in args.topks:
        assert 0.0 <= k < 1.0, f"--topks 는 [0,1) 이어야 함: {k}"

    # valid cell mask (1=observed 가정) — abs importance 후보에서 padding 제외
    m = mask_test if mask_test.dim() == 3 else \
        mask_test.unsqueeze(-1).expand(-1, -1, x_test.shape[-1])
    valid = (m > 0).cpu()                       # [N,T,D] bool

    # multiclass fold 에 모든 class 존재 확인
    if n_state > 2:
        miss = set(range(n_state)) - set(y_test.tolist())
        if miss:
            print(f"[warn] {args.data} f{args.fold}: y 에 없는 class {sorted(miss)} "
                  f"→ macro AUROC 불안정")

    # --- anchor 준비 (CPD 와 동일: 로드 우선, verify, 없으면 재도출) ---
    anchors_dict, cache = {}, None
    for which in ("pna", "na"):
        if which not in args.mask_refs:
            continue
        loaded = load_anchor_idx(which, x_test, args)
        if loaded is not None and not args.verify_anchors:
            anchors_dict[which] = loaded; continue
        if cache is None:
            cache = build_cache(x_train, classifier, args)
        derived = derive_anchors(which, x_test, cache, classifier, args)
        if loaded is not None and args.verify_anchors:
            max_abs = (loaded - derived).abs().max().item()
            print(f"[verify:{which}] max|loaded-derived|={max_abs:.3e} "
                  f"({'OK' if max_abs == 0.0 else 'MISMATCH!!'})")
            assert max_abs == 0.0
            anchors_dict[which] = loaded
        else:
            anchors_dict[which] = derived

    ref_fills = {ref: fills_for_ref(ref, x_test, anchors_dict, args)
                 for ref in args.mask_refs}

    x_cpu = x_test.cpu()

    # --- baseline AUROC (원본, 마스킹 없음): fold 당 1회만 ---
    base_probs = predict_proba(classifier, x_cpu, mask_test, args.testbs, device)
    base_auroc = auroc_score(base_probs, y_test, n_state)
    print(f"[{args.data} f{args.fold}] baseline AUROC = {base_auroc:.4f}")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output_file)
    fp = open(args.output_file, "a", newline="")
    w = csv.writer(fp)
    if write_header:
        w.writerow(["data", "fold", "seed", "method", "mask_ref", "reduce",
                    "k", "auroc", "auroc_drop", "base_auroc"])

    for method in args.methods:
        f = (f"{args.npy_dir}/{args.data}_{args.model_type}_{method}"
             f"_result_{args.fold}_{args.seed}.npy")
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing attribution file: {f}")
        attr_np = np.load(f)
        if attr_np.shape[0] != x_cpu.shape[0]:
            raise ValueError(f"attr N={attr_np.shape[0]} != eval N={x_cpu.shape[0]} ({f})")
        if attr_np.shape != tuple(x_cpu.shape):
            raise ValueError(f"attr shape {attr_np.shape} != x {tuple(x_cpu.shape)} ({f})")
        if not np.isfinite(attr_np).all():
            n_bad = int((~np.isfinite(attr_np)).sum())
            raise ValueError(f"attr has {n_bad} NaN/Inf: {f}")

        imp = importance_from_attr(th.from_numpy(attr_np).float().cpu(), valid)  # abs + padding 제외

        for ref in args.mask_refs:
            fills, reduce_tag = ref_fills[ref]
            for k in args.topks:
                aurocs = []
                for b in fills:
                    fill = b if isinstance(b, float) else b.cpu()
                    x_masked = mask_topk(x_cpu, imp, fill, k)
                    probs = predict_proba(classifier, x_masked, mask_test, args.testbs, device)
                    aurocs.append(auroc_score(probs, y_test, n_state))
                auroc = float(np.nanmean(aurocs))
                drop = base_auroc - auroc
                w.writerow([args.data, args.fold, args.seed, method, ref,
                            reduce_tag, k, auroc, drop, base_auroc])
                print(f"[{args.data} f{args.fold}] {method:48s} "
                      f"mask={ref:7s} k={k:.2f} AUROC={auroc:.4f} drop={drop:+.4f}")
    fp.close()

if __name__ == "__main__":
    main()