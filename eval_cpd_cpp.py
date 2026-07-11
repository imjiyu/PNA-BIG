"""
Re-evaluate CPD, CPP, accuracy, comprehensiveness, cross-entropy,
log-odds, sufficiency from saved attribution .npy files.

Table 3 default: --topk 0.1 --top 0 (10% feature masking), baseline=0.0.

Usage example (|T| and |R| for PAM):
    python real/eval_from_npy.py --data PAM --fold 0 --device cuda:0 \
        --methods timing_td_trend_seg10_min10_max600 \
                  timing_td_residual_seg10_min10_max600
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch as th
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root on path when invoked as `python real/eval_from_npy.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real.classifier import MimicClassifierNet
from real.cumulative_difference import cumulative_difference
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    log_odds,
    sufficiency,
)
from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer
from datasets.freezer import Freezer


# (datamodule, feature_size, n_state, n_timesteps, needs_n_folds_arg)
CFG = {
    "PAM":      (PAM,      17, 8, 600, False),
    "boiler":   (Boiler,   20, 2, 36,  False),
    "epilepsy": (Epilepsy, 1,  2, 178, False),
    "wafer":    (Wafer,    1,  2, 152, True),
    "freezer":  (Freezer,  1,  2, 301, True),
}


def build_datamodule(data, fold, seed):
    DM, _, _, _, needs_folds = CFG[data]
    if needs_folds:
        return DM(n_folds=5, fold=fold, seed=seed)
    return DM(fold=fold, seed=seed)


def compute_extra_metrics(classifier, x_test, mask_test, timesteps, attr,
                          baselines, topk, testbs, device):
    """Replicate main.py's per-batch metric loop for acc/comp/ce/lodds/suff."""
    test_loader = DataLoader(
        TensorDataset(x_test, mask_test), batch_size=testbs, shuffle=False
    )
    tot = {"acc": 0.0, "comp": 0.0, "ce": 0.0, "lodds": 0.0, "suff": 0.0}
    n_total = 0

    for batch_idx, (x_batch, m_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        m_batch = m_batch.to(device)
        bs = x_batch.shape[0]
        start = batch_idx * testbs
        end = start + bs
        ts_batch = timesteps[start:end]

        if isinstance(baselines, th.Tensor):
            b_batch = baselines[start:end].to(device)
        else:
            b_batch = baselines

        v_batch = attr[start:end].to(device)
        fwd_args = (m_batch, ts_batch, False)

        kw = dict(
            attributions=v_batch, baselines=b_batch,
            topk=topk, additional_forward_args=fwd_args,
        )
        tot["acc"]   += float(accuracy(classifier, x_batch, **kw))         * bs
        tot["comp"]  += float(comprehensiveness(classifier, x_batch, **kw)) * bs
        tot["ce"]    += float(cross_entropy(classifier, x_batch, **kw))    * bs
        tot["lodds"] += float(log_odds(classifier, x_batch, **kw))         * bs
        tot["suff"]  += float(sufficiency(classifier, x_batch, **kw))      * bs
        n_total += bs

    return {k: v / n_total for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, choices=list(CFG))
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_type", default="state")
    p.add_argument("--eval_split", default="test", choices=["test", "val"]) ### 추가
    p.add_argument("--testbs", type=int, default=30)
    p.add_argument("--topk", type=float, default=0.1, help="--areas in main.py")
    p.add_argument("--top", type=int, default=0)
    p.add_argument("--npy_dir", default="results_our")
    p.add_argument("--output_file", default="results_our/full_eval.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help=(
            "Method keys (the {key} part of "
            "{data}_{model_type}_{key}_result_{fold}_{seed}.npy). "
            "Example: timing_td_trend_seg10_min10_max600 "
            "timing_td_residual_seg10_min10_max600"
        ),
    )
    args = p.parse_args()

    seed_everything(args.seed, workers=True)
    device = th.device(args.device)
    _, feat, n_state, n_t, _ = CFG[args.data]

    classifier = MimicClassifierNet(
        feature_size=feat, n_state=n_state, n_timesteps=n_t,
        hidden_size=200, regres=True, loss="cross_entropy",
        lr=1e-4, l2=1e-3, model_type=args.model_type,
    )
    ckpt = (
        f"./model/{args.data}/{args.model_type}_classifier_"
        f"{args.fold}_{args.seed}_no_imputation"
    )
    classifier.load_state_dict(th.load(ckpt, map_location=device))
    classifier.eval().to(device)
    th.backends.cudnn.enabled = False  # required for RNN backward in captum

    dm = build_datamodule(args.data, args.fold, args.seed)
    _ = dm.preprocess(split="train")   # populate self._mean / self._std : mean, std 초기화용
    ###x_test = dm.preprocess(split="test")["x"].to(device)
    ###mask_test = dm.preprocess(split="test")["mask"].to(device)
    split = args.eval_split
    x_test = dm.preprocess(split=split)["x"].to(device)
    mask_test = dm.preprocess(split=split)["mask"].to(device)

    if args.eval_split == "val":
        idx_path = f"{args.npy_dir}/{args.data}_{args.model_type}_val_idx_{args.fold}_{args.seed}.npy"

        if not os.path.exists(idx_path):
            raise FileNotFoundError(
                f"Missing val index file: {idx_path}\n"
                f"Do not create a new random val index here. "
                f"Re-run attribution after saving val_idx in main_td.py."
            )

        idx = th.from_numpy(np.load(idx_path)).long().to(device)
        x_test = x_test[idx]
        mask_test = mask_test[idx]

    # timesteps used as additional_forward_args by metrics (per-batch sliced)
    data_len, t_len, _ = x_test.shape
    timesteps = (
        th.linspace(0, 1, t_len, device=device).unsqueeze(0).repeat(data_len, 1)
    )

    # Build exact list of .npy files from --methods (no glob)
    files = []
    for m in args.methods:
        f = (
            f"{args.npy_dir}/{args.data}_{args.model_type}_{m}"
            f"_result_{args.fold}_{args.seed}.npy"
        )
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing attribution file: {f}")
        files.append((m, f))

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output_file)
    fp = open(args.output_file, "a", newline="")
    w = csv.writer(fp)
    if write_header:
        w.writerow([
            "data", "fold", "seed", "method", "metric",
            "topk", "top",
            "cum_diff", "AUCC", "cum_50_diff",
            "accuracy", "comprehensiveness",
            "cross_entropy", "log_odds", "sufficiency",
        ])

    baselines = 0.0  # Table 3 setting
    for method, f in files:
        ###attr = th.from_numpy(np.load(f)).cpu()
        attr_np = np.load(f)

        if attr_np.shape[0] != x_test.shape[0]:
            raise ValueError(
                f"attr N={attr_np.shape[0]} but eval data N={x_test.shape[0]} "
                f"for {f}"
            )

        attr = th.from_numpy(attr_np).float().cpu()

        # CPD (largest=True) and CPP (largest=False)
        ### for metric_name, largest in [("CPD", True), ("CPP", False)]:
        for metric_name, largest in [("CPD", True)]:
            cum_diff, AUCC, cum_50_diff, _ = cumulative_difference(
                classifier, x_test,
                attributions=attr,
                baselines=baselines,
                topk=args.topk,
                top=args.top,
                testbs=args.testbs,
                largest=largest,
                ###additional_forward_args=(mask_test, None, False),
                additional_forward_args=(mask_test, timesteps, False),
            )

            # Extra metrics (replicate main.py's per-batch loop)
            extras = compute_extra_metrics(
                classifier, x_test, mask_test, timesteps,
                attr, baselines, args.topk, args.testbs, device,
            )

            w.writerow([
                args.data, args.fold, args.seed, method, metric_name,
                args.topk, args.top,
                float(cum_diff), float(AUCC), float(cum_50_diff),
                extras["acc"], extras["comp"],
                extras["ce"], extras["lodds"], extras["suff"],
            ])
            print(
                f"[{args.data} fold={args.fold}] {method:50s} {metric_name}: "
                f"cum_diff={float(cum_diff):.4f} AUCC={float(AUCC):.4f} "
                f"acc={extras['acc']:.4f} comp={extras['comp']:.4f} "
                f"ce={extras['ce']:.4f} lodds={extras['lodds']:.4f} "
                f"suff={extras['suff']:.4f}"
            )
    fp.close()


if __name__ == "__main__":
    main()
