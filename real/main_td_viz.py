import sys
from os import path
from pathlib import Path
print(path.dirname( path.dirname( path.abspath(__file__) ) ))
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))


import multiprocessing as mp
import numpy as np
import random
import torch as th
import torch.nn as nn
import os
import inspect
from utils.tools import print_results

from attribution.gate_mask import GateMask
from attribution.gatemasknn import *
from argparse import ArgumentParser
from tqdm import tqdm
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Lime, KernelShap, DeepLiftShap
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

from tint.attr import (
    DynaMask,
    ExtremalMask,
    Fit,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalOcclusion,
    Occlusion,
    FeatureAblation,
    TimeForwardTunnel,
)
from tint.attr.models import (
    ExtremalMaskNet,
    JointFeatureGeneratorNet,
    MaskNet,
    RetainNet,
)
from datasets.mimic3 import Mimic3
from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer
from datasets.freezer import Freezer
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    log_odds,
    sufficiency,
)

from real.cumulative_difference import cumulative_difference
from tint.models import MLP, RNN

from real.classifier import MimicClassifierNet


def main(
    explainers: List[str],
    data: str,
    areas: list,
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    is_train: bool = True,
    deterministic: bool = False,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    lambda_3: float = 1.0,
    num_segments: int = 50,
    min_seg_len: int = 1,
    max_seg_len: int = 48,
    mask_lr: float = 0.1,
    output_file: str = "results.csv",
    model_type: str = "state",
    testbs: int = 0,
    top: int = 50,
    skip_train_timex: bool = True,
    prob: float = 0.1,
    ### viz용 추가!!!
    viz_channels: list = None,
    viz: bool = False,
    viz_dir: str = "./viz_td",
    viz_n_samples: int = 5,
    viz_n_channels: int = 3,
):
    accelerator = device.split(":")[0]
    device_id = 1
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]

    lock = mp.Lock()

    if data == "mimic3":
        datamodule = Mimic3(n_folds=5, fold=fold, seed=seed)
        classifier = MimicClassifierNet(feature_size=32, n_state=2, n_timesteps=48, hidden_size=200, regres=True, loss="cross_entropy", lr=0.0001, l2=1e-3, model_type=model_type)
        num_features = 32; num_classes = 2; max_len = 48
    elif data == "PAM":
        datamodule = PAM(fold=fold, seed=seed)
        classifier = MimicClassifierNet(feature_size=17, n_state=8, n_timesteps=600, hidden_size=200, regres=True, loss="cross_entropy", lr=0.0001, l2=1e-3, model_type=model_type)
        num_features = 17; num_classes = 8; max_len = 600
    elif data == "boiler":
        datamodule = Boiler(fold=fold, seed=seed)
        classifier = MimicClassifierNet(feature_size=20, n_state=2, n_timesteps=36, hidden_size=200, regres=True, loss="cross_entropy", lr=0.0001, l2=1e-3, model_type=model_type)
        num_features = 20; num_classes = 2; max_len = 36
    elif data == "epilepsy":
        datamodule = Epilepsy(fold=fold, seed=seed)
        classifier = MimicClassifierNet(feature_size=1, n_state=2, n_timesteps=178, hidden_size=200, regres=True, loss="cross_entropy", lr=0.0001, l2=1e-3, model_type=model_type)
        num_features = 1; num_classes = 2; max_len = 178
    elif data == "freezer":
        datamodule = Freezer(n_folds=5, fold=fold, seed=seed)
        classifier = MimicClassifierNet(feature_size=1, n_state=2, n_timesteps=301, hidden_size=200, regres=True, loss="cross_entropy", lr=0.0001, l2=1e-3, model_type=model_type)
        num_features = 1; num_classes = 2; max_len = 301
    elif data == "wafer":
        datamodule = Wafer(n_folds=5, fold=fold, seed=seed)
        classifier = MimicClassifierNet(feature_size=1, n_state=2, n_timesteps=152, hidden_size=200, regres=True, loss="cross_entropy", lr=0.0001, l2=1e-3, model_type=model_type)
        num_features = 1; num_classes = 2; max_len = 152

    trainer = Trainer(
        max_epochs=100, accelerator=accelerator, devices=device_id,
        deterministic=deterministic,
        logger=TensorBoardLogger(save_dir=".", version=random.getrandbits(128)),
    )
    if is_train:
        trainer.fit(classifier, datamodule=datamodule)
        if not os.path.exists("./model/{}/".format(data)):
            os.makedirs("./model/{}/".format(data))
        th.save(classifier.state_dict(), "./model/{}/{}_classifier_{}_{}_no_imputation".format(data, model_type, fold, seed))
    else:
        classifier.load_state_dict(th.load("./model/{}/{}_classifier_{}_{}_no_imputation".format(data, model_type, fold, seed)))

    with lock:
        x_train = datamodule.preprocess(split="train")["x"].to(device)
        x_test  = datamodule.preprocess(split="test")["x"].to(device)
        y_train = datamodule.preprocess(split="train")["y"].to(device)
        y_test  = datamodule.preprocess(split="test")["y"].to(device)
        mask_train = datamodule.preprocess(split="train")["mask"].to(device)
        mask_test  = datamodule.preprocess(split="test")["mask"].to(device)

    classifier.eval()
    classifier.to(device)

    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    attr = dict()

    from torch.utils.data import DataLoader, TensorDataset

    test_indices = th.arange(x_test.shape[0], device=x_test.device)
    test_dataset = TensorDataset(x_test, mask_test, test_indices)
    test_loader  = DataLoader(test_dataset, batch_size=testbs, shuffle=False)

    if model_type == "state":
        temporal_additional_forward_args = (False, False, False)
    else:
        temporal_additional_forward_args = (False, False, False)

    data_mask = mask_test
    data_len, t_len, _ = x_test.shape
    timesteps = (
        th.linspace(0, 1, t_len, device=x_test.device)
        .unsqueeze(0).repeat(data_len, 1)
    )

    ###
    if "our_td" in explainers:
        from attribution.explainers_td_viz import OUR_TD
        #from attribution.explainers_td_rsd_viz import OUR_TD_VIZ as OUR_TD
        print("[DEBUG] OUR_TD file =", inspect.getfile(OUR_TD))
        print("[DEBUG] signature =", inspect.signature(OUR_TD.attribute_trend_residual_segments))
        explainer = OUR_TD(classifier.predict)

        trend_results, resid_results, fxc_results = [], [], []
        for batch in tqdm(test_loader):
            x_batch    = batch[0].to(device)
            data_mask  = batch[1].to(device)
            batch_ids  = batch[2].detach().cpu()
            batch_size = x_batch.shape[0]
            timesteps_b = timesteps[:batch_size, :]

            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier, x_batch,
                    additional_forward_args=(data_mask, timesteps_b, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            trend_attr, resid_attr, fxc = explainer.attribute_trend_residual_segments(
                x_batch,
                baselines=x_batch * 0,
                targets=partial_targets,
                additional_forward_args=(data_mask, timesteps_b, False),
                n_samples=50,
                num_segments=num_segments,
                min_seg_len=min_seg_len,
                max_seg_len=max_seg_len,
                kalman_obs_cov=1.0,
                kalman_trans_cov=0.01,
                n_alphas=50,
                alpha_chunk=10,
                ### viz용 추가!!!
                viz_dir=viz_dir if viz else None,
                viz_n_samples=viz_n_samples,
                viz_n_channels=viz_n_channels,
                viz_channels=viz_channels,
                tag=f"{data}_fold{fold}",
                sample_ids=batch_ids,
            )

            trend_results.append(trend_attr.detach().cpu())
            resid_results.append(resid_attr.detach().cpu())
            fxc_results.append(fxc.detach().cpu())

        SEG = f"kalman_seg{num_segments}_min{min_seg_len}_max{max_seg_len}"
        trend_signed = th.cat(trend_results, dim=0)
        resid_signed = th.cat(resid_results, dim=0)

        attr[f"timing_td_trend_{SEG}"]           = trend_signed.abs()
        attr[f"timing_td_residual_{SEG}"]        = resid_signed.abs()
        attr[f"timing_td_trend_signed_{SEG}"]    = trend_signed
        attr[f"timing_td_residual_signed_{SEG}"] = resid_signed
        attr[f"timing_td_fxc_{SEG}"]             = th.cat(fxc_results, dim=0)
        attr[f"timing_td_combined_{SEG}"]        = (trend_signed + resid_signed).abs()
        attr[f"timing_td_T_plus_R_{SEG}"]        = trend_signed.abs() + resid_signed.abs()
    ###

    if viz and "our_td" in explainers:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        def _viz_heatmap(x, attr_t, attr_r, save_dir, n_samp, channels, tag, signed):
            os.makedirs(save_dir, exist_ok=True)
            if hasattr(x,      "cpu"): x      = x.cpu().numpy()
            if hasattr(attr_t, "cpu"): attr_t = attr_t.cpu().numpy()
            if hasattr(attr_r, "cpu"): attr_r = attr_r.cpu().numpy()
            N, T, C  = x.shape
            n_samp   = min(n_samp, N)
            ch_list  = channels if isinstance(channels, list) else list(range(min(channels, C)))
            n_ch     = len(ch_list)
            cmap     = "RdBu_r" if signed else "Reds"
            fig = plt.figure(figsize=(max(14, T//20), n_samp * n_ch * 1.5))
            fig.suptitle(f"{tag} — Trend vs Residual ({'signed' if signed else 'abs'})", fontsize=13, y=1.01)
            outer = gridspec.GridSpec(n_samp, 1, figure=fig, hspace=0.6)
            for s in range(n_samp):
                inner = gridspec.GridSpecFromSubplotSpec(
                    n_ch, 3, subplot_spec=outer[s],
                    wspace=0.05, hspace=0.15, width_ratios=[6, 6, 0.3])
                for idx, c in enumerate(ch_list):
                    ax_t  = fig.add_subplot(inner[idx, 0])
                    ax_r  = fig.add_subplot(inner[idx, 1])
                    ax_cb = fig.add_subplot(inner[idx, 2])
                    t_row, r_row = attr_t[s,:,c], attr_r[s,:,c]
                    if signed:
                        vmax = max(np.abs(t_row).max(), np.abs(r_row).max(), 1e-9); vmin = -vmax
                    else:
                        vmin = 0.0; vmax = max(t_row.max(), r_row.max(), 1e-9)
                    im = ax_t.imshow(t_row[np.newaxis,:], aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
                    ax_r.imshow(r_row[np.newaxis,:], aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
                    plt.colorbar(im, cax=ax_cb)
                    for ax in (ax_t, ax_r): ax.set_yticks([]); ax.set_xticks([])
                    ax_t.set_ylabel(f"ch{c}", fontsize=7, rotation=0, labelpad=18)
                    ax_r.set_ylabel(f"ch{c}", fontsize=7, rotation=0, labelpad=18)
                    if idx == 0:
                        ax_t.set_title(f"Sample {s} — Trend",    fontsize=8)
                        ax_r.set_title(f"Sample {s} — Residual", fontsize=8)
            plt.tight_layout()
            suffix = "signed" if signed else "abs"
            plt.savefig(os.path.join(save_dir, f"heatmap_{suffix}.png"), dpi=120, bbox_inches="tight")
            plt.close()
            print(f"[viz] heatmap ({suffix}) → {save_dir}")

        SEG    = f"kalman_seg{num_segments}_min{min_seg_len}_max{max_seg_len}"
        t_key  = f"timing_td_trend_{SEG}"
        r_key  = f"timing_td_residual_{SEG}"
        ts_key = f"timing_td_trend_signed_{SEG}"
        rs_key = f"timing_td_residual_signed_{SEG}"
        viz_out = os.path.join(viz_dir, f"{data}_{model_type}_fold{fold}")
        ch_arg  = viz_channels if viz_channels is not None else viz_n_channels

        if t_key in attr and r_key in attr:
            _viz_heatmap(x_test[:viz_n_samples], attr[t_key][:viz_n_samples],
                         attr[r_key][:viz_n_samples], viz_out,
                         viz_n_samples, ch_arg, f"{data} fold{fold}", signed=False)

        if ts_key in attr and rs_key in attr:
            _viz_heatmap(x_test[:viz_n_samples], attr[ts_key][:viz_n_samples],
                         attr[rs_key][:viz_n_samples], viz_out,
                         viz_n_samples, ch_arg, f"{data} fold{fold}", signed=True)

        print(f"[viz] all figures saved under {viz_out}")

    print(f"{explainers} done")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--explainers", type=str, default=["gate_mask"], nargs="+", metavar="N")
    parser.add_argument("--data",       type=str, default="mimic3")
    parser.add_argument("--areas",      type=float, default=[0.1,0.2,0.3,0.4,0.5,0.6], nargs="+", metavar="N")
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--fold",       type=int, default=1)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--train",      type=bool, default=False)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--lambda-1",   type=float, default=0.001)
    parser.add_argument("--lambda-2",   type=float, default=0.01)
    parser.add_argument("--lambda-3",   type=float, default=0.01)
    parser.add_argument("--mask_lr",    type=float, default=0.01)
    parser.add_argument("--prob",       type=float, default=0.1)
    parser.add_argument("--output-file",type=str, default="results_gate.csv")
    parser.add_argument("--model_type", type=str, default="state", choices=["state","mtand","seft","transformer","cnn"])
    parser.add_argument("--testbs",     type=int, default=200)
    parser.add_argument("--top",        type=int, default=50)
    parser.add_argument("--num_segments",  type=int, default=50)
    parser.add_argument("--min_seg_len",   type=int, default=1)
    parser.add_argument("--max_seg_len",   type=int, default=48)
    parser.add_argument("--skip_train_timex", action="store_true")
    ### viz용 추가!!!
    parser.add_argument("--viz_channels", type=int, nargs="+", default=None,
                        help="시각화할 채널 인덱스 지정. 없으면 0~n_channels")
    parser.add_argument("--viz",            action="store_true")
    parser.add_argument("--viz_dir",        type=str, default="./viz_td")
    parser.add_argument("--viz_n_samples",  type=int, default=5)
    parser.add_argument("--viz_n_channels", type=int, default=3)
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    print(f"set seed as {seed}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(
        explainers=args.explainers,
        data=args.data,
        areas=args.areas,
        device=args.device,
        fold=args.fold,
        seed=args.seed,
        is_train=args.train,
        deterministic=args.deterministic,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        num_segments=args.num_segments,
        min_seg_len=args.min_seg_len,
        max_seg_len=args.max_seg_len,
        mask_lr=args.mask_lr,
        output_file=args.output_file,
        model_type=args.model_type,
        testbs=args.testbs,
        top=args.top,
        skip_train_timex=args.skip_train_timex,
        prob=args.prob,
        ### viz용 추가!!!
        viz_channels=args.viz_channels,
        viz=args.viz,
        viz_dir=args.viz_dir,
        viz_n_samples=args.viz_n_samples,
        viz_n_channels=args.viz_n_channels,
    )
