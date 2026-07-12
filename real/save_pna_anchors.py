"""
이미 생성된 attribution 에 대해, main_td.py 와 '완전히 동일한 결정적' 경로로
pool + anchor_idx 를 저장한다 (attribution 재생성 불필요).

- pool: seed 만 의존 (독립 generator). lam 무관.
- anchor_idx: lam 의존. select_pna_indices 로 test_loader 순서와 동일하게 산출.

저장물:
  results_pna/{data}_{mt}_pool_{fold}_{seed}.npy
  results_pna/{data}_{mt}_anchoridx_lam{l0}x{lf}_{fold}_{seed}.npy

Usage:
  CUDA_VISIBLE_DEVICES=0 python real/save_pna_anchors.py \
      --data wafer --fold 0 --device cuda:0 \
      --pna_lam0 10.0 --pna_lamf 10.0 --pna_ka 5 --eval_split test
"""
import argparse, os, sys
import numpy as np
import torch as th
from pytorch_lightning import seed_everything

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from real.classifier import MimicClassifierNet
from attribution.pna import build_pna_cache, select_pna_indices
from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer
from datasets.freezer import Freezer

CFG = {"PAM":(PAM,17,8,600,False),"boiler":(Boiler,20,2,36,False),
       "epilepsy":(Epilepsy,1,2,178,False),"wafer":(Wafer,1,2,152,True),
       "freezer":(Freezer,1,2,301,True)}

def dm_of(data, fold, seed):
    DM,_,_,_,nf = CFG[data]
    return DM(n_folds=5, fold=fold, seed=seed) if nf else DM(fold=fold, seed=seed)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, choices=list(CFG))
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_type", default="state")
    p.add_argument("--eval_split", default="test", choices=["test","val"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pna_feature", default="hidden", choices=["hidden","logits"])
    p.add_argument("--pna_ka", type=int, default=5)
    p.add_argument("--pna_lam0", type=float, default=10.0)
    p.add_argument("--pna_lamf", type=float, default=10.0)
    p.add_argument("--pool_size", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--out_dir", default="results_pna")
    p.add_argument("--testbs", type=int, default=30)
    args = p.parse_args()

    seed_everything(args.seed, workers=True)
    device = th.device(args.device)
    _, feat, n_state, n_t, _ = CFG[args.data]

    clf = MimicClassifierNet(feature_size=feat, n_state=n_state, n_timesteps=n_t,
                             hidden_size=200, regres=True, loss="cross_entropy",
                             lr=1e-4, l2=1e-3, model_type=args.model_type)
    ckpt = f"./model/{args.data}/{args.model_type}_classifier_{args.fold}_{args.seed}_no_imputation"
    clf.load_state_dict(th.load(ckpt, map_location=device))
    clf.eval().to(device)
    th.backends.cudnn.enabled = False   # main_td 와 동일 (forward 수치 일치)

    dm = dm_of(args.data, args.fold, args.seed)
    x_train = dm.preprocess(split="train")["x"].to(device)
    ev = dm.preprocess(split=args.eval_split)
    x_test = ev["x"].to(device)

    if args.eval_split == "val":
        idx_path = f"{args.out_dir}/{args.data}_{args.model_type}_val_idx_{args.fold}_{args.seed}.npy"
        vidx = th.from_numpy(np.load(idx_path)).long().to(device)
        x_test = x_test[vidx]

    # main_td 와 동일: 독립 generator pool
    g = th.Generator().manual_seed(args.seed)
    idx = th.randperm(x_train.shape[0], generator=g)[:min(args.pool_size, x_train.shape[0])]
    pool = x_train[idx]
    cache = build_pna_cache(pool, clf, feature=args.pna_feature,
                            lam0=args.pna_lam0, lamf=args.pna_lamf)

    idx_all = []
    for s in range(0, x_test.shape[0], args.chunk):
        xb = x_test[s:s+args.chunk]
        idx_all.append(select_pna_indices(xb, cache, clf, Ka=args.pna_ka).cpu())
    anchor_idx = th.cat(idx_all, 0)   # [N,Ka]

    os.makedirs(args.out_dir, exist_ok=True)
    pool_p = f"{args.out_dir}/{args.data}_{args.model_type}_pool_{args.fold}_{args.seed}.npy"
    idx_p  = (f"{args.out_dir}/{args.data}_{args.model_type}_anchoridx_"
              f"lam{args.pna_lam0}x{args.pna_lamf}_{args.fold}_{args.seed}.npy")
    np.save(pool_p, pool.detach().cpu().numpy())
    np.save(idx_p, anchor_idx.numpy())
    print(f"saved:\n  {pool_p}  pool{tuple(pool.shape)}\n  {idx_p}  idx{tuple(anchor_idx.shape)}")

if __name__ == "__main__":
    main()
