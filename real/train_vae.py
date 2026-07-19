import sys, os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import argparse
import torch as th
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.vae import TimeSeriesVAE, vae_loss
from datasets.mimic3 import Mimic3
from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer
from datasets.freezer import Freezer


DATASETS = {
    "PAM":      (PAM,      17, 600),
    "boiler":   (Boiler,   20, 36),
    "epilepsy": (Epilepsy,  1, 178),
    "freezer":  (Freezer,   1, 301),
    "wafer":    (Wafer,     1, 152),
}


def get_datamodule(data, fold, seed):
    DataCls, n_feat, n_t = DATASETS[data]
    try:
        dm = DataCls(n_folds=5, fold=fold, seed=seed)
    except TypeError:
        dm = DataCls(fold=fold, seed=seed)
    return dm, n_feat, n_t


def build_loaders(dm, batch_size):
    pre_train = dm.preprocess(split="train")
    pre_test  = dm.preprocess(split="test")

    x_tr, mask_tr = pre_train["x"], pre_train["mask"]
    x_te, mask_te = pre_test["x"],  pre_test["mask"]

    # 일부 데이터셋은 features가 [N, T] (univariate) 로 나올 수도 있으니 안전 처리
    if x_tr.dim() == 2:
        x_tr = x_tr.unsqueeze(-1)
        x_te = x_te.unsqueeze(-1)
        mask_tr = mask_tr.unsqueeze(-1) if mask_tr.dim() == 2 else mask_tr
        mask_te = mask_te.unsqueeze(-1) if mask_te.dim() == 2 else mask_te

    # mask shape을 x와 맞춰줌 ([N, T] -> [N, T, D])
    if mask_tr.shape != x_tr.shape:
        mask_tr = mask_tr.unsqueeze(-1).expand_as(x_tr).contiguous()
        mask_te = mask_te.unsqueeze(-1).expand_as(x_te).contiguous()

    train_ds = TensorDataset(x_tr, mask_tr)
    test_ds  = TensorDataset(x_te, mask_te)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="mimic3")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.1)
    args = p.parse_args()

    th.manual_seed(args.seed)
    device = args.device

    # ---- 같은 datamodule 인스턴스로 train/test 둘 다 만들기 ----
    dm, n_feat, n_t = get_datamodule(args.data, args.fold, args.seed)
    train_loader, val_loader = build_loaders(dm, args.batch_size)

    # 실제 데이터의 feature dim을 확인해서 매핑이 맞는지 sanity check
    sample_x, _ = next(iter(train_loader))
    print(f"[INFO] x shape from loader: {tuple(sample_x.shape)}, "
          f"expected feature_dim={n_feat}")
    actual_feat = sample_x.shape[-1]
    if actual_feat != n_feat:
        print(f"[WARN] feature_dim mismatch. Using actual: {actual_feat}")
        n_feat = actual_feat

    model = TimeSeriesVAE(
        feature_dim=n_feat,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    opt = th.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = f"./model/{args.data}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/vae_{args.fold}_{args.seed}.pt"

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for x, mask in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x = x.to(device).float()
            mask = mask.to(device).float()
            x_hat, mu, logvar, _ = model(x)
            loss, recon, kld = vae_loss(x, x_hat, mu, logvar,
                                        beta=args.beta, mask=mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
            n_tr += x.size(0)
        tr_loss /= max(n_tr, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with th.no_grad():
            for x, mask in val_loader:
                x = x.to(device).float()
                mask = mask.to(device).float()
                x_hat, mu, logvar, _ = model(x)
                loss, _, _ = vae_loss(x, x_hat, mu, logvar,
                                      beta=args.beta, mask=mask)
                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)
        val_loss /= max(n_val, 1)

        print(f"[Epoch {epoch}] train={tr_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            th.save({
                "model_state": model.state_dict(),
                "config": {
                    "feature_dim": n_feat,
                    "hidden_dim": args.hidden_dim,
                    "latent_dim": args.latent_dim,
                },
            }, save_path)
            print(f"  ↳ saved to {save_path}")


if __name__ == "__main__":
    main()
