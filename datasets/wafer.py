import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import torch as th
import warnings

from datetime import timedelta
from getpass import getpass
from sklearn.impute import SimpleImputer
from tslearn.utils import _load_txt_uea

from datasets.dataset import DataModule, Dataset

import sys, os

file_dir = os.path.dirname(__file__)

class Wafer(DataModule):
    def __init__(
        self,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "Wafer",
        ),
        batch_size: int = 32,
        prop_val: float = 0.2,
        n_folds: int = None,
        fold: int = None,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            prop_val=prop_val,
            n_folds=n_folds,
            fold=fold,
            num_workers=num_workers,
            seed=seed,
        )

        self._mean = None
        self._std = None

    def prepare_data(self):
        if not os.path.exists(
            os.path.join(self.data_dir, "Wafer_TRAIN.txt")
        ):
            raise RuntimeError("No data exists or wrong path")

    def preprocess(self, split: str = "train") -> dict:
        # ---------------------------------------------------------------
        #  Wafer 는 원래 TRAIN / TEST 파일만 존재 -> "val" split 이 없다!
        #  하이퍼파라미터 튜닝 때만 (환경변수 PNA_TUNE_VAL=1) TRAIN 을
        #  fold-aware stratified 로 (train_tune / val_tune) 로 나눠서
        #  val 을 만든다.  그 외 모든 실행에서는 원본 동작 그대로.
        #
        #  * PNA_TUNE_VAL 미설정(=일반/최종 test 실행): 100% TRAIN + 원본 test.
        #    -> 기존 attribution 생성/최종 평가에 아무 영향 없음.
        #  * PNA_TUNE_VAL=1 (sweep 스크립트에서만 export):
        #    -> train = fold 의 train_tune(≈80%), val = 그 fold 의 val_tune(≈20%).
        #       train / val 이 disjoint 이고 anchor pool 이 train_tune 에서만
        #       뽑히므로 "val 샘플이 자기 자신의 anchor 가 되는" leakage 없음.
        #       mean/std 도 train_tune 으로만 fit -> normalization leakage 없음.
        # ---------------------------------------------------------------
        dataset_name = "Wafer"
        X_train, y_train = _load_txt_uea(
            os.path.join(self.data_dir, dataset_name + "_TRAIN.txt")
        )
        X_test, y_test = _load_txt_uea(
            os.path.join(self.data_dir, dataset_name + "_TEST.txt")
        )

        tune_mode = os.environ.get("PNA_TUNE_VAL", "0") == "1"

        if tune_mode:
            from sklearn.model_selection import StratifiedKFold

            n_folds = self.n_folds if self.n_folds is not None else 5
            fold = self.fold if self.fold is not None else 0
            skf = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=self.seed
            )
            tr_idx, va_idx = list(skf.split(X_train, y_train))[fold]

            # mean/std 는 train_tune 으로만 fit (val leakage 방지)
            X_tr = th.Tensor(X_train)[tr_idx]
            self._mean = X_tr.mean(dim=(0, 1), keepdim=True)
            self._std = X_tr.std(dim=(0, 1), keepdim=True)

            if split == "train":
                features = X_tr
                y = th.Tensor(y_train)[tr_idx]
            elif split == "val":
                features = th.Tensor(X_train)[va_idx]
                y = th.Tensor(y_train)[va_idx]
            elif split == "test":
                features = th.Tensor(X_test)
                y = th.Tensor(y_test)
            else:
                raise RuntimeError(split)

        else:
            # ---- 원본 동작 (튜닝 아닐 때) ----
            if split == "train":
                features = th.Tensor(X_train)
                y = th.Tensor(y_train)
                self._mean = features.mean(dim=(0, 1), keepdim=True)
                self._std = features.std(dim=(0, 1), keepdim=True)
            elif split == "test":
                features = th.Tensor(X_test)
                y = th.Tensor(y_test)
            else:
                raise RuntimeError

        EPS = 1e-5
        features = (features - self._mean) / (self._std + EPS)

        y = y / 2 + 0.5
        return {
            "x": features.float(),
            "y": y.long(),
            "mask": th.ones_like(features),
        }
