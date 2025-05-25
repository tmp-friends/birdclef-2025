import logging
import os
from pathlib import Path
import glob
import gc
from tqdm import tqdm

import hydra

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import librosa
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from schedulefree import RAdamScheduleFree

from utils.utils import set_seed
from utils.pseudo_label import make_pl_windows
from conf.type import TrainConfig
from modules.birdclef_dataset import BirdCLEFDatasetFromNPY
from modules.birdclef_model import BirdCLEFModel
from modules.loss import FocalLossBCE


def _split_train_valid(cfg, df):
    """
    Hold-out / K-fold を自動で切替える split ヘルパ
    True → 5 % hold-out, False → K-fold list を返す
    """
    if cfg.use_holdout:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=cfg.valid_ratio, random_state=cfg.seed
        )
        train_idx, valid_idx = next(sss.split(df, df["primary_label"]))

        return [(train_idx, valid_idx)]  # 1 split のみ
    else:
        skf = StratifiedKFold(
            n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed
        )

        return list(skf.split(df, df["primary_label"]))


def collate_fn(batch):
    """Custom collate function to handle different sized spectrograms"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}

    result = {k: [] for k in batch[0].keys()}

    for item in batch:
        for k, v in item.items():
            result[k].append(v)

    for k in result:
        if k == "target" and isinstance(result[k][0], torch.Tensor):
            result[k] = torch.stack(result[k])
        elif k == "melspec" and isinstance(result[k][0], torch.Tensor):
            shapes = [v.shape for v in result[k]]
            if len(set(str(s) for s in shapes)) == 1:
                result[k] = torch.stack(result[k])

    return result


def get_optimizer(cfg: TrainConfig, model):
    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "RAdamScheduleFree":
        optimizer = RAdamScheduleFree(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")

    return optimizer


def get_scheduler(cfg: TrainConfig, optimizer):
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr
        )
    elif cfg.scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True,
        )
    elif cfg.scheduler == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=cfg.num_epochs // 3, gamma=0.5
        )
    elif cfg.scheduler == "OneCycleLR":
        scheduler = None
    else:
        scheduler = None

    return scheduler


def get_criterion(cfg: TrainConfig):
    if cfg.criterion == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.criterion == "FocalLossBCE":
        criterion = FocalLossBCE()
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")

    return criterion


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    if isinstance(optimizer, RAdamScheduleFree):
        optimizer.train()
    losses = []
    all_targets = []
    all_outputs = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for step, batch in pbar:
        if isinstance(batch["melspec"], list):
            batch_outputs = []
            batch_losses = []

            for i in range(len(batch["melspec"])):
                inputs = batch["melspec"][i].unsqueeze(0).to(device)
                target = batch["target"][i].unsqueeze(0).to(device)

                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()

                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())

            optimizer.step()
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch["target"].numpy()

        else:
            inputs = batch["melspec"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())

        pbar.set_postfix(
            {
                "train_loss": np.mean(losses[-10:]) if losses else 0,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, auc


def validate(model, loader, optimizer, criterion, device):
    model.eval()
    if isinstance(optimizer, RAdamScheduleFree):
        optimizer.eval()
    losses = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if isinstance(batch["melspec"], list):
                batch_outputs = []
                batch_losses = []

                for i in range(len(batch["melspec"])):
                    inputs = batch["melspec"][i].unsqueeze(0).to(device)
                    target = batch["target"][i].unsqueeze(0).to(device)

                    output = model(inputs)
                    loss = criterion(output, target)

                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())

                outputs = torch.cat(batch_outputs, dim=0).numpy()
                loss = np.mean(batch_losses)
                targets = batch["target"].numpy()

            else:
                inputs = batch["melspec"].to(device)
                targets = batch["target"].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, auc


def calculate_auc(targets, outputs):
    num_classes = targets.shape[1]
    aucs = []

    probs = 1 / (1 + np.exp(-outputs))

    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)

    return np.mean(aucs) if aucs else 0.0


def train_fold_once(
    cfg,
    fold,
    train_df,
    valid_df,
    taxonomy_df,
    spectrograms,
    init_weights=None,
    stage=1,
):
    train_dataset = BirdCLEFDatasetFromNPY(
        cfg=cfg,
        df=train_df,
        taxonomy_df=taxonomy_df,
        spectrograms=spectrograms,
        mode="train",
    )
    valid_dataset = BirdCLEFDatasetFromNPY(
        cfg=cfg,
        df=valid_df,
        taxonomy_df=taxonomy_df,
        spectrograms=spectrograms,
        mode="valid",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = BirdCLEFModel(
        cfg=cfg,
        num_classes=len(taxonomy_df),
        is_pretrained=True,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
    ).to(cfg.device)
    if init_weights is not None:
        model.load_state_dict(init_weights)

    # lr を小さく／epoch を短く
    cfg.lr = cfg.lr if stage == 1 else cfg.lr * cfg.pl_lr_scale
    n_epochs = cfg.num_epochs if stage == 1 else cfg.pl_epochs
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    best_auc = 0.0
    best_epoch = 0
    early_stopping_cnt = 0
    for epoch in range(n_epochs):
        train_loss, train_auc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            get_criterion(cfg),
            cfg.device,
            scheduler if cfg.scheduler == "OneCycleLR" else None,
        )
        valid_loss, valid_auc = validate(
            model, valid_loader, optimizer, get_criterion(cfg), cfg.device
        )

        LOGGER.info(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        LOGGER.info(f"Valid Loss: {valid_loss:.4f}, Valid AUC: {valid_auc:.4f}")

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch + 1
            LOGGER.info(f"New best AUC: {best_auc:.4f} at epoch {best_epoch}")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if scheduler
                    else None,
                    "epoch": epoch,
                    "valid_auc": valid_auc,
                    "train_auc": train_auc,
                    "cfg": cfg,
                },
                f"model_fold{fold}_stage{stage}.pth",
            )

            early_stopping_cnt = 0
        else:
            early_stopping_cnt += 1
            if early_stopping_cnt >= cfg.early_stopping:
                LOGGER.info(
                    f"Early stopping at epoch {epoch + 1}, no improvement in last {cfg.early_stopping} epochs"
                )
                break

        model.best_auc = best_auc

    return model


def run_training(cfg, df, taxonomy_df, spectrograms):
    """① hold-out で Stage-1/2 を回す
    ② full_data+PL で optional finetune"""
    splits = _split_train_valid(cfg, df)
    best_scores = []

    # ────────────────────────────────────────────────
    # 1. Stage-1/2 : hold-out (または K-fold) 学習
    # ────────────────────────────────────────────────
    for fold, (train_idx, valid_idx) in enumerate(splits):
        LOGGER.info(f"\n======== Fold {fold} ========")
        train_df, valid_df = (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[valid_idx].reset_index(drop=True),
        )

        # ----- Stage-1 -----
        model = train_fold_once(
            cfg,
            fold,
            train_df,
            valid_df,
            taxonomy_df,
            spectrograms,
            init_weights=None,
            stage=1,
        )
        stage1_auc = model.best_auc

        # ----- Stage-2 (PL) -----
        pl_df = make_pl_windows(cfg, model, taxonomy_df)  # ⚠ 1 回だけ
        LOGGER.info(pl_df.head())

        if len(pl_df):
            for r in pl_df.itertuples():
                spectrograms[r.filename.replace(".ogg", "")] = r.melspec
            train_df_pl = pd.concat(
                [train_df, pl_df[["filename", "primary_label"]]], ignore_index=True
            )

            model = train_fold_once(
                cfg,
                fold,
                train_df_pl,
                valid_df,
                taxonomy_df,
                spectrograms,
                init_weights=model.state_dict(),
                stage=2,
            )

        best_scores.append(model.best_auc)
        final_model = model  # hold-outでは 1 つだけ

    # ────────────────────────────────────────────────
    # 2. 目標 AUC を超えていれば full-data で追い込み
    # ────────────────────────────────────────────────
    if cfg.full_train_after_pl and max(best_scores) >= cfg.pl_success_threshold:
        LOGGER.info("\n==== Full-data finetune with PL ====")
        full_df = pd.concat(
            [df, pl_df[["filename", "primary_label"]]], ignore_index=True
        )

        # 学習用 DataFrame は全部 train、validation なし（Early-Stopping も無効）
        final_model = train_fold_once(
            cfg,
            fold=0,  # ダミー fold 番号
            train_df=full_df,
            valid_df=valid_df,  # valid_df はダミーでも可
            taxonomy_df=taxonomy_df,
            spectrograms=spectrograms,
            init_weights=final_model.state_dict(),
            stage=3,  # 追加ステージ
        )

    LOGGER.info(f"\nCV-Mean AUC (hold-out) : {np.mean(best_scores):.5f}")

    return final_model


@hydra.main(config_path="conf", config_name="train", version_base="1.1")
def main(cfg: TrainConfig):
    set_seed(cfg.seed)

    # Load data
    train_df = pd.read_csv(cfg.dir.train_csv)
    taxonomy_df = pd.read_csv(cfg.dir.taxonomy_csv)
    spectrograms = None
    try:
        spectrograms = np.load(cfg.spectrogram_npy_path, allow_pickle=True).item()
        LOGGER.info(f"Loaded {len(spectrograms)} pre-computed mel spectrograms")
    except Exception as e:
        LOGGER.info(f"Error loading pre-computed spectrograms: {e}")
        LOGGER.info("Will generate spectrograms on-the-fly instead.")

    run_training(
        cfg=cfg,
        df=train_df,
        taxonomy_df=taxonomy_df,
        spectrograms=spectrograms,
    )


if __name__ == "__main__":
    # Logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
    )
    LOGGER = logging.getLogger(Path(__file__).name)

    # For descriptive error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    main()
