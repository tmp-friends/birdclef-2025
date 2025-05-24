import logging
import os
from pathlib import Path
import gc
from tqdm import tqdm

import hydra

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from schedulefree import RAdamScheduleFree

from utils.utils import set_seed
from conf.type import TrainConfig
from modules.birdclef_dataset import BirdCLEFDatasetFromNPY
from modules.birdclef_model import BirdCLEFModel
from modules.loss import FocalLossBCE


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


def run_training(
    cfg: TrainConfig,
    df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    spectrograms: np.ndarray,
):
    """Training function that can either use pre-computed spectrograms

    Args:
        cfg (TrainConfig): 設定値
        df (pd.DataFrame): 学習データ
        taxonomy_df (pd.DataFrame): 学習データ
        spectrograms (np.ndarray): 学習データ
    """
    skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)

    best_scores = []
    for fold, (train_ix, valid_ix) in enumerate(skf.split(df, df["primary_label"])):
        LOGGER.info(f"\n{'=' * 30} Fold {fold} {'=' * 30}")

        train_df = df.iloc[train_ix].reset_index(drop=True)
        valid_df = df.iloc[valid_ix].reset_index(drop=True)

        LOGGER.info(f"Training set: {len(train_df)} samples")
        LOGGER.info(f"Validation set: {len(valid_df)} samples")

        LOGGER.info(f"Training set stats: {train_df.describe()}")

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
        optimizer = get_optimizer(cfg, model)
        criterion = get_criterion(cfg)

        if cfg.scheduler == "OneCycleLR":
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                steps_per_epoch=len(train_loader),
                epochs=cfg.num_epochs,
                pct_start=0.1,
            )
        else:
            scheduler = get_scheduler(cfg, optimizer)

        best_auc = 0
        best_epoch = 0
        early_stopping_cnt = 0

        for epoch in range(cfg.num_epochs):
            LOGGER.info(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")

            train_loss, train_auc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None,
            )

            valid_loss, valid_auc = validate(
                model, valid_loader, optimizer, criterion, cfg.device
            )

            if scheduler is not None and not isinstance(
                scheduler, lr_scheduler.OneCycleLR
            ):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()

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
                    f"model_fold{fold}.pth",
                )

                early_stopping_cnt = 0
            else:
                early_stopping_cnt += 1
                if early_stopping_cnt >= cfg.early_stopping:
                    LOGGER.info(
                        f"Early stopping at epoch {epoch + 1}, no improvement in last {cfg.early_stopping} epochs"
                    )
                    break

        best_scores.append(best_auc)
        LOGGER.info(f"\nBest AUC for fold {fold}: {best_auc:.4f} at epoch {best_epoch}")

        # Clear memory
        del model, optimizer, scheduler, train_loader, valid_loader
        torch.cuda.empty_cache()
        gc.collect()

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        LOGGER.info(f"Fold {fold}: {score:.4f}")
    LOGGER.info(f"Mean AUC: {np.mean(best_scores):.4f}")
    LOGGER.info("=" * 60)


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
