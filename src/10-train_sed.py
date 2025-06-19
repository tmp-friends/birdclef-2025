import logging
import os
from pathlib import Path
import gc
from tqdm import tqdm

import hydra

import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from schedulefree import RAdamScheduleFree

from utils.utils import set_seed
from conf.type import TrainConfig
from modules.birdclef_model import BirdCLEFSEDModel
from modules.loss import FocalLossBCE
from utils.augmentation import Augmentation


def rms_crop_shift(
    wav: np.ndarray, seg_len: int, sr: int, stride_s: float = 1.0, shift_s: float = 2.0
):
    """RMS 最大窓を求めた後 ±shift_s 秒ランダムにずらす"""
    if len(wav) <= seg_len:
        k = int(np.ceil(seg_len / len(wav)))
        return np.tile(wav, k)[:seg_len]

    stride = int(sr * stride_s)
    max_rms, max_idx = 0.0, 0
    for s in range(0, len(wav) - seg_len + 1, stride):
        win = wav[s : s + seg_len]
        rms = np.sqrt((win**2).mean())
        if rms > max_rms:
            max_rms, max_idx = rms, s

    # ±Shift
    max_shift = int(sr * shift_s)
    shift = np.random.randint(-max_shift, max_shift + 1)
    new_start = np.clip(max_idx + shift, 0, len(wav) - seg_len)
    return wav[new_start : new_start + seg_len]


class BirdCLEFWaveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
        df: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
        mode: str = "train",
    ):
        self.cfg = cfg
        self.df = df
        self.taxonomy_df = taxonomy_df
        self.mode = mode
        self.num_classes = len(taxonomy_df)
        self.seg_len = int(cfg.train_duration * cfg.fs)  # seconds to samples

        # Create label mapping
        self.label2idx = {
            label: idx for idx, label in enumerate(taxonomy_df["primary_label"])
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio file
        audio_path = os.path.join(self.cfg.dir.train_audio_dir, row["filename"])
        wav, _ = sf.read(audio_path, dtype="float32")
        
        # Apply RMS crop and shift
        if self.mode == "train":
            wav = rms_crop_shift(
                wav, self.seg_len, self.cfg.fs, shift_s=self.cfg.random_shift_s
            )
            
            # Apply audio-level augmentations during training
            wav = self._apply_audio_augmentations(wav)
        else:
            # For validation, use center crop
            if len(wav) > self.seg_len:
                start = (len(wav) - self.seg_len) // 2
                wav = wav[start : start + self.seg_len]
            else:
                k = int(np.ceil(self.seg_len / len(wav)))
                wav = np.tile(wav, k)[:self.seg_len]
        
        # Convert to tensor and add channel dimension
        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # (1, samples)

        # Create one-hot encoded target
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        if row["primary_label"] in self.label2idx:
            target[self.label2idx[row["primary_label"]]] = 1.0

        return {"wave": wav, "target": target}
    
    def _apply_audio_augmentations(self, wav: np.ndarray) -> np.ndarray:
        """Apply audio-level augmentations during training"""
        import random
        
        # Gaussian noise augmentation
        if hasattr(self.cfg.augmentation, 'gaussian_noise') and self.cfg.augmentation.gaussian_noise:
            if random.random() < getattr(self.cfg.augmentation, 'noise_prob', 0.3):
                noise_std = getattr(self.cfg.augmentation, 'noise_std', 0.01)
                noise = np.random.normal(0, noise_std, wav.shape).astype(np.float32)
                wav = wav + noise
        
        # Volume/gain augmentation  
        if hasattr(self.cfg.augmentation, 'volume_aug') and self.cfg.augmentation.volume_aug:
            if random.random() < getattr(self.cfg.augmentation, 'volume_prob', 0.3):
                gain_range = getattr(self.cfg.augmentation, 'gain_range', [0.7, 1.3])
                gain = random.uniform(gain_range[0], gain_range[1])
                wav = wav * gain
        
        # Time stretch augmentation (if librosa available)
        if hasattr(self.cfg.augmentation, 'time_stretch') and self.cfg.augmentation.time_stretch:
            if random.random() < getattr(self.cfg.augmentation, 'time_stretch_prob', 0.2):
                try:
                    import librosa
                    rate_range = getattr(self.cfg.augmentation, 'stretch_range', [0.9, 1.1])
                    rate = random.uniform(rate_range[0], rate_range[1])
                    wav = librosa.effects.time_stretch(wav, rate=rate)
                    
                    # Ensure length matches expected segment length
                    if len(wav) != self.seg_len:
                        if len(wav) > self.seg_len:
                            wav = wav[:self.seg_len]
                        else:
                            wav = np.pad(wav, (0, self.seg_len - len(wav)), mode='constant')
                except ImportError:
                    # Skip time stretch if librosa not available
                    pass
        
        # Clip to prevent overflow
        wav = np.clip(wav, -1.0, 1.0)
        
        return wav


def collate_fn(batch):
    waves = torch.stack([b["wave"] for b in batch])  # (B, 1, samples)
    targets = torch.stack([b["target"] for b in batch])  # (B, num_classes)
    return {"wave": waves, "target": targets}


def mixup_data(x, y, alpha=0.4):
    """Apply mixup to a batch of data and targets"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Apply mixup to the loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix to a batch of audio data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # For audio, we cut along time dimension (last dimension)
    time_length = x.size(-1)
    cut_length = int(time_length * (1 - lam))
    
    if cut_length > 0:
        # Random start position for the cut
        start_pos = np.random.randint(0, time_length - cut_length + 1)
        end_pos = start_pos + cut_length
        
        # Apply cutmix
        mixed_x = x.clone()
        mixed_x[:, :, start_pos:end_pos] = x[index, :, start_pos:end_pos]
    else:
        mixed_x = x
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


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
        # Default to Adam
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

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
        # Default to BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss()

    return criterion


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    if isinstance(optimizer, RAdamScheduleFree):
        optimizer.train()
    losses = []
    all_targets = []
    all_outputs = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for _, batch in pbar:
        inputs = batch["wave"].to(device)  # (B, 1, samples)
        targets = batch["target"].to(device)  # (B, num_classes)

        # Apply mixup or cutmix if enabled
        use_mixup = hasattr(model.cfg, 'augmentation') and getattr(model.cfg.augmentation, 'mixup', False)
        use_cutmix = hasattr(model.cfg, 'augmentation') and getattr(model.cfg.augmentation, 'cutmix', False)
        
        mixed = False
        if use_mixup and use_cutmix:
            # Randomly choose between mixup and cutmix
            if np.random.random() < 0.5:
                mixup_alpha = getattr(model.cfg.augmentation, 'mixup_alpha', 0.4)
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
                mixed = True
            else:
                cutmix_alpha = getattr(model.cfg.augmentation, 'cutmix_alpha', 1.0)
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)
                mixed = True
        elif use_mixup:
            mixup_alpha = getattr(model.cfg.augmentation, 'mixup_alpha', 0.4)
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            mixed = True
        elif use_cutmix:
            cutmix_alpha = getattr(model.cfg.augmentation, 'cutmix_alpha', 1.0)
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)
            mixed = True

        optimizer.zero_grad()
        outputs = model(inputs)  # (B, num_classes)

        # Check for NaN in model outputs
        if torch.any(torch.isnan(outputs)):
            print("Warning: NaN detected in model outputs during training")
            # Skip this batch
            continue

        # Calculate loss (with mixup/cutmix if applied)
        if mixed:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print("Warning: NaN detected in loss during training")
            # Skip this batch
            continue
            
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        losses.append(loss.item())

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
            inputs = batch["wave"].to(device)  # (B, 1, samples)
            targets = batch["target"].to(device)  # (B, num_classes)

            outputs = model(inputs)  # (B, num_classes)
            
            # Check for NaN in model outputs
            if torch.any(torch.isnan(outputs)):
                print("Warning: NaN detected in model outputs during validation")
                # Skip this batch
                continue
                
            loss = criterion(outputs, targets)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print("Warning: NaN detected in loss during validation")
                # Skip this batch
                continue

            all_outputs.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            losses.append(loss.item())

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, auc


def calculate_auc(targets, outputs):
    num_classes = targets.shape[1]
    aucs = []

    # Check for NaN or inf values in outputs
    if np.any(np.isnan(outputs)) or np.any(np.isinf(outputs)):
        print("Warning: NaN or inf detected in outputs. Replacing with 0.")
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)

    # sigmoid → prob (with clipping to prevent overflow)
    outputs_clipped = np.clip(outputs, -10, 10)  # More conservative clipping
    probs = 1 / (1 + np.exp(-outputs_clipped))

    for i in range(num_classes):
        y_true = targets[:, i]
        y_score = probs[:, i]

        # Check for NaN values in scores
        if np.any(np.isnan(y_score)):
            print(f"Warning: NaN detected in y_score for class {i}. Skipping.")
            continue

        # ① binary に丸める（>0.5 で 1）
        y_true_bin = (y_true > 0.5).astype(int)

        # ② 正例・負例の両方が存在しなければスキップ
        pos = y_true_bin.sum()
        neg = len(y_true_bin) - pos
        if pos == 0 or neg == 0:
            continue

        # ③ AUC
        try:
            auc_score = roc_auc_score(y_true_bin, y_score)
            if not np.isnan(auc_score):
                aucs.append(auc_score)
        except Exception as e:
            print(f"Warning: Failed to compute AUC for class {i}: {e}")
            continue

    # 全クラスで計算できなかった場合の fallback
    return float(np.mean(aucs)) if aucs else 0.0


def run_training(
    cfg: TrainConfig,
    df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
):
    """Training function for SED model
    
    Args:
        cfg (TrainConfig): 設定値
        df (pd.DataFrame): 学習データ
        taxonomy_df (pd.DataFrame): 分類体系データ
    """
    skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)

    best_scores = []
    for fold, (train_ix, valid_ix) in enumerate(skf.split(df, df["primary_label"])):
        if fold not in cfg.selected_folds:
            continue
            
        LOGGER.info(f"\n{'=' * 30} Fold {fold} {'=' * 30}")

        train_df = df.iloc[train_ix].reset_index(drop=True)
        valid_df = df.iloc[valid_ix].reset_index(drop=True)

        LOGGER.info(f"Training set: {len(train_df)} samples")
        LOGGER.info(f"Validation set: {len(valid_df)} samples")

        # Create datasets
        train_dataset = BirdCLEFWaveDataset(
            cfg=cfg,
            df=train_df,
            taxonomy_df=taxonomy_df,
            mode="train",
        )
        valid_dataset = BirdCLEFWaveDataset(
            cfg=cfg,
            df=valid_df,
            taxonomy_df=taxonomy_df,
            mode="valid",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Initialize SED model
        model = BirdCLEFSEDModel(cfg).to(cfg.device)
        
        # Get optimizer
        optimizer = get_optimizer(cfg, model)
        
        # Get criterion
        criterion = get_criterion(cfg)

        # Get scheduler if specified
        scheduler = None
        if hasattr(cfg, 'scheduler'):
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
                    f"sed_model_fold{fold}.pth",
                )

                early_stopping_cnt = 0
            else:
                early_stopping_cnt += 1
                if hasattr(cfg, 'early_stopping') and early_stopping_cnt >= cfg.early_stopping:
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

    if best_scores:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("Cross-Validation Results:")
        for i, score in enumerate(best_scores):
            LOGGER.info(f"Fold {cfg.selected_folds[i]}: {score:.4f}")
        LOGGER.info(f"Mean AUC: {np.mean(best_scores):.4f}")
        LOGGER.info("=" * 60)


@hydra.main(config_path="conf", config_name="train_sed", version_base="1.1")
def main(cfg: TrainConfig):
    set_seed(cfg.seed)

    # Load data
    train_df = pd.read_csv(cfg.dir.train_csv)
    taxonomy_df = pd.read_csv(cfg.dir.taxonomy_csv)

    run_training(
        cfg=cfg,
        df=train_df,
        taxonomy_df=taxonomy_df,
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