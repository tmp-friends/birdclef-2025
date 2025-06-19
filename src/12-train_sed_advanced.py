"""
Advanced SED training script for BirdCLEF 2025
Based on successful approaches from BirdCLEF 2023 2nd place solution

Features:
- Multi-stage training (pretrain_ce, train_bce, finetune)
- Advanced audio augmentations
- Mixup/CutMix with SpecAugment
- Multiple loss functions and optimization strategies
"""

import logging
import os
from pathlib import Path
import gc
from tqdm import tqdm
import yaml

import hydra
from omegaconf import DictConfig, OmegaConf

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

from utils.utils import set_seed
from conf.type import TrainConfig
from modules.birdclef_model import BirdCLEFSEDModel
from modules.loss import FocalLossBCE
from utils.audio_augmentations import SEDAugmentationPipeline, mixup_criterion


class AdvancedBirdCLEFWaveDataset(torch.utils.data.Dataset):
    """
    Advanced dataset with sophisticated augmentations for SED training
    """
    def __init__(
        self,
        cfg: DictConfig,
        df: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
        stage: str = "pretrain_ce",
        mode: str = "train",
    ):
        self.cfg = cfg
        self.df = df
        self.taxonomy_df = taxonomy_df
        self.stage = stage
        self.mode = mode
        self.num_classes = len(taxonomy_df)
        self.seg_len = int(cfg.train_duration * cfg.fs)
        
        # Stage-specific settings
        self.stage_config = cfg.training_stages[stage]
        
        # Create label mapping
        self.label2idx = {
            label: idx for idx, label in enumerate(taxonomy_df["primary_label"])
        }
        
        # Initialize augmentation pipeline
        if mode == "train":
            self.augmentation = SEDAugmentationPipeline(
                sample_rate=cfg.fs,
                use_audio_aug=cfg.augmentation.use_audio_aug,
                use_spec_aug=cfg.augmentation.use_spec_aug,
                use_mixup_cutmix=cfg.augmentation.use_mixup_cutmix,
                audio_aug_prob=cfg.augmentation.audio_aug_prob,
                spec_aug_prob=cfg.augmentation.spec_aug_prob,
                mixup_cutmix_prob=cfg.augmentation.mixup_cutmix_prob,
            )
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio file
        audio_path = os.path.join(self.cfg.dir.train_audio_dir, row["filename"])
        wav, _ = sf.read(audio_path, dtype="float32")
        
        # Apply RMS crop and shift
        if self.mode == "train":
            wav = self._rms_crop_shift(wav)
        else:
            wav = self._center_crop(wav)
        
        # Apply audio-level augmentations
        if self.augmentation is not None:
            wav = self.augmentation.apply_audio_augmentation(wav)
        
        # Convert to tensor and add channel dimension
        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # (1, samples)

        # Create target labels
        target = self._create_target(row)

        return {"wave": wav, "target": target, "filename": row["filename"]}
    
    def _rms_crop_shift(self, wav: np.ndarray) -> np.ndarray:
        """RMS cropping with random shift"""
        if len(wav) <= self.seg_len:
            k = int(np.ceil(self.seg_len / len(wav)))
            return np.tile(wav, k)[:self.seg_len]

        stride = int(self.cfg.fs * 1.0)  # 1 second stride
        max_rms, max_idx = 0.0, 0
        for s in range(0, len(wav) - self.seg_len + 1, stride):
            win = wav[s : s + self.seg_len]
            rms = np.sqrt((win**2).mean())
            if rms > max_rms:
                max_rms, max_idx = rms, s

        # Apply random shift
        max_shift = int(self.cfg.fs * self.cfg.random_shift_s)
        shift = np.random.randint(-max_shift, max_shift + 1)
        new_start = np.clip(max_idx + shift, 0, len(wav) - self.seg_len)
        return wav[new_start : new_start + self.seg_len]
    
    def _center_crop(self, wav: np.ndarray) -> np.ndarray:
        """Center crop for validation"""
        if len(wav) > self.seg_len:
            start = (len(wav) - self.seg_len) // 2
            wav = wav[start : start + self.seg_len]
        else:
            k = int(np.ceil(self.seg_len / len(wav)))
            wav = np.tile(wav, k)[:self.seg_len]
        return wav
    
    def _create_target(self, row) -> torch.Tensor:
        """Create target tensor based on stage configuration"""
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        
        # Primary label
        if row["primary_label"] in self.label2idx:
            if self.stage_config.criterion == 'CrossEntropyLoss':
                # For CE loss, return class index
                return torch.tensor(self.label2idx[row["primary_label"]], dtype=torch.long)
            else:
                # For BCE loss, use one-hot
                target[self.label2idx[row["primary_label"]]] = 1.0
        
        # Secondary labels (if enabled)
        if self.stage_config.use_secondary_labels and 'secondary_labels' in row:
            secondary_labels = row.get('secondary_labels', [])
            if isinstance(secondary_labels, str):
                secondary_labels = eval(secondary_labels)  # Convert string to list if needed
            
            for sec_label in secondary_labels:
                if sec_label in self.label2idx:
                    target[self.label2idx[sec_label]] = self.stage_config.secondary_label_weight
        
        return target


def collate_fn_advanced(batch):
    """Advanced collate function that handles both CE and BCE cases"""
    waves = torch.stack([b["wave"] for b in batch])
    targets = [b["target"] for b in batch]
    filenames = [b["filename"] for b in batch]
    
    # Handle different target types (CE vs BCE)
    if isinstance(targets[0], torch.Tensor) and targets[0].dim() == 0:
        # CE targets (class indices)
        targets = torch.stack(targets)
    else:
        # BCE targets (one-hot vectors)
        targets = torch.stack(targets)
    
    return {"wave": waves, "target": targets, "filename": filenames}


def get_optimizer(cfg: DictConfig, model, stage: str):
    """Get optimizer based on stage configuration"""
    stage_config = cfg.training_stages[stage]
    
    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(), 
            lr=stage_config.lr, 
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=stage_config.lr, 
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=stage_config.lr, 
            momentum=0.9, 
            weight_decay=cfg.weight_decay
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=stage_config.lr)

    return optimizer


def get_scheduler(cfg: DictConfig, optimizer, stage: str, steps_per_epoch: int):
    """Get scheduler based on configuration"""
    stage_config = cfg.training_stages[stage]
    
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=stage_config.epochs * steps_per_epoch,
            eta_min=stage_config.min_lr
        )
    elif cfg.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=stage_config.epochs * steps_per_epoch // 4,
            eta_min=stage_config.min_lr
        )
    elif cfg.scheduler == "OneCycleLR":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=stage_config.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=stage_config.epochs,
            pct_start=0.1,
        )
    else:
        scheduler = None

    return scheduler


def get_criterion(cfg: DictConfig, stage: str):
    """Get loss function based on stage configuration"""
    stage_config = cfg.training_stages[stage]
    
    if stage_config.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif stage_config.criterion == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif stage_config.criterion == "FocalLossBCE":
        criterion = FocalLossBCE(
            alpha=cfg.focal_loss_alpha,
            gamma=cfg.focal_loss_gamma
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    return criterion


def train_one_epoch_advanced(model, loader, optimizer, criterion, device, stage_config, augmentation=None, scheduler=None):
    """Advanced training loop with mixup/cutmix support"""
    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for step, batch in pbar:
        inputs = batch["wave"].to(device)
        targets = batch["target"].to(device)

        # Handle different target types
        is_ce_loss = len(targets.shape) == 1  # CE targets are 1D
        
        # Apply Mixup/CutMix if enabled and not CE loss
        mixed_inputs = inputs
        mixed_targets_a = targets
        mixed_targets_b = targets
        lam = 1.0
        aug_type = "none"
        
        if augmentation is not None and not is_ce_loss:
            # Convert to spectrogram for mixup/cutmix
            with torch.no_grad():
                spec = model.mel(mixed_inputs.squeeze(1))  # Remove channel dim for mel transform
                spec = model.db(spec)
                spec = (spec + 80) / 80
                spec = spec.unsqueeze(1)  # Add back channel dim
            
            # Apply mixup/cutmix on spectrogram
            mixed_spec, mixed_targets_a, mixed_targets_b, lam, aug_type = augmentation.apply_mixup_cutmix(spec, targets)
            
            # Apply spec augmentation
            mixed_spec = augmentation.apply_spec_augmentation(mixed_spec)
            
            # Forward pass with pre-computed spectrogram
            optimizer.zero_grad()
            feat = model._encode(mixed_spec)
            feat = torch.nn.functional.relu_(model.fc(feat.transpose(1, 2))).transpose(1, 2)
            outputs, _, _ = model.att_block(feat)
            # Remove torch.logit as it causes NaN with BCEWithLogitsLoss
        else:
            # Standard forward pass
            optimizer.zero_grad()
            outputs = model(mixed_inputs)

        # Check for NaN in model outputs
        if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
            print("Warning: NaN/inf detected in model outputs during training. Skipping batch.")
            continue

        # Calculate loss
        if aug_type != "none" and not is_ce_loss:
            loss = mixup_criterion(criterion, outputs, mixed_targets_a, mixed_targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/inf detected in loss during training. Skipping batch.")
            continue

        # Scale loss to prevent underflow
        scaled_loss = loss * 1000.0
        scaled_loss.backward()
        
        # Check gradients for NaN
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.any(torch.isnan(param.grad)):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print("Warning: NaN detected in gradients. Skipping batch.")
            optimizer.zero_grad()
            continue
        
        # Scale gradients back
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data /= 1000.0
        
        # Gradient clipping (more aggressive)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Store predictions for metrics
        if is_ce_loss:
            # For CE, convert to probabilities
            outputs_prob = torch.softmax(outputs, dim=1)
            targets_onehot = torch.zeros_like(outputs_prob)
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
            all_outputs.append(outputs_prob.detach().cpu().numpy())
            all_targets.append(targets_onehot.detach().cpu().numpy())
        else:
            # For BCE, convert to probabilities
            outputs_prob = torch.sigmoid(outputs)
            all_outputs.append(outputs_prob.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
        
        losses.append(loss.item())

        pbar.set_postfix({
            "train_loss": np.mean(losses[-10:]) if losses else 0,
            "lr": optimizer.param_groups[0]["lr"],
            "aug": aug_type,
        })

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, auc


def validate_advanced(model, loader, optimizer, criterion, device):
    """Advanced validation function"""
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            inputs = batch["wave"].to(device)
            targets = batch["target"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Handle different target types
            is_ce_loss = len(targets.shape) == 1
            
            if is_ce_loss:
                outputs_prob = torch.softmax(outputs, dim=1)
                targets_onehot = torch.zeros_like(outputs_prob)
                targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
                all_outputs.append(outputs_prob.cpu().numpy())
                all_targets.append(targets_onehot.cpu().numpy())
            else:
                outputs_prob = torch.sigmoid(outputs)
                all_outputs.append(outputs_prob.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
            
            losses.append(loss.item())

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, auc


def calculate_auc(targets, outputs):
    """Calculate AUC score"""
    num_classes = targets.shape[1]
    aucs = []

    # Check for NaN or inf values in outputs
    if np.any(np.isnan(outputs)) or np.any(np.isinf(outputs)):
        print("Warning: NaN or inf detected in outputs. Replacing with 0.0")
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)

    for i in range(num_classes):
        y_true = targets[:, i]
        y_score = outputs[:, i]

        # Additional check for NaN values in scores
        if np.any(np.isnan(y_score)) or np.any(np.isinf(y_score)):
            print(f"Warning: NaN/inf detected in y_score for class {i}. Skipping.")
            continue

        y_true_bin = (y_true > 0.5).astype(int)
        pos = y_true_bin.sum()
        neg = len(y_true_bin) - pos
        
        if pos == 0 or neg == 0:
            continue

        # Additional safety: clip values to prevent numerical issues
        y_score_safe = np.clip(y_score, 1e-8, 1.0 - 1e-8)
        
        try:
            auc_score = roc_auc_score(y_true_bin, y_score_safe)
            if not np.isnan(auc_score) and not np.isinf(auc_score):
                aucs.append(auc_score)
            else:
                print(f"Warning: Invalid AUC score for class {i}: {auc_score}")
        except Exception as e:
            print(f"Warning: Failed to compute AUC for class {i}: {e}")
            continue

    return float(np.mean(aucs)) if aucs else 0.0


def run_stage_training(cfg: DictConfig, stage: str, df: pd.DataFrame, taxonomy_df: pd.DataFrame):
    """Run training for a specific stage"""
    LOGGER.info(f"\n{'='*20} Starting Stage: {stage} {'='*20}")
    
    stage_config = cfg.training_stages[stage]
    skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)

    # Initialize augmentation for training
    augmentation = SEDAugmentationPipeline(
        sample_rate=cfg.fs,
        use_audio_aug=cfg.augmentation.use_audio_aug,
        use_spec_aug=cfg.augmentation.use_spec_aug,
        use_mixup_cutmix=cfg.augmentation.use_mixup_cutmix,
    ) if stage != "pretrain_ce" else None  # No mixup for CE stage

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
        train_dataset = AdvancedBirdCLEFWaveDataset(
            cfg=cfg,
            df=train_df,
            taxonomy_df=taxonomy_df,
            stage=stage,
            mode="train",
        )
        valid_dataset = AdvancedBirdCLEFWaveDataset(
            cfg=cfg,
            df=valid_df,
            taxonomy_df=taxonomy_df,
            stage=stage,
            mode="valid",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_advanced,
            drop_last=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_advanced,
        )

        # Initialize model
        model = BirdCLEFSEDModel(cfg).to(cfg.device)
        
        # Load checkpoint from previous stage if exists
        checkpoint_path = f"sed_model_fold{fold}_{stage}.pth"
        if stage != "pretrain_ce":
            # Try to load from previous stage
            prev_stages = ["pretrain_ce", "train_bce"]
            for prev_stage in prev_stages:
                # Search paths in order of priority
                search_paths = []
                
                # 1. Current directory
                search_paths.append(f"sed_model_fold{fold}_{prev_stage}.pth")
                
                # 2. Checkpoint save directory (for all stages run)
                if cfg.current_stage == "all" and hasattr(cfg, 'checkpoint_save_dir'):
                    search_paths.append(os.path.join(cfg.checkpoint_save_dir, f"sed_model_fold{fold}_{prev_stage}.pth"))
                
                # 3. Pretrain model directory
                if hasattr(cfg, 'pretrain_model_dir'):
                    search_paths.append(os.path.join(cfg.pretrain_model_dir, f"sed_model_fold{fold}_{prev_stage}.pth"))
                
                # Try each path
                for path in search_paths:
                    if os.path.exists(path):
                        LOGGER.info(f"Loading checkpoint from {path}")
                        checkpoint = torch.load(path, map_location=cfg.device, weights_only=False)
                        model.load_state_dict(checkpoint["model_state_dict"])
                        LOGGER.info(f"Successfully loaded {prev_stage} checkpoint for fold {fold}")
                        break
                else:
                    continue
                break
            else:
                LOGGER.warning(f"No checkpoint found for fold {fold}. Starting from scratch.")
        
        # Get optimizer, scheduler, criterion
        optimizer = get_optimizer(cfg, model, stage)
        scheduler = get_scheduler(cfg, optimizer, stage, len(train_loader))
        criterion = get_criterion(cfg, stage)

        best_auc = 0
        best_epoch = 0
        early_stopping_cnt = 0

        for epoch in range(stage_config.epochs):
            LOGGER.info(f"\nEpoch {epoch + 1}/{stage_config.epochs}")

            train_loss, train_auc = train_one_epoch_advanced(
                model,
                train_loader,
                optimizer,
                criterion,
                cfg.device,
                stage_config,
                augmentation,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None,
            )

            valid_loss, valid_auc = validate_advanced(
                model, valid_loader, optimizer, criterion, cfg.device
            )

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
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

                # Determine save paths
                save_paths = [checkpoint_path]  # Always save to current directory
                
                # If running all stages, also save to checkpoint_save_dir
                if cfg.current_stage == "all" and hasattr(cfg, 'checkpoint_save_dir'):
                    os.makedirs(cfg.checkpoint_save_dir, exist_ok=True)
                    shared_path = os.path.join(cfg.checkpoint_save_dir, checkpoint_path)
                    save_paths.append(shared_path)
                
                # Save checkpoint to all paths
                checkpoint_data = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch,
                    "valid_auc": valid_auc,
                    "train_auc": train_auc,
                    "stage": stage,
                    "cfg": cfg,
                }
                
                for save_path in save_paths:
                    torch.save(checkpoint_data, save_path)
                    LOGGER.info(f"Saved checkpoint to: {save_path}")

                early_stopping_cnt = 0
            else:
                early_stopping_cnt += 1
                if early_stopping_cnt >= cfg.early_stopping:
                    LOGGER.info(f"Early stopping at epoch {epoch + 1}")
                    break

        best_scores.append(best_auc)
        LOGGER.info(f"\nBest AUC for fold {fold}: {best_auc:.4f} at epoch {best_epoch}")

        # Clear memory
        del model, optimizer, scheduler, train_loader, valid_loader
        torch.cuda.empty_cache()
        gc.collect()

    if best_scores:
        LOGGER.info(f"\n{stage} Results:")
        LOGGER.info("="*60)
        for i, score in enumerate(best_scores):
            LOGGER.info(f"Fold {cfg.selected_folds[i]}: {score:.4f}")
        LOGGER.info(f"Mean AUC: {np.mean(best_scores):.4f}")
        LOGGER.info("="*60)
    
    return np.mean(best_scores) if best_scores else 0.0


@hydra.main(config_path="conf", config_name="train_sed_advanced", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    # Load data
    train_df = pd.read_csv(cfg.dir.train_csv)
    taxonomy_df = pd.read_csv(cfg.dir.taxonomy_csv)

    # Get training stages to run
    if cfg.current_stage == "all":
        stages = ["pretrain_ce", "train_bce", "finetune"]
    else:
        stages = [cfg.current_stage]
    
    stage_results = {}
    
    for stage in stages:
        if stage not in cfg.training_stages:
            LOGGER.warning(f"Stage {stage} not found in configuration, skipping.")
            continue
            
        mean_auc = run_stage_training(cfg, stage, train_df, taxonomy_df)
        stage_results[stage] = mean_auc
    
    # Print final results
    LOGGER.info("\n" + "="*60)
    LOGGER.info("FINAL RESULTS SUMMARY")
    LOGGER.info("="*60)
    for stage, auc in stage_results.items():
        LOGGER.info(f"{stage}: {auc:.4f}")
    LOGGER.info("="*60)


if __name__ == "__main__":
    # Logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
    )
    LOGGER = logging.getLogger(Path(__file__).name)

    # For descriptive error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    main()