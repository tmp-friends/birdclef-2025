"""
Train SED model with pseudo labels
Based on the finetune stage models
"""

import logging
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from utils.utils import set_seed
from modules.birdclef_dataset import BirdCLEFPseudoLabelDataset
from modules.birdclef_model import BirdCLEFSEDModel
# from modules.loss import BCEWithLogitsLoss
from utils.audio_augmentations import SEDAugmentationPipeline


def train_one_epoch_pseudo(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    cfg,
):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_is_pseudo = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, data in enumerate(progress_bar):
        waves = data["wave"].to(device)
        targets = data["target"].to(device)
        is_pseudo = data["is_pseudo"]
        confidences = data["confidence"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(waves)
        
        # Apply confidence weighting to loss
        loss = criterion(outputs, targets)
        
        # Weight loss by confidence for pseudo labels
        if cfg.use_confidence_weighting:
            loss = loss * confidences.unsqueeze(1)
            loss = loss.mean()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if hasattr(cfg, 'gradient_clip_val'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Store predictions for AUC calculation
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_is_pseudo.extend(is_pseudo.numpy())
        
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'lr': optimizer.param_groups[0]['lr']
        })
    
    # Calculate metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_is_pseudo = np.array(all_is_pseudo)
    
    # Calculate AUC for real and pseudo samples separately
    real_mask = ~all_is_pseudo
    pseudo_mask = all_is_pseudo
    
    train_loss = running_loss / len(dataloader)
    
    # Calculate AUC only for valid targets (avoid NaN issues)
    try:
        # Ensure binary targets for AUC calculation
        binary_targets = (all_targets > 0.5).astype(int)
        train_auc = roc_auc_score(binary_targets.ravel(), all_preds.ravel())
    except ValueError as e:
        LOGGER.warning(f"Could not calculate AUC: {e}")
        train_auc = 0.0
    
    metrics = {
        'train_loss': train_loss,
        'train_auc': train_auc,
    }
    
    if real_mask.sum() > 0:
        try:
            real_binary = (all_targets[real_mask] > 0.5).astype(int)
            real_auc = roc_auc_score(real_binary.ravel(), all_preds[real_mask].ravel())
            metrics['real_auc'] = real_auc
        except ValueError:
            metrics['real_auc'] = 0.0
    
    if pseudo_mask.sum() > 0:
        try:
            pseudo_binary = (all_targets[pseudo_mask] > 0.5).astype(int)
            pseudo_auc = roc_auc_score(pseudo_binary.ravel(), all_preds[pseudo_mask].ravel())
            metrics['pseudo_auc'] = pseudo_auc
        except ValueError:
            metrics['pseudo_auc'] = 0.0
    
    return metrics


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validation"):
            waves = data["wave"].to(device)
            targets = data["target"].to(device)
            
            outputs = model(waves)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    valid_loss = running_loss / len(dataloader)
    
    try:
        binary_targets = (all_targets > 0.5).astype(int)
        valid_auc = roc_auc_score(binary_targets.ravel(), all_preds.ravel())
    except ValueError:
        valid_auc = 0.0
    
    return valid_loss, valid_auc


def train_fold_pseudo(cfg, fold, train_df, valid_df, taxonomy_df):
    """Train one fold with pseudo labels"""
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = BirdCLEFPseudoLabelDataset(
        cfg=cfg,
        df=train_df,
        taxonomy_df=taxonomy_df,
        mode="train",
        pseudo_label_smoothing=cfg.pseudo_label_smoothing,
        pseudo_confidence_threshold=cfg.pseudo_confidence_threshold,
        pseudo_weight=cfg.pseudo_weight,
    )
    
    valid_dataset = BirdCLEFPseudoLabelDataset(
        cfg=cfg,
        df=valid_df,
        taxonomy_df=taxonomy_df,
        mode="val",
        pseudo_label_smoothing=0.0,  # No smoothing for validation
        pseudo_weight=1.0,  # Full weight for validation
    )
    
    # Create dataloaders
    if cfg.use_balanced_sampling:
        sampler = train_dataset.get_balanced_sampler(
            real_weight=cfg.real_sample_weight,
            pseudo_weight=cfg.pseudo_sample_weight
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    
    # Create model
    model = BirdCLEFSEDModel(cfg).to(device)
    
    # Load pretrained weights from finetune stage
    if cfg.pretrained_path:
        pretrained_path = os.path.join(
            cfg.pretrained_path, 
            f"sed_model_fold{fold}_finetune.pth"
        )
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            LOGGER.info(f"Loaded pretrained weights from {pretrained_path}")
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # Need 'none' for confidence weighting
    val_criterion = nn.BCEWithLogitsLoss(reduction='mean')  # Mean reduction for validation
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.T_max,
        eta_min=cfg.min_lr
    )
    
    # Training loop
    best_auc = 0.0
    best_epoch = 0
    
    for epoch in range(cfg.epochs):
        # Train
        train_metrics = train_one_epoch_pseudo(
            model, train_loader, criterion, optimizer, device, epoch, cfg
        )
        
        # Validate
        valid_loss, valid_auc = validate_epoch(
            model, valid_loader, val_criterion, device
        )
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        LOGGER.info(
            f"Fold {fold} Epoch {epoch}: "
            f"train_loss={train_metrics['train_loss']:.4f}, "
            f"train_auc={train_metrics['train_auc']:.4f}, "
            f"real_auc={train_metrics.get('real_auc', 0):.4f}, "
            f"pseudo_auc={train_metrics.get('pseudo_auc', 0):.4f}, "
            f"valid_loss={valid_loss:.4f}, valid_auc={valid_auc:.4f}"
        )
        
        # Save best model
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_auc': best_auc,
                'cfg': cfg,
            }
            
            save_path = f"sed_model_fold{fold}_pseudo.pth"
            torch.save(checkpoint, save_path)
            LOGGER.info(f"Saved best model to {save_path}")
        
        # Early stopping
        if epoch - best_epoch > cfg.early_stopping:
            LOGGER.info(f"Early stopping at epoch {epoch}")
            break
    
    return best_auc


@hydra.main(config_path="conf", config_name="train_sed_pseudo", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    
    # Load data
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.dir.taxonomy_csv)
    
    # Filter real samples for fold splitting
    real_df = train_df[~train_df['is_pseudo']].copy()
    pseudo_df = train_df[train_df['is_pseudo']].copy()
    
    LOGGER.info(f"Loaded {len(real_df)} real samples and {len(pseudo_df)} pseudo samples")
    
    # Create folds on real data only
    skf = StratifiedKFold(
        n_splits=cfg.num_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    
    real_df["fold"] = -1
    for fold, (_, valid_idx) in enumerate(
        skf.split(real_df, real_df["primary_label"])
    ):
        real_df.loc[real_df.index[valid_idx], "fold"] = fold
    
    # Process selected folds
    fold_scores = []
    
    for fold in cfg.selected_folds:
        LOGGER.info(f"\n{'='*50}")
        LOGGER.info(f"Training Fold {fold}")
        LOGGER.info(f"{'='*50}")
        
        # Split data
        train_real = real_df[real_df["fold"] != fold].copy()
        valid_real = real_df[real_df["fold"] == fold].copy()
        
        # Add pseudo labels to training only
        train_combined = pd.concat([train_real, pseudo_df], ignore_index=True)
        
        LOGGER.info(
            f"Fold {fold}: "
            f"train_real={len(train_real)}, "
            f"train_pseudo={len(pseudo_df)}, "
            f"valid={len(valid_real)}"
        )
        
        # Train fold
        fold_auc = train_fold_pseudo(
            cfg, fold, train_combined, valid_real, taxonomy_df
        )
        fold_scores.append(fold_auc)
        
        LOGGER.info(f"Fold {fold} Best AUC: {fold_auc:.4f}")
    
    # Summary
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    
    LOGGER.info(f"\n{'='*50}")
    LOGGER.info(f"Cross-validation Results:")
    LOGGER.info(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    LOGGER.info(f"Fold AUCs: {fold_scores}")
    

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
    )
    LOGGER = logging.getLogger(Path(__file__).name)
    
    main()