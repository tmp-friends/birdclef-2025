import numpy as np
import polars as pl
import torch
from torch import nn
import torch.nn.functional as F

import timm

from conf.type import TrainConfig


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg: TrainConfig, num_classes: int):
        super().__init__()

        self.cfg = cfg

        self.backbone = timm.create_model(
            model_name=cfg.model.name,
            pretrained=cfg.model.params.pretrained,
            in_chans=cfg.model.params.in_channels,
            drop_rate=cfg.model.params.drop_rate,
            drop_path_rate=cfg.model.params.drop_path_rate,
        )

        if "efficientnet" in cfg.model.name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif "resnet" in cfg.mode.name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, "")

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, num_classes)

        self.mixup_enabled = self.cfg.mixup_alpha > 0

    def forward(self, x, targets=None):
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self._mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None

        features = self.backbone(x)

        if isinstance(features, dict):
            features = features["features"]

        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)

        if self.training and self.mixup_enabled and targets is not None:
            loss = self._mixup_criterion(
                criterion=F.binary_cross_entropy_with_logits,
                pred=logits,
                y_a=targets_a,
                y_b=targets_b,
                lam=lam,
            )
            return logits, loss

        return logits

    def _mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]

        return mixed_x, targets, targets[indices], lam

    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
