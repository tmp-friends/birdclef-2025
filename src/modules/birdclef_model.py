import numpy as np
import polars as pl
import torch
from torch import nn
import torch.nn.functional as F

import timm

from conf.type import TrainConfig


class BirdCLEFModel(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig,
        num_classes: int,
        is_pretrained: bool = False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        self.cfg = cfg
        try:
            self.backbone = timm.create_model(
                cfg.model.name,
                pretrained=is_pretrained,
                in_chans=cfg.model.params.in_channels,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,  # ← EfficientViT は対応していない
            )
        except TypeError:
            self.backbone = timm.create_model(
                cfg.model.name,
                pretrained=is_pretrained,
                in_chans=cfg.model.params.in_channels,
            )

        # ── classifier を除去 & 出力次元を取得 ───────────────────────
        backbone_out = self._remove_head_and_get_outdim(self.backbone)

        # ── 後段層 ──────────────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(backbone_out, num_classes)

        # optional mix-up
        self.mixup_enabled = getattr(cfg, "mixup_alpha", 0) > 0

    def _remove_head_and_get_outdim(self, model: nn.Module) -> int:
        """
        timm のモデルから全結合層を Identity に置き換え、in_features を返す
        """
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            out_dim = model.classifier.in_features
            model.classifier = nn.Identity()
        elif hasattr(model, "head") and hasattr(model.head, "fc"):
            out_dim = model.head.fc.in_features
            model.head.fc = nn.Identity()
        else:  # Fallback (timm API)
            out_dim = model.get_classifier().in_features
            model.reset_classifier(0, "")

        return out_dim

    def forward(self, x, targets=None):
        if self.training and self.mixup_enabled and targets is not None:
            x, t_a, t_b, lam = self._mixup_data(x, targets)
        out = self.backbone(x)
        if isinstance(out, dict):  # ConvNeXt は dict を返す場合あり
            out = out["features"]
        if out.ndim == 4:  # (B,C,H,W) → GAP
            out = self.pool(out).flatten(1)
        logits = self.classifier(out)

        if self.training and self.mixup_enabled and targets is not None:
            loss = self._mixup_loss(logits, t_a, t_b, lam)
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
