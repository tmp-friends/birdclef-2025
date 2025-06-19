from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

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
        self.backbone = timm.create_model(
            cfg.model.name,
            pretrained=is_pretrained,
            in_chans=cfg.model.params.in_channels,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,  # ← EfficientViT は対応していない
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


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):  # timm と同じ初期化
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


# ---------- pooling layers ------------------------------------------
class GeM(nn.Module):
    """Generalized Mean Pooling"""
    
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


# ---------- attention head -------------------------------------------
class AttBlockV2(nn.Module):
    """(B,C,T) -> clipwise(B,C) + att(C,T) + frame-wise(C,T)"""

    def __init__(self, in_ch: int, out_ch: int, act: str = "sigmoid"):
        super().__init__()
        self.att = nn.Conv1d(in_ch, out_ch, 1)
        self.cla = nn.Conv1d(in_ch, out_ch, 1)
        init_layer(self.att)
        init_layer(self.cla)
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "linear":
            self.act = nn.Identity()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # x : (B, C, T)
        att = torch.softmax(torch.tanh(self.att(x)), dim=-1)  # (B,O,T)
        cla = self.act(self.cla(x))  # (B,O,T)
        clip = torch.sum(att * cla, dim=-1)  # (B,O)
        return clip, att, cla


class MultiHeadAttBlockV2(nn.Module):
    """Multi-head attention block for enhanced feature learning"""
    
    def __init__(self, in_ch: int, out_ch: int, num_heads: int = 4, act: str = "sigmoid"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_ch // num_heads
        assert in_ch % num_heads == 0, "in_ch must be divisible by num_heads"
        
        # Multi-head attention components
        self.att_heads = nn.ModuleList([
            nn.Conv1d(self.head_dim, out_ch, 1) for _ in range(num_heads)
        ])
        self.cla_heads = nn.ModuleList([
            nn.Conv1d(self.head_dim, out_ch, 1) for _ in range(num_heads)
        ])
        
        # Initialize layers
        for att, cla in zip(self.att_heads, self.cla_heads):
            init_layer(att)
            init_layer(cla)
        
        # Combination layer
        self.combine = nn.Conv1d(out_ch * num_heads, out_ch, 1)
        init_layer(self.combine)
        
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "linear":
            self.act = nn.Identity()
        else:
            self.act = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # x: (B, C, T)
        B, C, T = x.shape
        
        # Split into heads
        x_heads = x.view(B, self.num_heads, self.head_dim, T)
        
        head_outputs = []
        head_attentions = []
        head_classifications = []
        
        for i in range(self.num_heads):
            x_head = x_heads[:, i, :, :]  # (B, head_dim, T)
            
            att = torch.softmax(torch.tanh(self.att_heads[i](x_head)), dim=-1)
            cla = self.act(self.cla_heads[i](x_head))
            clip = torch.sum(att * cla, dim=-1)  # (B, out_ch)
            
            head_outputs.append(clip)
            head_attentions.append(att)
            head_classifications.append(cla)
        
        # Combine heads
        combined_output = torch.cat(head_outputs, dim=1)  # (B, out_ch * num_heads)
        final_output = self.combine(combined_output.unsqueeze(-1)).squeeze(-1)  # (B, out_ch)
        
        # Average attentions and classifications for visualization
        avg_att = torch.mean(torch.stack(head_attentions), dim=0)
        avg_cla = torch.mean(torch.stack(head_classifications), dim=0)
        
        return final_output, avg_att, avg_cla


# ---------- main model ----------------------------------------------
class BirdCLEFSEDModel(nn.Module):
    """Enhanced SED model with multiple resolution support and advanced features"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # taxonomy
        self.n_class = len(pd.read_csv(cfg.dir.taxonomy_csv))
        
        # Multi-resolution mel front-end
        self.image_size = getattr(cfg, 'image_size', 256)
        self.use_multi_resolution = getattr(cfg, 'use_multi_resolution', False)
        
        # Primary mel spectrogram
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.fs,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            norm="slaney",
            mel_scale="htk",
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        
        # Additional mel transforms for multi-resolution (if enabled)
        if self.use_multi_resolution:
            self.mel_high_res = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.fs,
                n_fft=cfg.n_fft * 2,  # Higher frequency resolution
                hop_length=cfg.hop_length // 2,  # Higher time resolution
                n_mels=cfg.n_mels,
                f_min=cfg.f_min,
                f_max=cfg.f_max,
                norm="slaney",
                mel_scale="htk",
            )
            self.in_channels = 2  # Two mel spectrograms
        else:
            self.in_channels = cfg.in_channels

        # Enhanced backbone with configurable input channels
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=self.in_channels,
            drop_rate=getattr(cfg, 'drop_rate', 0.2),
            drop_path_rate=getattr(cfg, 'drop_path_rate', 0.2),
        )
        
        # remove head and get feature dimension
        if hasattr(self.backbone, "classifier"):
            dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            dim = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, "")

        # Enhanced pooling strategies
        self.pool_type = getattr(cfg, 'pool_type', 'adaptive_avg')
        if self.pool_type == 'adaptive_avg':
            self.pool = nn.AdaptiveAvgPool2d((None, 1))  # keep time
        elif self.pool_type == 'adaptive_max':
            self.pool = nn.AdaptiveMaxPool2d((None, 1))
        elif self.pool_type == 'gem':
            # Generalized Mean Pooling
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d((None, 1))
        
        # Batch normalization for mel features
        self.bn0 = nn.BatchNorm2d(cfg.n_mels)
        init_bn(self.bn0)
        
        # Enhanced feature processing
        self.use_skip_connection = getattr(cfg, 'use_skip_connection', False)
        self.fc = nn.Linear(dim, dim, bias=True)
        
        if self.use_skip_connection:
            self.skip_fc = nn.Linear(dim, dim // 4, bias=True)
            self.combine_fc = nn.Linear(dim + dim // 4, dim, bias=True)
        
        # Multi-head attention block
        self.use_multi_head = getattr(cfg, 'use_multi_head_attention', False)
        if self.use_multi_head:
            self.att_block = MultiHeadAttBlockV2(dim, self.n_class, num_heads=4, act="linear")
        else:
            self.att_block = AttBlockV2(dim, self.n_class, act="linear")
        
        # Dropout for regularization
        self.dropout = nn.Dropout(getattr(cfg, 'final_dropout', 0.1))

    # ---------- helpers ------------------------------------------------
    @torch.no_grad()
    def _wav2spec(self, wav: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram(s)"""
        B = wav.size(0)
        
        # Ensure wav is (B, samples) - squeeze if (B, 1, samples)
        if wav.dim() == 3 and wav.size(1) == 1:
            wav = wav.squeeze(1)  # (B, samples)
        
        # Primary mel spectrogram
        spec = self.mel(wav)  # (B, n_mels, T)
        spec = self.db(spec)
        spec = (spec + 80) / 80  # 0-1
        
        if self.use_multi_resolution:
            # High resolution mel spectrogram
            spec_high = self.mel_high_res(wav)
            spec_high = self.db(spec_high)
            spec_high = (spec_high + 80) / 80
            
            # Resize to match primary spec time dimension
            if spec_high.size(-1) != spec.size(-1):
                # Add batch and channel dimensions for interpolation if needed
                if spec_high.dim() == 3:  # (B, F, T)
                    spec_high = spec_high.unsqueeze(1)  # (B, 1, F, T)
                
                spec_high = torch.nn.functional.interpolate(
                    spec_high, size=(spec.size(-2), spec.size(-1)),
                    mode='bilinear', align_corners=False
                )
                
                # Remove the channel dimension we added
                if spec_high.size(1) == 1:
                    spec_high = spec_high.squeeze(1)  # Back to (B, F, T)
            
            # Stack along channel dimension
            spec = torch.stack([spec, spec_high], dim=1)  # (B, 2, n_mels, T)
            spec = spec.view(B, 2, spec.size(-2), spec.size(-1))  # Ensure correct shape
        else:
            # Add channel dimension for single resolution
            spec = spec.unsqueeze(1)  # (B, 1, n_mels, T)
            
        return spec

    def _encode(self, spec: torch.Tensor):
        # BN & shape: (B,C,F,T) → encoder → (B,C,T)
        # Input spec should be (B, C, F, T)
        if spec.dim() == 3:
            # Add channel dimension if missing
            spec = spec.unsqueeze(1)  # (B, 1, F, T)
        elif spec.dim() == 5:
            # Squeeze out unnecessary dimension
            spec = spec.squeeze()
            if spec.dim() == 3:
                spec = spec.unsqueeze(1)
        
        B, C, F, T = spec.shape
        
        if C > 1 and self.use_multi_resolution:
            # For multi-resolution: reshape to (B*C, F, T) then add channel dim
            x = spec.view(B * C, F, T).unsqueeze(1)  # (B*C, 1, F, T)
        else:
            # For single resolution: ensure single channel
            x = spec[:, 0:1, :, :]  # (B, 1, F, T)
        
        # For BatchNorm2d(128), we need the F(128) dimension as the channel dimension
        # Current: (B, 1, F, T), need (B, F, 1, T) for BN
        x = x.transpose(1, 2)  # (B, F, 1, T)
        x = self.bn0(x)  # BatchNorm on F frequency bins
        x = x.transpose(1, 2)  # Back to (B, 1, F, T)
        
        # Pass through backbone
        x = self.backbone(x)  # (B, C_out, F', T')
        
        if x.dim() == 4:
            # Pool over frequency dimension, keep time
            x = self.pool(x)  # (B, C_out, F'_pooled, T')
            if x.size(2) > 1:  # If frequency dimension still exists
                x = torch.nn.functional.adaptive_avg_pool2d(x, (1, x.size(3)))  # (B, C_out, 1, T')
            x = x.squeeze(2)  # (B, C_out, T')
            
        if self.use_multi_resolution and C > 1:
            # Reshape back from multi-resolution processing
            _, C_out, T_out = x.shape
            x = x.view(B, C, C_out, T_out)
            # Average across resolution channels
            x = x.mean(dim=1)  # (B, C_out, T_out)
            
        return x

    # ---------- forward (train) ---------------------------------------
    def forward(self, wav: torch.Tensor):
        """
        Enhanced forward pass with multi-resolution support
        input  : (B, 1, samples) raw waveform 10 s
        output : clip-wise logit (B, C)
        """
        with torch.no_grad():
            spec = self._wav2spec(wav)

        feat = self._encode(spec)
        
        # Enhanced feature processing with skip connections
        feat_processed = torch.relu(self.fc(feat.transpose(1, 2))).transpose(1, 2)  # (B,C,T)
        
        if self.use_skip_connection:
            # Skip connection processing
            skip_feat = torch.relu(self.skip_fc(feat.transpose(1, 2))).transpose(1, 2)
            # Combine original and skip features
            combined_feat = torch.cat([feat_processed, skip_feat], dim=1)
            feat_processed = torch.relu(self.combine_fc(combined_feat.transpose(1, 2))).transpose(1, 2)
        
        # Apply dropout for regularization
        feat_processed = self.dropout(feat_processed.transpose(1, 2)).transpose(1, 2)
        
        # Attention block
        clip, _, _ = self.att_block(feat_processed)
        
        return clip  # Direct logits for BCEWithLogitsLoss

    # ---------- inference helper --------------------------------------
    def infer_rms_shift(self, wav, rms_crop_fn, shift_s=2.0):
        seg, idx = rms_crop_fn(
            wav, int(self.cfg.fs * self.cfg.train_duration), self.cfg.fs
        )
        logits_c = []
        for shift in (-shift_s, 0, shift_s):
            shift_samp = int(shift * self.cfg.fs)
            ss = max(0, idx + shift_samp)
            ee = ss + len(seg)
            seg2 = (
                wav[ss:ee] if ee <= len(wav) else np.pad(wav[ss:], (0, ee - len(wav)))
            )
            clip = torch.sigmoid(self.forward(seg2.unsqueeze(0))).cpu()
            logits_c.append(clip)
        return torch.mean(torch.stack(logits_c, 0), 0)
