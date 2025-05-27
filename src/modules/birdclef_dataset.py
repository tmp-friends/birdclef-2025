import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from conf.type import TrainConfig


class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
        df: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
        spectrograms: dict | None = None,
        mode: str = "train",
    ):
        super().__init__()
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.mode = mode

        # ── label map ──────────────────────────────────────────────
        self.species_ids = taxonomy_df["primary_label"].tolist()
        self.label2ix = {lb: ix for ix, lb in enumerate(self.species_ids)}
        self.num_classes = len(self.species_ids)

        # ── パス＆サンプル名補完 ──────────────────────────────────
        if "samplename" not in self.df.columns:
            self.df["samplename"] = self.df.filename.apply(
                lambda x: (
                    f"{x.split('/')[0]}-{x.split('/')[-1].split('.')[0]}"
                    if "/" in x
                    else x.split(".")[0]
                )
            )

        # ── キャッシュ ──────────────────────────────────────────
        self.spec_cache = spectrograms or {}
        self.has_melspec_col = "melspec" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row["samplename"]

        if samplename in self.spec_cache:
            spec = self.spec_cache[samplename]
        # 見つからなければ全 0（学習は継続可）
        else:
            spec = np.zeros(self.cfg.spec.target_shape, dtype=np.float32)
            if self.mode == "train":
                print(f"[WARN] spectrogram for {samplename} not found; using zeros")

        # ----- torch tensor & optional aug --------------------------------
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self._apply_spec_augmentations(spec)

        # ----- one-hot label ---------------------------------------------
        target = np.zeros(self.num_classes, dtype=np.float32)
        if row["primary_label"] in self.label2ix:
            target[self.label2ix[row["primary_label"]]] = 1.0

        # secondary labels (あれば)
        if "secondary_labels" in row and pd.notna(row["secondary_labels"]):
            sec = row["secondary_labels"]
            sec = eval(sec) if isinstance(sec, str) else sec
            for lb in sec:
                if lb in self.label2ix:
                    target[self.label2ix[lb]] = 1.0

        return {
            "melspec": spec,
            "target": torch.tensor(target, dtype=torch.float32),
            "filename": row["filename"],
        }

    def _apply_spec_augmentations(self, spec: torch.Tensor) -> torch.Tensor:
        """簡易 SpecAug (時間・周波数マスク+輝度+ガウシアンノイズ)"""
        C, H, W = spec.shape  # (1,H,W)
        # Time mask
        if random.random() < 0.5:
            w = random.randint(4, 32)
            s = random.randint(0, W - w)
            spec[:, :, s : s + w] = 0
        # Freq mask
        if random.random() < 0.5:
            h = random.randint(4, 32)
            s = random.randint(0, H - h)
            spec[:, s : s + h, :] = 0
        # Brightness / Contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec.mul_(gain).add_(bias).clamp_(0, 1)
        # Gaussian noise
        if random.random() < 0.5:
            std = random.uniform(0.01, 0.05)  # ノイズ強度は適宜調整
            noise = torch.normal(mean=0.0, std=std, size=spec.shape, device=spec.device)
            spec.add_(noise).clamp_(0, 1)
        # Pink noise
        if random.random() < 0.5:
            std = random.uniform(0.01, 0.05)
            # Pink noise: 1/f noise
            pink = torch.randn_like(spec)
            # FFT
            f = torch.fft.rfft(pink, dim=-1)
            freqs = torch.fft.rfftfreq(spec.shape[-1])
            # Avoid division by zero
            freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
            f = f / (freqs[None, None, :] ** 0.5)
            pink = torch.fft.irfft(f, n=spec.shape[-1], dim=-1)
            pink = pink * std
            spec.add_(pink).clamp_(0, 1)

        return spec
