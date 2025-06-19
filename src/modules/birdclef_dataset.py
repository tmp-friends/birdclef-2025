import numpy as np
import pandas as pd
import librosa
import os
from utils.sampling import rms_crop, rms_crop_shift
import torch
from torch.utils.data import Dataset
import warnings

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

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

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


class BirdCLEFDatasetWave(Dataset):
    """10 s の波形を返す Dataset（SED 用）"""

    def __init__(self, cfg, df):
        self.cfg = cfg
        self.df = df
        self.sr = cfg.spec.fs
        self.seg = int(cfg.train_duration * self.sr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav, _ = librosa.load(row.filepath, sr=self.sr, mono=True)
        if len(wav) < self.seg:  # circular pad
            k = int(np.ceil(self.seg / len(wav)))
            wav = np.tile(wav, k)[: self.seg]
        else:
            # RMS crop + random-shift
            wav, _ = rms_crop_shift(wav, self.seg, self.sr, shift_s=2.0)
        y = np.zeros(len(row.target))  # dummy, multi-hot 作るならここ
        return {
            "wave": torch.as_tensor(wav),
            "target": torch.tensor(y, dtype=torch.float32),
        }


class BirdCLEFSEDDataset(Dataset):
    """Base SED Dataset class for BirdCLEF with audio loading and label encoding"""
    
    def __init__(
        self,
        cfg: TrainConfig,
        df: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
        mode: str = "train",
    ):
        super().__init__()
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.taxonomy_df = taxonomy_df
        self.mode = mode
        
        # Sample rate and segment duration
        self.sr = cfg.fs
        self.seg_duration = cfg.train_duration  # in seconds
        self.seg_samples = int(self.seg_duration * self.sr)
        
        # Label mapping
        self.species_ids = taxonomy_df["primary_label"].tolist()
        self.label2ix = {lb: ix for ix, lb in enumerate(self.species_ids)}
        self.num_classes = len(self.species_ids)
        
    def __len__(self):
        return len(self.df)
    
    def load_audio(self, filepath, start_time=None, end_time=None):
        """Load audio with optional time bounds"""
        try:
            if start_time is not None and end_time is not None:
                # Load specific segment for pseudo labels
                duration = end_time - start_time
                wav, _ = librosa.load(
                    filepath, 
                    sr=self.sr, 
                    mono=True,
                    offset=start_time,
                    duration=duration
                )
            else:
                # Load entire file
                wav, _ = librosa.load(filepath, sr=self.sr, mono=True)
                
            # Ensure we have exactly seg_samples
            if len(wav) < self.seg_samples:
                # Circular pad if too short
                k = int(np.ceil(self.seg_samples / len(wav)))
                wav = np.tile(wav, k)[:self.seg_samples]
            elif len(wav) > self.seg_samples:
                # Crop if too long
                if self.mode == "train":
                    # RMS crop + random shift for training
                    wav, _ = rms_crop_shift(wav, self.seg_samples, self.sr, shift_s=2.0)
                else:
                    # Center crop for validation
                    start = (len(wav) - self.seg_samples) // 2
                    wav = wav[start:start + self.seg_samples]
                    
        except Exception as e:
            warnings.warn(f"Error loading audio from {filepath}: {e}")
            wav = np.zeros(self.seg_samples, dtype=np.float32)
            
        return wav
    
    def encode_labels(self, primary_label, secondary_labels=None):
        """Encode labels to multi-hot vector"""
        target = np.zeros(self.num_classes, dtype=np.float32)
        
        # Primary label
        if primary_label in self.label2ix:
            target[self.label2ix[primary_label]] = 1.0
            
        # Secondary labels if available
        if secondary_labels is not None and pd.notna(secondary_labels):
            if isinstance(secondary_labels, str):
                secondary_labels = eval(secondary_labels)
            for label in secondary_labels:
                if label in self.label2ix:
                    target[self.label2ix[label]] = 1.0
                    
        return target
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio
        wav = self.load_audio(row["filepath"])
        
        # Encode labels
        target = self.encode_labels(
            row["primary_label"],
            row.get("secondary_labels", None)
        )
        
        return {
            "wave": torch.tensor(wav, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "filename": row.get("filename", row.get("filepath", "")),
        }


class BirdCLEFPseudoLabelDataset(BirdCLEFSEDDataset):
    """Dataset for pseudo label training that handles both real and pseudo labeled samples"""
    
    def __init__(
        self,
        cfg: TrainConfig,
        df: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
        mode: str = "train",
        pseudo_label_smoothing: float = 0.1,
        pseudo_confidence_threshold: float = 0.5,
        pseudo_weight: float = 0.5,
    ):
        super().__init__(cfg, df, taxonomy_df, mode)
        
        # Pseudo label specific parameters
        self.pseudo_label_smoothing = pseudo_label_smoothing
        self.pseudo_confidence_threshold = pseudo_confidence_threshold
        self.pseudo_weight = pseudo_weight
        
        # Validate dataframe has required columns
        required_cols = ["is_pseudo"]
        pseudo_cols = ["confidence", "file_path", "start_time", "end_time"]
        
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
        # Check pseudo samples have required columns
        pseudo_mask = self.df["is_pseudo"]
        if pseudo_mask.any():
            pseudo_df = self.df[pseudo_mask]
            for col in pseudo_cols:
                if col not in pseudo_df.columns:
                    raise ValueError(f"Pseudo samples must have column: {col}")
                    
        # Separate real and pseudo indices for potential balanced sampling
        self.real_indices = self.df[~self.df["is_pseudo"]].index.tolist()
        self.pseudo_indices = self.df[self.df["is_pseudo"]].index.tolist()
        
        print(f"Dataset initialized with {len(self.real_indices)} real samples "
              f"and {len(self.pseudo_indices)} pseudo samples")
        
    def apply_label_smoothing(self, target, smoothing_factor):
        """Apply label smoothing to target"""
        if smoothing_factor > 0:
            target = target * (1 - smoothing_factor) + smoothing_factor / self.num_classes
        return target
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        is_pseudo = row["is_pseudo"]
        
        # Load audio
        if is_pseudo:
            # For pseudo samples, use file_path and time bounds
            wav = self.load_audio(
                row["file_path"], 
                start_time=row["start_time"],
                end_time=row["end_time"]
            )
        else:
            # For real samples, construct filepath from filename
            filename = row["filename"]
            filepath = os.path.join(self.cfg.dir.train_audio_dir, filename)
            wav = self.load_audio(filepath)
            
        # Encode labels
        target = self.encode_labels(
            row["primary_label"],
            row.get("secondary_labels", None)
        )
        
        # Apply modifications for pseudo labels
        if is_pseudo:
            confidence = row.get("confidence", 1.0)
            
            # Apply confidence weighting if confidence is below threshold
            if confidence < self.pseudo_confidence_threshold:
                target = target * confidence
                
            # Apply label smoothing to pseudo labels
            if self.pseudo_label_smoothing > 0:
                target = self.apply_label_smoothing(target, self.pseudo_label_smoothing)
                
            # Apply pseudo weight to reduce impact of pseudo labels
            target = target * self.pseudo_weight
            
        # For training, we might want to apply some label smoothing to real labels too
        elif self.mode == "train" and hasattr(self.cfg, "label_smoothing"):
            if self.cfg.label_smoothing > 0:
                target = self.apply_label_smoothing(target, self.cfg.label_smoothing)
                
        return {
            "wave": torch.tensor(wav, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "filename": row.get("filename", row.get("filepath", row.get("file_path", ""))),
            "is_pseudo": is_pseudo,
            "confidence": row.get("confidence", 1.0) if is_pseudo else 1.0,
        }
        
    def get_balanced_sampler(self, real_weight=1.0, pseudo_weight=1.0):
        """Create a weighted sampler to balance real and pseudo samples"""
        from torch.utils.data import WeightedRandomSampler
        
        weights = []
        for idx in range(len(self.df)):
            if self.df.iloc[idx]["is_pseudo"]:
                weights.append(pseudo_weight)
            else:
                weights.append(real_weight)
                
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
