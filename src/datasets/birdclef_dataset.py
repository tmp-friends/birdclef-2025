import random

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
        spectrograms=None,
        mode="train",
    ):
        super().__init__()

        self.cfg = cfg
        self.df = df
        self.spectrograms = spectrograms
        self.mode = mode

        self.species_ids = taxonomy_df["primary_label"].tolist()
        self.num_classes = len(self.species_ids)
        self.label2ix = {label: ix for ix, label in enumerate(self.species_ids)}

        if "filepath" not in self.df.columns:
            self.df["filepath"] = self.cfg.dir.train_audio_dir + "/" + self.df.filename

        if "samplename" not in self.df.columns:
            self.df["samplename"] = self.df.filename.map(
                lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0]
            )

        sample_names = set(self.df["samplename"])
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(
                f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        row = self.df.iloc[ix]
        samplename = row["samplename"]
        spec = self.spectrograms[samplename]

        if spec is None:
            spec = np.zeros(
                (self.cfg.spec.target_w, self.cfg.spec.target_h), dtype=np.float32
            )
            if self.mode == "train":  # 学習時のみ警告を出す
                print(
                    f"Warning: Spectrogram for {samplename} not found and could not be generated"
                )

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(
            0
        )  # チャネル次元の追加

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self._apply_spec_augmentations(spec)

        target = self._encode_label(row["primary_label"])

        if "secondary_labels" in row and row["secondary_labels"] not in [
            [""],
            None,
            np.nan,
        ]:
            if isinstance(row["secondary_labels"], str):
                secondary_labels = eval(row["secondary_labels"])
            else:
                secondary_labels = row["secondary_labels"]

            for label in secondary_labels:
                if label in self.label2ix:
                    target[self.label2ix[label]] = 1.0

        return {
            "melspec": spec,
            "target": torch.tensor(target, dtype=torch.float32),
            "filename": row["filename"],
        }

    def _apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram"""
        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start : start + width] = 0

        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start : start + height, :] = 0

        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)

        return spec

    def _encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label2ix:
            target[self.label2ix[label]] = 1.0
        return target
