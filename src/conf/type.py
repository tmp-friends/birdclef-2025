from dataclasses import dataclass
from typing import Any


@dataclass
class DirConfig:
    train_audio_dir: str
    train_csv: str
    taxonomy_csv: str
    submission_csv: str
    test_soundscapes_dir: str


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any]


@dataclass
class PreprocessConfig:
    dir: DirConfig
    fs: int
    num_fft: int
    hop_length: int
    num_mels: int
    fmin: int
    fmax: int
    target_duration: float
    target_w: int
    target_h: int


@dataclass
class TrainConfig:
    dir: DirConfig
    model: ModelConfig
    num_folds: int
    fold: int
    num_epochs: int
    train_batch_size: int
    valid_batch_size: int
    scheduler: str
    lr: float


@dataclass
class InferConfig:
    dir: DirConfig
    model: ModelConfig
    num_folds: int
    model_dir: str
    valid_batch_size: int
