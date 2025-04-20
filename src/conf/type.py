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
    seed: int
    device: str
    spectrogram_npy_path: str
    num_folds: int
    fold: int
    num_epochs: int
    train_batch_size: int
    valid_batch_size: int
    scheduler: str
    optimizer: str
    lr: float
    weight_decay: float
    T_max: int
    min_lr: float
    criterion: str
    target_w: int
    target_h: int
    mixup_alpha: float
    aug_prob: float


@dataclass
class InferConfig:
    dir: DirConfig
    model: ModelConfig
    seed: int
    device: str
    spectrogram_npy_path: str
    fs: int
    window_size: int
    uses_tta: bool
    tta_count: int
    num_folds: int
    model_dir: str
    valid_batch_size: int
    target_w: int
    target_h: int
