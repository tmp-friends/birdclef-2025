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
class SpecConfig:
    name: str
    fs: int
    window_size: int
    num_fft: int
    hop_length: int
    num_mels: int
    fmin: int
    fmax: int
    target_shape: int


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any]


@dataclass
class PreprocessConfig:
    dir: DirConfig
    spec: SpecConfig


@dataclass
class TrainConfig:
    dir: DirConfig
    spec: SpecConfig
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
    mixup_alpha: float
    aug_prob: float
    drop_rate: float
    drop_path_rate: float


@dataclass
class InferConfig:
    dir: DirConfig
    spec: SpecConfig
    model: ModelConfig
    seed: int
    device: str
    spectrogram_npy_path: str
    num_folds: int
    folds: list[int]
    model_dir: str
    uses_tta: bool
    tta_count: int
