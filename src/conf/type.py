from dataclasses import dataclass
from typing import Any


@dataclass
class DirConfig:
    train_audio_dir: str
    train_soundscapes_dir: str
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
class AugmentationConfig:
    rs_p: float  # Resize p
    sa_p: float  # SpecAug p
    fm_w: int
    fm_p: float  # FreqMask p
    tm_w: int
    tm_p: float  # TimeMask p
    ni_p: float  # NoiseInjection p
    gn_p: float  # GaussianNoise p
    pn_p: float  # PinkNoise p
    gb_p: float  # Gain/Bias p
    cm_p: float  # CutMix p
    cm_alpha: float


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
    augmentation: AugmentationConfig
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
    early_stopping: int
    mixup_alpha: float
    aug_prob: float
    drop_rate: float
    drop_path_rate: float
    pl_threshold: float
    pl_lr_scale: float
    pl_epochs: int
    use_holdout: bool
    valid_ratio: float
    full_train_after_pl: bool
    pl_success_threshold: float


@dataclass
class TrainAugConfig:
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
    early_stopping: int
    mixup_alpha: float
    fm_w: int
    fm_p: float
    tm_w: int
    tm_p: float
    gb_p: float
    rs_p: float
    cm_p: float
    cm_alpha: float
    drop_rate: float
    drop_path_rate: float
    pl_threshold: float
    pl_lr_scale: float
    pl_epochs: int
    use_holdout: bool
    valid_ratio: float
    full_train_after_pl: bool
    pl_success_threshold: float
    augmentation: AugmentationConfig


@dataclass
class InferConfig:
    dir: DirConfig
    spec: SpecConfig
    model: ModelConfig
    seed: int
    device: str
    num_folds: int
    folds: list[int]
    model_dir: str
    uses_tta: bool
    tta_count: int
