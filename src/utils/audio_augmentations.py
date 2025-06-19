"""
Advanced audio augmentations for BirdCLEF 2025 SED model
Based on successful approaches from BirdCLEF 2023 2nd place solution
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple

try:
    import audiomentations as AA
    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False
    print("Warning: audiomentations not available. Install with: pip install audiomentations")


class AdvancedAudioAugmentation:
    """
    Advanced audio augmentation pipeline combining audiomentations and custom transforms
    """
    
    def __init__(
        self,
        sample_rate: int = 32000,
        p_audio: float = 0.8,  # Probability of applying audio augmentations
        p_spec: float = 0.5,   # Probability of applying spectrogram augmentations
        max_noise_gain: float = 0.5,
        background_noise_dir: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self.p_audio = p_audio
        self.p_spec = p_spec
        
        # Audio augmentations using audiomentations
        if AUDIOMENTATIONS_AVAILABLE:
            self.audio_transforms = AA.Compose([
                # Gain and volume adjustments
                AA.Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.7),
                
                # Time-based transforms
                AA.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
                AA.PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
                
                # Noise injection
                AA.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                AA.AddGaussianSNR(min_snr_in_db=5.0, max_snr_in_db=20.0, p=0.3),
                
                # Filtering effects
                AA.BandPassFilter(min_frequency_fraction=0.1, max_frequency_fraction=0.9, p=0.3),
                AA.HighPassFilter(min_frequency_fraction=0.05, max_frequency_fraction=0.2, p=0.2),
                AA.LowPassFilter(min_frequency_fraction=0.7, max_frequency_fraction=0.95, p=0.2),
                
                # Room simulation
                AA.RoomImpulseResponse(p=0.2) if background_noise_dir else AA.Gain(0, 0, p=0),
                
                # Compression and limiting
                AA.Limiter(min_threshold_db=-16, max_threshold_db=-2, p=0.3),
                
                # Time masking
                AA.TimeMask(min_band_part=0.05, max_band_part=0.15, p=0.4),
                
            ], p=p_audio)
        else:
            self.audio_transforms = None
        
        # Custom spectrogram augmentations
        self.spec_augmentations = SpecAugment()
    
    def __call__(
        self, 
        audio: np.ndarray, 
        spectrogram: Optional[torch.Tensor] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
        """
        Apply augmentations to audio and/or spectrogram
        
        Args:
            audio: Input audio waveform (numpy array)
            spectrogram: Optional input spectrogram (torch tensor)
            
        Returns:
            Augmented audio (and spectrogram if provided)
        """
        # Apply audio augmentations
        if self.audio_transforms is not None and random.random() < self.p_audio:
            try:
                audio = self.audio_transforms(samples=audio, sample_rate=self.sample_rate)
            except Exception as e:
                print(f"Audio augmentation failed: {e}")
        
        # Apply spectrogram augmentations if provided
        if spectrogram is not None and random.random() < self.p_spec:
            spectrogram = self.spec_augmentations(spectrogram)
            return audio, spectrogram
        
        return audio if spectrogram is None else (audio, spectrogram)


class SpecAugment:
    """
    SpecAugment implementation for mel spectrograms
    Based on the paper: "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    """
    
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        freq_mask_num: int = 2,
        time_mask_num: int = 2,
        mask_value: float = 0.0,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.freq_mask_num = freq_mask_num
        self.time_mask_num = time_mask_num
        self.mask_value = mask_value
    
    def freq_mask(self, spec: torch.Tensor, num_masks: int = 1) -> torch.Tensor:
        """Apply frequency masking"""
        spec = spec.clone()
        num_mel_channels = spec.shape[-2]
        
        for _ in range(num_masks):
            f = random.randint(0, self.freq_mask_param)
            f_zero = random.randint(0, num_mel_channels - f)
            spec[..., f_zero:f_zero + f, :] = self.mask_value
        
        return spec
    
    def time_mask(self, spec: torch.Tensor, num_masks: int = 1) -> torch.Tensor:
        """Apply time masking"""
        spec = spec.clone()
        tau = spec.shape[-1]
        
        for _ in range(num_masks):
            t = random.randint(0, min(self.time_mask_param, tau))
            t_zero = random.randint(0, tau - t)
            spec[..., t_zero:t_zero + t] = self.mask_value
        
        return spec
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment"""
        spec = self.freq_mask(spec, self.freq_mask_num)
        spec = self.time_mask(spec, self.time_mask_num)
        return spec


class MixupCutmix:
    """
    Combined Mixup and CutMix augmentation for audio spectrograms
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob_mixup: float = 0.5,
        prob_cutmix: float = 0.5,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob_mixup = prob_mixup
        self.prob_cutmix = prob_cutmix
    
    def mixup(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply Mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation"""
        lam = np.random.beta(alpha, alpha)
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Get dimensions
        if len(x.shape) == 4:  # (B, C, H, W)
            _, _, H, W = x.shape
        else:  # (B, H, W) or (B, C, H, W)
            H, W = x.shape[-2], x.shape[-1]
        
        # Generate random box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        mixed_x = x.clone()
        mixed_x[..., bby1:bby2, bbx1:bbx2] = x[index][..., bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def __call__(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
        """
        Apply either Mixup or CutMix
        
        Returns:
            mixed_x, y_a, y_b, lam, aug_type
        """
        # Decide which augmentation to apply
        use_mixup = random.random() < self.prob_mixup
        use_cutmix = random.random() < self.prob_cutmix
        
        if use_mixup and not use_cutmix:
            mixed_x, y_a, y_b, lam = self.mixup(x, y, self.mixup_alpha)
            return mixed_x, y_a, y_b, lam, "mixup"
        elif use_cutmix and not use_mixup:
            mixed_x, y_a, y_b, lam = self.cutmix(x, y, self.cutmix_alpha)
            return mixed_x, y_a, y_b, lam, "cutmix"
        elif use_mixup and use_cutmix:
            # Randomly choose one
            if random.random() < 0.5:
                mixed_x, y_a, y_b, lam = self.mixup(x, y, self.mixup_alpha)
                return mixed_x, y_a, y_b, lam, "mixup"
            else:
                mixed_x, y_a, y_b, lam = self.cutmix(x, y, self.cutmix_alpha)
                return mixed_x, y_a, y_b, lam, "cutmix"
        else:
            # No augmentation
            return x, y, y, 1.0, "none"


class SEDAugmentationPipeline:
    """
    Complete augmentation pipeline for SED training
    Combines audio-level and spectrogram-level augmentations
    """
    
    def __init__(
        self,
        sample_rate: int = 32000,
        use_audio_aug: bool = True,
        use_spec_aug: bool = True,
        use_mixup_cutmix: bool = True,
        audio_aug_prob: float = 0.8,
        spec_aug_prob: float = 0.6,
        mixup_cutmix_prob: float = 0.7,
    ):
        self.use_audio_aug = use_audio_aug
        self.use_spec_aug = use_spec_aug
        self.use_mixup_cutmix = use_mixup_cutmix
        
        # Initialize augmentation modules
        if use_audio_aug:
            self.audio_aug = AdvancedAudioAugmentation(
                sample_rate=sample_rate,
                p_audio=audio_aug_prob,
                p_spec=0.0  # We handle spec aug separately
            )
        
        if use_spec_aug:
            self.spec_aug = SpecAugment()
            self.spec_aug_prob = spec_aug_prob
        
        if use_mixup_cutmix:
            self.mixup_cutmix = MixupCutmix()
            self.mixup_cutmix_prob = mixup_cutmix_prob
    
    def apply_audio_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio-level augmentations"""
        if self.use_audio_aug:
            return self.audio_aug(audio)
        return audio
    
    def apply_spec_augmentation(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply spectrogram-level augmentations"""
        if self.use_spec_aug and random.random() < self.spec_aug_prob:
            return self.spec_aug(spec)
        return spec
    
    def apply_mixup_cutmix(
        self, 
        spec: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
        """Apply Mixup/CutMix augmentations"""
        if self.use_mixup_cutmix and random.random() < self.mixup_cutmix_prob:
            return self.mixup_cutmix(spec, labels)
        return spec, labels, labels, 1.0, "none"


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)