import random, math
import numpy as np
import torch
import torch.nn.functional as F

import torchaudio


class _FreqMask:
    def __init__(self, max_w=24, p=0.5):
        self.max_w, self.p = max_w, p

    def __call__(self, x):
        if random.random() > self.p:
            return x
        w = random.randint(1, self.max_w)
        f0 = random.randint(0, x.size(-2) - w)
        x[..., f0 : f0 + w, :] = 0.0

        return x


class _TimeMask:
    def __init__(self, max_w=32, p=0.5):
        self.max_w, self.p = max_w, p

    def __call__(self, x):
        if random.random() > self.p:
            return x
        w = random.randint(1, self.max_w)
        t0 = random.randint(0, x.size(-1) - w)
        x[..., :, t0 : t0 + w] = 0.0

        return x


class _RandGainBias:
    def __init__(self, gain=(0.8, 1.2), bias=(-0.1, 0.1), p=0.5):
        self.gain, self.bias, self.p = gain, bias, p

    def __call__(self, x):
        if random.random() > self.p:
            return x
        g = random.uniform(*self.gain)
        b = random.uniform(*self.bias)

        return torch.clamp(x * g + b, 0.0, 1.0)


class _CutMixRect:
    """矩形を切り貼りしながらラベルも線形合成"""

    def __init__(self, p=0.5, alpha=1.0):
        self.p, self.alpha = p, alpha

    def __call__(self, spec, target, pool):
        if random.random() > self.p or len(pool) == 0:
            return spec, target
        lam = np.random.beta(self.alpha, self.alpha)
        h, w = spec.size(-2), spec.size(-1)
        rw, rh = int(w * math.sqrt(1.0 - lam)), int(h * math.sqrt(1.0 - lam))
        rx, ry = random.randint(0, w - rw), random.randint(0, h - rh)

        j = random.randint(0, len(pool) - 1)
        spec2, tgt2 = pool[j]
        spec2 = spec2.to(spec.device)

        spec[..., ry : ry + rh, rx : rx + rw] = spec2[..., ry : ry + rh, rx : rx + rw]
        lam = 1 - (rw * rh) / (w * h)
        target = target * lam + tgt2 * (1 - lam)

        return spec, target


class _RandResizeTime:
    """
    時軸方向にランダムリサイズして余剰は Crop，不足は右パディングする
    x : Tensor  … shape =  (C, H, W)  or  (B, C, H, W)
    """

    def __init__(self, scale=(0.8, 1.2), p=0.5):
        self.scale = scale
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        # ――― 入力の次元を統一（interpolate は 4-D を要求） ――― #
        squeeze_batch = False
        if x.dim() == 3:  # (C, H, W) → (1, C, H, W)
            x = x.unsqueeze(0)
            squeeze_batch = True

        B, C, H, W = x.shape
        r = random.uniform(*self.scale)
        new_W = max(1, int(W * r))

        # resize
        x = F.interpolate(x, size=(H, new_W), mode="bilinear", align_corners=False)

        # crop / pad to original width
        if new_W > W:  # 右側を切り捨て
            x = x[..., :W]
        elif new_W < W:  # 右側ゼロパディング
            pad = W - new_W
            x = F.pad(x, (0, pad))

        # 元が 3-D なら戻す
        if squeeze_batch:
            x = x.squeeze(0)

        return x


class _AddBackgroundNoise:
    def __init__(self, noise_paths, min_snr_in_db=3.0, max_snr_in_db=30.0, p=0.5):
        self.noise_paths = noise_paths
        self.min_snr = min_snr_in_db
        self.max_snr = max_snr_in_db
        self.p = p

    def __call__(self, x):
        if random.random() > self.p or not self.noise_paths:
            return x
        noise_path = random.choice(self.noise_paths)
        noise, _ = torchaudio.load(noise_path)
        if noise.shape[1] < x.shape[-1]:
            # pad noise
            repeat = int(np.ceil(x.shape[-1] / noise.shape[1]))
            noise = noise.repeat(1, repeat)[:, : x.shape[-1]]
        else:
            start = random.randint(0, noise.shape[1] - x.shape[-1])
            noise = noise[:, start : start + x.shape[-1]]
        snr = random.uniform(self.min_snr, self.max_snr)
        x_power = x.pow(2).mean()
        n_power = noise.pow(2).mean()
        factor = (x_power / (10 ** (snr / 10)) / (n_power + 1e-8)).sqrt()
        x = x + noise * factor

        return x


class _NoiseInjection:
    def __init__(self, p=0.5, max_noise_level=0.04):
        self.p = p
        self.max_noise_level = max_noise_level

    def __call__(self, x):
        if random.random() > self.p:
            return x
        noise_level = random.uniform(0, self.max_noise_level)
        noise = torch.randn_like(x) * noise_level
        return x + noise


class _GaussianNoiseSNR:
    def __init__(self, p=0.5, min_snr=3.0, max_snr=30.0):
        self.p = p
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __call__(self, x):
        if random.random() > self.p:
            return x
        snr = random.uniform(self.min_snr, self.max_snr)
        x_power = torch.mean(x**2)
        noise_power = x_power / (10 ** (snr / 10))
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise


class _PinkNoiseSNR:
    def __init__(self, p=0.5, min_snr=3.0, max_snr=30.0):
        self.p = p
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __call__(self, x):
        if random.random() > self.p:
            return x
        snr = random.uniform(self.min_snr, self.max_snr)
        x_power = torch.mean(x**2)
        noise_power = x_power / (10 ** (snr / 10))

        # Pink noise using torch operations
        device = x.device
        N = x.shape[-1]
        uneven = N % 2

        # Create complex random tensor
        real_part = torch.randn(N // 2 + 1 + uneven, device=device)
        imag_part = torch.randn(N // 2 + 1 + uneven, device=device)
        X_complex = torch.complex(real_part, imag_part)

        # Apply 1/f filter
        S = torch.sqrt(torch.arange(len(X_complex), device=device) + 1.0)
        X_filtered = X_complex / S

        # Inverse FFT (use torch.fft.irfft)
        y = torch.fft.irfft(X_filtered, n=N)

        # Normalize and scale to desired power
        y = y / torch.std(y)
        y = y * torch.sqrt(noise_power)

        # Reshape to match input dimensions
        if len(x.shape) > 1:
            for _ in range(len(x.shape) - 1):
                y = y.unsqueeze(0)
            # Expand to match dimensions
            y = y.expand_as(x)

        return x + y


class V2SAugment:
    def __init__(self, cfg):
        self.freq = _FreqMask(max_w=cfg.fm_w, p=cfg.fm_p)
        self.time = _TimeMask(max_w=cfg.tm_w, p=cfg.tm_p)
        self.gainbias = _RandGainBias(p=cfg.gb_p)
        self.resize = _RandResizeTime(p=cfg.rs_p)
        self.cutmix = _CutMixRect(p=cfg.cm_p, alpha=cfg.cm_alpha)
        self.noise_injection = _NoiseInjection(p=0.3)
        self.gaussian_noise = _GaussianNoiseSNR(p=0.3)
        self.pink_noise = _PinkNoiseSNR(p=0.3)

    def __call__(self, sample, pool=None):
        spec, target = sample["melspec"], sample["target"]

        # 1. リサイズ（時間軸方向の変形）
        spec = self.resize(spec)

        # 2. マスキング操作（周波数・時間領域）
        spec = self.freq(spec)
        spec = self.time(spec)

        # 3. ノイズ追加（マスク後、ゲイン調整前）
        noise_type = random.randint(0, 2)
        if noise_type == 0:
            spec = self.noise_injection(spec)
        elif noise_type == 1:
            spec = self.gaussian_noise(spec)
        else:
            spec = self.pink_noise(spec)

        # 4. ゲイン/バイアス調整（ノイズ追加後に振幅調整）
        spec = self.gainbias(spec)

        # 5. CutMix（最後に他サンプルとの混合）
        if pool is not None:
            spec, target = self.cutmix(spec, target, pool)

        sample["melspec"] = spec
        sample["target"] = target

        return sample
