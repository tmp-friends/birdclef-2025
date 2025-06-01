# augmentations_v2s.py
import random, math
import numpy as np
import torch
import torch.nn.functional as F

import torchaudio


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
        noise = np.random.randn(*x.shape) * noise_level

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
        x_power = np.mean(x**2)
        noise_power = x_power / (10 ** (snr / 10))
        noise = np.random.randn(*x.shape) * np.sqrt(noise_power)

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
        x_power = np.mean(x**2)
        noise_power = x_power / (10 ** (snr / 10))
        # Pink noise: 1/f noise
        N = x.shape[-1]
        uneven = N % 2
        X = np.random.randn(N // 2 + 1 + uneven) + 1j * np.random.randn(
            N // 2 + 1 + uneven
        )
        S = np.sqrt(np.arange(len(X)) + 1.0)
        y = (np.fft.irfft(X / S)).real
        y = y[:N]
        y = y / np.std(y)
        y = y * np.sqrt(noise_power)

        return x + y


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


class V2SAugment:
    def __init__(self, cfg):
        self.freq = _FreqMask(max_w=cfg.fm_w, p=cfg.fm_p)
        self.time = _TimeMask(max_w=cfg.tm_w, p=cfg.tm_p)
        self.gainbias = _RandGainBias(p=cfg.gb_p)
        self.resize = _RandResizeTime(p=cfg.rs_p)
        self.cutmix = _CutMixRect(p=cfg.cm_p, alpha=cfg.cm_alpha)

    def __call__(self, sample, pool=None):
        spec, target = sample["melspec"], sample["target"]
        spec = self.resize(spec)
        spec = self.freq(spec)
        spec = self.time(spec)
        spec = self.gainbias(spec)
        if pool is not None:  # pool は (spec,target) の list
            spec, target = self.cutmix(spec, target, pool)
        sample["melspec"] = spec
        sample["target"] = target

        return sample
