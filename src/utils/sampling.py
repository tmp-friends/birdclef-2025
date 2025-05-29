import numpy as np


def random_crop(wav: np.ndarray, seg_len: int, rng: np.random.Generator):
    """元配列 wav から長さ seg_len をランダムに切り出し
    (cropped_wave, crop_start_idx) を返す
    """
    if len(wav) <= seg_len:
        pad = seg_len - len(wav)
        y_pad = np.pad(wav, (0, pad), mode="reflect")
        return y_pad, 0  # start=0

    start = rng.integers(0, len(wav) - seg_len)

    return wav[start : start + seg_len].copy(), start


def rms_crop(
    wav: np.ndarray, seg_len: int, sr: int, stride_s: float = 1.0
) -> np.ndarray:
    """
    RMS(=エネルギー) が最も⼤きい窓を返す。

    Parameters
    ----------
    wav        : np.ndarray      1-D 波形
    seg_len    : int             必要サンプル数 (= cfg.spec.window_size * sr)
    sr         : int             サンプリングレート
    stride_s   : float           何秒ごとに RMS を計算するか (default=1.0s)
    """
    if len(wav) <= seg_len:  # ⻑さ不⾜なら円環 pad
        k = int(np.ceil(seg_len / len(wav)))
        return np.tile(wav, k)[:seg_len], 0

    stride = int(sr * stride_s)
    max_rms, max_idx = 0.0, 0
    for s in range(0, len(wav) - seg_len + 1, stride):
        win = wav[s : s + seg_len]
        e = np.sqrt(np.mean(win**2))  # RMS
        if e > max_rms:
            max_rms, max_idx = e, s

    return wav[max_idx : max_idx + seg_len], max_idx
