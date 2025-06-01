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
    if len(wav) <= seg_len:  # ⻑さ不⾜なら circular padding
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


def rms_crop_shift(
    wav: np.ndarray,
    seg_len: int,
    sr: int,
    stride_s: float = 1.0,
    shift_s: float = 5.0,
) -> tuple[np.ndarray, int]:
    """RMS が最大の窓を切り出し、さらに Random Shift で開始位置を変える

    Args:
        wav (np.ndarray): 1-D 波形
        seg_len (int): 必要サンプル数 (= cfg.spec.window_size * sr)
        sr (int): サンプリングレート
        stride_s (float): 何秒ごとに RMS を計算するか (default=1.0s)
    Returns:
        tuple[np.ndarray, int]: 切り出した波形と開始位置のインデックス
    """
    start_idx = _rms_pick_start(wav, seg_len, sr, stride_s)
    start_idx = _random_shift_start(start_idx, len(wav), sr, shift_s=1.0)
    cropped = _crop_with_wrap(wav, start_idx, seg_len)

    return cropped, start_idx


def _rms_pick_start(
    wav: np.ndarray,
    seg_len: int,
    sr: int,
    stride_s: float = 1.0,
) -> int:
    """RMS が最大の（開始）位置を返す

    Args:
        wav (np.ndarray): 1-D 波形
        seg_len (int): 必要サンプル数 (= cfg.spec.window_size * sr)
        sr (int): サンプリングレート
        stride_s (float): 何秒ごとに RMS を計算するか (default=1.0s)
    Returns:
        int: 最大 RMS の開始位置のインデックス
    """
    if len(wav) <= seg_len:
        return 0  # 長さ不⾜なら先頭

    stride = int(sr * stride_s)  # stride_s 秒ごとに計算
    max_rms, max_idx = 0.0, 0
    for s in range(0, len(wav) - seg_len + 1, stride):
        win = wav[s : s + seg_len]
        e = np.sqrt(np.mean(win**2))  # RMS
        if e > max_rms:
            max_rms, max_idx = e, s

    return max_idx


def _random_shift_start(
    start_idx: int,
    wav_len: int,
    sr: int,
    shift_s: float,
) -> int:
    """開始 idx を Random Shift 後の位置に変換

    Args:
        start_idx (int): 元の開始位置のインデックス
        wav_len (int): 波形の長さ
        sr (int): サンプリングレート
        shift_s (float): シフトする秒数
    Returns:
        int: シフト後の開始位置のインデックス
    """
    if shift_s <= 0:
        return start_idx

    rng = np.random.default_rng()
    delta = rng.integers(-int(sr * shift_s), int(sr * shift_s) + 1)
    new_start = start_idx + delta
    new_start %= wav_len  # wrap around

    return int(new_start)


def _crop_with_wrap(
    wav: np.ndarray,
    start_idx: int,
    seg_len: int,
) -> np.ndarray:
    """指定された開始位置から長さ seg_len の波形を切り出し、必要なら wrap around

    Args:
        wav (np.ndarray): 1-D 波形
        start_idx (int): 切り出し開始位置のインデックス
        seg_len (int): 切り出す長さ（サンプル数）
    Returns:
        np.ndarray: 切り出した波形
    """
    if len(wav) < seg_len:
        k = int(np.ceil(seg_len / len(wav)))
        wav = np.tile(wav, k)

    # circular padding
    end_idx = start_idx + seg_len
    if end_idx <= len(wav):
        return wav[start_idx:end_idx].copy()

    # wrap around
    wrapped_wave = np.concatenate((wav[start_idx:], wav[: end_idx - len(wav)]))

    return wrapped_wave
