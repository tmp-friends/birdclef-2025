import numpy as np


def random_crop(
    audio: np.ndarray, seg_len: int, rng: np.random.Generator
) -> np.ndarray:
    """
    audio    : 1-D waveform (float32/int16 など)
    seg_len  : 目標サンプル数 (= fs * window_size)
    rng      : numpy.random.Generator
    """
    # # 足りない場合はリピートしてからパディング
    # if len(audio) < seg_len:
    #     n_copy = int(np.ceil(seg_len / len(audio)))
    #     audio = np.tile(audio, n_copy)

    # ランダム開始位置（0〜len(audio)-seg_len の範囲）
    max_start = len(audio) - seg_len
    start_ix = rng.integers(0, max_start + 1, endpoint=True) if max_start > 0 else 0
    end_ix = start_ix + seg_len
    crop = audio[start_ix:end_ix]

    # # 念のため長さを確認（境界ギリギリのときに 1 サンプル欠けるのを防ぐ）
    # if len(crop) < seg_len:
    #     crop = np.pad(crop, (0, seg_len - len(crop)), mode="constant")

    return crop.astype(np.float32)
