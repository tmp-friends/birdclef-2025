import numpy as np
import librosa
import cv2

from conf.type import PreprocessConfig, InferConfig


def process_audio_segment(cfg: PreprocessConfig | InferConfig, audio_data: np.ndarray):
    """
    単一のオーディオセグメントを処理してメルスペクトログラムを生成します。

    Args:
        cfg (InferConfig): 推論設定を含む構成オブジェクト。サンプリング周波数 (fs)、ウィンドウサイズ (window_size)、および
                           出力スペクトログラムの目標幅 (target_w) と高さ (target_h) を含む。
        audio_data (np.ndarray): 処理対象のオーディオデータ。1次元のNumPy配列。

    Returns:
        np.ndarray: メルスペクトログラムを表す2次元のNumPy配列。データ型はfloat32。
    """
    target_len = int(cfg.spec.fs * cfg.spec.window_size)
    audio_data = _mirror_pad(y=audio_data, target_len=target_len, sr=cfg.spec.fs)

    mel_spec = _audio2melspec(cfg=cfg, audio_data=audio_data)  # (H, W)

    delta = librosa.feature.delta(mel_spec, order=1)
    delta2 = librosa.feature.delta(mel_spec, order=2)

    mel_spec3 = np.stack([mel_spec, delta, delta2], axis=0)  # (3, H, W)

    # Resize if needed (時間軸だけ調整)
    if mel_spec3.shape[2] != cfg.spec.target_shape[0]:
        mel_spec3 = cv2.resize(
            mel_spec3.transpose(1, 2, 0),  # (H, W, 3)
            cfg.spec.target_shape,
            interpolation=cv2.INTER_LINEAR,
        ).transpose(
            2, 0, 1
        )  # (3, H, W)

    return mel_spec3.astype(np.float32)


def _mirror_pad(y: np.ndarray, target_len: int, sr: int, fade_ms: int = 10) -> np.ndarray:
    """
    オーディオデータを鏡像パディングして、指定された長さに合わせます。

    Args:
        y (np.ndarray): オーディオデータ。
        target_len (int): パディング後の長さ。
        sr (int): サンプリングレート。
        fade_ms (int): フェードイン/アウトの長さ (ms)。

    Returns:
        np.ndarray: パディングされたオーディオデータ。
    """
    if len(y) >= target_len:
        return y[:target_len]

    # 反射 padding
    shortage = target_len - len(y)
    padding = np.pad(y, (0, shortage), mode="reflect")

    # フェードイン/アウト
    f = int(fade_ms / 1000 * sr)
    f = min(f, shortage)
    if f > 0:
        # フェードアウト効果を作成: 1から0までの線形な減衰を表す配列を生成
        ramp = np.linspace(1, 0, f, dtype=y.dtype)
        # パディングされた部分の先頭f個のサンプルにフェードアウトを適用
        # ramp[::-1]で減衰を反転させることで、元の音声から徐々にフェードアウトする
        padding[len(y) : len(y) + f] *= ramp[::-1]

    return padding[:target_len]


def _audio2melspec(cfg, audio_data):
    """
    オーディオデータを正規化されたメルスペクトログラムに変換します。

    この関数は生のオーディオデータを処理してメルスペクトログラムを生成し、
    正規化を行い、入力オーディオデータ内のNaN値を処理します。

    Args:
        cfg (InferConfig): メルスペクトログラム生成のためのパラメータを含む
            設定オブジェクト。以下を含みます:
            - fs (int): オーディオのサンプリングレート。
            - n_fft (int): FFTのコンポーネント数。
            - hop_length (int): フレーム間のサンプル数。
            - n_mels (int): 生成するメルバンドの数。
            - fmin (float): 最小周波数 (Hz)。
            - fmax (float): 最大周波数 (Hz)。
        audio_data (np.ndarray): 生のオーディオ信号を含む1次元のNumPy配列。

    Returns:
        np.ndarray: 正規化されたメルスペクトログラムを表す2次元のNumPy配列。
        値は0から1の範囲にスケーリングされています。

    Notes:
        - 入力オーディオデータにNaN値が含まれている場合、それらは信号内の
          非NaN値の平均値で置き換えられます。
        - メルスペクトログラムはデシベル(dB)スケールに変換され、
          [0, 1]の範囲に正規化されます。
    """
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.spec.fs,
        n_fft=cfg.spec.n_fft,
        hop_length=cfg.spec.hop_length,
        n_mels=cfg.spec.n_mels,
        fmin=cfg.spec.fmin,
        fmax=cfg.spec.fmax,
        power=2.0,
        pad_mode="reflect",  # 鏡映 padding
        norm="slaney",  # メルフィルタ正規化
        htk=True,  # メル尺度の定義
        center=True,  # STFT 中心 padding
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_norm
