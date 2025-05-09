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

    seg_len = int(cfg.spec.window_size * cfg.spec.fs)

    if len(audio_data) < seg_len:
        audio_data = np.pad(
            audio_data,
            (0, seg_len - len(audio_data)),
            mode="reflect",
        )

    mel_spec = _audio2melspec(cfg=cfg, audio_data=audio_data)

    # Resize if needed
    if mel_spec.shape != cfg.spec.target_shape:
        mel_spec = cv2.resize(
            mel_spec,
            cfg.spec.target_shape,
            interpolation=cv2.INTER_LINEAR,
        )

    return mel_spec.astype(np.float32)


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
        pad_mode="reflect",  # STFT 計算時の信号端の鏡映 padding
        norm="slaney",  # メルフィルタ正規化
        htk=True,  # メル尺度の定義
        center=True,  # STFT 中心 padding
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (
        mel_spec_db.max() - mel_spec_db.min() + 1e-8
    )

    return mel_spec_norm
