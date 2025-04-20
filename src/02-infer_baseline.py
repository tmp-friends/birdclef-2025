import logging
import os
from pathlib import Path
import re

import hydra

import numpy as np
import pandas as pd
import torch
import cv2
import librosa

from utils.utils import set_seed
from conf.type import InferConfig
from models.birdclef_model import BirdCLEFModel


def load_models(cfg: InferConfig, num_classes: int):
    models = []
    # ファイル名のパターン定義
    pattern = "model_fold(\d+).pth"

    # ディレクトリ内のファイルを事前にフィルタリング
    files = [f for f in os.listdir(cfg.model_dir) if re.match(pattern, f)]

    # 使用するフォールドを決定
    folds_to_load = (
        cfg.folds if hasattr(cfg, "folds") and cfg.folds else range(cfg.num_folds)
    )

    for fold in folds_to_load:
        # フォルダ内のファイルに対して正規表現マッチング
        for file in files:
            match = re.match(pattern, file)
            if match:
                (fold_number,) = match.groups()
                # フォールド番号が現在のフォールドと一致するかチェック
                if int(fold_number) == fold:
                    LOGGER.info(f"Loading model file: {file}")

                    model_path = os.path.join(cfg.model_dir, file)
                    model = BirdCLEFModel(cfg=cfg, num_classes=num_classes)
                    # Load the checkpoint
                    checkpoint = torch.load(model_path, map_location=cfg.device)

                    # Extract only the model's state_dict
                    if "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        model.load_state_dict(checkpoint)

                    model.to(cfg.device)
                    model.eval()
                    models.append(model)
                    break
        else:
            LOGGER.warning(f"No model found for fold {fold}.")

    return models


def process_audio_segment(cfg: InferConfig, audio_data: np.ndarray):
    """
    単一のオーディオセグメントを処理してメルスペクトログラムを生成します。

    Args:
        cfg (InferConfig): 推論設定を含む構成オブジェクト。サンプリング周波数 (fs)、ウィンドウサイズ (window_size)、および
                           出力スペクトログラムの目標幅 (target_w) と高さ (target_h) を含む。
        audio_data (np.ndarray): 処理対象のオーディオデータ。1次元のNumPy配列。

    Returns:
        np.ndarray: メルスペクトログラムを表す2次元のNumPy配列。データ型はfloat32。
    """
    if len(audio_data) < cfg.fs * cfg.window_size:
        audio_data = np.pad(
            audio_data,
            (0, cfg.fs * cfg.window_size - len(audio_data)),
            mode="constant",
        )

    mel_spec = audio2melspec(cfg=cfg, audio_data=audio_data)

    # Resize if needed
    if mel_spec.shape != (cfg.target_w, cfg.target_h):
        mel_spec = cv2.resize(
            mel_spec,
            (cfg.target_w, cfg.target_h),
            interpolation=cv2.INTER_LINEAR,
        )

    return mel_spec.astype(np.float32)


def audio2melspec(cfg: InferConfig, audio_data: np.ndarray):
    """
    オーディオデータを正規化されたメルスペクトログラムに変換します。

    この関数は生のオーディオデータを処理してメルスペクトログラムを生成し、
    正規化を行い、入力オーディオデータ内のNaN値を処理します。

    Args:
        cfg (InferConfig): メルスペクトログラム生成のためのパラメータを含む
            設定オブジェクト。以下を含みます:
            - fs (int): オーディオのサンプリングレート。
            - num_fft (int): FFTのコンポーネント数。
            - hop_length (int): フレーム間のサンプル数。
            - num_mels (int): 生成するメルバンドの数。
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
        sr=cfg.fs,
        n_fft=cfg.num_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.num_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (
        mel_spec_db.max() - mel_spec_db.min() + 1e-8
    )

    return mel_spec_norm


def apply_tta(mel_spec, tta_ix):
    """
    Test Time Augmentation (TTA) をメルスペクトログラムに適用する関数。
    返却時に .copy() して負ストライド問題を回避する。

    Args:
        mel_spec (np.ndarray): Augmentation を適用するメルスペクトログラム。
        tta_ix (int): 適用する TTA 手法のインデックス。
            - 0: 変更なし
            - 1: 水平方向の反転
            - 2: 垂直方向の反転

    Returns:
        np.ndarray: Augmentation が適用されたメルスペクトログラム。
    """
    if tta_ix == 0:
        return mel_spec
    elif tta_ix == 1:
        # Horizontal flip
        return np.flip(mel_spec, axis=1).copy()
    elif tta_ix == 2:
        # Vertical flip
        return np.flip(mel_spec, axis=0).copy()
    else:
        return mel_spec.copy()


def _forward(models: list[torch.nn.Module], x: torch.Tensor) -> torch.Tensor:
    """すべてのモデルで推論 → 平均 → Sigmoid"""
    with torch.no_grad():
        if len(models) == 1:
            return torch.sigmoid(models[0](x))
        preds = [torch.sigmoid(m(x)) for m in models]
        return torch.mean(torch.stack(preds, dim=0), dim=0)


def _segment_audio(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    """指定位置のオーディオ切り出し（長さ不足時はゼロ埋め）"""
    end = start + length
    if end > len(audio):
        pad = end - len(audio)
        audio = np.pad(audio, (0, pad))
    return audio[start:end]


def _predict_for_segment(cfg, segment_audio: np.ndarray, models) -> np.ndarray:
    """1セグメントに対する予測（TTA を考慮）"""
    # TTA ありなら複数 mel を作り平均、無しなら 1 つ
    tta_indices = range(cfg.tta_count) if cfg.uses_tta else [0]

    segment_preds = []
    for tta_ix in tta_indices:
        mel = process_audio_segment(cfg, segment_audio)
        mel = apply_tta(mel, tta_ix)

        mel_tensor = (
            torch.tensor(mel, dtype=torch.float32)
            .unsqueeze(0)  # (1, H, W)
            .unsqueeze(0)  # (1, 1, H, W)
            .to(cfg.device)
        )

        pred = _forward(models, mel_tensor)
        segment_preds.append(pred.cpu().numpy().squeeze())

    return np.mean(segment_preds, axis=0)


def predict_on_spectrogram(
    cfg, audio_path: str, models: list
) -> tuple[list[str], list[np.ndarray]]:
    """
    1 つのサウンドスケープ (.ogg) に対して推論を実行し、
    セグメント毎の row_id と予測確率を返す。
    """
    soundscape_id = Path(audio_path).stem
    row_ids, preds = [], []

    try:
        LOGGER.info(f"Processing {soundscape_id}")
        audio, _ = librosa.load(audio_path, sr=cfg.fs)

        seg_len = cfg.fs * cfg.window_size
        total_segs = int(np.ceil(len(audio) / seg_len))

        for seg_ix in range(total_segs):
            start_sample = seg_ix * seg_len
            segment = _segment_audio(audio, start_sample, seg_len)

            # 推論
            pred = _predict_for_segment(cfg, segment, models)

            # メタ情報
            end_sec = (seg_ix + 1) * cfg.window_size
            row_ids.append(f"{soundscape_id}_{end_sec}")
            preds.append(pred)

    except Exception as e:
        LOGGER.error(f"Error processing {audio_path}: {e}")

    return row_ids, preds


def run_inference(cfg: InferConfig, models):
    """Run inference on all test soundscapes"""
    test_files = list(Path(cfg.dir.test_soundscapes_dir).glob("*.ogg"))
    # DEBUG
    # test_files = list(Path(cfg.dir.train_audio_dir + "/21038").glob("*.ogg"))
    LOGGER.info(f"Found {len(test_files)} test")

    all_row_ids = []
    all_preds = []
    for audio_path in test_files:
        row_ids, preds = predict_on_spectrogram(
            cfg=cfg,
            audio_path=str(audio_path),
            models=models,
        )
        all_row_ids.extend(row_ids)
        all_preds.extend(preds)

    return all_row_ids, all_preds


def create_submission(cfg: InferConfig, all_row_ids, all_preds, species_ids):
    """Create submission file in the *wide* format (one row per row_id,
    one column per species) identical to the sample submission."""

    # ── build wide dataframe ────────────────────────────────────────────────
    submission_df = pd.DataFrame(all_preds, columns=species_ids)
    submission_df["row_id"] = all_row_ids
    submission_df.set_index("row_id", inplace=True)

    # ── align with sample submission ───────────────────────────────────────
    sample_sub = pd.read_csv(cfg.dir.submission_csv, index_col="row_id")

    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        LOGGER.warning(
            f"Missing {len(missing_cols)} species columns - filling with 0.0"
        )
        for col in missing_cols:
            submission_df[col] = 0.0

    # exact column order
    submission_df = submission_df[sample_sub.columns]

    # ── save ───────────────────────────────────────────────────────────────
    submission_df = submission_df.reset_index()
    submission_df.to_csv("submission.csv", index=False)


@hydra.main(config_path="conf", config_name="infer", version_base="1.1")
def main(cfg: InferConfig):
    set_seed(cfg.seed)

    # Load data
    taxonomy_df = pd.read_csv(cfg.dir.taxonomy_csv)
    spectrograms = None
    try:
        spectrograms = np.load(cfg.spectrogram_npy_path, allow_pickle=True).item()
        LOGGER.info(f"Loaded {len(spectrograms)} pre-computed mel spectrograms")
    except Exception as e:
        LOGGER.info(f"Error loading pre-computed spectrograms: {e}")
        LOGGER.info("Will generate spectrograms on-the-fly instead.")

    num_classes = len(taxonomy_df)
    species_ids = taxonomy_df["primary_label"].tolist()

    # Load model
    models = load_models(cfg, num_classes=num_classes)

    # infer
    all_row_ids, all_preds = run_inference(cfg=cfg, models=models)

    create_submission(
        cfg=cfg,
        all_row_ids=all_row_ids,
        all_preds=all_preds,
        species_ids=species_ids,
    )


if __name__ == "__main__":
    # Logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
    )
    LOGGER = logging.getLogger(Path(__file__).name)

    # For descriptive error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    main()
