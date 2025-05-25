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
from utils.audio2melspec import process_audio_segment
from modules.birdclef_model import BirdCLEFModel


def load_models(cfg: InferConfig, num_classes: int):
    models = []

    model_path = os.path.join(cfg.model_dir, "model_fold0_stage3.pth")
    model = BirdCLEFModel(cfg=cfg, num_classes=num_classes)
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=cfg.device, weights_only=False)

    # Extract only the model's state_dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(cfg.device)
    model.eval()
    models.append(model)

    return models


def _apply_tta(mel_spec, tta_ix):
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


def _predict_for_segment(cfg, segment_audio: np.ndarray, models) -> np.ndarray:
    """1セグメントに対する予測（TTA を考慮）"""
    # TTA ありなら複数 mel_spec を作り平均、無しなら 1 つ
    tta_indices = range(cfg.tta_count) if cfg.uses_tta else [0]

    segment_preds = []
    for tta_ix in tta_indices:
        mel_spec = process_audio_segment(cfg, segment_audio)
        mel_spec = _apply_tta(mel_spec, tta_ix)

        mel_tensor = (
            torch.tensor(mel_spec, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(cfg.device)  # (1, H, W)  # (1, 1, H, W)
        )

        pred = _forward(models, mel_tensor)
        segment_preds.append(pred.cpu().numpy().squeeze())

    return np.mean(segment_preds, axis=0)


def _predict_on_spectrogram(
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
        audio_data, _ = librosa.load(audio_path, sr=cfg.spec.fs)

        seg_len = cfg.spec.fs * cfg.spec.window_size
        total_segs = int(np.ceil(len(audio_data) / seg_len))

        for seg_ix in range(total_segs):
            start_sample = seg_ix * seg_len
            end_sample = start_sample + seg_len
            segment_audio = audio_data[start_sample:end_sample]

            # 推論
            pred = _predict_for_segment(cfg, segment_audio, models)

            # メタ情報
            end_sec = (seg_ix + 1) * cfg.spec.window_size
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
        row_ids, preds = _predict_on_spectrogram(
            cfg=cfg,
            audio_path=str(audio_path),
            models=models,
        )
        all_row_ids.extend(row_ids)
        all_preds.extend(preds)

    return all_row_ids, all_preds


def create_submission(
    cfg: InferConfig, all_row_ids, all_preds, species_ids
) -> pd.DataFrame:
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

    submission_df = submission_df.reset_index()

    return submission_df


def apply_time_smoothing(
    submission_df: pd.DataFrame, weights: list[float] | None = [0.2, 0.6, 0.2]
) -> pd.DataFrame:
    """
    Apply time smoothing to the predictions in a submission file.
    ref: https://www.kaggle.com/code/salmanahmedtamu/labels-tta-efficientnet-b0-pytorch-inference/notebook

    Args:
        submission_path (str): Path to the input submission CSV file.
        output_path (str): Path to save the smoothed submission CSV file.
    Returns:
        None
    """
    cols = submission_df.columns[1:]  # Prediction columns
    groups = (
        submission_df["row_id"].astype(str).str.rsplit("_", n=1).str[0].values
    )  # Extract group IDs

    # Apply smoothing for each group
    for group in np.unique(groups):
        sub_group = submission_df[group == groups]
        predictions = sub_group[cols].values
        new_predictions = predictions.copy()

        # Apply smoothing
        for i in range(1, predictions.shape[0] - 1):
            new_predictions[i] = (
                (predictions[i - 1] * weights[0])
                + (predictions[i] * weights[1])
                + (predictions[i + 1] * weights[2])
            )
        new_predictions[0] = (predictions[0] * (weights[1] + weights[0])) + (
            predictions[1] * weights[2]
        )
        new_predictions[-1] = (predictions[-1] * (weights[1] + weights[2])) + (
            predictions[-2] * weights[0]
        )

        # Update the smoothed predictions
        sub_group[cols] = new_predictions
        submission_df[group == groups] = sub_group

    return submission_df


@hydra.main(config_path="conf", config_name="infer", version_base="1.1")
def main(cfg: InferConfig):
    set_seed(cfg.seed)

    # Load data
    taxonomy_df = pd.read_csv(cfg.dir.taxonomy_csv)
    num_classes = len(taxonomy_df)
    species_ids = taxonomy_df["primary_label"].tolist()

    # Load model
    models = load_models(cfg, num_classes=num_classes)

    # infer
    all_row_ids, all_preds = run_inference(cfg=cfg, models=models)

    submission_df = create_submission(
        cfg=cfg,
        all_row_ids=all_row_ids,
        all_preds=all_preds,
        species_ids=species_ids,
    )

    submission_df = apply_time_smoothing(submission_df=submission_df)
    submission_df.to_csv("submission.csv", index=False)

    LOGGER.info("Inference completed successfully!")


if __name__ == "__main__":
    # Logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
    )
    LOGGER = logging.getLogger(Path(__file__).name)

    # For descriptive error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    main()
