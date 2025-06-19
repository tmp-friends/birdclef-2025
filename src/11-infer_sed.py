import logging
import os
from pathlib import Path
import re

import hydra

import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf

from utils.utils import set_seed
from conf.type import InferConfig
from modules.birdclef_model import BirdCLEFSEDModel


def load_models(cfg: InferConfig, num_classes: int):
    """Load SED models for inference"""
    models = []

    # Check if model_dir exists
    if not os.path.exists(cfg.model_dir):
        raise ValueError(f"Model directory not found: {cfg.model_dir}")

    # Define filename patterns for models (support both SED and regular models)
    sed_pattern = r"sed_model_fold(\d+).pth"
    sed_advanced_pattern = (
        r"sed_model_fold(\d+)_(\w+).pth"  # For advanced models with stage
    )
    regular_pattern = r"model_fold(\d+).pth"

    # Filter files in directory that match any pattern
    all_files = os.listdir(cfg.model_dir)
    sed_files = [f for f in all_files if re.match(sed_pattern, f)]
    sed_advanced_files = [f for f in all_files if re.match(sed_advanced_pattern, f)]
    regular_files = [f for f in all_files if re.match(regular_pattern, f)]

    # Determine which pattern to use
    if sed_advanced_files:
        # For advanced models, prefer 'pseudo' stage if available
        pseudo_files = [f for f in sed_advanced_files if "_pseudo.pth" in f]
        if pseudo_files:
            files = pseudo_files
            pattern = r"sed_model_fold(\d+)_pseudo.pth"
            LOGGER.info("Using SED advanced model files (pseudo stage)")
        elif "_finetune" in str(sed_advanced_files):
            # Prefer 'finetune' stage if available
            finetune_files = [f for f in sed_advanced_files if "_finetune.pth" in f]
            files = finetune_files
            pattern = r"sed_model_fold(\d+)_finetune.pth"
            LOGGER.info("Using SED advanced model files (finetune stage)")
        else:
            # Fall back to train_bce stage
            bce_files = [f for f in sed_advanced_files if "_train_bce.pth" in f]
            if bce_files:
                files = bce_files
                pattern = r"sed_model_fold(\d+)_train_bce.pth"
                LOGGER.info("Using SED advanced model files (train_bce stage)")
            else:
                # Fall back to pretrain_ce stage
                ce_files = [f for f in sed_advanced_files if "_pretrain_ce.pth" in f]
                files = ce_files
                pattern = r"sed_model_fold(\d+)_pretrain_ce.pth"
                LOGGER.info("Using SED advanced model files (pretrain_ce stage)")
    elif sed_files:
        files = sed_files
        pattern = sed_pattern
        LOGGER.info("Using SED model files")
    elif regular_files:
        files = regular_files
        pattern = regular_pattern
        LOGGER.info("Using regular model files (will adapt for SED)")
    else:
        files = []
        pattern = sed_pattern

    # Determine folds to load
    folds_to_load = (
        cfg.folds if hasattr(cfg, "folds") and cfg.folds else cfg.selected_folds
    )

    LOGGER.info(f"Looking for SED models in: {cfg.model_dir}")
    LOGGER.info(f"Available files: {files}")
    LOGGER.info(f"Folds to load: {folds_to_load}")

    loaded_folds = []
    for fold in folds_to_load:
        # Look for matching file in the directory
        found = False
        for file in files:
            match = re.match(pattern, file)
            if match:
                (fold_number,) = match.groups()
                # Check if fold number matches current fold
                if int(fold_number) == fold:
                    LOGGER.info(f"Loading SED model file: {file}")

                    model_path = os.path.join(cfg.model_dir, file)

                    # Create model instance
                    model = BirdCLEFSEDModel(cfg)

                    # Load checkpoint
                    checkpoint = torch.load(
                        model_path, map_location=cfg.device, weights_only=False
                    )

                    # Extract model state dict
                    if "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        model.load_state_dict(checkpoint)

                    model.to(cfg.device)
                    model.eval()
                    models.append(model)
                    loaded_folds.append(fold)
                    found = True
                    break

        if not found:
            LOGGER.warning(f"Model file not found for fold {fold}")

    if not models:
        raise ValueError(
            f"No SED models loaded from {cfg.model_dir}. Check model paths and fold numbers."
        )

    LOGGER.info(
        f"Successfully loaded {len(models)} SED models for folds: {loaded_folds}"
    )
    return models


def _forward(models: list[torch.nn.Module], x: torch.Tensor) -> torch.Tensor:
    """Forward pass through all models and average predictions"""
    with torch.no_grad():
        if len(models) == 1:
            # For SED model, output is already logits, apply sigmoid
            return torch.sigmoid(models[0](x))

        preds = [torch.sigmoid(m(x)) for m in models]
        return torch.mean(torch.stack(preds, dim=0), dim=0)


def _apply_power_to_low_ranked_cols(
    p: np.ndarray, top_k: int = 30, exponent: int | float = 2.0, inplace: bool = True
) -> np.ndarray:
    """
    各列の最大値で列を順位付けし、top_k 以降の列だけを p^exponent で強調／平坦化する。
    ref: https://www.kaggle.com/code/myso1987/post-processing-with-power-adjustment-for-low-rank

    Args:
        p (np.ndarray): 予測確率配列 (rows x classes)
        top_k (int): 上位k個のクラスは変更しない
        exponent (float): 低ランククラスに適用する指数
        inplace (bool): インプレースで変更するか

    Returns:
        np.ndarray: 後処理された予測配列
    """
    if not inplace:
        p = p.copy()

    tail_cols = np.argsort(-p.max(axis=0))[top_k:]
    p[:, tail_cols] = p[:, tail_cols] ** exponent

    return p


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
        submission_df (pd.DataFrame): Submission dataframe with row_id and prediction columns
        weights (list[float]): Smoothing weights [previous, current, next]
    Returns:
        pd.DataFrame: Smoothed submission dataframe
    """
    # Create a copy to avoid SettingWithCopyWarning
    submission_df = submission_df.copy()
    
    cols = submission_df.columns[1:]  # Prediction columns
    groups = (
        submission_df["row_id"].astype(str).str.rsplit("_", n=1).str[0].values
    )  # Extract group IDs

    # Apply smoothing for each group
    for group in np.unique(groups):
        group_mask = (groups == group)
        group_indices = submission_df.index[group_mask]
        predictions = submission_df.loc[group_indices, cols].values
        new_predictions = predictions.copy()

        # Apply smoothing only if we have more than 1 prediction
        if len(predictions) > 1:
            # Apply smoothing for middle elements
            for i in range(1, predictions.shape[0] - 1):
                new_predictions[i] = (
                    (predictions[i - 1] * weights[0])
                    + (predictions[i] * weights[1])
                    + (predictions[i + 1] * weights[2])
                )
            
            # Handle boundary cases
            new_predictions[0] = (predictions[0] * (weights[1] + weights[0])) + (
                predictions[1] * weights[2]
            )
            new_predictions[-1] = (predictions[-1] * (weights[1] + weights[2])) + (
                predictions[-2] * weights[0]
            )

        # Update the smoothed predictions using .loc to avoid warning
        submission_df.loc[group_indices, cols] = new_predictions

    return submission_df


def _predict_for_segment(cfg, segment_audio: np.ndarray, models) -> np.ndarray:
    """Predict for a single audio segment using SED model"""
    # Convert to tensor with correct shape: (1, 1, samples)
    wav_tensor = (
        torch.tensor(segment_audio, dtype=torch.float32)
        .unsqueeze(0)  # Add batch dimension
        .unsqueeze(0)  # Add channel dimension
        .to(cfg.device)  # Shape: (1, 1, samples)
    )

    # Get predictions
    pred = _forward(models, wav_tensor)
    return pred.cpu().numpy().squeeze()


def _predict_on_audio(
    cfg: InferConfig,
    audio: np.ndarray,
    models: list[torch.nn.Module],
) -> np.ndarray:
    """
    音声全体に対して sliding window で予測を実行する関数
    SED model は音声波形を直接入力として受け取る

    Args:
        cfg (InferConfig): 設定
        audio (np.ndarray): 音声データ
        models (list[torch.nn.Module]): 使用するモデルのリスト

    Returns:
        np.ndarray: 各セグメントの予測結果
    """
    duration_samples = int(cfg.train_duration * cfg.fs)
    
    segment_preds = []

    # Calculate expected number of segments for 60s audio with 5s stride
    # This should be 12 segments (5, 10, 15, ..., 60)
    expected_segments = 12
    
    # Generate predictions for each 5-second end time
    for i in range(expected_segments):
        end_time = (i + 1) * 5  # 5, 10, 15, ..., 60
        
        # Calculate the center of the window for this end time
        # For 10s window ending at time t, start at t-10
        end_idx = int(end_time * cfg.fs)
        start_idx = max(0, end_idx - duration_samples)
        
        # Extract segment
        if start_idx + duration_samples <= len(audio):
            segment_audio = audio[start_idx : start_idx + duration_samples]
        else:
            # For the last segments, take the last 10 seconds
            start_idx = max(0, len(audio) - duration_samples)
            segment_audio = audio[start_idx:]
            # Pad if necessary
            if len(segment_audio) < duration_samples:
                segment_audio = np.pad(
                    segment_audio,
                    (0, duration_samples - len(segment_audio)),
                    mode="constant",
                )
        
        # Predict for this segment
        segment_pred = _predict_for_segment(cfg, segment_audio, models)
        segment_preds.append(segment_pred)

    segment_preds = np.array(segment_preds)
    
    # Verify we have exactly 12 predictions
    assert len(segment_preds) == expected_segments, f"Expected {expected_segments} segments, got {len(segment_preds)}"

    # Apply post-processing if there are multiple segments
    if len(segment_preds) > 1:
        # Apply power adjustment to low-ranked columns
        if hasattr(cfg, "post_processing") and cfg.post_processing.get(
            "apply_power_adjustment", True
        ):
            top_k = cfg.post_processing.get("power_top_k", 30)
            exponent = cfg.post_processing.get("power_exponent", 2.0)
            segment_preds = _apply_power_to_low_ranked_cols(
                segment_preds, top_k=top_k, exponent=exponent, inplace=True
            )

    return segment_preds


def inference(
    cfg: InferConfig,
    df: pd.DataFrame,
    models: list[torch.nn.Module],
    mode: str = "validation",
    num_classes: int = None,
):
    """
    推論を実行する関数

    Args:
        cfg (InferConfig): 設定
        df (pd.DataFrame): 推論対象のデータフレーム
        models (list[torch.nn.Module]): 使用するモデルのリスト
        mode (str): 推論モード ("validation" or "test")
        num_classes (int): クラス数
    """
    predictions = []

    for _, row in df.iterrows():
        if mode == "validation":
            audio_path = os.path.join(cfg.dir.train_audio_dir, row["filename"])
        else:
            audio_path = os.path.join(cfg.dir.test_audio_dir, row["filename"])

        # Load audio
        try:
            audio, sr = sf.read(audio_path, dtype="float32")

            # Resample if necessary
            if sr != cfg.fs:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.fs)

            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Get predictions using sliding window
            preds = _predict_on_audio(cfg, audio, models)

            # Average predictions across all windows
            avg_pred = np.mean(preds, axis=0)
            predictions.append(avg_pred)

        except Exception as e:
            LOGGER.error(f"Error processing {row['filename']}: {e}")
            # Return zero predictions in case of error
            predictions.append(np.zeros(num_classes))

    return np.array(predictions)


def run_soundscape_inference(
    cfg: InferConfig, soundscape_files: list[str], models: list[torch.nn.Module]
):
    """
    サウンドスケープファイルに対する推論（テスト用）

    Args:
        cfg (InferConfig): 設定
        soundscape_files (list[str]): サウンドスケープファイルのパス一覧
        models (list[torch.nn.Module]): 使用するモデルのリスト

    Returns:
        tuple: (row_ids, predictions)
    """
    all_row_ids = []
    all_predictions = []

    for soundscape_path in soundscape_files:
        soundscape_id = Path(soundscape_path).stem
        LOGGER.info(f"Processing soundscape: {soundscape_id}")

        try:
            # Load full soundscape audio
            audio, _ = librosa.load(soundscape_path, sr=cfg.fs)

            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Get predictions for all segments using sliding window
            segment_preds = _predict_on_audio(cfg, audio, models)

            # Create row_ids for each segment
            # Must generate exactly 12 rows per soundscape (5, 10, 15, ..., 60)
            for i, pred in enumerate(segment_preds):
                end_time = (i + 1) * 5  # 5, 10, 15, ..., 60
                row_id = f"{soundscape_id}_{end_time}"
                all_row_ids.append(row_id)
                all_predictions.append(pred)
            
            # Verify we have 12 predictions per soundscape
            LOGGER.info(f"Generated {len(segment_preds)} predictions for {soundscape_id}")

        except Exception as e:
            LOGGER.error(f"Error processing {soundscape_path}: {e}")

    return all_row_ids, all_predictions


def run_inference(cfg: InferConfig, mode: str = "validation"):
    """
    推論のメイン実行関数

    Args:
        cfg (InferConfig): 設定
        mode (str): 推論モード ("validation" or "test")
    """
    # Load taxonomy to get number of classes
    taxonomy_df = pd.read_csv(cfg.dir.taxonomy_csv)
    num_classes = len(taxonomy_df)

    # Load models
    models = load_models(cfg, num_classes)
    if not models:
        raise ValueError("No models loaded. Check model paths.")

    # Save predictions
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "validation":
        # For validation mode, use train soundscapes to test inference behavior
        # (same behavior as test mode, but using train_soundscapes directory)
        LOGGER.info(f"Looking for train soundscapes in: {getattr(cfg.dir, 'train_soundscapes_dir', 'NOT_SET')}")
        LOGGER.info(f"Directory exists: {os.path.exists(getattr(cfg.dir, 'train_soundscapes_dir', ''))}")
        
        if hasattr(cfg.dir, 'train_soundscapes_dir') and os.path.exists(cfg.dir.train_soundscapes_dir):
            all_files = list(Path(cfg.dir.train_soundscapes_dir).glob("*.ogg"))
            # Limit to 10 samples for quick validation testing
            test_files = all_files[:10] if len(all_files) >= 10 else all_files
            LOGGER.info(f"Using {len(test_files)} train soundscape files for validation (limited from {len(all_files)} total)")
        else:
            # Fallback: create empty submission for validation testing
            LOGGER.warning("No train soundscape directory found. Creating empty submission for validation.")
            test_files = []
        
        if test_files:
            row_ids, soundscape_predictions = run_soundscape_inference(
                cfg, [str(f) for f in test_files], models
            )
        else:
            row_ids = []
            soundscape_predictions = []
        
        # Create submission using the same logic as test mode
        species_ids = taxonomy_df["primary_label"].tolist()
        submission = create_submission(
            cfg=cfg,
            all_row_ids=row_ids,
            all_preds=soundscape_predictions,
            species_ids=species_ids,
        )
        
        # Apply time smoothing
        submission = apply_time_smoothing(submission_df=submission)
        
        # Save submission for validation
        submission.to_csv("validation_submission.csv", index=False)
        LOGGER.info("Saved validation submission to validation_submission.csv")

    else:
        # For test mode, directly process soundscape files
        if hasattr(cfg.dir, "test_soundscapes_dir") and os.path.exists(
            cfg.dir.test_soundscapes_dir
        ):
            # Test soundscape inference
            test_files = list(Path(cfg.dir.test_soundscapes_dir).glob("*.ogg"))
            LOGGER.info(f"Found {len(test_files)} test soundscape files")

            if test_files:
                row_ids, soundscape_predictions = run_soundscape_inference(
                    cfg, [str(f) for f in test_files], models
                )
            else:
                LOGGER.warning(
                    "No test soundscape files found. Creating empty submission."
                )
                row_ids = []
                soundscape_predictions = []
        else:
            LOGGER.warning(
                "No test soundscape directory found. Creating empty submission."
            )
            row_ids = []
            soundscape_predictions = []

        # Create submission using the same logic as baseline
        species_ids = taxonomy_df["primary_label"].tolist()
        submission = create_submission(
            cfg=cfg,
            all_row_ids=row_ids,
            all_preds=soundscape_predictions,
            species_ids=species_ids,
        )

        # Apply time smoothing
        submission = apply_time_smoothing(submission_df=submission)
        
        # Final validation for Kaggle format
        expected_rows_per_soundscape = 12
        n_soundscapes = len(submission) // expected_rows_per_soundscape
        if len(submission) % expected_rows_per_soundscape != 0:
            LOGGER.error(f"Invalid submission format: {len(submission)} rows is not divisible by {expected_rows_per_soundscape}")
        else:
            LOGGER.info(f"Submission format validated: {len(submission)} rows for {n_soundscapes} soundscapes")
        
        # Save submission directly to the Hydra output directory
        # Use the current working directory which is already set by Hydra
        submission.to_csv("submission.csv", index=False)
        LOGGER.info("Inference completed successfully!")


@hydra.main(config_path="conf", config_name="infer_sed", version_base="1.1")
def main(cfg: InferConfig):
    set_seed(cfg.seed)

    # Run inference
    run_inference(cfg, mode=cfg.mode)


# Global logger for create_submission function
LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    # Logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
    )
    LOGGER = logging.getLogger(Path(__file__).name)

    # For descriptive error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    main()
