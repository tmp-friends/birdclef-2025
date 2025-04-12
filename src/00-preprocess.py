import logging
import os
from pathlib import Path
import time
import math
import concurrent.futures

import hydra

import numpy as np
import pandas as pd
import librosa
import cv2
import matplotlib.pyplot as plt

from utils.utils import set_seed
from conf.type import PreprocessConfig


def audio2melspec(cfg, audio_data):
    """Convert audio data to mel spectrogram"""
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


def process_audio(cfg, row, target_samples):
    """Process a single audio file to get the mel spectrogram"""

    try:
        # サンプリングレートを指定して、音声データを読み込む
        audio_data, _ = librosa.load(row["filepath"], sr=cfg.fs)

        # 音声が短い場合のリピート補正
        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        # 音声データの中央部分のみ抽出
        start_ix = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_ix = max(len(audio_data), start_ix + target_samples)
        center_audio = audio_data[start_ix:end_ix]

        if len(center_audio) < target_samples:
            center_audio = np.pad(
                center_audio,
                (0, target_samples - len(center_audio)),
                mode="constant",
            )

        mel_spec = audio2melspec(cfg=cfg, audio_data=center_audio)

        if mel_spec.shape != (cfg.target_w, cfg.target_h):
            mel_spec = cv2.resize(
                mel_spec,
                (cfg.target_w, cfg.target_h),
                interpolation=cv2.INTER_LINEAR,
            )

        return row["samplename"], mel_spec.astype(np.float32)

    except Exception as e:
        LOGGER.error(e)


@hydra.main(config_path="conf", config_name="preprocess", version_base="1.1")
def main(cfg: PreprocessConfig):
    """
    audio data を mel-spectrogram data へ変換。
    2D CNN を学習するために必要。

    ref: https://www.kaggle.com/code/kadircandrisolu/transforming-audio-to-mel-spec-birdclef-25
    """
    # Load data
    train_df = pd.read_csv(cfg.dir.train_csv)
    taxonomy_df = pd.read_csv(cfg.dir.taxonomy_csv)
    species_class_map = dict(
        zip(taxonomy_df["primary_label"], taxonomy_df["class_name"])
    )
    LOGGER.info(train_df.head())

    # mapping 辞書の作成
    labels = sorted(train_df["primary_label"].unique())
    label_ids = list(range(len(labels)))
    label2id = dict(zip(labels, label_ids))
    id2label = dict(zip(label_ids, labels))

    working_df = train_df[["primary_label", "rating", "filename"]].copy()
    working_df["target"] = working_df.primary_label.map(label2id)
    working_df["filepath"] = cfg.dir.train_audio_dir + "/" + working_df.filename
    working_df["samplename"] = working_df.filename.map(
        lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0]
    )
    working_df["class"] = working_df.primary_label.map(
        lambda x: species_class_map.get(x, "Unknown")
    )

    total_samples = len(working_df)
    LOGGER.info(
        f"Total samples to process: {total_samples} out of {len(working_df)} available"
    )
    LOGGER.info(f"Samples by class: {working_df['class'].value_counts()}")

    # 目標とする録音時間にサンプリングレートを乗算することで必要なサンプル数を算出
    target_samples = int(cfg.target_duration * cfg.fs)

    # 並列処理で実行
    # librosa の I/O や Numpy の処理は GIL の影響を受けにくいので恩恵がある
    LOGGER.info("Starting audio processing...")
    start_time = time.time()

    all_bird_data = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_audio, cfg, row, target_samples)
            for _, row in working_df.iterrows()
        ]

        # 並列タスクの完了を待ち、結果を収集
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                samplename, mel_spec = result
                all_bird_data[samplename] = mel_spec

    end_time = time.time()

    LOGGER.info(f"Processing completed in {end_time - start_time:.2f} seconds")
    LOGGER.info(
        f"Successfully processed {len(all_bird_data)} files out of {total_samples} total"
    )
    # npy で保存
    np.save("all_bird_data.npy", all_bird_data, allow_pickle=True)

    # mel_spec を可視化
    samples = []
    displayed_classes = set()

    max_samples = min(4, len(all_bird_data))
    for i, row in working_df.iterrows():
        if i >= len(working_df):
            break

        if row["samplename"] in all_bird_data:
            if row["class"] not in displayed_classes:
                samples.append((row["samplename"], row["class"], row["primary_label"]))
                displayed_classes.add(row["class"])

            if len(samples) >= max_samples:
                break

    if samples:
        plt.figure(figsize=(16, 12))

        for i, (samplename, class_name, species) in enumerate(samples):
            plt.subplot(2, 2, i + 1)
            plt.imshow(
                all_bird_data[samplename], aspect="auto", origin="lower", cmap="viridis"
            )
            plt.title(f"{class_name}: {species}")
            plt.colorbar(format="%+2.0f dB")

        plt.tight_layout()
        plt.savefig("melspec_examples.png")
        # plt.show()


if __name__ == "__main__":
    # Logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
    )
    LOGGER = logging.getLogger(Path(__file__).name)

    # For descriptive error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    set_seed()

    main()
