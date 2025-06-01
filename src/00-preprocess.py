import logging
import os
from pathlib import Path
import time
import pickle
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
from utils.audio2melspec import process_audio_segment
from utils.sampling import random_crop, rms_crop, rms_crop_shift


VOICE_DATA_PATH = "/home/tomoya/kaggle/birdclef-2025/output/eda/train_voice_data.pkl"
if os.path.exists(VOICE_DATA_PATH):
    with open(VOICE_DATA_PATH, "rb") as f:
        VOICE_DATA = pickle.load(f)
else:
    VOICE_DATA = {}


def _mask_voice_with_noise(
    arr: np.ndarray, s: int, e: int, fade: int = 256, noise_db: float = -60.0
):
    """区間[s:e] を微小ホワイトノイズ + フェードでマスク"""
    n = e - s
    if n <= 0:
        return
    # -60 dB 相当の振幅
    amp = 10 ** (noise_db / 20.0)
    noise = np.random.randn(n) * amp
    arr[s:e] = noise

    # フェードイン / フェードアウト
    f = min(fade, n // 2)
    if f > 0:
        win = np.linspace(0, 1, f, dtype=arr.dtype)
        arr[s : s + f] *= win  # in
        arr[e - f : e] *= win[::-1]  # out


def process_audio(cfg, row):
    """1 ファイル → random crop → 人声マスク → mel"""
    try:
        # --- load ---------------------------------------------------------
        wav, _ = librosa.load(row["filepath"], sr=cfg.spec.fs, mono=True)
        seg_len = int(cfg.spec.window_size * cfg.spec.fs)

        # center_audio, crop_start = random_crop(wav, seg_len, RNG)
        # center_audio, crop_start = rms_crop(wav, seg_len, cfg.spec.fs)
        center_audio, crop_start = rms_crop_shift(
            wav=wav, seg_len=seg_len, sr=cfg.spec.fs
        )

        real_len = len(center_audio)
        # --- human-voice masking -----------------------------------------
        voice_segments = VOICE_DATA.get(row["filepath"])
        if voice_segments:
            real_len = len(center_audio)
            for v in voice_segments:
                g_s = int(v["start"] * cfg.spec.fs)
                g_e = int(v["end"] * cfg.spec.fs)

                s = g_s - crop_start
                e = g_e - crop_start

                s = np.clip(s, 0, real_len)
                e = np.clip(e, 0, real_len)

                if e - s > 0:
                    _mask_voice_with_noise(center_audio, s, e)

        # --- mel ----------------------------------------------------------
        mel_spec = process_audio_segment(cfg, center_audio)
        return row["samplename"], mel_spec.astype(np.float32)

    except Exception as e:
        LOGGER.error(f"{row['filename']} でエラー: {e}")


@hydra.main(config_path="conf", config_name="preprocess", version_base="1.1")
def main(cfg: PreprocessConfig):
    """
    audio data を mel-spectrogram data へ変換。
    2D CNN を学習するために必要。

    ref: https://www.kaggle.com/code/kadircandrisolu/transforming-audio-to-mel-spec-birdclef-25
    """
    # Log the current configuration for the sweep
    LOGGER.info(
        f"Running with configuration: n_mels={cfg.spec.n_mels}, hop_length={cfg.spec.hop_length}, fmin={cfg.spec.fmin}, fmax={cfg.spec.fmax}"
    )

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

    # 並列処理で実行
    # librosa の I/O や Numpy の処理は GIL の影響を受けにくいので恩恵がある
    LOGGER.info("Starting audio processing...")
    start_time = time.time()

    all_bird_data = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_audio, cfg, row) for _, row in working_df.iterrows()
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
