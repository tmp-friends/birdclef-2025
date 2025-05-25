import os
from tqdm import tqdm
from pathlib import Path
import glob
import pickle

import numpy as np
import pandas as pd
import librosa
import torch

from utils.audio2melspec import process_audio_segment

VOICE_DATA_PATH = "/home/tomoya/kaggle/birdclef-2025/output/eda/ss_voice_data.pkl"
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


def make_pl_windows(cfg, model, taxonomy_df):
    """
    unlabeled soundscapes から 5 s 窓を切り出して pseudo-label を付与する

    Returns
    -------
    pl_df : pd.DataFrame
        filename, primary_label (= predicted id), melspec の列をもつ
    """
    model.eval()
    device = cfg.device

    rows = []
    soundscape_paths = sorted(glob.glob(f"{cfg.dir.train_soundscapes_dir}/*.ogg"))

    seg_len = int(cfg.spec.window_size * cfg.spec.fs)
    id2label = dict(enumerate(taxonomy_df["primary_label"]))

    with torch.no_grad():
        for path in tqdm(soundscape_paths, desc="pseudo-label"):
            wav, _ = librosa.load(path, sr=cfg.spec.fs, mono=True)

            # ★ ① 人声マスクを全長に先掛けしておく
            voice_segments = VOICE_DATA.get(path)
            if voice_segments:
                for v in voice_segments:
                    s = int(v["start"] * cfg.spec.fs)
                    e = int(v["end"] * cfg.spec.fs)
                    _mask_voice_with_noise(wav, s, e)

            n_segs = int(np.ceil(len(wav) / seg_len))
            for k in range(n_segs):
                seg = wav[k * seg_len : (k + 1) * seg_len]
                if len(seg) < seg_len:  # 末尾パディング
                    seg = np.pad(seg, (0, seg_len - len(seg)), mode="reflect")

                mel = process_audio_segment(cfg, seg)

                x = (
                    torch.tensor(mel, dtype=torch.float32)
                    .unsqueeze(0)  # C
                    .unsqueeze(0)  # B
                    .to(device)
                )
                logits = model(x).sigmoid().cpu().numpy()[0]
                best_idx = logits.argmax()
                best_prob = logits[best_idx]

                # ★ ② しきい値を超えたものだけ pseudo-label へ
                if best_prob >= cfg.pl_threshold:
                    row_id = f"{Path(path).stem}_{(k + 1) * cfg.spec.window_size}"
                    fname = f"pl_{row_id}.ogg"  # ダミー名

                    rows.append(
                        {
                            "filename": fname,
                            "primary_label": id2label[best_idx],
                            "melspec": mel.astype(np.float32),
                            "prob": best_prob,
                        }
                    )

    pl_df = pd.DataFrame(rows)

    return pl_df
