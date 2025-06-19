# 過去コンペの上位 Solution
## 2024
### 1st
https://www.kaggle.com/competitions/birdclef-2024/discussion/512197

パート | キーアイデア | 具体的な技術
-- | -- | --
データ & 前処理 | - 公式 train_audio + 疑似ラベル付き unlabeled_soundscapes <br> - Google Bird Vocalization Classifier で ノイズ判定 & ラベル補正 <br> - 統計 T = std+var+rms+pwr の 0.8 分位で「うるさ過ぎ／静か過ぎ」クリップを除外 | 182 種（NoCall なし）<br>重複クリップ除去・短音は Cyclic padding
入力 | 10 秒チャンク = 5 s+5 s をモデル入力にし、ラベルは 2つの 5 s ウィンドウを平均→ 完全な鳴き声周期を含める | Mel: 128 bin / hop 500 sample (~15.6 ms) <br>サイズ 1 × 128 × 640
モデル | - EfficientNet-B0 ×3 <br>- RegNetY_008 ×3（全て ImageNet 初期重み） | 損失 CrossEntropy（マルチクラス扱い）学習 7–12 epoch・BS 96・AdamW・CosineLR
Augmentation | - XY-Masking (SpecAug) <br>- Horizontal CutMix <br>- ランダム 5 s 抽出 | BCE は性能↓、CE+Aug が安定
疑似ラベル | - Google Classifier 予測を 0.05 重みで混ぜ「ラベルスムージング」<br>- Soundscapes は自モデル+Google でアンサンブルラベル付け | 2 段階 PL → Private +0.05 AUC
後処理 | 10 s ウィンドウで推論 → 3 ウィンドウ平均（chunk n±1）min-ensemble で不確実クラスを抑制 | CE 学習でも推論は SigmoidXY Masking & CutMix のノイズを平滑化
アンサンブル | - Eff-B0 3 枚 + RegNetY 3 枚 <br>- Reduction = mean （モデル間）<br>- 1st Place提出は Eff-B0×6 min-reduce もトライ | Private AUC 0.690 (mean 3+3)
推論最適化 | - Mel を joblib 並列計算 & RAMキャッシュ<br>- OpenVINO INT8 コンパイル | 単体モデル 18 min / 全テスト <br>6 モデルでも 2 h (Kaggle CPU limit 内)
効果が薄かった/悪化 | BCE / Focal Loss, 1 D Aug, Mixup noise, STFT, 外部 XC データ、マルチステージ訓練 | 「シンプルが強い」と結論

### 2nd

https://www.kaggle.com/competitions/birdclef-2024/discussion/512340

| **パート** | **キーアイデア** | **具体的な技術** |
|---|---|---|
| データ前処理 | 先頭 5 秒だけ使いノイズを抑制 | ・`librosa.load` → 5 s crop<br>・短尺 reflect pad<br>・重複除外 |
| 擬似ラベル生成 | **テスト領域 5 s クリップに擬似ラベル追加** | ・自モデル ensemble で予測<br>・25–45 % の確率で mix<br>・`ampFactor = 10**U(min,max)` で音量ランダム |
| 入力特徴 | log-mel + Δ + ΔΔ の 3ch 画像化 | ・`n_mels` 64/128<br>・`hop` 512/1024<br>・サイズ 256×256 / 128×128 / 64×64 |
| モデル設計 | 軽量 EfficientNet B0 を高速化 | ・`tf_efficientnet_b0_ns`<br>・Dropout×5 (fc 前)<br>・Checkpoint-Soup (Ep 13–50) |
| 損失関数 | BCE と Focal をハイブリッド | 0.5 × BCE + 0.5 × FocalLoss |
| 学習設定 | 速い収束・過学習抑制 | CosineAnnealingLR + 3 warm-up<br>LR 1e-3 / Epoch 50 / Batch 64 |
| データ拡張 | メル画像上で多方向のゆらぎ | HorizontalFlip<br>CoarseDropout<br>Mixup<br>局所＋全体 Time/Freq Stretch |
| アンサンブル | 6 モデル平均で安定化 | Mel param / 画像サイズ / データ subset / pseudoLabelChance / AmpFactor を変化 |
| ポストプロセス | 時間平滑化で誤検出抑制 | 窓 t の確率 +0.5×(t-1) +0.5×(t+1) |
| 推論最適化 | CPU 2 h 制限内 | マルチスレッドで Mel 生成共有<br>小画像モデル併用 |

### 3rd

https://www.kaggle.com/competitions/birdclef-2024/discussion/511905

| **パート** | **キーアイデア** | **具体的な技術** |
|---|---|---|
| データ前処理 | 今年＋過去＋Xeno-Canto を統合し<br>不均衡を抑える | ・種ごと **≤500** clip に制限<br>・頻度 <10 のクラスを up-sample<br>・各録音から **先頭 6 s または末尾 6 s 内でランダム 5 s crop**<br>・不足はランダム pad（中心 2–3 s に来るよう調整）<br>・Additive mix-up（sec.label は結合） |
| 擬似ラベル生成 / 蒸留 | **1st レベル多様モデルで unlabeled 5 s を推論 → 学習データに注入** | ・First-level（EfficientVit 系 etc.）を 5seed 学習<br>・unlabeled soundscape→5 s 切り出し→確率ラベル<br>・Second-level 学習時 **① 128 real +192 pseudo / ② 128 pseudo only** の大バッチ |
| 入力特徴 | 画像モデル用 log-mel (224×224 / 288×288) | ・Mel param をモデルごとに最適化<br>・波形を std=1 に正規化 |
| モデル設計 | **二段階アーキ**：大モデルで pseudo、軽モデルで提出 | *First level* : EfficientVit-b0,-b1,-m3, EfficientNet, MobileNet, Aves 等<br>*Second level* : **efficientvit-b0** (高速) ＋ mnasnet-100 |
| 学習設定 | 長期学習で distillation、2 段で Epoch 差 | ・1st: 30 epoch<br>・2nd: 88 epoch<br>・Optimizer AdamW / Ranger |
| 損失関数 | sec-label の曖昧さを無視 | **BCEWithLogits で primary のみ勾配**<br>secondary label → loss ×0 |
| データ拡張 | 軽量 aug 中心で CPU 友好 | ・1 s time-shift<br>・Additive mix-up<br>・一部モデルは BirdCLEF23 SED aug パック採用 |
| アンサンブル | チーム 3 人×パイプラインで **14 重み** | Pipeline1: vit-b0→vit-b0<br>Pipeline2: 多様 CNN → 2×mnasnet<br>Pipeline3: vit-b1 / SED → 5×vit-b0<br>→ **mean** 合成 |
| ポストプロセス | サウンドスケープ文脈を注入 | ① soundscape-max ブースト `P + (...)*0.8`（効果小）<br>② Convolution 平滑 [0.1 0.2 0.4 0.2 0.1]（+0.01 公開LB） |
| 推論最適化 | CPU 提出を前提に計算共有 | ・Mel 変換を 1 回だけ実行し全モデルに再利用<br>・ONNX 化で +10 %、OpenVINO は速いが精度↓で不採用 |
| 結果 | 大規模蒸留 + 軽量モデルで高速高精度 | **Public 0.742 / Private 0.690** (ensemble) |

