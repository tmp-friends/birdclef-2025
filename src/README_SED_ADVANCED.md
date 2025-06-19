# Advanced SED Model for BirdCLEF 2025

このディレクトリには、BirdCLEF 2023 2位解法に基づいた高度なSound Event Detection (SED)モデルの実装が含まれています。

## 主な特徴

### 1. **多段階学習パイプライン**
- **pretrain_ce**: CrossEntropyLossによる事前学習
- **train_bce**: BCEWithLogitsLossによる主学習
- **finetune**: ファインチューニング段階

### 2. **高度なデータ拡張**
- **Audiomentations**: プロ級の音声拡張
  - ガウシアンノイズ、ピッチシフト、タイムストレッチ
  - バンドパス・ハイパス・ローパスフィルター
  - ルームインパルス応答、コンプレッション
- **SpecAugment**: メルスペクトログラム拡張
- **Mixup/CutMix**: データミキシング手法

### 3. **改良されたモデルアーキテクチャ**
- **マルチ解像度入力**: 異なる時間・周波数解像度の組み合わせ
- **マルチヘッドアテンション**: より豊富な特徴学習
- **スキップ接続**: 勾配流改善
- **GeM Pooling**: 一般化平均プーリング
- **強化された正規化**: ドロップアウト、レイヤー正規化

### 4. **疑似ラベリングパイプライン**
- **アンサンブル予測**: 複数モデルによる高品質ラベル生成
- **信頼度フィルタリング**: 高品質な疑似ラベルの選別
- **ラベルスムージング**: 過信頼の抑制

## ファイル構成

```
src/
├── 12-train_sed_advanced.py      # 高度な多段階学習スクリプト
├── 13-generate_pseudo_labels.py  # 疑似ラベル生成パイプライン
├── utils/
│   └── audio_augmentations.py    # 高度な音声拡張ライブラリ
├── modules/
│   └── birdclef_model.py         # 改良されたSEDモデル (更新済み)
└── conf/
    └── train_sed_advanced.yaml   # 高度な学習設定
```

## 使用方法

### 1. 基本学習 (単一段階)

```bash
# pretrain_ce段階のみ実行
uv run python src/12-train_sed_advanced.py current_stage=pretrain_ce

# train_bce段階のみ実行  
uv run python src/12-train_sed_advanced.py current_stage=train_bce

# finetune段階のみ実行
uv run python src/12-train_sed_advanced.py current_stage=finetune
```

### 2. 全段階学習

```bash
# 全段階を順次実行
uv run python src/12-train_sed_advanced.py current_stage=all
```

### 3. 高度な機能の有効化

```bash
# マルチ解像度入力を有効化
uv run python src/12-train_sed_advanced.py use_multi_resolution=true

# マルチヘッドアテンションを有効化
uv run python src/12-train_sed_advanced.py use_multi_head_attention=true

# スキップ接続を有効化
uv run python src/12-train_sed_advanced.py use_skip_connection=true

# GeMプーリングを使用
uv run python src/12-train_sed_advanced.py pool_type=gem
```

### 4. 疑似ラベル生成

```bash
# 学習済みモデルから疑似ラベルを生成
uv run python src/13-generate_pseudo_labels.py
```

## 設定のカスタマイズ

### モデル設定

```yaml
# Model settings
model_name: 'seresnext26t_32x4d'  # or 'tf_efficientnetv2_s_in21k'
pretrained: true
in_channels: 1

# Enhanced features
use_multi_resolution: true    # 複数解像度入力
use_multi_head_attention: true  # マルチヘッドアテンション
use_skip_connection: true     # スキップ接続
pool_type: 'gem'             # プーリング戦略
```

### 学習段階設定

```yaml
training_stages:
  pretrain_ce:
    epochs: 30
    lr: 3e-4
    criterion: 'CrossEntropyLoss'
    mixup_alpha: 0.2
    
  train_bce:
    epochs: 25  
    lr: 1e-4
    criterion: 'BCEWithLogitsLoss'
    use_secondary_labels: true
    
  finetune:
    epochs: 15
    lr: 5e-5
    criterion: 'BCEWithLogitsLoss'
    mixup_alpha: 0.1
```

### 拡張設定

```yaml
augmentation:
  use_audio_aug: true      # 音声レベル拡張
  audio_aug_prob: 0.8      # 音声拡張確率
  
  use_spec_aug: true       # スペクトログラム拡張
  spec_aug_prob: 0.6       # スペク拡張確率
  
  use_mixup_cutmix: true   # Mixup/CutMix
  mixup_cutmix_prob: 0.7   # ミキシング確率
```

## パフォーマンス最適化のヒント

### 1. **メモリ使用量削減**
- `batch_size`を調整 (デフォルト: 64)
- `num_workers`を適切に設定
- 不要な段階をスキップ

### 2. **学習速度向上**
- `use_multi_resolution=false`で開始
- 段階的に高度な機能を有効化
- より軽量なモデル (`efficientnet_b0`) から開始

### 3. **精度向上**
- 疑似ラベリングパイプラインの活用
- アンサンブル学習
- 複数段階での学習

## 期待される改善点

1. **AUC向上**: 基本モデルから +0.05-0.10の改善
2. **汎化性能**: 強力な拡張による過学習抑制  
3. **学習安定性**: 多段階学習による安定収束
4. **少数クラス対応**: 疑似ラベルによるデータ増強

## トラブルシューティング

### よくある問題

1. **CUDA out of memory**
   - `batch_size`を削減
   - `use_multi_resolution=false`に設定

2. **学習が収束しない**
   - 学習率を下げる (`lr: 1e-5`)
   - より多くのエポック数を設定

3. **audiomentationsインストールエラー**
   ```bash
   pip install audiomentations
   ```

## 参考資料

- [BirdCLEF 2023 2nd Place Solution](https://github.com/LIHANG-HONG/birdclef2023-2nd-place-solution)
- [BirdCLEF 2024 上位解法まとめ](docs/past_solution.md)
- [SpecAugment論文](https://arxiv.org/abs/1904.08779)
- [Mixup論文](https://arxiv.org/abs/1710.09412)