"""
Pseudo labeling pipeline for BirdCLEF 2025
Based on successful approaches from BirdCLEF 2023/2024 solutions

Features:
- Multi-model ensemble for pseudo label generation
- Confidence-based filtering
- Label smoothing and mixing strategies
- Integration with unlabeled soundscape data
"""

import logging
import os
from pathlib import Path
import glob
from tqdm import tqdm
import pickle

import hydra
from omegaconf import DictConfig

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from utils.utils import set_seed
from modules.birdclef_model import BirdCLEFSEDModel
from utils.audio_augmentations import SEDAugmentationPipeline


class UnlabeledSoundscapeDataset(torch.utils.data.Dataset):
    """
    Dataset for processing unlabeled soundscape data for pseudo labeling
    """
    def __init__(
        self,
        cfg: DictConfig,
        soundscape_files: list,
        chunk_duration: float = 5.0,
        overlap_ratio: float = 0.5,
    ):
        self.cfg = cfg
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * cfg.fs)
        self.overlap_samples = int(self.chunk_samples * overlap_ratio)
        
        # Index soundscape files as metadata only (memory efficient)
        self.chunks_meta = []
        self._index_soundscapes(soundscape_files)
        
        LOGGER.info(f"Indexed {len(self.chunks_meta)} chunks from {len(soundscape_files)} soundscape files")
    
    def _index_soundscapes(self, soundscape_files):
        """Index soundscape files without loading audio data"""
        for file_path in tqdm(soundscape_files, desc="Indexing soundscapes"):
            try:
                # Get file info without loading audio
                info = sf.info(file_path)
                sr = info.samplerate
                nframes = info.frames
                
                # Skip files with different sample rate (would need resampling)
                if sr != self.cfg.fs:
                    LOGGER.debug(f"Skipping {file_path}: sample rate {sr} != {self.cfg.fs}")
                    continue
                
                # Extract chunk indices with overlap
                stride = self.chunk_samples - self.overlap_samples
                for start_idx in range(0, nframes - self.chunk_samples + 1, stride):
                    # Store metadata only (file path + start position)
                    self.chunks_meta.append((file_path, start_idx))
                        
            except Exception as e:
                LOGGER.warning(f"Failed to index {file_path}: {e}")
    
    def __len__(self):
        return len(self.chunks_meta)
    
    def __getitem__(self, idx):
        file_path, start_idx = self.chunks_meta[idx]
        
        try:
            # Load only the required 5-second chunk
            audio, sr = sf.read(
                file_path,
                start=start_idx,
                frames=self.chunk_samples,
                dtype='float32'
            )
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Pad or trim to exact length
            if len(audio) < self.chunk_samples:
                audio = np.pad(audio, (0, self.chunk_samples - len(audio)))
            elif len(audio) > self.chunk_samples:
                audio = audio[:self.chunk_samples]
            
            # Skip chunks that are too quiet (likely background noise)
            rms = np.sqrt(np.mean(audio**2))
            if rms <= 0.001:  # Minimum RMS threshold
                audio = np.zeros_like(audio)  # Return silent chunk instead of skipping
            
            # Convert to tensor and add channel dimension
            audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            
            return {
                'wave': audio_tensor,
                'file_path': file_path,
                'start_time': start_idx / self.cfg.fs,
                'end_time': (start_idx + self.chunk_samples) / self.cfg.fs,
            }
            
        except Exception as e:
            LOGGER.warning(f"Failed to load chunk from {file_path} at {start_idx}: {e}")
            # Return silent chunk on error
            audio = np.zeros(self.chunk_samples, dtype=np.float32)
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            
            return {
                'wave': audio_tensor,
                'file_path': file_path,
                'start_time': start_idx / self.cfg.fs,
                'end_time': (start_idx + self.chunk_samples) / self.cfg.fs,
            }


class PseudoLabelGenerator:
    """
    Generate pseudo labels using ensemble of trained models
    """
    def __init__(
        self,
        cfg: DictConfig,
        model_checkpoints: list,
        taxonomy_df: pd.DataFrame,
        confidence_threshold: float = 0.5,
        ensemble_method: str = "mean",  # "mean", "max", "vote"
    ):
        self.cfg = cfg
        self.taxonomy_df = taxonomy_df
        self.confidence_threshold = confidence_threshold
        self.ensemble_method = ensemble_method
        self.num_classes = len(taxonomy_df)
        
        # Load models
        self.models = []
        self._load_models(model_checkpoints)
        
        # Create label mapping
        self.label2idx = {
            label: idx for idx, label in enumerate(taxonomy_df["primary_label"])
        }
        self.idx2label = {v: k for k, v in self.label2idx.items()}
    
    def _load_models(self, model_checkpoints):
        """Load ensemble of trained models"""
        LOGGER.info(f"Loading {len(model_checkpoints)} models for ensemble")
        
        for checkpoint_path in model_checkpoints:
            if not os.path.exists(checkpoint_path):
                LOGGER.warning(f"Checkpoint not found: {checkpoint_path}")
                continue
            
            try:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.cfg.device)
                
                # Create model
                model = BirdCLEFSEDModel(self.cfg).to(self.cfg.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                
                self.models.append(model)
                LOGGER.info(f"Loaded model from {checkpoint_path}")
                
            except Exception as e:
                LOGGER.error(f"Failed to load model from {checkpoint_path}: {e}")
        
        LOGGER.info(f"Successfully loaded {len(self.models)} models")
    
    def predict_batch(self, batch):
        """Generate predictions for a batch using model ensemble"""
        waves = batch['wave'].to(self.cfg.device)
        
        # Get predictions from all models
        all_predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(waves)
                pred = torch.sigmoid(pred)  # Convert to probabilities
                all_predictions.append(pred.cpu().numpy())
        
        # Ensemble predictions
        all_predictions = np.stack(all_predictions, axis=0)  # (num_models, batch_size, num_classes)
        
        if self.ensemble_method == "mean":
            ensemble_pred = np.mean(all_predictions, axis=0)
        elif self.ensemble_method == "max":
            ensemble_pred = np.max(all_predictions, axis=0)
        elif self.ensemble_method == "vote":
            # Binary voting (threshold at 0.5, then majority vote)
            binary_preds = (all_predictions > 0.5).astype(int)
            ensemble_pred = np.mean(binary_preds, axis=0)
        else:
            ensemble_pred = np.mean(all_predictions, axis=0)
        
        return ensemble_pred
    
    def generate_pseudo_labels(self, unlabeled_dataset, output_path: str):
        """Generate pseudo labels for unlabeled data"""
        # Optimize DataLoader for memory efficiency with I/O intensive workload
        batch_size = min(self.cfg.batch_size, 32)  # Smaller batches for memory efficiency
        num_workers = min(self.cfg.num_workers, 4)  # Limit workers to avoid too many file handles
        
        dataloader = DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive to avoid restart overhead
            prefetch_factor=2,        # Reduce prefetch to save memory
        )
        
        pseudo_labels = []
        flush_every = 10000  # Flush every 10K pseudo labels to save memory
        
        for batch in tqdm(dataloader, desc="Generating pseudo labels"):
            # Get ensemble predictions
            predictions = self.predict_batch(batch)
            
            # Process each sample in the batch
            for i in range(len(batch['file_path'])):
                pred = predictions[i]
                
                # Find confident predictions
                confident_classes = np.where(pred > self.confidence_threshold)[0]
                
                if len(confident_classes) > 0:
                    # Create pseudo label entry
                    primary_class_idx = confident_classes[np.argmax(pred[confident_classes])]
                    primary_label = self.idx2label[primary_class_idx]
                    primary_confidence = pred[primary_class_idx]
                    
                    # Secondary labels (other confident predictions)
                    secondary_labels = []
                    secondary_confidences = []
                    for class_idx in confident_classes:
                        if class_idx != primary_class_idx:
                            secondary_labels.append(self.idx2label[class_idx])
                            secondary_confidences.append(pred[class_idx])
                    
                    pseudo_label = {
                        'file_path': batch['file_path'][i],
                        'start_time': batch['start_time'][i].item(),
                        'end_time': batch['end_time'][i].item(),
                        'primary_label': primary_label,
                        'primary_confidence': primary_confidence,
                        'secondary_labels': secondary_labels,
                        'secondary_confidences': secondary_confidences,
                        'all_predictions': pred.tolist(),
                    }
                    
                    pseudo_labels.append(pseudo_label)
            
            # Periodic flush to save memory
            if len(pseudo_labels) >= flush_every:
                LOGGER.info(f"Flushing {len(pseudo_labels)} pseudo labels to disk...")
                temp_path = output_path.replace('.pkl', f'_temp_{len(pseudo_labels)}.pkl')
                with open(temp_path, 'wb') as f:
                    pickle.dump(pseudo_labels, f)
                pseudo_labels.clear()
                import gc
                gc.collect()
        
        # Collect all temp files and merge
        import glob
        temp_files = glob.glob(output_path.replace('.pkl', '_temp_*.pkl'))
        all_pseudo_labels = pseudo_labels.copy()  # Final batch
        
        # Load and merge temp files
        for temp_file in temp_files:
            try:
                with open(temp_file, 'rb') as f:
                    temp_labels = pickle.load(f)
                all_pseudo_labels.extend(temp_labels)
                os.remove(temp_file)  # Clean up temp file
                LOGGER.info(f"Merged {len(temp_labels)} labels from {temp_file}")
            except Exception as e:
                LOGGER.warning(f"Failed to load temp file {temp_file}: {e}")
        
        # Save all pseudo labels
        LOGGER.info(f"Generated {len(all_pseudo_labels)} pseudo labels total")
        with open(output_path, 'wb') as f:
            pickle.dump(all_pseudo_labels, f)
        
        # Also save as CSV for inspection
        csv_data = []
        for pl in all_pseudo_labels:
            csv_data.append({
                'file_path': pl['file_path'],
                'start_time': pl['start_time'],
                'end_time': pl['end_time'],
                'primary_label': pl['primary_label'],
                'primary_confidence': pl['primary_confidence'],
                'num_secondary': len(pl['secondary_labels']),
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = output_path.replace('.pkl', '.csv')
        csv_df.to_csv(csv_path, index=False)
        
        LOGGER.info(f"Saved pseudo labels to {output_path} and {csv_path}")
        return all_pseudo_labels


class PseudoLabelIntegrator:
    """
    Integrate pseudo labels with original training data
    """
    def __init__(
        self,
        cfg: DictConfig,
        taxonomy_df: pd.DataFrame,
        pseudo_label_weight: float = 0.7,
        label_smoothing: float = 0.1,
    ):
        self.cfg = cfg
        self.taxonomy_df = taxonomy_df
        self.pseudo_label_weight = pseudo_label_weight
        self.label_smoothing = label_smoothing
        self.num_classes = len(taxonomy_df)
        
        self.label2idx = {
            label: idx for idx, label in enumerate(taxonomy_df["primary_label"])
        }
    
    def create_pseudo_training_data(
        self,
        original_train_df: pd.DataFrame,
        pseudo_labels: list,
        mixing_ratio: float = 0.3,  # Ratio of pseudo to real data
    ) -> pd.DataFrame:
        """
        Create augmented training dataset with pseudo labels
        """
        # Convert pseudo labels to DataFrame format
        pseudo_df_data = []
        
        for pl in pseudo_labels:
            # Create filename-like identifier for pseudo samples
            filename = f"pseudo_{Path(pl['file_path']).stem}_{pl['start_time']:.1f}_{pl['end_time']:.1f}.wav"
            
            pseudo_df_data.append({
                'filename': filename,
                'primary_label': pl['primary_label'],
                'secondary_labels': pl['secondary_labels'],
                'is_pseudo': True,
                'confidence': pl['primary_confidence'],
                'file_path': pl['file_path'],
                'start_time': pl['start_time'],
                'end_time': pl['end_time'],
            })
        
        pseudo_df = pd.DataFrame(pseudo_df_data)
        
        # Filter by confidence if needed
        high_confidence_pseudo = pseudo_df[pseudo_df['confidence'] > 0.7].copy()
        
        # Sample pseudo data according to mixing ratio
        num_pseudo_samples = int(len(original_train_df) * mixing_ratio)
        if len(high_confidence_pseudo) > num_pseudo_samples:
            # Sample high confidence pseudo labels
            sampled_pseudo = high_confidence_pseudo.sample(
                n=num_pseudo_samples, 
                random_state=self.cfg.seed
            ).copy()
        else:
            sampled_pseudo = high_confidence_pseudo.copy()
        
        # Add is_pseudo flag to original data
        original_train_df = original_train_df.copy()
        original_train_df['is_pseudo'] = False
        
        # Combine original and pseudo data
        combined_df = pd.concat([original_train_df, sampled_pseudo], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=self.cfg.seed).reset_index(drop=True)
        
        LOGGER.info(f"Created combined dataset with {len(original_train_df)} real + {len(sampled_pseudo)} pseudo samples")
        
        return combined_df
    
    def apply_label_smoothing(self, labels: torch.Tensor, smoothing: float = None) -> torch.Tensor:
        """Apply label smoothing to reduce overconfidence"""
        if smoothing is None:
            smoothing = self.label_smoothing
        
        # Convert hard labels to soft labels
        num_classes = labels.shape[-1]
        smooth_labels = labels * (1 - smoothing) + smoothing / num_classes
        
        return smooth_labels


def find_model_checkpoints(checkpoint_dir: str, pattern: str = "sed_model_fold*.pth") -> list:
    """Find model checkpoint files"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    return sorted(checkpoint_files)


def find_unlabeled_soundscapes(soundscape_dir: str, extensions: list = ['.wav', '.ogg', '.mp3']) -> list:
    """Find unlabeled soundscape files"""
    soundscape_files = []
    for ext in extensions:
        pattern = os.path.join(soundscape_dir, f"*{ext}")
        soundscape_files.extend(glob.glob(pattern))
    
    return sorted(soundscape_files)


@hydra.main(config_path="conf", config_name="train_sed_advanced", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    
    # Load taxonomy
    taxonomy_df = pd.read_csv(cfg.dir.taxonomy_csv)
    
    # Find model checkpoints
    checkpoint_dir = cfg.get('checkpoint_dir', './checkpoints')
    model_checkpoints = find_model_checkpoints(checkpoint_dir)
    
    if not model_checkpoints:
        LOGGER.error(f"No model checkpoints found in {checkpoint_dir}")
        return
    
    LOGGER.info(f"Found {len(model_checkpoints)} model checkpoints")
    
    # Find unlabeled soundscape data (use train_soundscapes_dir)
    soundscape_dir = getattr(cfg.dir, 'train_soundscapes_dir', './train_soundscapes')
    if not os.path.exists(soundscape_dir):
        LOGGER.warning(f"Soundscape directory not found: {soundscape_dir}")
        return
    
    soundscape_files = find_unlabeled_soundscapes(soundscape_dir)
    if not soundscape_files:
        LOGGER.warning(f"No soundscape files found in {soundscape_dir}")
        return
    
    # Limit soundscape files for testing (set to None for all files)
    max_files = getattr(cfg, 'max_soundscape_files', None)  # None = use all files
    if max_files is not None and len(soundscape_files) > max_files:
        soundscape_files = soundscape_files[:max_files]
        LOGGER.info(f"Limited to {max_files} soundscape files for processing")
    
    LOGGER.info(f"Found {len(soundscape_files)} unlabeled soundscape files")
    
    # Create unlabeled dataset
    unlabeled_dataset = UnlabeledSoundscapeDataset(
        cfg=cfg,
        soundscape_files=soundscape_files,
        chunk_duration=5.0,  # 5-second chunks
        overlap_ratio=0.5,   # 50% overlap
    )
    
    # Generate pseudo labels
    pseudo_label_generator = PseudoLabelGenerator(
        cfg=cfg,
        model_checkpoints=model_checkpoints,
        taxonomy_df=taxonomy_df,
        confidence_threshold=0.6,  # Higher threshold for quality
        ensemble_method="mean",
    )
    
    output_path = "pseudo_labels.pkl"
    pseudo_labels = pseudo_label_generator.generate_pseudo_labels(
        unlabeled_dataset, 
        output_path
    )
    
    # Integrate with original training data
    original_train_df = pd.read_csv(cfg.dir.train_csv)
    
    integrator = PseudoLabelIntegrator(
        cfg=cfg,
        taxonomy_df=taxonomy_df,
        pseudo_label_weight=0.7,
        label_smoothing=0.1,
    )
    
    combined_df = integrator.create_pseudo_training_data(
        original_train_df,
        pseudo_labels,
        mixing_ratio=0.3,  # 30% pseudo data
    )
    
    # Save combined dataset
    combined_path = "combined_train_with_pseudo.csv"
    combined_df.to_csv(combined_path, index=False)
    LOGGER.info(f"Saved combined training data to {combined_path}")
    
    # Print statistics
    LOGGER.info("\nPseudo Labeling Statistics:")
    LOGGER.info(f"Original training samples: {len(original_train_df)}")
    LOGGER.info(f"Generated pseudo labels: {len(pseudo_labels)}")
    LOGGER.info(f"High-confidence pseudo labels used: {len(combined_df[combined_df['is_pseudo']])}")
    LOGGER.info(f"Total combined samples: {len(combined_df)}")
    
    # Class distribution
    LOGGER.info("\nClass distribution in pseudo labels:")
    if pseudo_labels:
        pseudo_classes = [pl['primary_label'] for pl in pseudo_labels]
        class_counts = pd.Series(pseudo_classes).value_counts()
        LOGGER.info(f"Top 10 classes: {class_counts.head(10).to_dict()}")


if __name__ == "__main__":
    # Logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
    )
    LOGGER = logging.getLogger(Path(__file__).name)
    
    main()