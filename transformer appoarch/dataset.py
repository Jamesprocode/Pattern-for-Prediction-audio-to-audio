"""
Dataset for mel spectrogram continuation task.
Handles loading audio files, converting to mel spectrograms,
and creating training pairs (context → target).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
from pathlib import Path


class MelContinuationDataset(Dataset):
    """
    Dataset for mel spectrogram continuation.
    
    Each sample returns:
    - context: mel spectrogram frames (n_mels, context_frames)
    - target: mel spectrogram frames to predict (n_mels, predict_frames)
    """
    
    def __init__(
        self,
        audio_files,
        sr=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=80,
        context_frames=100,
        predict_frames=50,
        min_duration=3.0,  # Minimum audio duration in seconds
        normalize=True
    ):
        """
        Args:
            audio_files: List of paths to audio files
            sr: Sample rate for loading audio
            n_fft: FFT window size
            hop_length: Number of samples between frames
            n_mels: Number of mel frequency bins
            context_frames: Number of frames to use as context
            predict_frames: Number of frames to predict
            min_duration: Minimum audio duration (seconds)
            normalize: Whether to normalize mel spectrograms
        """
        self.audio_files = audio_files
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.context_frames = context_frames
        self.predict_frames = predict_frames
        self.min_duration = min_duration
        self.normalize = normalize
        
        # Filter out audio files that are too short
        self.valid_files = self._filter_valid_files()
        
        print(f"Dataset initialized:")
        print(f"  Total files: {len(audio_files)}")
        print(f"  Valid files: {len(self.valid_files)}")
        print(f"  Context frames: {context_frames}")
        print(f"  Predict frames: {predict_frames}")
        print(f"  Mel bins: {n_mels}")
    
    def _filter_valid_files(self):
        """Filter files that are long enough"""
        valid = []
        min_samples = self.min_duration * self.sr
        
        for file in self.audio_files:
            try:
                duration = librosa.get_duration(path=file, sr=self.sr)
                if duration >= self.min_duration:
                    valid.append(file)
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
        
        return valid
    
    def _audio_to_mel(self, audio):
        """
        Convert audio waveform to mel spectrogram.
        
        Args:
            audio: Audio waveform (numpy array)
        
        Returns:
            mel_db: Mel spectrogram in dB scale (n_mels, time_frames)
        """
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=20,  # Minimum frequency
            fmax=8000  # Maximum frequency (good for piano)
        )
        
        # Convert to dB scale
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [-1, 1] range
        if self.normalize:
            mel_db = mel_db / 80.0  # Typical dB range is ~80dB
        
        return mel_db
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        """
        Get one training sample.
        
        Returns:
            context: (n_mels, context_frames)
            target: (n_mels, predict_frames)
        """
        # Load audio
        audio, _ = librosa.load(self.valid_files[idx], sr=self.sr)
        
        # Convert to mel spectrogram
        mel_db = self._audio_to_mel(audio)
        
        total_frames = mel_db.shape[1]
        required_frames = self.context_frames + self.predict_frames
        
        # If audio is too short, pad it
        if total_frames < required_frames:
            pad_width = required_frames - total_frames
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1.0)
            total_frames = required_frames
        
        # Random window selection (for data augmentation)
        max_start = total_frames - required_frames
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        
        # Extract context and target
        context = mel_db[:, start:start + self.context_frames]
        target = mel_db[:, start + self.context_frames:start + required_frames]
        
        # Convert to tensors
        context = torch.FloatTensor(context)
        target = torch.FloatTensor(target)
        
        return context, target


def collect_audio_files(data_dir, extensions=['.wav', '.mp3', '.flac', '.m4a']):
    """
    Recursively collect all audio files from a directory.
    
    Args:
        data_dir: Root directory to search
        extensions: List of valid audio file extensions
    
    Returns:
        List of audio file paths
    """
    audio_files = []
    data_path = Path(data_dir)
    
    for ext in extensions:
        audio_files.extend(list(data_path.rglob(f'*{ext}')))
    
    return [str(f) for f in audio_files]


def create_dataloaders(
    train_dir,
    val_dir=None,
    batch_size=32,
    num_workers=4,
    **dataset_kwargs
):
    """
    Create train and validation dataloaders.
    
    Args:
        train_dir: Directory with training audio files
        val_dir: Directory with validation audio files (optional)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments for MelContinuationDataset
    
    Returns:
        train_loader, val_loader (val_loader is None if val_dir not provided)
    """
    # Collect audio files
    print("Collecting audio files...")
    train_files = collect_audio_files(train_dir)
    print(f"Found {len(train_files)} training files")
    
    # Create training dataset and loader
    train_dataset = MelContinuationDataset(train_files, **dataset_kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create validation dataset if provided
    val_loader = None
    if val_dir is not None:
        val_files = collect_audio_files(val_dir)
        print(f"Found {len(val_files)} validation files")
        
        val_dataset = MelContinuationDataset(val_files, **dataset_kwargs)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing MelContinuationDataset...")
    print("=" * 60)
    
    # Create a dummy audio file for testing
    print("Creating test audio file...")
    test_dir = "/tmp/test_audio"
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate 5 seconds of sine wave (simulating piano note)
    sr = 22050
    duration = 5.0
    freq = 440  # A4 note
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    
    # Save as wav file
    import soundfile as sf
    test_file = os.path.join(test_dir, "test.wav")
    sf.write(test_file, audio, sr)
    print(f"Created test file: {test_file}")
    print()
    
    # Create dataset
    dataset = MelContinuationDataset(
        audio_files=[test_file],
        sr=22050,
        n_mels=80,
        context_frames=100,
        predict_frames=50
    )
    
    print(f"Dataset length: {len(dataset)}")
    print()
    
    # Get one sample
    print("Loading one sample...")
    context, target = dataset[0]
    
    print(f"Context shape: {context.shape}")
    print(f"  - Expected: (80, 100)")
    print(f"Target shape: {target.shape}")
    print(f"  - Expected: (80, 50)")
    print()
    
    # Check value ranges
    print(f"Context value range: [{context.min():.3f}, {context.max():.3f}]")
    print(f"Target value range: [{target.min():.3f}, {target.max():.3f}]")
    print()
    
    # Test dataloader
    print("Testing DataLoader...")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch_context, batch_target in loader:
        print(f"Batch context shape: {batch_context.shape}")
        print(f"Batch target shape: {batch_target.shape}")
        break
    
    print()
    print("✓ Dataset test passed!")
    print("=" * 60)