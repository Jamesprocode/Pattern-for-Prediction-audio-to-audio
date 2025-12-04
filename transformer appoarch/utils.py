"""
Utilities for inference and visualization.
"""

import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from model import MelContinuationTransformer


def load_model(checkpoint_path, device='cpu'):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model (you may need to adjust these parameters)
    model = MelContinuationTransformer(
        n_mels=80,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    return model


def audio_to_mel(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=80, normalize=True):
    """
    Convert audio to mel spectrogram.
    
    Args:
        audio: Audio waveform (numpy array)
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bins
        normalize: Whether to normalize
    
    Returns:
        mel_db: Mel spectrogram in dB (n_mels, time)
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20,
        fmax=8000
    )
    
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    if normalize:
        mel_db = mel_db / 80.0
    
    return mel_db


def mel_to_audio(mel_db, sr=22050, hop_length=512, n_iter=32, normalize=True):
    """
    Convert mel spectrogram back to audio using Griffin-Lim algorithm.
    
    Args:
        mel_db: Mel spectrogram in dB (n_mels, time)
        sr: Sample rate
        hop_length: Hop length
        n_iter: Number of Griffin-Lim iterations
        normalize: Whether mel was normalized
    
    Returns:
        audio: Audio waveform
    """
    if normalize:
        mel_db = mel_db * 80.0
    
    # Convert from dB to power
    mel = librosa.db_to_power(mel_db)
    
    # Invert to audio using Griffin-Lim
    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sr,
        hop_length=hop_length,
        n_iter=n_iter
    )
    
    return audio


def predict_continuation(
    model,
    audio_path,
    context_duration=2.0,
    predict_duration=1.0,
    sr=22050,
    device='cpu'
):
    """
    Predict audio continuation from an audio file.
    
    Args:
        model: Trained model
        audio_path: Path to input audio file
        context_duration: Duration of context in seconds
        predict_duration: Duration to predict in seconds
        sr: Sample rate
        device: Device to run on
    
    Returns:
        context_audio: Original context audio
        predicted_audio: Predicted continuation audio
        context_mel: Context mel spectrogram
        predicted_mel: Predicted mel spectrogram
    """
    hop_length = 512
    
    # Calculate frame counts
    context_frames = int(context_duration * sr / hop_length)
    predict_frames = int(predict_duration * sr / hop_length)
    
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # Take only the context portion
    context_samples = int(context_duration * sr)
    context_audio = audio[:context_samples]
    
    # Convert to mel
    context_mel = audio_to_mel(context_audio, sr=sr, hop_length=hop_length)
    
    # Ensure we have enough frames
    if context_mel.shape[1] < context_frames:
        pad_width = context_frames - context_mel.shape[1]
        context_mel = np.pad(context_mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1.0)
    else:
        context_mel = context_mel[:, :context_frames]
    
    # Convert to tensor
    context_tensor = torch.FloatTensor(context_mel).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        predicted_tensor = model(context_tensor, predict_length=predict_frames)
    
    # Convert back to numpy
    predicted_mel = predicted_tensor.squeeze(0).cpu().numpy()
    
    # Convert mel to audio
    predicted_audio = mel_to_audio(predicted_mel, sr=sr, hop_length=hop_length)
    
    return context_audio, predicted_audio, context_mel, predicted_mel


def visualize_prediction(context_mel, predicted_mel, ground_truth_mel=None, save_path=None):
    """
    Visualize mel spectrogram prediction.
    
    Args:
        context_mel: Context mel spectrogram (n_mels, time)
        predicted_mel: Predicted mel spectrogram (n_mels, time)
        ground_truth_mel: Ground truth mel (optional)
        save_path: Path to save figure (optional)
    """
    n_plots = 3 if ground_truth_mel is not None else 2
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Plot context
    img1 = axes[0].imshow(context_mel, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Context Mel Spectrogram')
    axes[0].set_ylabel('Mel Frequency Bin')
    axes[0].set_xlabel('Time Frame')
    plt.colorbar(img1, ax=axes[0])
    
    # Plot prediction
    img2 = axes[1].imshow(predicted_mel, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Predicted Mel Spectrogram')
    axes[1].set_ylabel('Mel Frequency Bin')
    axes[1].set_xlabel('Time Frame')
    plt.colorbar(img2, ax=axes[1])
    
    # Plot ground truth if provided
    if ground_truth_mel is not None:
        img3 = axes[2].imshow(ground_truth_mel, aspect='auto', origin='lower', cmap='viridis')
        axes[2].set_title('Ground Truth Mel Spectrogram')
        axes[2].set_ylabel('Mel Frequency Bin')
        axes[2].set_xlabel('Time Frame')
        plt.colorbar(img3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_and_save(
    model,
    audio_path,
    output_path,
    context_duration=2.0,
    predict_duration=1.0,
    sr=22050,
    device='cpu',
    save_visualization=True
):
    """
    Generate continuation and save as audio file.
    
    Args:
        model: Trained model
        audio_path: Input audio path
        output_path: Output audio path
        context_duration: Context duration in seconds
        predict_duration: Prediction duration in seconds
        sr: Sample rate
        device: Device to use
        save_visualization: Whether to save mel spectrogram visualization
    """
    print(f"Processing: {audio_path}")
    
    # Predict
    context_audio, predicted_audio, context_mel, predicted_mel = predict_continuation(
        model, audio_path, context_duration, predict_duration, sr, device
    )
    
    # Concatenate context and prediction
    full_audio = np.concatenate([context_audio, predicted_audio])
    
    # Save audio
    sf.write(output_path, full_audio, sr)
    print(f"Audio saved to: {output_path}")
    
    # Save visualization
    if save_visualization:
        viz_path = output_path.replace('.wav', '_mel.png')
        visualize_prediction(context_mel, predicted_mel, save_path=viz_path)
    
    return full_audio


def compute_metrics(predicted_mel, target_mel):
    """
    Compute evaluation metrics between predicted and target mel spectrograms.
    
    Args:
        predicted_mel: Predicted mel spectrogram
        target_mel: Target mel spectrogram
    
    Returns:
        Dictionary of metrics
    """
    mse = np.mean((predicted_mel - target_mel) ** 2)
    mae = np.mean(np.abs(predicted_mel - target_mel))
    
    # Spectral convergence
    spectral_conv = np.linalg.norm(target_mel - predicted_mel, ord='fro') / (np.linalg.norm(target_mel, ord='fro') + 1e-8)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'spectral_convergence': spectral_conv
    }
    
    return metrics


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    print("=" * 60)
    
    # Create test audio
    sr = 22050
    duration = 3.0
    freq = 440  # A4
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    
    print("Test audio created (3 seconds of A4 note)")
    
    # Convert to mel
    mel = audio_to_mel(audio, sr=sr)
    print(f"Mel spectrogram shape: {mel.shape}")
    
    # Convert back to audio
    reconstructed = mel_to_audio(mel, sr=sr)
    print(f"Reconstructed audio shape: {reconstructed.shape}")
    
    # Visualize
    print("Creating visualization...")
    visualize_prediction(mel[:, :100], mel[:, 100:150], save_path='/tmp/test_mel.png')
    
    print("\n✓ Utilities test passed!")
    print("=" * 60)