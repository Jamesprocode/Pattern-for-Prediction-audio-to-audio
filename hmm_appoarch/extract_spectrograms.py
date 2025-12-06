"""
Extract spectrograms from normalized audio segmented into 16th notes.
Apply PCA dimensionality reduction for HMM training.
Each 16th note becomes one observation for HMM training.
"""

import librosa
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA


def calculate_target_time_steps(bpm=120, sr=22050, hop_length=512):
    """
    Calculate the target time steps for a 16th note based on BPM and audio parameters.

    Args:
        bpm: Beats per minute
        sr: Sample rate
        hop_length: Hop length for STFT

    Returns:
        int: Recommended target time steps (rounded up)
    """
    # At given BPM: 1 beat = 60/bpm seconds
    # 1 16th note = (60/bpm) / 4 seconds
    seconds_per_16th = (60.0 / bpm) / 4.0
    samples_per_16th = seconds_per_16th * sr
    frames_per_16th = samples_per_16th / hop_length

    # Round up to ensure we capture the full 16th note
    target_steps = int(np.ceil(frames_per_16th))

    return target_steps


def extract_16th_note_spectrograms(audio_path, sr=22050, n_mels=80,
                                    n_fft=2048, hop_length=512):
    """
    Extract mel spectrograms for each 16th note in the audio.

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        List of mel spectrograms (one per 16th note), shape (n_mels, time_steps)
    """
    y, _ = librosa.load(audio_path, sr=sr)

    # Get beat times
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 2:
        return None

    # Create 16th note times (4 divisions per beat)
    sixteenth_times = []
    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]
        beat_duration = beat_end - beat_start

        for j in range(4):  # 4 sixteenth notes per beat
            sixteenth_times.append(beat_start + (j / 4) * beat_duration)

    sixteenth_times.append(beat_times[-1])  # Add last beat

    # Extract spectrogram for each 16th note segment
    specs = []
    for i in range(len(sixteenth_times) - 1):
        start_time = sixteenth_times[i]
        end_time = sixteenth_times[i + 1]

        # Convert time to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Extract segment
        segment = y[start_sample:end_sample]

        if len(segment) == 0:
            continue

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=20,
            fmax=8000
        )

        # Convert to dB
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize
        mel_db = (mel_db + 80) / 80.0  # Scale to [0, 1]

        specs.append(mel_db)

    return specs


def pad_spectrograms(specs, target_time_steps=10, n_mels=80):
    """
    Pad or truncate spectrograms to consistent dimensions.

    Args:
        specs: List of spectrograms with shape (n_mels, time_steps)
        target_time_steps: Target time dimension
        n_mels: Expected number of mel bins

    Returns:
        List of padded spectrograms with shape (n_mels, target_time_steps)
    """
    padded = []

    for spec in specs:
        # Validate mel dimension
        if spec.shape[0] != n_mels:
            raise ValueError(f"Expected {n_mels} mel bins, got {spec.shape[0]}")

        if spec.shape[1] < target_time_steps:
            # Pad with zeros
            pad_width = ((0, 0), (0, target_time_steps - spec.shape[1]))
            spec_padded = np.pad(spec, pad_width, mode='constant', constant_values=0)
        else:
            # Truncate
            spec_padded = spec[:, :target_time_steps]

        # Final validation
        assert spec_padded.shape == (n_mels, target_time_steps), \
            f"Spectrogram shape mismatch: expected ({n_mels}, {target_time_steps}), got {spec_padded.shape}"

        padded.append(spec_padded)

    return padded


def process_dataset_spectrograms(dataset_dir, output_dir,
                                 target_time_steps=None,
                                 n_mels=80,
                                 n_pca_components=20,
                                 use_pca=True,
                                 bpm=120,
                                 sr=22050,
                                 hop_length=512):
    """
    Process all audio files: extract spectrograms, optionally apply PCA, save for HMM.

    Args:
        dataset_dir: Path to normalized_dataset directory
        output_dir: Path to save processed data
        target_time_steps: Target time dimension for spectrograms. If None, calculated from BPM.
        n_mels: Number of mel bins
        n_pca_components: Number of PCA components (ignored if use_pca=False)
        use_pca: Whether to apply PCA dimensionality reduction (default True)
        bpm: Beats per minute (used to calculate target_time_steps if not provided)
        sr: Sample rate
        hop_length: Hop length for STFT

    Returns:
        Dictionary with processing stats, train_features, test_features, pca_model
    """
    # Calculate target_time_steps from BPM if not provided
    if target_time_steps is None:
        target_time_steps = calculate_target_time_steps(bpm=bpm, sr=sr, hop_length=hop_length)
        print(f"\nCalculated target_time_steps from BPM: {target_time_steps}")
        print(f"  (BPM={bpm}, 16th note ≈ {target_time_steps} frames at {sr}Hz with hop_length={hop_length})\n")

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    # Find all audio files
    train_files = sorted(train_dir.glob("*.wav"))
    test_files = sorted(test_dir.glob("*.wav"))

    print(f"Found {len(train_files)} training files")
    print(f"Found {len(test_files)} test files\n")

    stats = {
        'train_count': 0,
        'test_count': 0,
        'failed': 0,
        'total_specs': 0,
        'target_time_steps': target_time_steps,
        'n_mels': n_mels,
        'use_pca': use_pca,
        'n_pca_components': n_pca_components if use_pca else None
    }

    # Process training files
    train_specs = []
    train_names = []

    print("=" * 70)
    print("Processing TRAINING files")
    print("=" * 70)

    for audio_file in tqdm(train_files, desc="Train", unit="file"):
        try:
            specs = extract_16th_note_spectrograms(
                str(audio_file),
                sr=sr,
                n_mels=n_mels,
                hop_length=hop_length
            )

            if specs is None or len(specs) == 0:
                stats['failed'] += 1
                continue

            # Pad spectrograms
            specs_padded = pad_spectrograms(specs, target_time_steps, n_mels)

            # Store with filename
            for spec in specs_padded:
                train_specs.append(spec)
                train_names.append(audio_file.stem)

            stats['train_count'] += 1
            stats['total_specs'] += len(specs_padded)

        except Exception as e:
            stats['failed'] += 1
            print(f"  Error processing {audio_file.name}: {e}")

    # Process test files
    test_specs = []
    test_names = []

    print("\n" + "=" * 70)
    print("Processing TEST files")
    print("=" * 70)

    for audio_file in tqdm(test_files, desc="Test", unit="file"):
        try:
            specs = extract_16th_note_spectrograms(
                str(audio_file),
                sr=sr,
                n_mels=n_mels,
                hop_length=hop_length
            )

            if specs is None or len(specs) == 0:
                stats['failed'] += 1
                continue

            # Pad spectrograms
            specs_padded = pad_spectrograms(specs, target_time_steps, n_mels)

            # Store with filename
            for spec in specs_padded:
                test_specs.append(spec)
                test_names.append(audio_file.stem)

            stats['test_count'] += 1
            stats['total_specs'] += len(specs_padded)

        except Exception as e:
            stats['failed'] += 1
            print(f"  Error processing {audio_file.name}: {e}")

    # Convert to numpy arrays
    train_specs = np.array(train_specs)  # Shape: (num_specs, n_mels, time_steps)
    test_specs = np.array(test_specs)

    # Validate shapes
    assert train_specs.ndim == 3, f"Train spectrograms should be 3D, got {train_specs.ndim}D"
    assert train_specs.shape[1] == n_mels, f"Train spectrograms should have {n_mels} mel bins, got {train_specs.shape[1]}"
    assert train_specs.shape[2] == target_time_steps, f"Train spectrograms should have {target_time_steps} time steps, got {train_specs.shape[2]}"

    assert test_specs.ndim == 3, f"Test spectrograms should be 3D, got {test_specs.ndim}D"
    assert test_specs.shape[1] == n_mels, f"Test spectrograms should have {n_mels} mel bins, got {test_specs.shape[1]}"
    assert test_specs.shape[2] == target_time_steps, f"Test spectrograms should have {target_time_steps} time steps, got {test_specs.shape[2]}"

    print("\n" + "=" * 70)
    print("Step 1: Spectrogram Extraction")
    print("=" * 70)
    print(f"Training files processed: {stats['train_count']}")
    print(f"Test files processed: {stats['test_count']}")
    print(f"Total spectrograms extracted: {stats['total_specs']}")
    print(f"Train spectrograms shape: {train_specs.shape}")
    print(f"Test spectrograms shape: {test_specs.shape}")

    # Flatten spectrograms
    print("\n" + "=" * 70)
    if use_pca:
        print("Step 2: Flatten and Apply PCA")
    else:
        print("Step 2: Flatten Spectrograms (PCA disabled)")
    print("=" * 70)

    train_specs_flat = train_specs.reshape(train_specs.shape[0], -1)
    test_specs_flat = test_specs.reshape(test_specs.shape[0], -1)

    print(f"Flattened train shape: {train_specs_flat.shape}")
    print(f"Flattened test shape: {test_specs_flat.shape}")

    # Apply PCA or use raw spectrograms
    pca = None
    if use_pca:
        # Fit PCA on training data
        pca = PCA(n_components=n_pca_components)
        train_features = pca.fit_transform(train_specs_flat)
        test_features = pca.transform(test_specs_flat)

        # Validate PCA features shapes
        assert train_features.shape[1] == n_pca_components, \
            f"Train features should have {n_pca_components} components, got {train_features.shape[1]}"
        assert test_features.shape[1] == n_pca_components, \
            f"Test features should have {n_pca_components} components, got {test_features.shape[1]}"
        assert train_features.shape[0] == len(train_names), \
            f"Number of train features ({train_features.shape[0]}) doesn't match train_names ({len(train_names)})"
        assert test_features.shape[0] == len(test_names), \
            f"Number of test features ({test_features.shape[0]}) doesn't match test_names ({len(test_names)})"

        print(f"\nPCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")
        print(f"PCA features shape - Train: {train_features.shape}, Test: {test_features.shape}")
    else:
        # Use raw flattened spectrograms
        train_features = train_specs_flat
        test_features = test_specs_flat

        # Validate features shapes
        assert train_features.shape[0] == len(train_names), \
            f"Number of train features ({train_features.shape[0]}) doesn't match train_names ({len(train_names)})"
        assert test_features.shape[0] == len(test_names), \
            f"Number of test features ({test_features.shape[0]}) doesn't match test_names ({len(test_names)})"

        feature_dim = n_mels * target_time_steps
        print(f"\nUsing raw spectrograms (no PCA)")
        print(f"Raw spectrogram features shape - Train: {train_features.shape}, Test: {test_features.shape}")
        print(f"Feature dimension: {feature_dim} ({n_mels} mels × {target_time_steps} time steps)")

    # Save data and model
    print("\n" + "=" * 70)
    print("Step 3: Save Data")
    print("=" * 70)

    train_file = output_dir / "train_features.npy"
    test_file = output_dir / "test_features.npy"
    train_names_file = output_dir / "train_names.pkl"
    test_names_file = output_dir / "test_names.pkl"

    np.save(train_file, train_features)
    np.save(test_file, test_features)

    with open(train_names_file, 'wb') as f:
        pickle.dump(train_names, f)

    with open(test_names_file, 'wb') as f:
        pickle.dump(test_names, f)

    print(f"Data saved to: {output_dir}")
    print(f"  - {train_file.name}: {train_features.shape}")
    print(f"  - {test_file.name}: {test_features.shape}")
    print(f"  - {train_names_file.name}")
    print(f"  - {test_names_file.name}")

    # Save PCA model only if used
    if use_pca:
        pca_file = output_dir / "pca_model.pkl"
        with open(pca_file, 'wb') as f:
            pickle.dump(pca, f)
        print(f"  - {pca_file.name}")

    # Save stats
    stats_file = output_dir / "feature_stats.json"
    if use_pca:
        stats['variance_explained'] = float(pca.explained_variance_ratio_.sum())
    else:
        stats['variance_explained'] = None
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to: {stats_file}")

    return stats, train_features, test_features, pca


if __name__ == "__main__":
    DATASET_DIR = "/Users/jameswang/workspace/Pattern for Prediction audio to audio/hmm_appoarch/normalized_dataset"
    OUTPUT_DIR = "/Users/jameswang/workspace/Pattern for Prediction audio to audio/hmm_appoarch/spectrogram_data"

    # Extract spectrograms with PCA (target_time_steps auto-calculated from BPM)
    # process_dataset_spectrograms(
    #     DATASET_DIR,
    #     OUTPUT_DIR,
    #     # target_time_steps auto-calculated as 6 frames for 120 BPM
    #     n_mels=80,
    #     n_pca_components=20,
    #     use_pca=True,
    #     bpm=120  # All MIDI files normalized to this BPM
    # )

    # Examples of other configurations:

    # Option 1: Train HMM directly on raw spectrograms without PCA
    process_dataset_spectrograms(
        DATASET_DIR,
        "/Users/jameswang/workspace/Pattern for Prediction audio to audio/hmm_appoarch/spectrogram_data_no_pca",
        n_mels=80,
        use_pca=False,
        bpm=120
    )

    # Option 2: Override auto-calculated target_time_steps with manual value
    # process_dataset_spectrograms(
    #     DATASET_DIR,
    #     OUTPUT_DIR,
    #     target_time_steps=5,  # Manual override (usually not needed)
    #     n_mels=80,
    #     use_pca=True,
    #     bpm=120
    # )
