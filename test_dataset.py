"""
Test script for the MelContinuationDataset with your prime_eave audio data.
"""

import torch
from dataset import MelContinuationDataset, collect_audio_files, create_dataloaders
from pathlib import Path

def test_single_file():
    """Test loading a single audio file"""
    print("=" * 60)
    print("TEST 1: Single File Loading")
    print("=" * 60)

    # Path to your dataset
    data_dir = "/Users/jameswang/Downloads/PPDD-Sep2018_aud_mono_small"

    # Find all audio files
    print(f"\nSearching for audio files in: {data_dir}")
    audio_files = collect_audio_files(data_dir)

    if not audio_files:
        print("ERROR: No audio files found!")
        print("Make sure you have .wav, .mp3, .flac, or .m4a files in the directory")
        return False

    print(f"Found {len(audio_files)} audio files")
    print(f"First file: {audio_files[0]}")
    print()

    # Create dataset with just one file
    print("Creating dataset with first file...")
    dataset = MelContinuationDataset(
        audio_files=[audio_files[0]],
        sr=22050,
        n_mels=80,
        context_frames=100,
        predict_frames=50,
        min_duration=3.0
    )

    if len(dataset) == 0:
        print("ERROR: Dataset is empty! File might be too short.")
        return False

    print(f"\nDataset created successfully!")
    print(f"Dataset length: {len(dataset)}")
    print()

    # Get one sample
    print("Loading sample from dataset...")
    context, target = dataset[0]

    print(f"✓ Context shape: {context.shape} (expected: [80, 100])")
    print(f"✓ Target shape: {target.shape} (expected: [80, 50])")
    print(f"✓ Context range: [{context.min():.3f}, {context.max():.3f}]")
    print(f"✓ Target range: [{target.min():.3f}, {target.max():.3f}]")
    print()

    return True


def test_dataloader():
    """Test creating a DataLoader with multiple files"""
    print("=" * 60)
    print("TEST 2: DataLoader with Batching")
    print("=" * 60)

    # Path to your dataset
    data_dir = "/Users/jameswang/Downloads/PPDD-Sep2018_aud_mono_small"

    print(f"\nSearching for audio files in: {data_dir}")
    audio_files = collect_audio_files(data_dir)

    if not audio_files:
        print("ERROR: No audio files found!")
        return False

    print(f"Found {len(audio_files)} audio files")

    # Limit to first 10 files for quick testing
    test_files = audio_files[:min(10, len(audio_files))]
    print(f"Using {len(test_files)} files for testing")
    print()

    # Create dataset
    print("Creating dataset...")
    dataset = MelContinuationDataset(
        audio_files=test_files,
        sr=22050,
        n_mels=80,
        context_frames=100,
        predict_frames=50,
        min_duration=3.0
    )

    print(f"\nValid samples: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: No valid samples! Files might be too short.")
        return False

    # Create DataLoader
    print("\nCreating DataLoader...")
    from torch.utils.data import DataLoader

    batch_size = 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )

    print(f"DataLoader created with batch_size={batch_size}")
    print()

    # Test one batch
    print("Loading one batch...")
    for batch_context, batch_target in loader:
        print(f"✓ Batch context shape: {batch_context.shape}")
        print(f"   Expected: [batch_size, 80, 100]")
        print(f"✓ Batch target shape: {batch_target.shape}")
        print(f"   Expected: [batch_size, 80, 50]")
        print(f"✓ Batch context range: [{batch_context.min():.3f}, {batch_context.max():.3f}]")
        print(f"✓ Batch target range: [{batch_target.min():.3f}, {batch_target.max():.3f}]")
        break

    print()
    return True


def test_with_model():
    """Test dataset with the actual model"""
    print("=" * 60)
    print("TEST 3: Integration with Model")
    print("=" * 60)

    from model import MelContinuationTransformer

    # Create a small dataset
    data_dir = "/Users/jameswang/Downloads/PPDD-Sep2018_aud_mono_small"
    audio_files = collect_audio_files(data_dir)

    if not audio_files:
        print("ERROR: No audio files found!")
        return False

    # Use first file only
    dataset = MelContinuationDataset(
        audio_files=[audio_files[0]],
        sr=22050,
        n_mels=80,
        context_frames=100,
        predict_frames=50,
        min_duration=3.0
    )

    if len(dataset) == 0:
        print("ERROR: No valid samples!")
        return False

    print(f"\nDataset created with {len(dataset)} sample(s)")

    # Create model
    print("Creating model...")
    model = MelContinuationTransformer(
        n_mels=80,
        d_model=256,  # Smaller for testing
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1
    )

    print("Model created successfully")
    print()

    # Get a sample and run through model
    print("Testing forward pass...")
    context, target = dataset[0]

    # Add batch dimension
    context_batch = context.unsqueeze(0)  # (1, 80, 100)

    print(f"Input shape: {context_batch.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(context_batch, predict_length=50)

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Target shape: {target.unsqueeze(0).shape}")
    print(f"✓ Shapes match: {output.shape == target.unsqueeze(0).shape}")
    print()

    return True


def explore_dataset():
    """Explore the dataset structure"""
    print("=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)

    data_dir = "/Users/jameswang/Downloads/PPDD-Sep2018_aud_mono_small"
    print(f"\nDataset directory: {data_dir}")

    # Check if prime_eave folder exists
    prime_eave_path = Path(data_dir) / "prime_eave"
    if prime_eave_path.exists():
        print(f"✓ Found prime_eave folder: {prime_eave_path}")

    # Check if prime_csv folder exists
    prime_csv_path = Path(data_dir) / "prime_csv"
    if prime_csv_path.exists():
        print(f"✓ Found prime_csv folder: {prime_csv_path}")
        csv_files = list(prime_csv_path.glob("*.csv"))
        print(f"  - Contains {len(csv_files)} CSV files")

    print()

    # Find audio files
    audio_files = collect_audio_files(data_dir)
    print(f"Total audio files found: {len(audio_files)}")

    if audio_files:
        print(f"\nFirst 5 audio files:")
        for i, f in enumerate(audio_files[:5], 1):
            print(f"  {i}. {Path(f).name}")
    else:
        print("\n⚠ No audio files found!")
        print("Expected extensions: .wav, .mp3, .flac, .m4a")

    print()


if __name__ == "__main__":
    print("\n🎵 Testing MelContinuationDataset with Prime Eave Dataset\n")

    # First explore the dataset
    explore_dataset()

    # Run tests
    tests = [
        ("Single File", test_single_file),
        ("DataLoader", test_dataloader),
        ("Model Integration", test_with_model)
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✓ PASSED" if success else "✗ FAILED"
        except Exception as e:
            results[test_name] = f"✗ ERROR: {str(e)}"
            import traceback
            traceback.print_exc()
        print()

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    print("=" * 60)
