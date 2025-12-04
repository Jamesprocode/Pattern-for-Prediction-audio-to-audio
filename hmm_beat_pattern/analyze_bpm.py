"""
Script to analyze BPM consistency across your dataset.
Run this to check if all audio files have similar BPM.
"""

import sys
from pathlib import Path
import json
from beat_utils import analyze_dataset_bpm, collect_audio_files


def main():
    """Main analysis function."""

    # Specify your dataset directory here
    dataset_dir = "/Users/jameswang/Downloads/PPDD-Jul2018_aud_mono_small/prime_wav"

    if not Path(dataset_dir).exists():
        print(f"Error: Directory not found: {dataset_dir}")
        sys.exit(1)

    # Collect all audio files
    print(f"\nScanning for audio files in: {dataset_dir}")
    audio_files = collect_audio_files(dataset_dir)

    if not audio_files:
        print("No audio files found!")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files\n")

    # Analyze BPM
    stats = analyze_dataset_bpm(audio_files, sr=22050)

    # Save results
    output_file = "bpm_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    if stats['std_bpm'] is not None:
        if stats['std_bpm'] < 5:
            print("\n✓ Good news! BPM is very consistent across your dataset.")
            print("  You can use a fixed BPM for beat aggregation.")
        elif stats['std_bpm'] < 15:
            print("\n⚠ BPM varies moderately across files.")
            print("  Consider analyzing each file individually or normalizing BPM.")
        else:
            print("\n⚠ BPM varies significantly across files.")
            print("  You should handle each file with its own BPM separately.")


def collect_audio_files(data_dir, extensions=['.wav', '.mp3', '.flac', '.m4a']):
    """Recursively collect all audio files from a directory."""
    audio_files = []
    data_path = Path(data_dir)

    for ext in extensions:
        audio_files.extend(list(data_path.rglob(f'*{ext}')))

    return [str(f) for f in audio_files]


if __name__ == "__main__":
    main()
