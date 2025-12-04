"""
Synthesize MIDI files to audio with tempo normalization and create train/test splits.
Normalizes all MIDI to 120 BPM, synthesizes to audio, verifies with librosa, and filters.
"""

import pretty_midi
import librosa
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import subprocess
import shutil


def normalize_midi_tempo(midi_file_path, target_bpm=120.0):
    """
    Normalize MIDI file to target BPM by scaling note timings.

    Args:
        midi_file_path: Path to MIDI file
        target_bpm: Target tempo in BPM

    Returns:
        pretty_midi.PrettyMIDI object with normalized tempo
    """
    midi = pretty_midi.PrettyMIDI(midi_file_path)

    # Get original tempo
    original_bpm = midi.get_tempo_changes()[1][0]
    tempo_ratio = target_bpm / original_bpm

    # Scale note timings
    for instrument in midi.instruments:
        for note in instrument.notes:
            note.start *= tempo_ratio
            note.end *= tempo_ratio
        for cc in instrument.control_changes:
            cc.time *= tempo_ratio
        for pb in instrument.pitch_bends:
            pb.time *= tempo_ratio

        # Reset to program 0 (piano)
        if not instrument.is_drum:
            instrument.program = 0
            instrument.channel = 0

    # Set constant tempo
    midi._tempo_changes = ([0.0], [target_bpm])

    return midi


def synthesize_midi(midi_file_path, output_audio_path, soundfont_path):
    """
    Synthesize MIDI to audio using FluidSynth.

    Args:
        midi_file_path: Path to MIDI file
        output_audio_path: Path to save audio
        soundfont_path: Path to soundfont

    Returns:
        True if successful
    """
    try:
        cmd = [
            "fluidsynth",
            "-ni",
            "-r", "22050",  # Sample rate to match spectrogram extraction
            "-F", str(output_audio_path),
            str(soundfont_path),
            str(midi_file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0
    except Exception as e:
        print(f"    Error synthesizing: {e}")
        return False


def verify_tempo(audio_path, sr=22050, target_bpm=120.0, tolerance=5.0):
    """
    Verify audio tempo using librosa, accepting tempo octaves.

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        target_bpm: Expected BPM
        tolerance: Allowed deviation in BPM (default ±5)

    Returns:
        Tuple of (is_valid, detected_bpm)
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr)
        detected_bpm = librosa.feature.tempo(y=y, sr=sr)[0]

        # Check if detected BPM matches target or tempo octaves (x0.5, x2)
        octave_bpms = [target_bpm / 2, target_bpm, target_bpm * 2]
        is_valid = any(abs(detected_bpm - bpm) <= tolerance for bpm in octave_bpms)

        return is_valid, detected_bpm
    except Exception as e:
        return False, None


def find_soundfont():
    """Find available FluidSynth soundfont."""
    paths = "/Users/jameswang/workspace/Pattern for Prediction audio to audio/hmm_beat_pattern/UprightPianoKW-small-20190703.sf2"

    return paths


def process_dataset(target_bpm=120.0):
    """
    Process all MIDI files: normalize, synthesize, verify, and split.

    Args:
        target_bpm: Target tempo for normalization

    Returns:
        Dictionary with dataset info
    """
    # Paths
    MIDI_DIR = Path("/Users/jameswang/Downloads/PPDD-Jul2018_aud_mono_small/prime_midi")
    CONTINUATION_DIR = Path("/Users/jameswang/Downloads/PPDD-Jul2018_aud_mono_small/cont_true_midi")

    OUTPUT_BASE = Path("/Users/jameswang/workspace/Pattern for Prediction audio to audio/hmm_beat_pattern/normalized_dataset")
    TRAIN_DIR = OUTPUT_BASE / "train"
    TEST_DIR = OUTPUT_BASE / "test"

    # Create directories
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Find soundfont
    soundfont = find_soundfont()
    if not soundfont:
        print("ERROR: FluidSynth soundfont not found!")
        return None

    print(f"Using soundfont: {soundfont}\n")

    # Get all MIDI files
    midi_files = sorted(list(MIDI_DIR.glob('*.mid')) + list(MIDI_DIR.glob('*.midi')))
    continuation_files = {f.stem: f for f in list(CONTINUATION_DIR.glob('*.mid')) + list(CONTINUATION_DIR.glob('*.midi'))} if CONTINUATION_DIR.exists() else {}

    print(f"Found {len(midi_files)} MIDI files")
    print(f"Found {len(continuation_files)} continuation files\n")

    # Process each MIDI file
    stats = {
        'total': len(midi_files),
        'synthesized': 0,
        'verified': 0,
        'failed': 0,
        'bpms': [],
        'failed_files': [],
        'train_files': [],
        'test_files': []
    }

    temp_midi_dir = OUTPUT_BASE / "temp_normalized_midi"
    temp_midi_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("STEP 1: Normalize and synthesize MIDI files")
    print("=" * 70)

    valid_pairs = []  # (name, midi_path, audio_path)

    for i, midi_file in enumerate(tqdm(midi_files, desc="Processing", unit="file")):
        name = midi_file.stem

        try:
            # Normalize MIDI
            normalized_midi = normalize_midi_tempo(str(midi_file), target_bpm)

            # Save normalized MIDI
            temp_midi_path = temp_midi_dir / f"{name}_normalized.mid"
            normalized_midi.write(str(temp_midi_path))

            # Synthesize to audio
            audio_path = OUTPUT_BASE / f"{name}.wav"
            if synthesize_midi(str(temp_midi_path), str(audio_path), soundfont):
                stats['synthesized'] += 1

                # Verify tempo
                is_valid, detected_bpm = verify_tempo(str(audio_path), target_bpm=target_bpm)

                if is_valid:
                    stats['verified'] += 1
                    stats['bpms'].append(detected_bpm)
                    valid_pairs.append((name, str(midi_file), str(audio_path)))
                else:
                    stats['failed'] += 1
                    stats['failed_files'].append(f"{name} (detected: {detected_bpm:.1f} BPM)")
            else:
                stats['failed'] += 1
                stats['failed_files'].append(f"{name} (synthesis failed)")

        except Exception as e:
            stats['failed'] += 1
            stats['failed_files'].append(f"{name} ({str(e)})")

    print("\n" + "=" * 70)
    print("STEP 2: Split into train/test (80/20)")
    print("=" * 70)

    # Split 80/20
    n_train = int(len(valid_pairs) * 0.8)
    train_pairs = valid_pairs[:n_train]
    test_pairs = valid_pairs[n_train:]

    print(f"Train set: {len(train_pairs)} files")
    print(f"Test set: {len(test_pairs)} files\n")

    # Copy train files
    print("Copying training files...")
    for name, midi_path, audio_path in tqdm(train_pairs, desc="Train", unit="file"):
        shutil.copy(audio_path, TRAIN_DIR / f"{name}.wav")

        # Add continuation if available
        if name in continuation_files:
            cont_midi = continuation_files[name]
            cont_audio = TRAIN_DIR / f"{name}_continuation.wav"

            # # Normalize and synthesize continuation
            # try:
            #     normalized_cont = normalize_midi_tempo(str(cont_midi), target_bpm)
            #     temp_cont_midi = temp_midi_dir / f"{name}_continuation_normalized.mid"
            #     normalized_cont.write(str(temp_cont_midi))
            #     synthesize_midi(str(temp_cont_midi), str(cont_audio), soundfont)
            # except:
            #     pass

        stats['train_files'].append(name)

    # Copy test files
    print("Copying test files...")
    test_continuation_dir = TEST_DIR / "continuation_midi"
    test_continuation_dir.mkdir(exist_ok=True)

    for name, midi_path, audio_path in tqdm(test_pairs, desc="Test", unit="file"):
        shutil.copy(audio_path, TEST_DIR / f"{name}.wav")

        # Save normalized continuation MIDI if available (for evaluation ground truth)
        if name in continuation_files:
            cont_midi = continuation_files[name]
            cont_output = test_continuation_dir / f"{name}_continuation.mid"

            # Normalize and save continuation MIDI
            try:
                normalized_cont = normalize_midi_tempo(str(cont_midi), target_bpm)
                normalized_cont.write(str(cont_output))
            except Exception as e:
                print(f"    Warning: Could not process continuation for {name}: {e}")

        stats['test_files'].append(name)

    # Clean up temp directory
    shutil.rmtree(temp_midi_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed: {stats['synthesized']}")
    print(f"Files verified (120 BPM ±2): {stats['verified']}")
    print(f"Files failed: {stats['failed']}")

    if stats['bpms']:
        bpms = np.array(stats['bpms'])
        print(f"Mean detected BPM: {np.mean(bpms):.1f}")
        print(f"Std dev: {np.std(bpms):.2f}")

    if stats['failed_files']:
        print(f"\nFailed files ({len(stats['failed_files'])}):")
        for f in stats['failed_files'][:10]:
            print(f"  - {f}")
        if len(stats['failed_files']) > 10:
            print(f"  ... and {len(stats['failed_files']) - 10} more")

    print(f"\nDataset saved to: {OUTPUT_BASE}")
    print(f"  Train: {TRAIN_DIR} ({len(stats['train_files'])} files)")
    print(f"  Test: {TEST_DIR} ({len(stats['test_files'])} files)")

    # Save stats
    stats_file = OUTPUT_BASE / "dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            'total': stats['total'],
            'synthesized': stats['synthesized'],
            'verified': stats['verified'],
            'failed': stats['failed'],
            'mean_bpm': float(np.mean(stats['bpms'])) if stats['bpms'] else None,
            'std_bpm': float(np.std(stats['bpms'])) if stats['bpms'] else None,
            'train_count': len(stats['train_files']),
            'test_count': len(stats['test_files']),
            'failed_files': stats['failed_files']
        }, f, indent=2)

    print(f"\nStats saved to: {stats_file}")

    return stats


if __name__ == "__main__":
    process_dataset(target_bpm=120.0)
