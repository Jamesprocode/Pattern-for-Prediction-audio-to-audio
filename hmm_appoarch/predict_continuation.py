"""
Predict continuations using trained HMM and evaluate against ground truth.
Uses test features to generate continuations and compares with ground truth MIDI.
"""

import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm
import pretty_midi
import librosa

def group_features_by_file(test_features, test_names):
    """Group test features by file name."""
    file_groups = defaultdict(list)
    for i, name in enumerate(test_names):
        file_groups[name].append((i, test_features[i]))
    return file_groups

def predict_continuation(hmm_model, file_features, n_continuation=50):
    """
    Predict continuation for a test file using HMM.

    Args:
        hmm_model: Trained Gaussian HMM
        file_features: Features for one test file, shape (n_specs, 20)
        n_continuation: Number of observations to generate

    Returns:
        continuation_features: Generated features, shape (n_continuation, 20)
        state_path: Hidden state path
    """
    # Find most likely state path using Viterbi algorithm
    state_path = hmm_model.predict(file_features)
    final_state = state_path[-1]

    # Generate continuation starting from final state
    # Sample n_continuation observations conditioned on final state
    generated = []
    current_state = final_state

    for _ in range(n_continuation):
        # Sample observation from current state's distribution
        mean = hmm_model.means_[current_state].copy()
        cov = hmm_model.covars_[current_state]

        # Handle both diagonal and full covariance matrices
        if cov.ndim == 1:
            # Diagonal covariance: convert to matrix
            cov_matrix = np.diag(np.abs(cov) + 1e-6)
        else:
            # Full covariance: ensure positive definite
            cov_matrix = cov + np.eye(cov.shape[0]) * 1e-6

        obs = np.random.multivariate_normal(mean, cov_matrix)
        generated.append(obs)

        # Transition to next state based on transition matrix
        transition_probs = hmm_model.transmat_[current_state]
        current_state = np.random.choice(len(transition_probs), p=transition_probs)

    return np.array(generated), state_path

def spectrograms_to_midi(spectrograms, sr=22050, hop_length=512,
                         fmin=20, fmax=8000, n_mels=80):
    """
    Convert spectrograms back to MIDI using onset detection + pitch estimation.

    This is a simplified conversion - in practice you'd use librosa's
    piptrack or other pitch extraction methods.

    Args:
        spectrograms: Array of shape (n_frames, n_mels, time_steps)
        sr: Sample rate
        hop_length: Hop length used in spectrogram extraction
        fmin: Min frequency
        fmax: Max frequency
        n_mels: Number of mel bins

    Returns:
        midi: pretty_midi.PrettyMIDI object
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, is_drum=False)

    # Convert mel freq to Hz
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

    # Simple heuristic: for each frame, find peak frequency
    # and create note if above threshold
    onset_strength = np.mean(spectrograms, axis=(1, 2))
    threshold = np.mean(onset_strength) + np.std(onset_strength)

    onsets = np.where(onset_strength > threshold)[0]

    for onset_idx in onsets:
        spec = spectrograms[onset_idx]  # (n_mels, time_steps)

        # Find peak frequency bin
        peak_mel_bin = np.argmax(np.mean(spec, axis=1))
        peak_freq = mel_freqs[peak_mel_bin]

        # Convert frequency to MIDI note
        if peak_freq > 0:
            midi_note = librosa.hz_to_midi(peak_freq)
            midi_note = int(np.round(midi_note))
            midi_note = np.clip(midi_note, 0, 127)

            # Create note
            start_time = onset_idx * hop_length / sr
            end_time = (onset_idx + 1) * hop_length / sr

            note = pretty_midi.Note(
                velocity=80,
                pitch=midi_note,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi

def cardinality_score(predicted_midi, ground_truth_midi):
    """
    Calculate cardinality score: ratio of correctly predicted notes.

    Args:
        predicted_midi: Predicted pretty_midi.PrettyMIDI
        ground_truth_midi: Ground truth pretty_midi.PrettyMIDI

    Returns:
        score: 0 to 1, higher is better
    """
    try:
        pred_notes = predicted_midi.instruments[0].notes if predicted_midi.instruments else []
        gt_notes = ground_truth_midi.instruments[0].notes if ground_truth_midi.instruments else []

        if len(gt_notes) == 0:
            return 0.0

        # Count matching notes (within tolerance)
        matches = 0
        for pred_note in pred_notes:
            for gt_note in gt_notes:
                if abs(pred_note.pitch - gt_note.pitch) <= 1:  # Within 1 semitone
                    matches += 1
                    break

        return min(matches / len(gt_notes), 1.0)
    except:
        return 0.0

def pitch_accuracy(predicted_midi, ground_truth_midi):
    """
    Calculate average pitch accuracy of predicted notes.

    Args:
        predicted_midi: Predicted pretty_midi.PrettyMIDI
        ground_truth_midi: Ground truth pretty_midi.PrettyMIDI

    Returns:
        accuracy: 0 to 1, higher is better
    """
    try:
        pred_notes = predicted_midi.instruments[0].notes if predicted_midi.instruments else []
        gt_notes = ground_truth_midi.instruments[0].notes if ground_truth_midi.instruments else []

        if len(pred_notes) == 0 or len(gt_notes) == 0:
            return 0.0

        accuracies = []
        for pred_note in pred_notes:
            # Find closest ground truth note
            min_pitch_diff = min(abs(pred_note.pitch - gt_note.pitch) for gt_note in gt_notes)
            # Accuracy decreases with pitch difference
            acc = max(0, 1 - min_pitch_diff / 12)  # Normalize by octave
            accuracies.append(acc)

        return np.mean(accuracies) if accuracies else 0.0
    except:
        return 0.0

def main():
    """Predict continuations and evaluate."""

    print("=" * 70)
    print("HMM Continuation Prediction & Evaluation")
    print("=" * 70)

    data_dir = Path("/Users/jameswang/workspace/Pattern for Prediction audio to audio/hmm_beat_pattern/spectrogram_data")
    test_dir = Path("/Users/jameswang/workspace/Pattern for Prediction audio to audio/hmm_beat_pattern/normalized_dataset/test")

    # Load data
    print("\nLoading trained model and data...")
    with open(data_dir / "hmm_model.pkl", 'rb') as f:
        model_data = pickle.load(f)
        hmm_model = model_data['model']
        pca_model = model_data['pca_model']

    test_features = np.load(data_dir / "test_features.npy")
    test_names = pickle.load(open(data_dir / "test_names.pkl", 'rb'))

    print(f"Loaded HMM model with {hmm_model.n_components} hidden states")
    print(f"Test features shape: {test_features.shape}")

    # Group test features by file
    print("\nGrouping test features by file...")
    file_groups = group_features_by_file(test_features, test_names)
    print(f"Found {len(file_groups)} unique test files")

    # Results storage
    results = {}

    # Predict for each test file
    print("\n" + "=" * 70)
    print("Generating Continuations")
    print("=" * 70)

    for file_name in sorted(file_groups.keys()):
        indices_and_features = file_groups[file_name]
        file_features = np.array([f for _, f in indices_and_features])

        # Generate continuation
        n_continuation = len(file_features)  # Generate same length as input
        continuation_features, state_path = predict_continuation(
            hmm_model, file_features, n_continuation
        )

        # Convert back to spectrogram space
        continuation_specs_flat = pca_model.inverse_transform(continuation_features)
        continuation_specs = continuation_specs_flat.reshape(n_continuation, 80, 10)

        # Convert to MIDI
        predicted_midi = spectrograms_to_midi(continuation_specs)

        # Load ground truth continuation MIDI
        gt_midi_path = test_dir / "continuation_midi" / f"{file_name}_continuation.mid"

        if gt_midi_path.exists():
            try:
                ground_truth_midi = pretty_midi.PrettyMIDI(str(gt_midi_path))

                # Calculate metrics
                card_score = cardinality_score(predicted_midi, ground_truth_midi)
                pitch_acc = pitch_accuracy(predicted_midi, ground_truth_midi)

                results[file_name] = {
                    'cardinality_score': float(card_score),
                    'pitch_accuracy': float(pitch_acc),
                    'n_predicted_notes': len(predicted_midi.instruments[0].notes if predicted_midi.instruments else []),
                    'n_ground_truth_notes': len(ground_truth_midi.instruments[0].notes if ground_truth_midi.instruments else [])
                }

                print(f"{file_name}: Card={card_score:.3f}, Pitch={pitch_acc:.3f}")

            except Exception as e:
                print(f"{file_name}: Error loading ground truth - {e}")
                results[file_name] = {'error': str(e)}
        else:
            print(f"{file_name}: No ground truth found")
            results[file_name] = {'error': 'No ground truth MIDI'}

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    valid_results = [r for r in results.values() if 'error' not in r]

    if valid_results:
        card_scores = [r['cardinality_score'] for r in valid_results]
        pitch_accs = [r['pitch_accuracy'] for r in valid_results]

        print(f"Files evaluated: {len(valid_results)}")
        print(f"Cardinality Score: {np.mean(card_scores):.3f} ± {np.std(card_scores):.3f}")
        print(f"Pitch Accuracy: {np.mean(pitch_accs):.3f} ± {np.std(pitch_accs):.3f}")
        print(f"Best Cardinality: {np.max(card_scores):.3f}")
        print(f"Best Pitch Accuracy: {np.max(pitch_accs):.3f}")
    else:
        print("No valid results to summarize")

    # Save results
    results_file = data_dir / "prediction_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
