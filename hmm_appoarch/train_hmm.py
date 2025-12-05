"""
Train Gaussian HMM on PCA-reduced spectrogram features.
The HMM learns transition probabilities and emission distributions.
"""

import numpy as np
import pickle
from pathlib import Path
from hmmlearn import hmm
import json


def main():
    """Train Gaussian HMM on spectrogram features."""

    print("=" * 70)
    print("HMM Training on Spectrogram Features")
    print("=" * 70)

    # Load PCA-reduced training features
    data_dir = Path("/Users/jameswang/workspace/Pattern for Prediction audio to audio/hmm_beat_pattern/spectrogram_data")

    print("\nLoading training data...")
    train_features = np.load(data_dir / "train_features.npy")  # Shape: (12848, 20)
    train_names = pickle.load(open(data_dir / "train_names.pkl", 'rb'))
    pca_model = pickle.load(open(data_dir / "pca_model.pkl", 'rb'))

    print(f"Train features shape: {train_features.shape}")
    print(f"  Observations: {train_features.shape[0]}")
    print(f"  Feature dimension: {train_features.shape[1]}")
    print(f"  Unique files: {len(set(train_names))}")

    # Initialize Gaussian HMM
    print("\n" + "=" * 70)
    print("Training Gaussian HMM")
    print("=" * 70)

    n_hidden_states = 10
    n_iter = 200

    print(f"\nHMM Configuration:")
    print(f"  Hidden states: {n_hidden_states}")
    print(f"  Covariance type: diag (diagonal)")
    print(f"  Max iterations: {n_iter}")
    print(f"  Random state: 42")

    # Create and train HMM
    model = hmm.GaussianHMM(
        n_components=n_hidden_states,
        covariance_type='diag',
        n_iter=n_iter,
        random_state=42,
        verbose=1
    )

    print(f"\nFitting HMM on {len(train_features)} observations...")
    model.fit(train_features)

    print(f"\n✓ Training Complete!")
    print(f"  Converged: {model.monitor_.converged}")
    print(f"  Log-likelihood: {model.score(train_features):.4f}")

    # Print learned transition matrix info
    print(f"\nLearned Transition Probabilities:")
    print(f"  Shape: {model.transmat_.shape}")
    print(f"  Max probability: {model.transmat_.max():.4f}")
    print(f"  Min probability: {model.transmat_.min():.4f}")

    # Print learned emission distribution info
    print(f"\nLearned Emission Distribution:")
    print(f"  Means shape: {model.means_.shape}")
    print(f"  Covariances shape: {model.covars_.shape}")

    # Save trained HMM and related data
    print("\n" + "=" * 70)
    print("Saving Model")
    print("=" * 70)

    model_path = data_dir / "hmm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'n_hidden_states': n_hidden_states,
            'n_features': train_features.shape[1],
            'pca_model': pca_model
        }, f)

    print(f"✓ Model saved to: {model_path}")

    # Test generation
    print("\n" + "=" * 70)
    print("Testing Generation")
    print("=" * 70)

    n_generate = 100
    print(f"\nGenerating {n_generate} observations from the HMM...")
    generated_features, states = model.sample(n_generate)

    print(f"✓ Generated sequence:")
    print(f"  Shape: {generated_features.shape}")
    print(f"  State sequence (first 20): {states[:20]}")
    print(f"  Unique states visited: {len(np.unique(states))}")

    # Convert back to spectrogram space using PCA inverse
    print(f"\nConverting generated features back to spectrogram space...")
    generated_specs = pca_model.inverse_transform(generated_features)
    print(f"  Generated spectrograms shape: {generated_specs.shape}")
    print(f"  (Can be reshaped to (100, 80, 10) for audio synthesis)")

    # Save generation test results
    results = {
        'n_generated': n_generate,
        'generated_features_shape': list(generated_features.shape),
        'generated_specs_shape': list(generated_specs.shape),
        'states_visited': len(np.unique(states)),
        'log_likelihood': float(model.score(train_features))
    }

    results_file = data_dir / "hmm_generation_test.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print("\n" + "=" * 70)
    print("HMM Training Complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Use the trained HMM to generate predictions")
    print(f"  2. Convert predictions to audio via PCA inverse + MIDI synthesis")
    print(f"  3. Evaluate against test set continuation ground truth")


if __name__ == "__main__":
    main()
