# HMM Beat Pattern Prediction

An audio-to-audio pattern prediction system using Hidden Markov Models (HMM) to generate musical continuations from input audio.

## Overview

This project learns temporal patterns in musical sequences using a Gaussian HMM trained on spectral features. Given an audio input, the model generates a continuation that matches the learned pattern.

**Complete Pipeline:**
```
MIDI Files (100, varying BPM)
    ↓
Tempo Normalization → 120 BPM
    ↓
FluidSynth Synthesis → 22050 Hz WAV
    ↓
Verification & Train/Test Split (60/15)
    ↓
Beat Detection + Spectrogram Extraction
    ↓
16th-Note Observations (80 mel × 10 steps)
    ↓
PCA Dimensionality Reduction (800D → 20D, 89.38% variance)
    ↓
Gaussian HMM Training (10 hidden states)
    ↓
Continuation Prediction (Viterbi + Conditional Sampling)
    ↓
Griffin-Lim Phase Reconstruction
    ↓
Audio Output
```

## Project Structure

```
hmm_beat_pattern/
├── normalized_dataset/            # this will be in the shared google drive
│   ├── train/                     # 60 normalized & synthesized WAV files
│   ├── test/                      # 15 test WAV files + continuation MIDI
│   └── dataset_stats.json         # Dataset statistics (100 → 75 verified)
├── spectrogram_data/              # find in the shared google drive
│   ├── train_features.npy         # PCA features (12848, 20)
│   ├── test_features.npy          # PCA features (3184, 20)
│   ├── train_names.pkl            # File names per feature
│   ├── test_names.pkl             # File names per feature
│   ├── pca_model.pkl              # Fitted PCA (800D → 20D)
│   ├── hmm_model.pkl              # Trained HMM model
│   ├── hmm_generation_test.json   # Generation test results
│   └── prediction_results.json    # Evaluation metrics (15 files)
├── synthesize_and_split.py        # MIDI → Audio pipeline
├── extract_spectrograms.py        # Audio → PCA features pipeline
├── train_hmm.py                   # HMM training
├── predict_continuation.py        # Evaluation script (cardinality + pitch accuracy)
├── test_single_file.py            # Interactive single-file testing
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8+
- conda (recommended)
- FluidSynth (for MIDI synthesis)

### Install FluidSynth

**macOS:**
```bash
brew install fluidsynth
```

### Install Python Dependencies

```bash
conda create -n p4p python=3.10
conda activate p4p
pip install -r requirements.txt
```

## Detailed Workflow

### Step 1: MIDI Normalization & Synthesis (synthesize_and_split.py)

**Problem:** Input MIDI files have extreme BPM variability (μ=116.9 BPM, σ=31.0)

**Solution:** Normalize all files to 120 BPM before processing

```bash
python synthesize_and_split.py
```

**What it does:**

1. **Tempo Normalization:**
   - Loads each MIDI file
   - Calculates: `tempo_ratio = 120.0 / original_bpm`
   - Scales all note timings: `note.start *= tempo_ratio`, `note.end *= tempo_ratio`
   - Preserves musical relationships while unifying tempo

2. **Audio Synthesis:**
   - Uses FluidSynth command-line tool
   - Configuration: `fluidsynth -ni -r 22050 -F output.wav soundfont.sf2 input.mid`
   - Sample rate: 22050 Hz (matches spectrogram extraction)

3. **Tempo Verification:**
   - Uses librosa.feature.tempo() for re-detection
   - Acceptance criteria: 120 BPM ± 5 BPM or tempo octaves (60, 240 BPM)
   - Tolerance accounting for synthesis artifacts

4. **Train/Test Split:**
   - 80/20 split on verified files
   - 60 training files → normalized_dataset/train/
   - 15 test files → normalized_dataset/test/
   - Continuation MIDI saved as ground truth

**Input:**
- 100 MIDI files from `/Downloads/PPDD-Jul2018_aud_mono_small/prime_midi/`
- 100 continuation MIDI files (ground truth)

**Output:**
- 75 verified WAV files (60 train, 15 test)
- dataset_stats.json with BPM statistics

**Results:**
```
Total: 100
Synthesized: 100
Verified: 75 (mean BPM: 113.09 ± 15.67)
Failed: 25 (detected BPMs outside tolerance)
```

### Step 2: Spectrogram Extraction & PCA (extract_spectrograms.py)

**Goal:** Convert audio into fixed-length spectral observations

```bash
python extract_spectrogram.py
```

process_dataset_spectrograms(
    dataset_dir="normalized_dataset",
    output_dir="spectrogram_data",
    target_time_steps=10,
    n_mels=80,
    n_pca_components=20
)
```

**Pipeline:**

1. **Beat Detection:**
   - Uses librosa.beat.beat_track()
   - Detects beat times in audio

2. **16th-Note Segmentation:**
   - Creates 16th-note times (4 per beat)
   - Musically meaningful temporal scale
   - Fixed-length windows per segment

3. **Spectrogram Extraction:**
   - Mel spectrogram: 80 mel bins, n_fft=2048, hop_length=512
   - Converts to dB: librosa.power_to_db()
   - Normalizes to [0, 1]: `(mel_db + 80) / 80.0`
   - Each observation: 80 mel bins × 10 time steps = 800 dimensions

4. **Spectrogram Validation:**
   - Ensures consistent shape: (80, 10)
   - Pads with zeros if too short
   - Truncates if too long
   - Validates all features match dimensions

5. **PCA Dimensionality Reduction:**
   - Flattens: (80, 10) → (800,) per observation
   - Fits PCA on training data only (12,848 observations)
   - Transforms to 20 principal components
   - Variance explained: **89.38%** (97.5% dimension reduction)

**Output:**
```
train_features.npy:  (12848, 20) - 60 files, 214 specs/file avg
test_features.npy:   (3184, 20)  - 15 files, 212 specs/file avg
pca_model.pkl:       Fitted PCA transformer for inverse transform
train_names.pkl:     File name per feature (for grouping)
test_names.pkl:      File name per feature (for grouping)
```

### Step 3: HMM Training (train_hmm.py)

**Goal:** Learn temporal patterns in spectrogram sequences

```bash
python train_hmm.py
```

**Configuration:**
```
Model Type:        Gaussian HMM
Hidden States:     10
Covariance Type:   full
Iterations:        200
Random Seed:       42
```

**What it learns:**

1. **Transition Matrix (10×10):**
   - P(next_state | current_state)
   - How musical states transition

2. **Means (10×20):**
   - μᵢ for each state
   - Average spectrogram signature per state

3. **Covariances (10×20):**
   - Σᵢ for each state
   - Variability of spectrogram features per state

**Training Results:**
```
Converged:              Yes
Log-Likelihood:         -125,286.56
States Visited (test):  6/10
```

**Output:**
```
hmm_model.pkl:            Trained HMM + PCA model
hmm_generation_test.json: 100 generated samples, 6 states visited
```

### Step 4: Continuation Prediction & Evaluation (predict_continuation.py)

**Goal:** Generate continuations and measure accuracy

```bash
python predict_continuation.py
```

**Pipeline for Each Test File:**

1. **Input:** Test audio features (n_frames, 20)

2. **Viterbi Algorithm:**
   ```python
   state_path = hmm_model.predict(file_features)
   final_state = state_path[-1]
   ```
   - Finds most likely hidden state sequence
   - Extracts final state for continuation

3. **Continuation Generation:**
   ```python
   for _ in range(n_continuation):
       mean = hmm_model.means_[current_state]
       cov = hmm_model.covars_[current_state]

       # Create covariance matrix (handles full covariance)
       if cov.ndim == 1:
           cov_matrix = np.diag(np.abs(cov) + 1e-6)
       else:
           cov_matrix = cov + np.eye(cov.shape[0]) * 1e-6

       # Sample observation from state's Gaussian
       obs = np.random.multivariate_normal(mean, cov_matrix)
       generated.append(obs)

       # Transition to next state
       transition_probs = hmm_model.transmat_[current_state]
       current_state = np.random.choice(len(transition_probs), p=transition_probs)
   ```

4. **Inverse PCA:**
   ```python
   specs_flat = pca_model.inverse_transform(continuation_features)
   specs = specs_flat.reshape(n_continuation, 80, 10)
   ```

5. **Spectrogram → Audio (Griffin-Lim):**
   ```python
   mel_spec_full = np.concatenate(specs, axis=1)  # Stack spectrograms
   mel_spec_linear = librosa.db_to_power(mel_spec_full)
   spec = librosa.feature.inverse.mel_to_stft(mel_spec_linear, sr=sr, n_fft=2048)
   y = librosa.griffinlim(spec, hop_length=512, n_iter=32)
   ```
   - Converts dB → linear magnitude
   - Reconstructs full spectrogram
   - Recovers phase using Griffin-Lim iterations

6. **Evaluation Metrics:**
   - **Cardinality Score:** Note matching (±1 semitone tolerance)
     - Score = min(matches / len(ground_truth_notes), 1.0)
   - **Pitch Accuracy:** Mean pitch correctness
     - Per-note accuracy = max(0, 1 - |pitch_diff| / 12)
     - Octave-normalized (12 semitones)

**Evaluation Results (15 test files):**

| Metric | Mean | Std | Best | Worst |
|--------|------|-----|------|-------|
| Cardinality Score | 0.339 | 0.391 | 1.000 | 0.000 |
| Pitch Accuracy | 0.375 | 0.284 | 0.825 | 0.000 |

**Best Performers (Cardinality = 1.0):**
- dd1e320f: Pitch accuracy 0.766
- ef1c3b2b: Pitch accuracy 0.825 ⭐
- e600b304: Pitch accuracy 0.669

**Output:**
```
prediction_results.json:  Per-file scores for all 15 test files
Console:                  Summary statistics
```

## Interactive Testing: Single File

Use this notebook workflow to test individual audio files:

```bash
test_hmm.ipynb
```

## Results Summary

Successfully built complete HMM-based audio pattern prediction system:

✓ **Data Preparation:** 100 MIDI files with 54.9-198.8 BPM variability → 75 verified at 120 BPM
✓ **Feature Engineering:** 15,000+ beat-synchronous spectrograms → 20D PCA features (89.38% variance)
✓ **Model Training:** Gaussian HMM with 10 states learning temporal patterns
✓ **Continuation Generation:** Viterbi + conditional sampling + Griffin-Lim reconstruction
✓ **Evaluation:** 15 test files evaluated on cardinality and pitch accuracy metrics

**Key Achievement:** Model achieves perfect note matching (cardinality = 1.0) on 3/15 test files with pitch accuracy up to 0.825, demonstrating viability of HMM-based music continuation.



---

**Last Updated:** December 2024
**Status:** Working prototype with full evaluation results
**Authors:** Pattern for Prediction Team
