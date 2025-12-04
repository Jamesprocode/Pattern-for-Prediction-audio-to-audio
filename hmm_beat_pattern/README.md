# Beat Aggregation + HMM Pattern Prediction

This project implements a **Hidden Markov Model (HMM)** approach for audio pattern prediction using beat-level aggregation.

## Design Overview

```
1. Audio Input
   ↓
2. Beat Detection (librosa)
   ↓
3. Beat Segmentation (split audio by beat times)
   ↓
4. Spectrogram Extraction (one mel-spectrogram per beat)
   ↓
5. HMM Training (learn transitions between spectrograms)
   ↓
6. HMM Generation (predict next beat spectrograms)
   ↓
7. Audio Synthesis (convert spectrograms back to audio)
```

## Workflow

### Step 1: Analyze Dataset BPM

Check if your audio files have consistent BPM (important for beat detection):

```bash
python analyze_bpm.py
```

This will:
- Scan your dataset for audio files
- Estimate BPM for each file
- Report consistency statistics
- Tell you if files have similar or varying BPM

**Expected output:**
```
Files processed: 50
Mean BPM: 120.5
Std Dev: 2.3
Range: 115.0 - 126.0 BPM
Consistency: HIGH (±5 BPM)
```

### Step 2: Extract and Visualize Beat Spectrograms

Test different window sizes and see how they affect spectrogram quality:

```bash
python extract_beat_specs.py
```

This will:
- Load an audio file
- Extract beat spectrograms with different FFT window sizes
- Visualize the first 16 beats
- Save comparison images

**Window configurations to test:**
- **Small window (n_fft=1024)**: Better time resolution, more detail
- **Medium window (n_fft=2048)**: Balanced frequency/time resolution (default)
- **Large window (n_fft=4096)**: Better frequency resolution, smoother

### Step 3: Train HMM

Once you've chosen optimal settings, train the HMM:

```bash
python train_hmm.py
```

The HMM learns:
- Probability distributions of spectrograms
- Transitions between consecutive beats
- Hidden state patterns

### Step 4: Generate Predictions

Use the trained HMM to predict next beats (to be implemented):

```python
from train_hmm import BeatHMM

# Load trained model
hmm_model = BeatHMM()
hmm_model.load("beat_hmm_model.pkl")

# Generate predictions
generated_specs = hmm_model.generate_sequence(n_steps=10)

# Convert back to audio (using Griffin-Lim or other phase reconstruction)
```

## Key Parameters to Experiment With

### Beat Detection
- `sr` (sample rate): 22050 Hz (standard)
- Librosa's beat tracking is automatic

### Spectrogram Extraction
- `n_fft`: FFT window size
  - Larger = better frequency resolution
  - Smaller = better time resolution
- `hop_length`: Samples between frames (usually n_fft/4)
- `n_mels`: Number of mel frequency bins (80 is standard)
- `fmin/fmax`: Frequency range to capture

### HMM Training
- `n_hidden_states`: Number of hidden states (5-10 typically)
  - More states = more complex patterns
  - Fewer states = more generalization
- `n_components`: Gaussian components per state
- `covariance_type`: 'diag' (default), 'full', 'tied', 'spherical'
- `n_iter`: EM iterations (50-100 typically)

## Important Notes

1. **BPM Consistency**: If your dataset has varying BPM, beat alignment might be difficult. Check with Step 1 first.

2. **Spectrogram Shape**: Beats will have variable length spectrograms (different number of time frames). HMM handles this by:
   - Flattening spectrograms to vectors
   - Training on variable-length sequences
   - Using sequence lengths to track boundaries

3. **Phase Reconstruction**: Converting spectrograms back to audio requires phase information:
   - Griffin-Lim algorithm (simple, fast)
   - STFT inversion (faster, better quality)
   - Neural vocoder (best, but slower)

4. **Audio Quality**: Current approach uses magnitude spectrograms only. Consider:
   - Adding phase information
   - Using log-magnitude for better numerical stability
   - Normalizing by beat duration

## File Structure

```
hmm_beat_pattern/
├── beat_utils.py          # Core utilities for beat extraction
├── analyze_bpm.py         # Dataset BPM analysis script
├── extract_beat_specs.py  # Beat spectrogram extraction & visualization
├── train_hmm.py          # HMM training and generation
└── README.md             # This file
```

## Requirements

```
librosa>=0.9.0
numpy
matplotlib
hmmlearn
scikit-learn
soundfile (for audio I/O)
```

Install with:
```bash
pip install librosa numpy matplotlib hmmlearn scikit-learn soundfile
```

## Next Steps

1. ✓ Prepare beat extraction utilities
2. → Check BPM consistency in your dataset
3. → Experiment with different spectrogram window sizes
4. → Train HMM on beat sequences
5. → Implement audio synthesis from HMM predictions
6. → Evaluate quality and iterate

## References

- Librosa Documentation: https://librosa.org/
- hmmlearn Documentation: https://hmmlearn.readthedocs.io/
- Beat Tracking: http://music-ir.org/mirex/abstracts/2014/ellis_beat.pdf
