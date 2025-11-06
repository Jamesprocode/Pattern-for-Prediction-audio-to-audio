# Training Guide - Mel Continuation Transformer

## Setup Weights & Biases

1. **Install wandb** (if not already installed):
```bash
pip install wandb
```

2. **Login to W&B**:
```bash
wandb login
```
This will prompt you to enter your API key from https://wandb.ai/authorize

## Start Training

### Quick Test Run (Small Model)
```bash
python train.py \
  --train_dir "/Users/jameswang/Downloads/PPDD-Sep2018_aud_mono_small/prime_eave" \
  --output_dir "./checkpoints_test" \
  --num_encoder_layers 2 \
  --num_decoder_layers 2 \
  --d_model 256 \
  --batch_size 4 \
  --num_epochs 5 \
  --num_workers 0 \
  --wandb_name "test-run-small"
```

### Full Training Run
```bash
python train.py \
  --train_dir "/Users/jameswang/Downloads/PPDD-Sep2018_aud_mono_small/prime_eave" \
  --output_dir "./checkpoints" \
  --batch_size 16 \
  --num_epochs 100 \
  --num_workers 4 \
  --wandb_name "full-training-v1"
```

### Training with Custom Project Name
```bash
python train.py \
  --train_dir "/Users/jameswang/Downloads/PPDD-Sep2018_aud_mono_small/prime_eave" \
  --wandb_project "my-audio-project" \
  --wandb_name "experiment-1"
```

### Disable W&B (use only TensorBoard)
```bash
python train.py \
  --train_dir "/Users/jameswang/Downloads/PPDD-Sep2018_aud_mono_small/prime_eave" \
  --no_wandb
```

## What Gets Logged to W&B

### Metrics (per epoch):
- `train/loss` - Training loss (epoch average)
- `train/batch_loss` - Training loss (per batch, real-time)
- `val/loss` - Validation loss
- `learning_rate` - Current learning rate
- `epoch` - Current epoch number

### Config (hyperparameters):
- Model architecture: d_model, nhead, num_encoder_layers, etc.
- Data parameters: context_frames, predict_frames, sr, n_mels
- Training parameters: batch_size, learning_rate, num_epochs
- Total model parameters

### Model Monitoring:
- Gradients and parameters (logged every 100 steps)
- Best validation loss (saved in run summary)

## Viewing Results

### W&B Dashboard
Your training will appear at: `https://wandb.ai/<your-username>/<project-name>`

You'll see:
- Real-time loss curves
- Learning rate schedule
- System metrics (GPU, CPU, memory)
- Model gradients and weights
- Hyperparameter comparison across runs

### TensorBoard (still available)
```bash
tensorboard --logdir checkpoints/logs
```

## Resume Training from Checkpoint
```bash
python train.py \
  --train_dir "/Users/jameswang/Downloads/PPDD-Sep2018_aud_mono_small/prime_eave" \
  --resume_from "./checkpoints/checkpoint_epoch_20.pt" \
  --wandb_name "resumed-training"
```

## All Command Line Options

```
Data:
  --train_dir          Directory with training audio
  --val_dir            Directory with validation audio (optional)
  --output_dir         Where to save checkpoints (default: ./checkpoints)

Model Architecture:
  --n_mels             Number of mel bins (default: 80)
  --d_model            Transformer dimension (default: 512)
  --nhead              Number of attention heads (default: 8)
  --num_encoder_layers Encoder layers (default: 6)
  --num_decoder_layers Decoder layers (default: 6)
  --dim_feedforward    FFN dimension (default: 2048)
  --dropout            Dropout rate (default: 0.1)

Data Processing:
  --context_frames     Input context frames (default: 100)
  --predict_frames     Frames to predict (default: 50)
  --sr                 Sample rate (default: 22050)

Training:
  --batch_size         Batch size (default: 32)
  --num_epochs         Training epochs (default: 100)
  --learning_rate      Learning rate (default: 1e-4)
  --num_workers        Data loader workers (default: 4)

W&B Logging:
  --use_wandb          Enable W&B (default: True)
  --no_wandb           Disable W&B
  --wandb_project      W&B project name (default: mel-continuation-transformer)
  --wandb_name         W&B run name (optional)

Other:
  --resume_from        Checkpoint path to resume from
  --device             Device: cuda/cpu (auto-detected if not specified)
```

## Tips

1. **Start small**: Use the test run first to make sure everything works
2. **Monitor W&B**: Check wandb.ai to see real-time training progress
3. **GPU recommended**: Training will be much faster with CUDA
4. **Adjust batch_size**: Reduce if you get OOM errors
5. **Use validation set**: Split your data into train/val for better monitoring
