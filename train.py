"""
Training script for Mel Continuation Transformer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from pathlib import Path
import argparse
import wandb

from model import MelContinuationTransformer, count_parameters
from dataset import create_dataloaders


class MelLoss(nn.Module):
    """
    Combined loss for mel spectrogram prediction.
    Uses MSE for basic reconstruction + optional spectral loss.
    """
    def __init__(self, use_spectral_loss=True, spectral_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.use_spectral_loss = use_spectral_loss
        self.spectral_weight = spectral_weight
    
    def forward(self, predicted, target):
        """
        Args:
            predicted: (batch, n_mels, time)
            target: (batch, n_mels, time)
        """
        # Basic MSE loss
        mse_loss = self.mse(predicted, target)
        
        if not self.use_spectral_loss:
            return mse_loss
        
        # Spectral convergence loss (encourages matching frequency structure)
        spectral_loss = torch.norm(target - predicted, p='fro') / (torch.norm(target, p='fro') + 1e-8)
        
        return mse_loss + self.spectral_weight * spectral_loss


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_wandb=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (context, target) in enumerate(train_loader):
        # Move to device
        context = context.to(device)
        target = target.to(device)

        # Forward pass
        predicted = model(context, predict_length=target.size(2))
        loss = criterion(predicted, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (helps training stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Log to wandb every step
        if use_wandb:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/epoch': epoch,
            })

        # Print progress
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.6f} '
                  f'Time: {elapsed:.1f}s')

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for context, target in val_loader:
            context = context.to(device)
            target = target.to(device)
            
            predicted = model(context, predict_length=target.size(2))
            loss = criterion(predicted, target)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.6f}")
    return epoch, loss


def train(
    train_dir,
    val_dir=None,
    output_dir="./checkpoints",
    # Model parameters
    n_mels=80,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    # Data parameters
    context_frames=100,
    predict_frames=50,
    sr=22050,
    # Training parameters
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-4,
    num_workers=4,
    # Other
    resume_from=None,
    device=None,
    use_wandb=True,
    wandb_project="mel-continuation-transformer",
    wandb_name=None
):
    """
    Main training function.

    Args:
        train_dir: Directory with training audio files
        val_dir: Directory with validation audio files
        output_dir: Directory to save checkpoints and logs
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_name: W&B run name (optional)
        ... (other parameters documented in argparse below)
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize W&B
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                'n_mels': n_mels,
                'd_model': d_model,
                'nhead': nhead,
                'num_encoder_layers': num_encoder_layers,
                'num_decoder_layers': num_decoder_layers,
                'dim_feedforward': dim_feedforward,
                'dropout': dropout,
                'context_frames': context_frames,
                'predict_frames': predict_frames,
                'sr': sr,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
            }
        )

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # Create model
    print("\nCreating model...")
    model = MelContinuationTransformer(
        n_mels=n_mels,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    if use_wandb:
        wandb.config.update({'num_parameters': num_params})
        wandb.watch(model, log='all', log_freq=100)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        sr=sr,
        n_mels=n_mels,
        context_frames=context_frames,
        predict_frames=predict_frames
    )
    
    # Create loss function and optimizer
    criterion = MelLoss(use_spectral_loss=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduces LR when loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from is not None:
        start_epoch, _ = load_checkpoint(resume_from, model, optimizer)
        start_epoch += 1
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_wandb=use_wandb)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.6f}")

            # Log to tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # Log to W&B
            if use_wandb:
                wandb.log({
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })

            # Update learning rate
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(output_dir, 'best_model.pt')
                )
                if use_wandb:
                    wandb.run.summary['best_val_loss'] = best_val_loss
        else:
            # No validation set
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # Log to W&B
            if use_wandb:
                wandb.log({
                    'train/loss': train_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
            )
    
    print("\n" + "=" * 60)
    print("Training complete!")

    # Save final model
    save_checkpoint(
        model, optimizer, num_epochs - 1, train_loss,
        os.path.join(output_dir, 'final_model.pt')
    )

    writer.close()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Mel Continuation Transformer')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Directory with training audio files')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Directory with validation audio files')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints and logs')
    
    # Model arguments
    parser.add_argument('--n_mels', type=int, default=80,
                       help='Number of mel frequency bins')
    parser.add_argument('--d_model', type=int, default=512,
                       help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                       help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                       help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Data processing arguments
    parser.add_argument('--context_frames', type=int, default=100,
                       help='Number of context frames')
    parser.add_argument('--predict_frames', type=int, default=50,
                       help='Number of frames to predict')
    parser.add_argument('--sr', type=int, default=22050,
                       help='Sample rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Other arguments
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')

    # W&B arguments
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Use Weights & Biases logging')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mel-continuation-transformer',
                       help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='W&B run name (optional)')

    args = parser.parse_args()

    # Train
    train(**vars(args))