"""
Simple Mel Spectrogram Continuation Transformer
Direct flattening approach - no CNNs needed!
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    Uses sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MelContinuationTransformer(nn.Module):
    """
    Transformer for mel spectrogram continuation.
    
    Architecture:
    1. Project mel frames (n_mels) to d_model dimension
    2. Add positional encoding
    3. Process with transformer encoder (understand context)
    4. Use transformer decoder to generate future frames
    5. Project back to mel space (n_mels)
    """
    
    def __init__(
        self,
        n_mels=80,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_predict_frames=100
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.d_model = d_model
        self.max_predict_frames = max_predict_frames
        
        # Input projection: mel space → d_model space
        self.input_projection = nn.Linear(n_mels, d_model)
        
        # Output projection: d_model space → mel space
        self.output_projection = nn.Linear(d_model, n_mels)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important! Expects (batch, seq, feature)
        )
        
        # Learnable queries for future frames
        # These are the "questions" we ask: "What should the next frame be?"
        self.future_queries = nn.Parameter(
            torch.randn(max_predict_frames, d_model) * 0.02
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training"""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, mel_context, predict_length=50):
        """
        Forward pass for mel continuation.
        
        Args:
            mel_context: Input mel spectrogram (batch, n_mels, context_frames)
                        e.g., (32, 80, 100) - 32 samples, 100 frames of context
            predict_length: Number of future frames to predict (int)
                           e.g., 50 - predict next 50 frames
        
        Returns:
            mel_future: Predicted mel spectrogram (batch, n_mels, predict_length)
                       e.g., (32, 80, 50) - prediction for next 50 frames
        """
        batch_size = mel_context.size(0)
        
        # Step 1: Transpose to (batch, time, n_mels)
        # We treat each time frame as a token
        src = mel_context.transpose(1, 2)  # (B, context_time, n_mels)
        
        # Step 2: Project mel frames to d_model dimension
        src = self.input_projection(src)  # (B, context_time, d_model)
        
        # Step 3: Add positional encoding
        src = self.pos_encoder(src)  # (B, context_time, d_model)
        
        # Step 4: Create target queries for future frames
        # These are learnable parameters that ask "what comes next?"
        tgt = self.future_queries[:predict_length].unsqueeze(0).repeat(batch_size, 1, 1)
        # tgt: (B, predict_length, d_model)
        
        # Add positional encoding to queries too
        tgt = self.pos_encoder(tgt)
        
        # Step 5: Run through transformer
        # Encoder processes the context, decoder generates future
        output = self.transformer(src, tgt)  # (B, predict_length, d_model)
        
        # Step 6: Project back to mel space
        mel_future = self.output_projection(output)  # (B, predict_length, n_mels)
        
        # Step 7: Transpose back to (batch, n_mels, time)
        mel_future = mel_future.transpose(1, 2)  # (B, n_mels, predict_length)
        
        return mel_future
    
    def generate(self, mel_context, predict_length=50, temperature=1.0):
        """
        Generate continuation with optional temperature control.
        
        Args:
            mel_context: Input context (batch, n_mels, context_frames)
            predict_length: Number of frames to generate
            temperature: Controls randomness (not used in regression, but useful for sampling)
        
        Returns:
            Generated mel spectrogram (batch, n_mels, predict_length)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(mel_context, predict_length)


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing Mel Continuation Transformer...")
    print("=" * 60)
    
    # Create model
    model = MelContinuationTransformer(
        n_mels=80,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    )
    
    print(f"Model created!")
    print(f"Total parameters: {count_parameters(model):,}")
    print()
    
    # Test with dummy data
    batch_size = 4
    context_frames = 100
    predict_frames = 50
    
    # Create random mel spectrogram (simulating real data)
    mel_input = torch.randn(batch_size, 80, context_frames)
    
    print(f"Input shape: {mel_input.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Mel bins: 80")
    print(f"  - Context frames: {context_frames}")
    print()
    
    # Forward pass
    print("Running forward pass...")
    mel_output = model(mel_input, predict_length=predict_frames)
    
    print(f"Output shape: {mel_output.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Mel bins: 80")
    print(f"  - Predicted frames: {predict_frames}")
    print()
    
    print("✓ Model test passed!")
    print("=" * 60)