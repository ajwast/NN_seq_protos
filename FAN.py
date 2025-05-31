import torch
import torch.nn as nn
import torch.nn.functional as F

# Fourier feature layer: encodes periodicity explicitly
class FANFourierLayer(nn.Module):
    def __init__(self, input_dim, num_frequencies=16, output_dim=64):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(num_frequencies, input_dim))  # (F, D)
        self.linear = nn.Linear(2 * num_frequencies, output_dim)

    def forward(self, x):
        # x: (batch, steps, input_dim)
        B, S, D = x.size()
        x_flat = x.view(B * S, D)
        proj = x_flat @ self.W_in.T  # (B*S, F)
        sin_cos = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B*S, 2F)
        out = self.linear(sin_cos).view(B, S, -1)  # (B, S, output_dim)
        return out

# Rhythm generation model: predicts onsets and groove offsets
class RhythmFAN(nn.Module):
    def __init__(self, input_dim, num_frequencies=16, hidden_dim=64):
        super().__init__()
        self.fourier = FANFourierLayer(input_dim, num_frequencies, hidden_dim)
        self.onset_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # (B, S, 1)
        )
        self.groove_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x):
        # x: (batch, 16, input_dim)
        h = self.fourier(x)
        onset_logits = self.onset_head(h).squeeze(-1)    # (B, 16)
        groove_offsets = self.groove_head(h).squeeze(-1) # (B, 16)
        return onset_logits, groove_offsets

# Dummy input: batch of 2 sequences, 16 steps, 3 features (e.g., step_index, beat_strength, style_id)
batch_size = 2
steps = 16
input_dim = 3
x = torch.randn(batch_size, steps, input_dim)

# Initialize model and forward
model = RhythmFAN(input_dim=input_dim)
onset_logits, groove_offsets = model(x)

# Convert logits to probabilities
onset_probs = torch.sigmoid(onset_logits)

# Threshold to get binary onset predictions
onset_predictions = (onset_probs > 0.5).float()

# Outputs
print("Onset predictions:\n", onset_predictions)
print("Groove offsets:\n", groove_offsets)

