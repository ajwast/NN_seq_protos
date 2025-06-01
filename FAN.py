import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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


def compute_beat_strength(step_idx):
    if step_idx % 4 == 0:
        return 1.0 if step_idx == 0 else 0.8
    elif step_idx % 2 == 0:
        return 0.5
    else:
        return 0.2

def build_fan_inputs_from_32xN(data_array):
    """
    Converts 32xN input array into FAN-ready input tensors.
    
    Args:
        data_array (np.ndarray): shape = (32, N_examples)
            - First 16 rows: binary onsets
            - Next 16 rows: groove float offsets (0.0 to 1.0)
    
    Returns:
        inputs: Tensor of shape (N_examples, 16, 4)
        onset_targets: Tensor of shape (N_examples, 16)
        groove_targets: Tensor of shape (N_examples, 16)
    """
    assert data_array.shape[0] == 32, "Expected shape (32, N), got {}".format(data_array.shape)

    N = data_array.shape[1]
    step_positions = np.linspace(0, 1, 16)  # Normalized step indices
    beat_strengths = [compute_beat_strength(i) for i in range(16)]

    all_inputs = []
    all_onsets = []
    all_grooves = []

    for i in range(N):
        onset_row = data_array[:16, i]       # shape (16,)
        groove_row = data_array[16:, i]      # shape (16,)

        example_inputs = []

        for t in range(16):
            prev_onset = onset_row[t - 1] if t > 0 else 0.0
            next_onset = onset_row[t + 1] if t < 15 else 0.0

            feature_vec = [
                step_positions[t],           # step_index (normalized)
                beat_strengths[t],           # beat_strength
                prev_onset,                  # previous onset
                next_onset                   # next onset
            ]
            example_inputs.append(feature_vec)

        all_inputs.append(example_inputs)
        all_onsets.append(onset_row)
        all_grooves.append(groove_row)

    inputs = torch.tensor(all_inputs, dtype=torch.float32)          # (N, 16, 4)
    onset_targets = torch.tensor(all_onsets, dtype=torch.float32)   # (N, 16)
    groove_targets = torch.tensor(all_grooves, dtype=torch.float32) # (N, 16)

    return inputs, onset_targets, groove_targets


def train_rhythm_fan(
    model,
    dataloader,
    num_epochs=10,
    lr=1e-3,
    groove_loss_weight=1.0,
    device='cpu'
):
    """
    Trains the RhythmFAN model.

    Args:
        model: RhythmFAN model instance.
        dataloader: PyTorch DataLoader yielding (inputs, onset_targets, groove_targets)
        num_epochs: Number of training epochs.
        lr: Learning rate.
        groove_loss_weight: Weighting factor for groove loss.
        device: 'cuda' or 'cpu'.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        total_onset_loss = 0.0
        total_groove_loss = 0.0

        for inputs, onset_targets, groove_targets in dataloader:
            inputs = inputs.to(device)                   # (B, 16, input_dim)
            onset_targets = onset_targets.to(device)     # (B, 16)
            groove_targets = groove_targets.to(device)   # (B, 16)

            # Forward pass
            onset_logits, groove_pred = model(inputs)

            # Losses
            loss_onset = bce_loss(onset_logits, onset_targets)
            loss_groove = mse_loss(groove_pred, groove_targets)

            loss = loss_onset + groove_loss_weight * loss_groove

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_onset_loss += loss_onset.item()
            total_groove_loss += loss_groove.item()

        avg_onset = total_onset_loss / len(dataloader)
        avg_groove = total_groove_loss / len(dataloader)
        print(f"Epoch {epoch+1:02d}: Onset Loss = {avg_onset:.4f}, Groove Loss = {avg_groove:.4f}")

class RhythmDataset(Dataset):
    def __init__(self, inputs, onset_targets, groove_targets):
        self.inputs = inputs
        self.onsets = onset_targets
        self.grooves = groove_targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.onsets[idx], self.grooves[idx]


dataset = RhythmDataset(inputs, onsets, grooves)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = RhythmFAN(input_dim=4, num_frequencies=16, hidden_dim=64)

train_rhythm_fan(model, dataloader, num_epochs=20, lr=1e-3)
