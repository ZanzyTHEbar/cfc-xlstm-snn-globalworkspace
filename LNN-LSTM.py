import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint_adjoint as odeint

# sine wave test dataset
seq_length = 50
num_sequences = 100
test_split = 0.2

x = np.linspace(0, num_sequences * np.pi, num_sequences * seq_length)
sine_wave = np.sin(x)

data = []
targets = []
for i in range(len(sine_wave) - seq_length):
    data.append(sine_wave[i : i + seq_length])
    targets.append(sine_wave[i + 1 : i + seq_length + 1])

data = np.array(data)
targets = np.array(targets)

split_idx = int(len(data) * (1 - test_split))
train_data = data[:split_idx]
train_targets = targets[:split_idx]
test_data = data[split_idx:]
test_targets = targets[split_idx:]

train_data = torch.tensor(train_data, dtype=torch.float32)
train_targets = torch.tensor(train_targets, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_targets = torch.tensor(test_targets, dtype=torch.float32)



# CfC-xLSTM Cell (NeuroFlexNet)
class NeuroFlexCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuroFlexCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)  # xLSTM equivalent
        self.A = nn.Parameter(torch.Tensor(hidden_size))
        self.w_tau = nn.Parameter(torch.Tensor(hidden_size))

        nn.init.uniform_(self.A, -0.1, 0.1)
        nn.init.uniform_(self.w_tau, -0.1, 0.1)

        self.learning_rate = 0.0001  # Reduced learning rate
        self.dT = 1.0
        self.epsilon = 1e-5  # Small epsilon to prevent numerical instability

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_t, hx):
        h_t, c_t = hx
        
        # xLSTM: Standard LSTM update
        x_LSTM_t, c_t = self.lstm_cell(input_t, (h_t, c_t))

        # CfC: Continuous dynamics with spike-driven updates
        I_t = input_t
        f = torch.sigmoid
        exponent = -(self.w_tau + f(I_t)) * self.dT + self.epsilon
        x_combined = (
            (x_LSTM_t - self.A) * torch.exp(exponent) * f(-I_t) + self.A
        ) + h_t

        # Apply layer normalization for stability
        x_combined = self.layer_norm(x_combined)

        h_t = x_combined  # Combine CfC and LSTM updates
        return x_combined, (h_t, c_t)

    def update_weights(self, loss):
        grads = torch.autograd.grad(
            loss, self.parameters(), retain_graph=True, allow_unused=True
        )
        with torch.no_grad():
            for param, grad in zip(self.parameters(), grads):
                if grad is not None:
                    param -= self.learning_rate * grad * self.dT


# NeuroFlexNet Model (CfC + xLSTM)
class NeuroFlexNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuroFlexNet, self).__init__()
        self.hidden_size = hidden_size
        self.neuroflex_cell = NeuroFlexCell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, h_t=None, c_t=None):
        seq_len, batch_size, _ = inputs.size()

        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        if c_t is None:
            c_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        outputs = []
        for t in range(seq_len):
            input_t = inputs[t]
            x_combined, (h_t, c_t) = self.neuroflex_cell(input_t, (h_t, c_t))
            output_t = self.output_layer(x_combined)
            outputs.append(output_t)
        outputs = torch.stack(outputs, dim=0)
        return outputs, (h_t, c_t)

    def update_weights(self, loss):
        self.neuroflex_cell.update_weights(loss)


# Prepare DataLoaders
train_dataset = TensorDataset(train_data, train_targets)
test_dataset = TensorDataset(test_data, test_targets)

batch_size = 16  # Reduced batch size
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
)

# Instantiate the model
input_size = 1
hidden_size = 20
output_size = 1

model = NeuroFlexNet(input_size, hidden_size, output_size).to("cuda")
optimizer = optim.Adam(
    model.parameters(), lr=0.0001, weight_decay=1e-4
)  # Added L2 regularization

criterion = nn.MSELoss()

# Train the model with gradient clipping
num_epochs = 10  # Increase the number of epochs

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data_batch, target_batch) in enumerate(train_loader):
        data_batch = data_batch.unsqueeze(-1).transpose(0, 1).to("cuda")
        target_batch = target_batch.unsqueeze(-1).transpose(0, 1).to("cuda")
        h_t = None
        c_t = None
        outputs, (h_t, c_t) = model(data_batch, h_t, c_t)
        print(outputs[0][0])
        print(target_batch[0][0])

        loss = criterion(outputs, target_batch)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # Apply gradient clipping
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.6f}")

    # Inference
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        for data_batch, target_batch in test_loader:
            data_batch = data_batch.unsqueeze(-1).transpose(0, 1).to("cuda")
            target_batch = target_batch.unsqueeze(-1).transpose(0, 1).to("cuda")
            outputs, (h_t, c_t) = model(data_batch, h_t, c_t)
            loss = criterion(outputs, target_batch)
            total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.6f}")

# Visualization
model.eval()
with torch.no_grad():
    all_outputs = []
    all_targets = []
    for data_batch, target_batch in test_loader:
        data_batch = data_batch.unsqueeze(-1).transpose(0, 1).to("cuda")
        target_batch = target_batch.unsqueeze(-1).transpose(0, 1).to("cuda")
        h_t = None
        c_t = None
        outputs, (h_t, c_t) = model(data_batch, h_t, c_t)
        all_outputs.append(outputs.detach().cpu().numpy())
        all_targets.append(target_batch.detach().cpu().numpy())
        break
    all_outputs = np.concatenate(all_outputs, axis=1).squeeze()
    all_targets = np.concatenate(all_targets, axis=1).squeeze()

plt.figure(figsize=(12, 6))
plt.plot(all_targets.flatten(), label="Actual")
plt.plot(all_outputs.flatten(), label="Predicted")
plt.title("Sine Wave Prediction - NeuroFlexNet")
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
