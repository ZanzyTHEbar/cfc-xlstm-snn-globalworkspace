import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from collections import namedtuple

# --------------------------
# NamedTuple for Hidden States
# --------------------------

HiddenStates = namedtuple(
    "HiddenStates",
    [
        "h_perc",
        "c_perc",
        "h_mirror_obs",
        "c_mirror_obs",
        "h_mirror_act",
        "c_mirror_act",
        "h_gw",
        "c_gw",
        "h_act",
        "c_act",
    ],
)

# --------------------------
# Dataset Preparation
# --------------------------

# Sine wave parameters
seq_length = 50
num_sequences = 1000
test_split = 0.2

# Generate sine wave data
# x = np.linspace(0, num_sequences * 2 * np.pi, num_sequences * seq_length)
# sine_wave = np.sin(x)
#
## Create sequences
# data = []
# targets = []
# for i in range(len(sine_wave) - seq_length):
#    data.append(sine_wave[i : i + seq_length])
#    targets.append(sine_wave[i + 1 : i + seq_length + 1])
#
# data = np.array(data)
# targets = np.array(targets)

# --------------------------
# Clean Stock Price Data
# --------------------------


def clean_stock_data(df):
    # Remove '$' and ',' from the 'Close' column and convert to float
    df["Close"] = df["Close"].replace({"\$": "", ",": ""}, regex=True).astype(float)
    return df


# --------------------------
# Load Stock Data
# --------------------------


def load_stock_data(file_path, seq_length):
    df = pd.read_csv(file_path)
    # Clean the data to ensure 'Close' column is numeric
    df = clean_stock_data(df)

    # Extract the 'Close' price and forward fill any missing values
    closing_prices = df["Close"].ffill().values

    # Ensure that the data is converted to a numeric type (float32)
    closing_prices = closing_prices.astype(np.float32)
    data = []
    targets = []
    for i in range(len(closing_prices) - seq_length):
        data.append(closing_prices[i : i + seq_length])
        targets.append(closing_prices[i + 1 : i + seq_length + 1])
    return np.array(data), np.array(targets)


# Split into training and testing
# split_idx = int(len(data) * (1 - test_split))
# train_data = data[:split_idx]
# train_targets = targets[:split_idx]
# test_data = data[split_idx:]
# test_targets = targets[split_idx:]


# --------------------------
# Prepare Data for PyTorch
# --------------------------


def prepare_tensors(train_data, train_targets, test_data, test_targets):
    # Convert to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(-1)
    train_targets = torch.tensor(train_targets, dtype=torch.float32).unsqueeze(-1)
    test_data = torch.tensor(test_data, dtype=torch.float32).unsqueeze(-1)
    test_targets = torch.tensor(test_targets, dtype=torch.float32).unsqueeze(-1)

    return train_data, train_targets, test_data, test_targets


# Load training data (e.g., AAPL stock data)
train_data, train_targets = load_stock_data("AAPL.csv", seq_length=seq_length)

# Load testing data (e.g., MSFT stock data for generalization testing)
test_data, test_targets = load_stock_data("MSFT.csv", seq_length=seq_length)

# Normalize data (this can improve performance)
train_mean = np.mean(train_data)
train_std = np.std(train_data)

train_data = (train_data - train_mean) / train_std
train_targets = (train_targets - train_mean) / train_std

test_data = (test_data - train_mean) / train_std
test_targets = (test_targets - train_mean) / train_std

# Prepare the data for PyTorch
train_data, train_targets, test_data, test_targets = prepare_tensors(
    train_data, train_targets, test_data, test_targets
)


# DataLoaders
batch_size = 32
train_dataset = TensorDataset(train_data, train_targets)
test_dataset = TensorDataset(test_data, test_targets)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
)

# --------------------------
# CfC-xLSTM Cell (NeuroFlexNet)
# --------------------------


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


# --------------------------
# Mirror Neuron Module
# --------------------------


class MirrorNeuronModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MirrorNeuronModule, self).__init__()
        self.hidden_size = hidden_size

        # Observation Pathway
        self.obs_cell = NeuroFlexCell(input_size, hidden_size)

        # Action Execution Pathway
        self.act_cell = NeuroFlexCell(input_size, hidden_size)

        # Mirror Integration Layer
        self.mirror_layer = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self, obs_input, act_input, hx_obs=None, cx_obs=None, hx_act=None, cx_act=None
    ):

        # Check for is None and initialize hidden states if necessary

        if hx_obs is None:
            hx_obs = torch.zeros(batch_size, self.hidden_size, device=obs_input.device)
        if cx_obs is None:
            cx_obs = torch.zeros(batch_size, self.hidden_size, device=obs_input.device)

        if hx_act is None:
            hx_act = torch.zeros(batch_size, self.hidden_size, device=act_input.device)
        if cx_act is None:
            cx_act = torch.zeros(batch_size, self.hidden_size, device=act_input.device)

        # Observation Pathway
        obs_output, (hx_obs, cx_obs) = self.obs_cell(obs_input, (hx_obs, cx_obs))

        # Action Execution Pathway
        act_output, (hx_act, cx_act) = self.act_cell(act_input, (hx_act, cx_act))

        # Mirror Integration
        combined = torch.cat((obs_output, act_output), dim=1)
        mirror_output = torch.sigmoid(self.mirror_layer(combined))

        return mirror_output, (hx_obs, cx_obs, hx_act, cx_act)


# --------------------------
# Global Workspace Core
# --------------------------


class GlobalWorkspace(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GlobalWorkspace, self).__init__()
        self.hidden_size = hidden_size
        self.cell = NeuroFlexCell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_t, hx=None, cx=None):
        output, (hx, cx) = self.cell(input_t, hx)
        decision = self.output_layer(output)
        return decision, (hx, cx)


# --------------------------
# Full NeuroFlexNet Model
# --------------------------


class NeuroFlexNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuroFlexNet, self).__init__()
        self.hidden_size = hidden_size

        # Modules
        self.perception = NeuroFlexCell(input_size, hidden_size)
        self.mirror_neuron = MirrorNeuronModule(hidden_size, hidden_size)
        self.global_workspace = GlobalWorkspace(hidden_size, hidden_size, hidden_size)
        self.action_execution = NeuroFlexCell(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        inputs,
        h_perc=None,
        c_perc=None,
        h_mirror_obs=None,
        c_mirror_obs=None,
        h_mirror_act=None,
        c_mirror_act=None,
        h_gw=None,
        c_gw=None,
        h_act=None,
        c_act=None,
    ):
        seq_len, batch_size, _ = inputs.size()
        outputs = []

        if h_perc is None:
            h_perc = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        if c_perc is None:
            c_perc = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        # Similarly initialize h_gw, h_mirror_obs, h_mirror_act, h_act, etc.

        if h_gw is None:
            h_gw = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        if c_gw is None:
            c_gw = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        if h_act is None:
            h_act = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        if c_act is None:
            c_act = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        if h_mirror_obs is None:
            h_mirror_obs = torch.zeros(
                batch_size, self.hidden_size, device=inputs.device
            )
        if c_mirror_obs is None:
            c_mirror_obs = torch.zeros(
                batch_size, self.hidden_size, device=inputs.device
            )

        if c_mirror_act is None:
            c_mirror_act = torch.zeros(
                batch_size, self.hidden_size, device=inputs.device
            )
        if h_mirror_act is None:
            h_mirror_act = torch.zeros(
                batch_size, self.hidden_size, device=inputs.device
            )

        for t in range(seq_len):
            input_t = inputs[t]
            perc_output, (h_perc, c_perc) = self.perception(input_t, (h_perc, c_perc))
            gw_decision, (h_gw, c_gw) = self.global_workspace(perc_output, (h_gw, c_gw))
            mirror_output, (h_mirror_obs, c_mirror_obs, h_mirror_act, c_mirror_act) = (
                self.mirror_neuron(
                    perc_output,
                    gw_decision,
                    h_mirror_obs,
                    c_mirror_obs,
                    h_mirror_act,
                    c_mirror_act,
                )
            )
            act_output, (h_act, c_act) = self.action_execution(
                mirror_output, (h_act, c_act)
            )
            final_output = self.output_layer(act_output)
            outputs.append(final_output)

        outputs = torch.stack(outputs, dim=0)
        # Package all hidden states into HiddenStates namedtuple
        hidden_states = HiddenStates(
            h_perc=h_perc,
            c_perc=c_perc,
            h_mirror_obs=h_mirror_obs,
            c_mirror_obs=c_mirror_obs,
            h_mirror_act=h_mirror_act,
            c_mirror_act=c_mirror_act,
            h_gw=h_gw,
            c_gw=c_gw,
            h_act=h_act,
            c_act=c_act,
        )
        return outputs, hidden_states

    """ def update_weights(self, loss):
        self.perception.update_weights(loss)
        self.mirror_neuron.obs_cell.update_weights(loss)
        self.mirror_neuron.act_cell.update_weights(loss)
        self.mirror_neuron.mirror_layer.weight -= (
            self.mirror_neuron.mirror_layer.weight.grad
            * self.mirror_neuron.mirror_layer.learning_rate
        )
        self.global_workspace.cell.update_weights(loss)
        self.global_workspace.output_layer.weight -= (
            self.global_workspace.output_layer.weight.grad
            * self.global_workspace.output_layer.learning_rate
        )
        self.action_execution.update_weights(loss)
        self.output_layer.weight -= (
            self.output_layer.weight.grad * self.output_layer.learning_rate
        ) """

    def reset_hidden_states(self):
        self.h_perc = None
        self.c_perc = None
        self.h_mirror_obs = None
        self.c_mirror_obs = None
        self.h_mirror_act = None
        self.c_mirror_act = None
        self.h_gw = None
        self.c_gw = None
        self.h_act = None
        self.c_act = None


# --------------------------
# Instantiate the Model
# --------------------------

input_size = 1
hidden_size = 32
output_size = 1

model = NeuroFlexNet(input_size, hidden_size, output_size).to("cuda")

# Added L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
criterion = nn.MSELoss()

# --------------------------
# Training Loop
# --------------------------

num_epochs = 20

print("Training the NeuroFlexNet model...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data_batch, target_batch) in enumerate(train_loader):
        data_batch = data_batch.transpose(0, 1).to("cuda")
        target_batch = target_batch.transpose(0, 1).to("cuda")

        outputs, hidden_states = model(data_batch)
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}:")
            print(f"Output: {outputs[0][0]}")
            print(f"Target: {target_batch[0][0]}")
            print(f"Outputs shape: {outputs.shape}")
            print(f"Targets shape: {target_batch.shape}")

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

    # Evaluation
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        for data_batch, target_batch in test_loader:
            data_batch = data_batch.transpose(0, 1).to("cuda")
            target_batch = target_batch.transpose(0, 1).to("cuda")
            outputs, hidden_states = model(data_batch)
            loss = criterion(outputs, target_batch)
            total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.6f}")

# --------------------------
# Visualization
# --------------------------

model.eval()
with torch.no_grad():
    all_outputs = []
    all_targets = []
    for data_batch, target_batch in test_loader:
        data_batch = data_batch.transpose(0, 1).to("cuda")
        target_batch = target_batch.transpose(0, 1).to("cuda")
        outputs, hidden_states = model(data_batch)
        all_outputs.append(outputs.detach().cpu().numpy())
        all_targets.append(target_batch.detach().cpu().numpy())
        break
    all_outputs = np.concatenate(all_outputs, axis=1).squeeze()
    all_targets = np.concatenate(all_targets, axis=1).squeeze()

plt.figure(figsize=(12, 6))
plt.plot(all_targets.flatten(), label="Actual")
plt.plot(all_outputs.flatten(), label="Predicted")
plt.title("Stock Prediction - NeuroFlexNet With Mirror Neurons")
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
