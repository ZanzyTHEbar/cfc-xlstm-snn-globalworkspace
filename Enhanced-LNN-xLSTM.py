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


class AdjointCfCNeuron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AdjointCfCNeuron, self).__init__()
        self.cfc_neuron = CfCNeuronODE(input_size, hidden_size)

    def forward(self, input_sequence, hidden_state):
        # Define the time points
        t = torch.linspace(0., 1., steps=input_sequence.size(0)).to(hidden_state.device)

        # Integrate the ODE
        z = odeint(self.cfc_neuron, hidden_state, t, method='dopri5')

        return z


class AdaptiveCfCNeuron(nn.Module):
    def __init__(self, input_size, hidden_size, min_dt=0.01, max_dt=1.0):
        super(AdaptiveCfCNeuron, self).__init__()
        self.A = nn.Parameter(torch.Tensor(hidden_size))
        self.w_tau = nn.Parameter(torch.Tensor(hidden_size))
        self.min_dt = min_dt
        self.max_dt = max_dt

        # Initialize parameters
        nn.init.uniform_(self.A, -0.1, 0.1)
        nn.init.uniform_(self.w_tau, -0.1, 0.1)

    def forward(self, input_t, hidden_state, prev_input):
        # Calculate the rate of change in the input
        delta_I = torch.abs(input_t - prev_input)
        # Adjust the time step based on the rate of change
        delta_t = torch.clamp(delta_I, self.min_dt, self.max_dt)

        # Compute the closed-form CfC update with adaptive time-step
        f = torch.sigmoid
        exponent = -(self.w_tau + f(input_t)) * delta_t
        x_combined = (
            (hidden_state - self.A) * torch.exp(exponent) * f(-input_t)
        ) + self.A
        return x_combined, delta_t

class NeuroFlexNetAdjoint(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuroFlexNetAdjoint, self).__init__()
        self.hidden_size = hidden_size
        self.adjoint_neuroflex_cell = AdjointCfCNeuron(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, h_t=None):
        seq_len, batch_size, _ = inputs.size()

        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        # Pass the entire input sequence to the adjoint neuron
        z = self.adjoint_neuroflex_cell(inputs, h_t)

        # Take the final hidden state
        h_final = z[-1]

        # Generate output
        output = self.output_layer(h_final).unsqueeze(0).repeat(seq_len, 1, 1)

        return output, h_final


class NeuroFlexCellAdaptive(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuroFlexCellAdaptive, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.adaptive_cfc = AdaptiveCfCNeuron(input_size, hidden_size)

        self.A = nn.Parameter(torch.Tensor(hidden_size))
        self.w_tau = nn.Parameter(torch.Tensor(hidden_size))

        nn.init.uniform_(self.A, -0.1, 0.1)
        nn.init.uniform_(self.w_tau, -0.1, 0.1)

        self.learning_rate = 0.0001
        self.epsilon = 1e-5

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_t, hx, prev_input):
        h_t, c_t = hx
        # LSTM Update
        x_LSTM_t, c_t = self.lstm_cell(input_t, (h_t, c_t))

        # Adaptive CfC Update
        x_combined, delta_t = self.adaptive_cfc(input_t, h_t, prev_input)

        # Combine CfC and LSTM updates
        x_combined = self.layer_norm(x_combined)

        h_t = x_combined
        return x_combined, (h_t, c_t), delta_t

class ComprehensiveNeuroFlexNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, flow_hidden_size):
        super(ComprehensiveNeuroFlexNet, self).__init__()
        self.hidden_size = hidden_size
        self.adjoint_neuroflex_cell = AdjointCfCNeuron(input_size, hidden_size)
        self.cnf = ContinuousNormalizingFlow(hidden_size, flow_hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, timestamps, h_t=None):
        seq_len, batch_size, _ = inputs.size()

        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        # Pass the entire input sequence to the adjoint neuron
        z = self.adjoint_neuroflex_cell(inputs, h_t)

        # Apply CNF and generate outputs
        outputs = []
        for t in range(seq_len):
            z_t = z[t]
            delta_t = timestamps[t] - (timestamps[t-1] if t > 0 else 0)
            z_t = self.cnf(z_t, delta_t)
            z_t = self.layer_norm(z_t)
            output_t = self.output_layer(z_t)
            outputs.append(output_t)

        outputs = torch.stack(outputs, dim=0)
        return outputs, z[-1]

class StableCfCNeuron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StableCfCNeuron, self).__init__()
        self.cfc_neuron = CfCNeuronODE(input_size, hidden_size)

    def forward(self, input_t, hidden_state):
        # Define time points for integration
        t = torch.tensor([0.0, 1.0]).to(hidden_state.device)  # Example: single step

        # Integrate using an adaptive solver like Runge-Kutta 4
        z = odeint(self.cfc_neuron, hidden_state, t, method='rk4')

        return z[-1]

class ContinuousMemoryxLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ContinuousMemoryxLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        # Separate linear layers for gates
        self.forget_gate = nn.Linear(hidden_size, hidden_size)
        self.input_gate = nn.Linear(hidden_size, hidden_size)
        self.candidate_memory = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_t, hidden_state, cell_state, delta_t):
        h_t, c_t = self.lstm_cell(input_t, (hidden_state, cell_state))

        # Compute gates
        f_t = torch.sigmoid(self.forget_gate(h_t))
        i_t = torch.sigmoid(self.input_gate(h_t)) 
        \tilde{C}_t = torch.tanh(self.candidate_memory(h_t))

        # Compute derivative of cell state
        dC_dt = f_t * c_t + i_t * \tilde{C}_t

        # Euler integration
        c_t_new = c_t + delta_t * dC_dt

        return h_t, c_t_new
    
class NeuroFlexCellContinuous(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuroFlexCellContinuous, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.adaptive_cfc = AdaptiveCfCNeuron(input_size, hidden_size)
        self.continuous_memory = ContinuousMemoryxLSTM(input_size, hidden_size)

        self.A = nn.Parameter(torch.Tensor(hidden_size))
        self.w_tau = nn.Parameter(torch.Tensor(hidden_size))

        nn.init.uniform_(self.A, -0.1, 0.1)
        nn.init.uniform_(self.w_tau, -0.1, 0.1)

        self.learning_rate = 0.0001
        self.epsilon = 1e-5

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_t, hx, prev_input):
        h_t, c_t = hx
        # LSTM Update
        x_LSTM_t, c_t = self.lstm_cell(input_t, (h_t, c_t))

        # Adaptive CfC Update
        x_combined, delta_t = self.adaptive_cfc(input_t, h_t, prev_input)

        # Continuous Memory Update
        h_t, c_t = self.continuous_memory(input_t, h_t, c_t, delta_t)

        # Combine CfC and LSTM updates
        x_combined = self.layer_norm(x_combined)

        h_t = x_combined
        return x_combined, (h_t, c_t), delta_t

class IrregularTimeCfCNeuron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(IrregularTimeCfCNeuron, self).__init__()
        self.A = nn.Parameter(torch.Tensor(hidden_size))
        self.w_tau = nn.Parameter(torch.Tensor(hidden_size))

        nn.init.uniform_(self.A, -0.1, 0.1)
        nn.init.uniform_(self.w_tau, -0.1, 0.1)

    def forward(self, input_t, hidden_state, delta_t):
        f = torch.sigmoid
        exponent = -(self.w_tau + f(input_t)) * delta_t
        x_combined = ((hidden_state - self.A) * torch.exp(exponent) * f(-input_t)) + self.A
        return x_combined

class NeuroFlexNetIrregular(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuroFlexNetIrregular, self).__init__()
        self.hidden_size = hidden_size
        self.neuroflex_cell = NeuroFlexCellContinuous(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, timestamps, h_t=None, c_t=None, prev_input=None):
        seq_len, batch_size, _ = inputs.size()

        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        if c_t is None:
            c_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        outputs = []
        prev_input = torch.zeros_like(inputs[0])
        prev_timestamp = torch.zeros(batch_size, 1, device=inputs.device)

        for t in range(seq_len):
            input_t = inputs[t]
            current_timestamp = timestamps[t]
            delta_t = current_timestamp - prev_timestamp
            x_combined, (h_t, c_t), delta_t = self.neuroflex_cell(input_t, (h_t, c_t), prev_input)
            output_t = self.output_layer(x_combined)
            outputs.append(output_t)
            prev_input = input_t
            prev_timestamp = current_timestamp

        outputs = torch.stack(outputs, dim=0)
        return outputs, (h_t, c_t)

class ContinuousNormalizingFlow(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ContinuousNormalizingFlow, self).__init__()
        self.flow_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, z, delta_t):
        # Compute the derivative
        dz_dt = self.flow_net(z)
        # Simple Euler integration step
        z_new = z + delta_t * dz_dt
        return z_new

class NeuroFlexNetCNF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, flow_hidden_size):
        super(NeuroFlexNetCNF, self).__init__()
        self.hidden_size = hidden_size
        self.neuroflex_cell = NeuroFlexCellContinuous(input_size, hidden_size)
        self.cnf = ContinuousNormalizingFlow(hidden_size, flow_hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, timestamps, h_t=None, c_t=None, prev_input=None):
        seq_len, batch_size, _ = inputs.size()

        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        if c_t is None:
            c_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        outputs = []
        prev_input = torch.zeros_like(inputs[0])
        prev_timestamp = torch.zeros(batch_size, 1, device=inputs.device)

        for t in range(seq_len):
            input_t = inputs[t]
            current_timestamp = timestamps[t]
            delta_t = current_timestamp - prev_timestamp
            x_combined, (h_t, c_t), delta_t = self.neuroflex_cell(input_t, (h_t, c_t), prev_input)
            # Apply CNF
            z = x_combined
            z = self.cnf(z, delta_t)
            output_t = self.output_layer(z)
            outputs.append(output_t)
            prev_input = input_t
            prev_timestamp = current_timestamp

        outputs = torch.stack(outputs, dim=0)
        return outputs, (h_t, c_t)

class CfCNeuronODE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CfCNeuronODE, self).__init__()
        self.A = nn.Parameter(torch.Tensor(hidden_size))
        self.w_tau = nn.Parameter(torch.Tensor(hidden_size))

        nn.init.uniform_(self.A, -0.1, 0.1)
        nn.init.uniform_(self.w_tau, -0.1, 0.1)

    def forward(self, t, hidden_state):
        input_t = torch.zeros_like(hidden_state)  # Modify as per your input handling
        f = torch.sigmoid
        exponent = -(self.w_tau + f(input_t)) * 1.0  # Assume dt=1.0 or integrate accordingly
        x_combined = ((hidden_state - self.A) * torch.exp(exponent) * f(-input_t)) + self.A
        return x_combined
    
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