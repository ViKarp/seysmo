import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x : (batch_size, timestep, feature)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # lstm_out: (batch_size, sequence_length, hidden_size)
        out = self.fc(out[:, -1, :])
        return out


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(input_size, input_size//8),
            nn.Tanh(),
            nn.Linear(input_size//8, input_size//16),
            nn.LeakyReLU(),
            nn.Linear(input_size//16, input_size//64),
            nn.LeakyReLU(),
            nn.Linear(input_size//64, output_size)
        )

    def forward(self, x):
        # x : (batch_size, timestep, feature)
        x = x.view(x.size(0), -1)
        out = self.MLP(x)
        return out


class ParallelLSTMConv2DModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, dropout, conv_in_channels, conv_out_channels,
                 conv_kernel_size, stride, padding,
                 pool_kernel_size, fc1_output_size, fc2_output_size, input_shape):
        super(ParallelLSTMConv2DModel, self).__init__()

        self.hidden_size = lstm_hidden_size
        self.num_layers = lstm_num_layers

        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, batch_first=True, dropout=dropout)

        self.conv = nn.Conv2d(conv_in_channels, conv_out_channels, conv_kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)

        cnn_output_height = ((input_shape[0] - conv_kernel_size + 2 * padding) // stride + 1) // pool_kernel_size
        cnn_output_width = ((input_shape[1] - conv_kernel_size + 2 * padding) // stride + 1) // pool_kernel_size
        cnn_output_size = conv_out_channels * cnn_output_height * cnn_output_width

        self.fc1_input_size = lstm_hidden_size + cnn_output_size
        self.fc1 = nn.Linear(self.fc1_input_size, fc1_output_size)
        self.fc2 = nn.Linear(fc1_output_size, fc2_output_size)

    def forward(self, x):
        # x : (batch_size, timestep, feature)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: (batch_size, sequence_length, hidden_size)
        lstm_out = lstm_out[:, -1, :]

        conv_out = torch.relu(self.conv(x.unsqueeze(1)))
        conv_out = self.pool(conv_out)
        conv_out = conv_out.view(conv_out.size(0), -1)  # conv_out: (batch_size, linear)

        # Combining outputs
        combined = torch.cat((lstm_out, conv_out), dim=1)

        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)

        return x


class CNN(nn.Module):
    def __init__(self, in_channels, conv_channels, kernel_size, pool_size, fc_layers, input_shape):
        super(CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(len(conv_channels)):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else conv_channels[i - 1],
                    out_channels=conv_channels[i],
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )

        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

        conv_output_height = input_shape[0] // (pool_size ** len(conv_channels))
        conv_output_width = input_shape[1] // (pool_size ** len(conv_channels))
        linear_input_size = conv_channels[-1] * conv_output_height * conv_output_width

        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_layers)):
            self.fc_layers.append(
                nn.Linear(
                    in_features=linear_input_size if i == 0 else fc_layers[i - 1],
                    out_features=fc_layers[i]
                )
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        # x : (batch_size, timestep, feature)
        for conv in self.conv_layers:
            x = self.pool(self.relu(conv(x)))

        x = x.view(x.size(0), -1)

        for i, fc in enumerate(self.fc_layers):
            x = self.relu(fc(x)) if i < len(self.fc_layers) - 1 else fc(x)

        return x


class CNNLSTMNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, hidden_size, num_layers, dropout,
                 linear_input_size, linear_output_size, input_shape):
        super(CNNLSTMNetwork, self).__init__()

        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.relu = nn.ReLU()

        # Calculate the size of the output after the CNN layer
        cnn_output_height = (input_shape[0] - kernel_size + 2 * padding) // stride + 1
        cnn_output_width = (input_shape[1] - kernel_size + 2 * padding) // stride + 1
        cnn_output_size = out_channels * cnn_output_height * cnn_output_width

        self.lstm = nn.LSTM(cnn_output_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.fc1 = nn.Linear(hidden_size, linear_input_size)
        self.fc2 = nn.Linear(linear_input_size, linear_output_size)

    def forward(self, x):
        # x : (batch_size, timestep, feature)
        x = x.unsqueeze(1)  # x : (batch_size, channel, timestep, feature)
        x = self.cnn(x)
        x = self.relu(x)  # x: (batch_size, out_channels, height, width)

        # Reshape the CNN output to fit the LSTM input
        batch_size, out_channels, height, width = x.size()
        x = x.view(batch_size, -1, out_channels * height * width)  # x: (batch_size, timestep (1), feature)

        # Apply LSTM
        x, _ = self.lstm(x)  # without resetting the internal neurons

        # Take the output from the last time step
        x = x[:, -1, :]  # x: (batch_size, hidden_size)

        # Apply linear layers
        x = self.fc1(x)  # x: (batch_size, linear_input_size)
        x = self.relu(x)
        x = self.fc2(x)  # x: (batch_size, linear_output_size)

        return x


class CNNLSTMNetwork2(nn.Module):
    def __init__(self, in_channels, conv_channels, kernel_size, pool_size, hidden_size, lstm_num_layers, dropout,
                 fc_layers, input_shape):
        super(CNNLSTMNetwork2, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(conv_channels)):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else conv_channels[i - 1],
                    out_channels=conv_channels[i],
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )

        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

        # Линейные слои
        conv_output_height = input_shape[0] // (pool_size ** len(conv_channels))
        conv_output_width = input_shape[1] // (pool_size ** len(conv_channels))
        cnn_output_size = conv_channels[-1] * conv_output_height * conv_output_width

        self.lstm = nn.LSTM(cnn_output_size, hidden_size, lstm_num_layers, batch_first=True, dropout=dropout)

        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_layers)):
            self.fc_layers.append(
                nn.Linear(
                    in_features=hidden_size if i == 0 else fc_layers[i - 1],
                    out_features=fc_layers[i]
                )
            )

        # Функция активации
        self.relu = nn.ReLU()

    def forward(self, x):
        # x : (batch_size, timestep, feature)
        x = x.unsqueeze(1)  # x : (batch_size, channel, timestep, feature)
        for conv in self.conv_layers:
            x = self.pool(self.relu(conv(x)))  # x: (batch_size, out_channels, height, width)

        batch_size, out_channels, height, width = x.size()
        x = x.view(batch_size, -1, out_channels * height * width)

        # Apply LSTM
        x, _ = self.lstm(x)  # without resetting the internal neurons

        # Take the output from the last time step
        x = x[:, -1, :]

        # Применение линейных слоёв с активацией
        for i, fc in enumerate(self.fc_layers):
            x = self.relu(fc(x)) if i < len(self.fc_layers) - 1 else fc(x)

        return x
