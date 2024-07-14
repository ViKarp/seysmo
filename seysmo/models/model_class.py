import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 32, batch_first=True)
        self.CNN = nn.Sequential(
            # nn.Linear(32, 64),
            # nn.ReLU(),
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Flatten()
        )
        self.MLP = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        out = self.CNN(x)
        out, _ = self.lstm(out)
        out = self.MLP(out[-1, :])
        return out


class HybridNN(nn.Module):

    def __init__(self, embedding_length, output_size, batch_size, n_hidden=256, n_layers=2, n_hidden_channels_1=8,
                 n_hidden_channels_2=16, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.embedding_length = embedding_length
        self.output_size = output_size
        self.batch_size = batch_size
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_hidden_channels_1 = n_hidden_channels_1
        self.n_hidden_channels_2 = n_hidden_channels_2
        self.lr = lr
        conv_output_length = ((embedding_length - 3 + 1) // 2 - 3 + 1) // 2

        # Полносвязные слои для объединения выходов
        self.MLP_input_size = n_hidden + n_hidden_channels_2 * conv_output_length

        self.lstm = nn.LSTM(embedding_length, n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        self.CNN = nn.Sequential(
            nn.Conv1d(1, self.n_hidden_channels_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(self.n_hidden_channels_1, n_hidden_channels_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )
        self.MLP = nn.Sequential(
            nn.Linear(self.MLP_input_size, self.MLP_input_size // 2),
            nn.ReLU(),
            nn.Linear(self.MLP_input_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )

        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = self._prepare_batch(batch)
        return self(x)

    def _common_step(self, batch, batch_idx, stage: str):
        x, labels = self._prepare_batch(batch)
        loss = self.criterion(self(x), labels)
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss

    def _prepare_batch(self, batch):
        # Ignore label
        x, labels = batch
        # Input shape should be (batch_size, seq_length, input_size)
        return x.view(x.size(0), -1, self.input_size), labels

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))
        # getting the last time step output
        lstm_out = lstm_output[:, -1, :]

        conv_out = x.permute(0, 2, 1)  # Перестановка для Conv1D: (batch_size, channels, sequence_length)
        conv_out = self.CNN(conv_out)
        conv_out = conv_out.view(conv_out.size(0), -1)

        combined = torch.cat((lstm_out, conv_out), dim=1)

        # Полносвязные слои
        out = self.MLP(combined)

        return out


class ParallelLSTMConv1DModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, conv_in_channels, conv_out_channels, conv_kernel_size,
                 pool_kernel_size, fc1_output_size, fc2_output_size, sequence_length):
        super(ParallelLSTMConv1DModel, self).__init__()

        # Параметры LSTM
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size

        # Параметры Conv1D
        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size

        # Параметры Fully Connected слоев
        self.fc1_output_size = fc1_output_size
        self.fc2_output_size = fc2_output_size

        # LSTM путь
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, batch_first=True)

        # Conv1D путь
        self.conv1 = nn.Conv1d(in_channels=conv_in_channels, out_channels=conv_out_channels,
                               kernel_size=conv_kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)

        # Вычисление размера выхода Conv1D пути после pooling
        conv_output_length = (sequence_length - conv_kernel_size + 1) // pool_kernel_size

        # Полносвязные слои для объединения выходов
        self.fc1_input_size = lstm_hidden_size + conv_out_channels * conv_output_length
        self.fc1 = nn.Linear(self.fc1_input_size, fc1_output_size)
        self.fc2 = nn.Linear(fc1_output_size, fc2_output_size)

    def forward(self, x):
        # LSTM путь
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, sequence_length, hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Использование выхода последнего временного шага

        # Conv1D путь
        conv_out = x.permute(0, 2, 1)  # Перестановка для Conv1D: (batch_size, channels, sequence_length)
        conv_out = torch.relu(self.conv1(conv_out))
        conv_out = self.pool(conv_out)
        conv_out = conv_out.reshape(conv_out.size(0), -1)  # Преобразование в плоский вид

        # Объединение выходов
        combined = torch.cat((lstm_out, conv_out), dim=1)

        # Полносвязные слои
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)

        return x
