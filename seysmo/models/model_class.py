import random

import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMPredictor(nn.Module):
    def __init__(self, cfg):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = cfg.model.hidden_size
        self.num_layers = cfg.model.num_layers
        self.lstm = nn.LSTM(cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_layers, batch_first=True, dropout=cfg.model.dropout)
        self.fc = nn.Linear(cfg.model.hidden_size, cfg.model.output_size)

    def forward(self, x):
        # x : (batch_size, timestep, feature)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # lstm_out: (batch_size, sequence_length, hidden_size)
        out = self.fc(out[:, -1, :])
        return out


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(cfg.model.input_size, cfg.model.input_size // 8),
            nn.Tanh(),
            nn.Linear(cfg.model.input_size // 8, cfg.model.input_size // 16),
            nn.LeakyReLU(),
            nn.Linear(cfg.model.input_size // 16, cfg.model.input_size // 64),
            nn.LeakyReLU(),
            nn.Linear(cfg.model.input_size // 64, cfg.model.output_size)
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


class LeNet5(nn.Module):
    def __init__(self, cfg):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(3600, 3600 // 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600 // 2, 3600 // 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(3600 // 4, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class LeNet5_2(nn.Module):
    def __init__(self):
        super(LeNet5_2, self).__init__()
        self.num_classes = 10
        # Сверточные слои
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_input_size = 3600

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_input_size // 2),
            nn.Tanh(),
            nn.Linear(self.fc_input_size // 2, self.fc_input_size // 4),
            nn.ReLU(),
            nn.Linear(self.fc_input_size // 4, 1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_input_size // 2),
            nn.Tanh(),
            nn.Linear(self.fc_input_size // 2, self.fc_input_size // 4),
            nn.ReLU(),
            nn.Linear(self.fc_input_size // 4, 1)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_input_size // 2),
            nn.Tanh(),
            nn.Linear(self.fc_input_size // 2, self.fc_input_size // 4),
            nn.ReLU(),
            nn.Linear(self.fc_input_size // 4, 1)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_input_size // 2),
            nn.Tanh(),
            nn.Linear(self.fc_input_size // 2, self.fc_input_size // 4),
            nn.ReLU(),
            nn.Linear(self.fc_input_size // 4, 1)
        )

        self.fc5 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_input_size // 2),
            nn.Tanh(),
            nn.Linear(self.fc_input_size // 2, self.fc_input_size // 4),
            nn.ReLU(),
            nn.Linear(self.fc_input_size // 4, 1)
        )

        self.fc6 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_input_size // 2),
            nn.Tanh(),
            nn.Linear(self.fc_input_size // 2, self.fc_input_size // 4),
            nn.ReLU(),
            nn.Linear(self.fc_input_size // 4, 1)
        )

        self.fc7 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_input_size // 2),
            nn.Tanh(),
            nn.Linear(self.fc_input_size // 2, self.fc_input_size // 4),
            nn.ReLU(),
            nn.Linear(self.fc_input_size // 4, 1)
        )

        self.fc8 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_input_size // 2),
            nn.Tanh(),
            nn.Linear(self.fc_input_size // 2, self.fc_input_size // 4),
            nn.ReLU(),
            nn.Linear(self.fc_input_size // 4, 1)
        )

        self.fc9 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_input_size // 2),
            nn.Tanh(),
            nn.Linear(self.fc_input_size // 2, self.fc_input_size // 4),
            nn.ReLU(),
            nn.Linear(self.fc_input_size // 4, 1)
        )

        self.fc0 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_input_size // 2),
            nn.Tanh(),
            nn.Linear(self.fc_input_size // 2, self.fc_input_size // 4),
            nn.ReLU(),
            nn.Linear(self.fc_input_size // 4, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Разворачиваем тензор для полносвязного слоя

        # Параллельные полносвязные слои
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)
        out4 = self.fc4(x)
        out5 = self.fc5(x)
        out6 = self.fc6(x)
        out7 = self.fc7(x)
        out8 = self.fc8(x)
        out9 = self.fc9(x)
        out0 = self.fc0(x)

        # Объединение выходов
        out = torch.cat((out1, out2, out3, out4, out5, out6, out7, out8, out9, out0), dim=1)

        return out


class WaveNetModel(nn.Module):
    def __init__(self, input_size, sequence_length, num_filters=64, kernel_size=4, dilation_rates=[1, 2, 4, 8, 16],
                 fc_units=32, output_size=10):
        super(WaveNetModel, self).__init__()

        # Dilated Convolutional layers
        self.dilated_convs = nn.ModuleList([nn.Conv1d(in_channels=input_size if i == 0 else num_filters,
                                                      out_channels=num_filters,
                                                      kernel_size=kernel_size,
                                                      dilation=dilation_rate,
                                                      padding=(kernel_size - 1) * dilation_rate)
                                            for i, dilation_rate in enumerate(dilation_rates)])

        # Fully connected layers
        self.fc1 = nn.Linear(25984, fc_units)
        self.fc2 = nn.Linear(fc_units, output_size)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # Apply Dilated Conv1D
        for conv in self.dilated_convs:
            x = x.permute(0, 2, 1)  # (batch_size, input_size, sequence_length) for Conv1D
            x = self.relu(conv(x))
            x = x.permute(0, 2, 1)  # (batch_size, sequence_length, num_filters) after Conv1D

        # Flatten the output for fully connected layers
        x = x.flatten(start_dim=1)  # (batch_size, sequence_length * num_filters)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ConvFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(ConvFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(cfg.model.feature_extractor.input_size, cfg.model.feature_extractor.extracted_feature_size, kernel_size=5, stride=2, padding=1),  # conv1
            nn.ReLU(),
            nn.Conv2d(cfg.model.feature_extractor.extracted_feature_size, cfg.model.feature_extractor.extracted_feature_size, kernel_size=3, stride=1, padding=1),  # conv2
            nn.ReLU(),
            nn.Conv2d(cfg.model.feature_extractor.extracted_feature_size, cfg.model.feature_extractor.extracted_feature_size, kernel_size=3, stride=1, padding=1),  # conv3
            nn.ReLU(),
            nn.Conv2d(cfg.model.feature_extractor.extracted_feature_size, cfg.model.feature_extractor.extracted_feature_size, kernel_size=(13, 3)),  # conv4
            # nn.ReLU(),
            # nn.Conv1d(extracted_feature_size, extracted_feature_size, kernel_size=4, stride=2, padding=1),  # conv5
            # nn.ReLU(),
            # nn.Conv1d(extracted_feature_size, extracted_feature_size, kernel_size=4, stride=2, padding=1)  # conv6
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        return x


class ContextualTransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(ContextualTransformerEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cfg.model.encoder.embed_dim, nhead=cfg.model.encoder.num_heads),
            num_layers=cfg.model.encoder.num_layers
        )
        self.conv1d = nn.Conv1d(154, 154, kernel_size=5, stride=1, padding=2)

    def positional_encoding(self, x):
        pos_emb = self.conv1d(x)
        all_emb = pos_emb + x
        return all_emb

    def forward(self, x):
        x = self.positional_encoding(x)
        # (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        # (seq_len, batch, embed_dim) -> (batch, seq_len, embed_dim)
        x = x.permute(1, 0, 2)
        return x


class GumbelVectorQuantizer(nn.Module):
    def __init__(self, num_code_vector_groups, num_code_vectors_per_group, extracted_feature_size, code_vector_size,
                 gumbel_init_temperature):
        super().__init__()
        self.num_groups = num_code_vector_groups
        self.num_vectors = num_code_vectors_per_group

        self.linear = nn.Linear(
            extracted_feature_size,
            self.num_groups * self.num_vectors
        )
        self.code_book = nn.Parameter(
            torch.FloatTensor(1, self.num_groups, self.num_vectors, code_vector_size // self.num_groups)
        )

        self.temperature = gumbel_init_temperature

    @staticmethod
    def _compute_perplexity(probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs (torch.Tensor): with shape `(B, L, G, V)`

        Returns:
            torch.Tensor with shape `(G, V)`
        """

        num_values = probs.size(0)
        perplexity = probs.sum(0) / num_values

        return perplexity

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D1)`
            lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            code_vectors (torch.Tensor): with shape `(B, L, D2)`
            perplexity (torch.Tensor): with shape `(G, V)`
            )
        """

        batch_size, length, _ = hidden_states.shape

        hidden_states = self.linear(hidden_states)
        # `(B, L, G * V)` -> `(B * L * G, V)`
        hidden_states = hidden_states.view(batch_size * length * self.num_groups, -1)

        code_vector_probs = nn.functional.gumbel_softmax(
            hidden_states.float(), tau=self.temperature, hard=True
        ).type_as(hidden_states)
        code_vector_soft_dist = torch.softmax(
            hidden_states.view(batch_size, length, self.num_groups, -1).float(), dim=-1
        )
        perplexity = self._compute_perplexity(code_vector_soft_dist)

        code_vector_probs = code_vector_probs.view(batch_size * length, self.num_groups, -1).unsqueeze(-1)

        code_vectors = code_vector_probs * self.code_book
        # `(B * L, G, V, D)` -> `(B, L, G * D)`
        code_vectors = code_vectors.sum(-2).view(batch_size, length, -1)

        return code_vectors, perplexity


class Wav2Vec2Framework(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mask_time_prob = cfg.model.mask_time_prob
        self.num_mask_time_steps = cfg.model.num_mask_time_steps
        feat_ext = getattr(__import__(__name__).models.model_class, cfg.model.feature_extractor.name)
        self.feature_extractor: nn.Module = feat_ext(cfg)
        enc = getattr(__import__(__name__).models.model_class, cfg.model.encoder.name)
        self.encoder: nn.Module = enc(cfg)

        self.quantizer = GumbelVectorQuantizer(cfg.model.num_code_vector_groups, cfg.model.num_code_vectors_per_group,
                                               cfg.model.feature_extractor.extracted_feature_size, cfg.model.code_vector_size,
                                               cfg.model.gumbel_init_temperature)
        self.out_linear = nn.Linear(cfg.model.encoder_hidden_size, cfg.model.code_vector_size)

    def forward(self, input_values: torch.Tensor):
        """
        Args:
            input_values (torch.Tensor): with shape `(B, T, D1)`

        Returns:
            tuple(
            hidden_states (torch.Tensor): with shape `(B, L, D2)`
            quantized_features (torch.Tensor): with shape `(B, L, D2)`
            perplexity (torch.Tensor): with shape `(G, V)`
            time_mask (torch.BoolTensor): with shape `(B, L)`
            )
        """

        hidden_states = self.feature_extractor(input_values)
        masked_hidden_states, time_mask_indices = self.time_masking(hidden_states.clone())

        quantized_features, perplexity = self.quantizer(hidden_states)

        encoder_out = self.encoder(masked_hidden_states)

        encoder_out = self.out_linear(encoder_out)

        return encoder_out, quantized_features, perplexity, time_mask_indices

    def time_masking(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.BoolTensor]:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`
            lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            Masked hidden states (torch.Tensor with shape `(B, L, D)`),
            Time mask (torch.BoolTensor with `(B, L)`)
            )
        """

        batch_size, num_steps, hidden_size = hidden_states.size()

        # non mask: 0, mask: 1
        time_mask_indices = torch.zeros(
            batch_size, num_steps + self.num_mask_time_steps,
            device=hidden_states.device, dtype=torch.bool
        )

        for batch in range(batch_size):
            time_mask_idx_candidates = list(range(num_steps))
            k = int(self.mask_time_prob * num_steps)
            start_time_idx_array = torch.tensor(
                random.sample(time_mask_idx_candidates, k=k), device=hidden_states.device
            )
            for i in range(self.num_mask_time_steps):
                time_mask_indices[batch, start_time_idx_array + i] = 1

        time_mask_indices = time_mask_indices[:, :-self.num_mask_time_steps]
        num_masks = sum(time_mask_indices.flatten())

        # Maks hidden states
        mask_values = torch.zeros(num_masks, hidden_size, device=hidden_states.device)
        hidden_states[time_mask_indices] = mask_values

        return hidden_states, time_mask_indices


class Wav2vec2Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k = cfg.loss.contrastive_loss_temperature
        self.K = cfg.loss.num_contrastive_loss_negative_samples
        self.cos = nn.CosineSimilarity(dim=-1)
        self.G = cfg.loss.num_code_vector_groups
        self.V = cfg.loss.num_code_vectors_per_group
        self.a = cfg.loss.loss_alpha

    def forward(self, encoder_out, quantized_features, perplexity, time_mask_indices):
        target_encoder_out = encoder_out[time_mask_indices]
        labels = quantized_features[time_mask_indices]

        # Number of targets per batch
        num_targets_per_batch = [int(time_mask_indices[i].sum()) for i in range(time_mask_indices.size(0))]

        # Make negative samples
        negative_samples = self.negative_sampler(labels, num_targets_per_batch)
        negative_samples = torch.cat([labels.unsqueeze(1), negative_samples], dim=1)

        contrastive_loss = self.contrastive_loss(target_encoder_out, labels, negative_samples)
        diversity_loss = self.diversity_loss(perplexity)

        loss = contrastive_loss + self.a * diversity_loss

        return loss

    def contrastive_loss(
            self,
            targets: torch.Tensor,
            labels: torch.Tensor,
            negative_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            targets (torch.Tensor): with shape `(N, D)`
            labels (torch.Tensor): with shape `(N, D)`
            negative_samples (torch.Tensor): with shape `(N, K, D)`

        Returns:
            torch.Tensor with shape `(1)`
        """

        similarity = torch.exp(self.cos(targets, labels) / self.k)
        negative_similarity = torch.sum(torch.exp((self.cos(targets.unsqueeze(1), negative_samples) / self.k)),
                                        dim=1)

        contrastive_loss = -torch.log(similarity / negative_similarity).mean()

        return contrastive_loss

    def diversity_loss(self, perplexity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            perplexity (torch.Tensor): with shape `(G, V)`

        Returns:
            torch.Tensor with shape `(1)`
        """
        log_perplexity = torch.log(torch.clamp(perplexity, min=1e-9))
        entropy = torch.sum(perplexity * log_perplexity, dim=-1)
        diversity_loss = torch.sum(entropy) / (self.G * self.V)

        return diversity_loss

    def negative_sampler(self, label: torch.Tensor, num_targets_per_batch: list[int]):
        """
        Args:
            label (torch.Tensor): with shape `(N, D)`
            num_targets_per_batch (list[int]): Number of targets per batch.

        Returns:
            torch.Tensor with shape `(N, K, D)'

        """
        negative_samples = []
        start_idx = 0
        for num_targets in num_targets_per_batch:
            negative_sample_candidate_indices = torch.arange(
                num_targets, device=label.device
            ).unsqueeze(0).repeat(num_targets, 1)

            diagonal = torch.eye(num_targets)

            # Pull yourself from the list of candidates. `(N, N)` -> `(N, N-1)`
            negative_sample_candidate_indices = negative_sample_candidate_indices[diagonal == 0].view(num_targets,
                                                                                                      -1)
            negative_sample_candidate_indices += start_idx

            where_negative_sample = (
                torch.tensor([i for i in range(num_targets) for _ in range(self.K)]),
                torch.tensor(
                    [random.sample(list(range(num_targets - 1)), k=self.K) for _ in range(num_targets)]).flatten()
            )

            # `(K * N)`
            negative_sample_indices = negative_sample_candidate_indices[where_negative_sample]

            negative_samples.append(label[negative_sample_indices])
            start_idx += num_targets

        negative_samples = torch.cat(negative_samples).view(label.size(0), self.K, -1)

        return negative_samples