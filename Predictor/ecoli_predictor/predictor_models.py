"""Defines neural network architectures for the E. coli activity predictor."""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class WeightedMSELoss(nn.Module):
    def __init__(self, weight):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight

    def forward(self, y_pred, y_true):
        loss = torch.mean(self.weight * (y_true - y_pred)**2)
        return loss

class NonLocalBlock1D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock1D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(

            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):


        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class SAModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, lambda_l2):
        super(SAModel, self).__init__()
        self.hidden_size = hidden_size
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, dilation=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lambda_l2 = lambda_l2
        self.non_local_block = NonLocalBlock1D(hidden_size)
        self.pooling = nn.MaxPool1d(kernel_size=3)
        self.bigru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.non_local_block(x)
        x = x.permute(0,2,1)
        output, _ = self.bigru(x)
        output = torch.cat((output[:, -1, :self.hidden_size], output[:, 0, self.hidden_size:]), dim=-1)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight
        return output

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, lambda_l2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, dilation=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lambda_l2 = lambda_l2
        self.non_local_block = NonLocalBlock1D(hidden_size)
        self.pooling = nn.MaxPool1d(kernel_size=3)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.non_local_block(x)
        x = x.permute(0,2,1)
        _, hidden = self.gru(x)
        hidden = hidden.squeeze(0)
        hidden = self.fc1(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        output = output * self.weight
        return output


class OnlyCNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5, lambda_l2=1e-5):
        super(OnlyCNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )


        self.fc1 = nn.Linear(3840, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)


        x = x.view(x.size(0), -1)

        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight
        return output


class onlyGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, lambda_l2):
        super(onlyGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, dilation=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lambda_l2 = lambda_l2
        self.pooling = nn.MaxPool1d(kernel_size=3)
        self.bigru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):
        output, _ = self.bigru(x)
        output = torch.cat((output[:, -1, :self.hidden_size], output[:, 0, self.hidden_size:]), dim=-1)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight
        return output

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5, lambda_l2=1e-5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=17, stride=1, padding=4),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=13, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=9, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )


        self.lambda_l2 = lambda_l2
        self.non_local_block = NonLocalBlock1D(hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)

        x = self.non_local_block(x)
        x = x.permute(0, 2, 1)
        output, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        output = hidden.view(hidden.size(0), -1)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight
        return output

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5, lambda_l2=1e-5):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )


        self.non_local_block = NonLocalBlock1D(hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.non_local_block(x)
        x = x.permute(0, 2, 1)
        output, (hidden, _) = self.bilstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.fc1(hidden)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight
        return output

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=8, num_layers=2, dropout_rate=0.5, lambda_l2=1e-5):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size


        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )


        encoder_layers = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout_rate)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)


        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight
        return output

class LSTM_Transformer_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=8, num_layers=2, dropout_rate=0.2):
        super(LSTM_Transformer_Model, self).__init__()
        self.hidden_size = hidden_size


        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )


        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)


        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)


        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)


        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)


        x = self.transformer_encoder(x)


        x = x.mean(dim=1)
        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight

        return output

class LSTM_Transformer_large(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=8, num_layers=2, dropout_rate=0.5):
        super(LSTM_Transformer_large, self).__init__()
        self.hidden_size = hidden_size


        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=17, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=13, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=9, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=1),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )


        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)


        encoder_layers = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout_rate)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)


        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F


class Densenet(nn.Module):
    def __init__(self, input_nc, growth_rate=32, block_config=(2, 2, 4, 2), num_init_features=64, bn_size=4,
                 drop_rate=0.2):
        super(Densenet, self).__init__()


        self.features = nn.Sequential(
            nn.Conv1d(input_nc, num_init_features, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )


        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                                           growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = self._make_transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2


        self.features.add_module('norm_final', nn.BatchNorm1d(num_features))


        self.classifier = nn.Linear(num_features, 1)

    def _make_dense_block(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        return nn.Sequential(*layers)

    def _make_transition(self, num_input_features, num_output_features):
        return nn.Sequential(
            nn.BatchNorm1d(num_input_features),
            nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1',
                        nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),

        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2',
                        nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
