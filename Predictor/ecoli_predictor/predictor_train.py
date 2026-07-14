"""Trains and applies the E. coli sequence activity predictor."""
import os
import glob
import time
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from contextlib import nullcontext

from Predictor.ecoli_predictor.pytorchtools import EarlyStopping
from Predictor.ecoli_predictor.predictor_models import *


class WeightedMSELoss(nn.Module):


    def __init__(self, weight: torch.Tensor):
        super().__init__()

        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float32)
        self.register_buffer("weight", weight.float())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target) ** 2


        return loss.mean() * self.weight.view(-1)[0]


class PREDICT():
    def __init__(self, run_name='predictor_', dataset='e.coli', model_name='LSTMModel',
                 data_path=None,
                 train_data_path=None, predict_data_path=None, model_ckpt_path=None,
                 val_ratio=0.1, mode='train'):


        self.model_name = model_name
        self.patience = 20
        self.val_acc_list = []
        self.dataset = dataset
        self.mode = mode
        self.val_ratio = val_ratio


        self.train_data_path = train_data_path if train_data_path else data_path
        self.predict_data_path = predict_data_path
        self.model_ckpt_path = model_ckpt_path


        self.batch_size = 16
        self.hidden_size = 256
        self.conv_hidden = 128
        self.seq_len = 165
        self.output_size = 1
        self.lambda_l2 = 0.001
        self.dropout_rate = 0.2


        self.input_size = 4


        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        self.use_gpu = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')


        if self.mode == 'train':
            self.seq1, self.exp = self.data_load(self.train_data_path)

            self.seq = self.seq_onehot(self.seq1)


            self.input_size = self.seq.shape[-1]


        self.build_model()


        self.checkpoint_dir = './checkpoint/' + run_name + dataset + '/'
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def data_load(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as data:

            seq = []
            exp = []
            for item in data:
                item = item.strip().split(",")
                seq.append(item[0])
                exp.append(item[1])
        expression = np.zeros((len(exp), 1))
        for i in range(len(exp)):
            expression[i] = float(exp[i])
        return seq, expression

    def string_to_array(self, my_string):
        my_string = my_string.lower()
        my_string = re.sub('[^acgt]', 'z', my_string)
        my_array = np.array(list(my_string))
        return my_array

    def one_hot_encode(self, my_array):
        label_encoder = LabelEncoder()
        label_encoder.fit(np.array(['a', 'c', 'g', 't', 'z']))
        integer_encoded = label_encoder.transform(my_array)
        onehot_encoder = OneHotEncoder(sparse_output=False, dtype=int)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        onehot_encoded = np.delete(onehot_encoded, -1, 1)
        return onehot_encoded

    def seq_onehot(self, seq):


        onehot_seq = []
        valid_chars = re.compile('[^ACGTacgt]')
        for s in seq:

            s = re.sub(valid_chars, '', s.strip())
            if not s:
                s = 'A' * self.seq_len
            arr = torch.tensor(self.one_hot_encode(self.string_to_array(s)), dtype=torch.float32)


            if arr.shape[1] < self.input_size:
                pad_c = self.input_size - arr.shape[1]
                arr = pad(arr, (0, pad_c), value=0)
            elif arr.shape[1] > self.input_size:
                arr = arr[:, :self.input_size]

            onehot_seq.append(arr)


        max_length = max(m.shape[0] for m in onehot_seq)
        padded_list = [pad(m, (0, 0, 0, max_length - m.shape[0]), value=0) for m in onehot_seq]
        return torch.stack(padded_list, dim=0)

    def build_model(self):
        if self.model_name == 'OnlyCNNModel':
            self.model = OnlyCNNModel(self.input_size, self.hidden_size, self.output_size, self.dropout_rate, self.lambda_l2)
        elif self.model_name == 'GRUModel':
            self.model = GRUModel(self.input_size, self.hidden_size, self.output_size, self.dropout_rate, self.lambda_l2)
        elif self.model_name == 'LSTMModel':
            self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size, self.dropout_rate, self.lambda_l2)
        elif self.model_name == 'BiLSTMModel':
            self.model = BiLSTMModel(self.input_size, self.hidden_size, self.output_size, self.dropout_rate, self.lambda_l2)
        elif self.model_name == 'TransformerModel':
            self.model = TransformerModel(self.input_size, self.hidden_size, self.output_size, nhead=8, num_layers=2, dropout_rate=self.dropout_rate)
        elif self.model_name == 'LSTM_Transformer_Model':
            self.model = LSTM_Transformer_Model(self.input_size, self.hidden_size, self.output_size, nhead=8, num_layers=2, dropout_rate=self.dropout_rate)
        elif self.model_name == 'LSTM_Transformer_large':
            self.model = LSTM_Transformer_large(self.input_size, self.hidden_size, self.output_size, nhead=8, num_layers=2, dropout_rate=self.dropout_rate)
        elif self.model_name == 'Densenet':
            self.model = Densenet(input_nc=self.input_size, growth_rate=32, block_config=(2, 2, 4, 2),
                                  num_init_features=64, bn_size=4, drop_rate=self.dropout_rate)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=self.lambda_l2)
        self.criterion = nn.MSELoss()

    def train(self):

        seq = self.seq.numpy()
        expression = self.exp
        X_train, X_val, y_train, y_val = train_test_split(
            seq, expression, test_size=self.val_ratio, random_state=42
        )
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        train_data = TensorDataset(X_train, y_train)
        val_data = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        num_epochs = 100


        weighted_loss_function = WeightedMSELoss(weight=torch.tensor([10.0], dtype=torch.float32, device=self.device))

        early_stopping = EarlyStopping(
            patience=self.patience,
            verbose=True,
            path=os.path.join(self.checkpoint_dir, self.model_name + '.pth'),
            stop_order='max'
        )


        autocast_ctx = torch.cuda.amp.autocast if self.use_gpu else nullcontext

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_feature, batch_label in train_loader:
                batch_feature = batch_feature.to(self.device)
                batch_label = batch_label.to(self.device)

                self.optimizer.zero_grad()
                with autocast_ctx():
                    output = self.model(batch_feature)
                    loss = weighted_loss_function(output, batch_label)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(train_loader))


            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_preds, val_labels = [], []
                for val_feature, val_label in val_loader:
                    val_feature = val_feature.to(self.device)
                    val_label = val_label.to(self.device)

                    val_output = self.model(val_feature)
                    loss = weighted_loss_function(val_output, val_label)
                    val_loss += loss.item()

                    val_preds.extend(val_output.detach().cpu().numpy())
                    val_labels.extend(val_label.detach().cpu().numpy())

                avg_val_loss = val_loss / max(1, len(val_loader))
                val_rho = spearmanr(np.array(val_labels).flatten(), np.array(val_preds).flatten())[0]
                val_cor = pearsonr(np.array(val_labels).flatten(), np.array(val_preds).flatten())[0]

            print(f"Epoch:{epoch} train_loss:{avg_loss:.4f} val_loss:{avg_val_loss:.4f} spearman:{val_rho:.4f} pearson:{val_cor:.4f}")


            early_stopping(val_loss=val_cor, model=self.model)
            if early_stopping.early_stop:
                print('Early Stopping......')
                break

    def load_model(self):

        path = os.path.join(self.checkpoint_dir, self.model_name + '.pth')
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f'Loaded model from {path}')

    def predict_file(self):


        ckpt_state = torch.load(self.model_ckpt_path, map_location='cpu')
        if 'cnn1.copy.weight' in ckpt_state:
            C_ckpt = ckpt_state['cnn1.copy.weight'].shape[1]
        elif 'features.0.weight' in ckpt_state:
            C_ckpt = ckpt_state['features.0.weight'].shape[1]
        else:
            C_ckpt = self.input_size
        print(f"[INFO] Inferred input channels from checkpoint: {C_ckpt}")


        if C_ckpt != self.input_size:
            self.input_size = C_ckpt
            self.build_model()


        self.model.load_state_dict(ckpt_state, strict=True)
        self.model.eval()
        print(f"[INFO] Loaded model ({self.model_name}) from {self.model_ckpt_path} | input_channels={self.input_size}")


        with open(self.predict_data_path, "r", encoding="utf-8") as f:
            seqs = [line.strip() for line in f if len(line.strip()) > 0]
        print(f"[INFO] Loaded {len(seqs)} sequences from {self.predict_data_path}")


        x = self.seq_onehot(seqs).to(self.device)


        with torch.no_grad():
            preds = self.model(x).cpu().numpy()


        save_path = "./Predictor/ecoli_predictor/outputs/predictions.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            for s, p in zip(seqs, preds):
                f.write(s + "," + str(p[0]) + "\n")

        print(f"[INFO] Prediction results saved to: {save_path}")


if __name__ == '__main__':
    mode = 'predict'
    predictor = PREDICT(
        run_name='predictor_',
        dataset='e.coli',
        model_name='LSTMModel',
        train_data_path='./DPO/data/real_train.csv',

        predict_data_path='./Predictor/ecoli_predictor/data/sequences.txt',
        model_ckpt_path='./Predictor/ecoli_predictor/checkpoints/LSTMModel.pth',
        mode=mode
    )
    if mode == 'train':
        time_start = time.time()
        predictor.train()
        time_end = time.time()
        print("Total training time: {:.2f}s".format(time_end - time_start))
    elif mode == 'predict':
        predictor.predict_file()
