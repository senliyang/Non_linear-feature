import numpy as np
from numpy import matlib as nm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim, sparsity_target=0.1, sparsity_weight=1e-3):
        super(SparseAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 350),
            nn.ReLU(),
            nn.Linear(350, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.Sigmoid()  # 使用 Sigmoid 激活函数以确保输出在 [0, 1] 范围内
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 350),
            nn.ReLU(),
            nn.Linear(350, 400),
            nn.ReLU(),
            nn.Linear(400, input_size),
            nn.Sigmoid()
        )

        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def sparse_loss(self, x):
        # 计算KL散度
        sparsity = F.mse_loss(x, self.sparsity_target * torch.ones_like(x))
        return self.sparsity_weight * sparsity

def train_autoencoder(model, train_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for data in train_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            sparsity_loss = model.sparse_loss(model.encoder[2].weight)
            total_loss = loss + sparsity_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return model


def drug_auto_encoder(y_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 436
    encoding_dim = 128
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    dataset = TensorDataset(y_train_tensor)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = SparseAutoencoder(input_size=input_size, encoding_dim=encoding_dim).to(device)
    trained_model = train_autoencoder(model, train_loader)

    # 获取编码向量
    # drug_encoded_vector = trained_model.encoder(y_train_tensor).cpu().detach().numpy()
    drug_encoded_vector = trained_model.encoder(y_train_tensor).cpu()

    return drug_encoded_vector





def Sparse_Auto_encoder(d_sim):
    dtrain, label = data_process(d_sim)
    d_features = drug_auto_encoder(dtrain)
    return d_features


def data_process(d_sim):
    A = pd.read_csv("association.csv", index_col=0).to_numpy()
    R_A = np.repeat(A, repeats=218, axis=0)  # 271*218,218
    sd = nm.repmat(d_sim, 271, 1)  # 218*271,218
    train1 = np.concatenate((R_A, sd), axis=1)
    label = A.reshape((59078, 1))
    return train1, label
