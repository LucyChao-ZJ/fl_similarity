import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from torch.utils.data import DataLoader, Subset
import logging

device = torch.device("mps")

logging.basicConfig(filename="./output.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 100)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Client:
    def __init__(self, data_loader, model, c_id):
        self.his_model = [model]
        self.id = c_id
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.loss_function = nn.CrossEntropyLoss()

    def train_local_model(self, epochs):
        # 本地训练
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.his_model.append(model)
                running_loss += loss.item()

            print(f"Client {self.id} - Epoch {epoch + 1}, Loss: {running_loss / len(self.data_loader)}")


# 计算相似度
def calculate_similarity(model1, model2, metric="cosine"):
    params1 = get_model_params(model1)
    params2 = get_model_params(model2)

    if metric == "cosine":
        return cosine_similarity([params1], [params2])[0][0]
    elif metric == "euclidean":
        return euclidean(params1, params2)
    elif metric == "kl":
        return np.sum(entropy(params1, params2))
    else:
        raise ValueError("Unsupported metric")


def get_model_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().flatten())
    return np.concatenate(params)


def split_cifar100(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    client_data = []
    vis = {_: 0 for _ in range(60001)}
    for client_id in range(6):
        if client_id < 3:  # 客户端 A, B, C
            cnt = {_: 0 for _ in range(101)}
            indices = []
            for i in range(len(train_data)):
                if cnt.get(train_data.targets[i]) < 100 and not vis[i]:
                    vis[i] = 1
                    indices.append(i)
                    cnt[train_data.targets[i]] += 1
            indices = np.array(indices)
        else:  # 客户端 D，E,F
            indices = np.array([i for i in range(len(train_data)) if (train_data.targets[i] >= (client_id - 3) * 20) & (
                    train_data.targets[i] < (client_id - 2) * 20)])

        client_data.append(Subset(train_data, indices))

    client_loaders = [DataLoader(client_data[i], batch_size=batch_size, shuffle=True) for i in range(6)]
    print("Split Data Over......")
    return client_loaders


def train_and_evaluate(clients, epochs):
    n_clients = len(clients)
    similarities = {"cosine": [[None for _ in range(n_clients)] for _ in range(n_clients)],
                    "euclidean": [[None for _ in range(n_clients)] for _ in range(n_clients)],
                    "kl": [[None for _ in range(n_clients)] for _ in range(n_clients)]}

    for client in clients:
        client.train_local_model(epochs)

    for i, client1 in enumerate(clients):
        for j, client2 in enumerate(clients):
            if i <= j:
                cos_sim = calculate_similarity(client1.model, client2.model, "cosine")
                euclidean_sim = calculate_similarity(client1.model, client2.model, "euclidean")
                kl_sim = calculate_similarity(client1.model, client2.model, "kl")

                similarities["cosine"][i][j] = cos_sim
                similarities["euclidean"][i][j] = euclidean_sim
                similarities["kl"][i][j] = kl_sim

                similarities["cosine"][j][i] = cos_sim
                similarities["euclidean"][j][i] = euclidean_sim
                similarities["kl"][j][i] = kl_sim

                logging.info(f"Similarity between Client {client1.id} and Client {client2.id}:")
                logging.info(f"  Cosine Similarity: {cos_sim}")
                logging.info(f"  Euclidean Distance: {euclidean_sim}")
                logging.info(f"  KL Divergence: {kl_sim}")

    return similarities


model = CNN().to(device)
client_loaders = split_cifar100()

clients = [Client(loader, copy.deepcopy(model), i) for i, loader in enumerate(client_loaders)]

similarity_results = train_and_evaluate(clients, epochs=2)
print(similarity_results)


def plot_similarity_matrices(similarities, client_ids):
    for metric, matrix in similarities.items():
        if metric == "kl":
            continue
        matrix = np.array(matrix)

        plt.figure(figsize=(8, 6))
        plt.imshow(matrix)  # 选择颜色映射

        plt.colorbar(label=f"{metric.capitalize()} Similarity")

        plt.xticks(ticks=np.arange(len(client_ids)), labels=client_ids, rotation=45)
        plt.yticks(ticks=np.arange(len(client_ids)), labels=client_ids)

        plt.title(f"{metric.capitalize()} Matrix")
        plt.xlabel("Client ID")
        plt.ylabel("Client ID")

        # 显示数值在热力图上
        for i in range(len(client_ids)):
            for j in range(len(client_ids)):
                value = matrix[i][j]
                if value is not None:
                    plt.text(j, i, f"{value:.2f}", ha='center', va='center', color="white" if value < 0.5 else "black")

        plt.tight_layout()
        plt.show()


plot_similarity_matrices(similarity_results, [0, 1, 2, 3, 4, 5])
