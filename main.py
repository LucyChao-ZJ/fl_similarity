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
EPOCHS = 50
logging.basicConfig(filename="./output.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def subtract_model(model1, model2):
    model_diff = CNN()
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    state_dict_diff = {key: state_dict1[key] - state_dict2[key] for key in state_dict1.keys()}
    model_diff.load_state_dict(state_dict_diff)
    return model_diff


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
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Client:
    def __init__(self, data_loader, model, c_id):
        self.id = c_id
        self.data_loader = data_loader
        self.model = model
        self.pre_model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.loss_function = nn.CrossEntropyLoss()
        self.cur_round = 0

    def train_local_model(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            self.cur_round += 1
            running_loss = 0.0
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                self.pre_model = copy.deepcopy(self.model)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f"Client {self.id} - Epoch {self.cur_round}, Loss: {running_loss / len(self.data_loader)}")

# 计算相似度
def cal_model_sim(model1, model2, metric="cos"):
    params1 = get_model_params(model1)
    params2 = get_model_params(model2)

    if metric == "cos":
        return cosine_similarity([params1], [params2])[0][0]
    elif metric == "euc":
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
        if client_id < 3:  # 客户端 A, B, C为数据分布相似
            cnt = {_: 0 for _ in range(101)}
            indices = []
            for i in range(len(train_data)):
                if cnt.get(train_data.targets[i]) < 100 and not vis[i]:
                    vis[i] = 1
                    indices.append(i)
                    cnt[train_data.targets[i]] += 1
            indices = np.array(indices)
        else:  # 客户端 D，E,F为数据分布不相似
            indices = np.array([i for i in range(len(train_data)) if (train_data.targets[i] >= (client_id - 3) * 20) & (
                    train_data.targets[i] < (client_id - 2) * 20)])

        client_data.append(Subset(train_data, indices))

    client_loaders = [DataLoader(client_data[i], batch_size=batch_size, shuffle=True) for i in range(6)]
    print("Split Data Over......")
    return client_loaders


# def aggregate(clients):
#     if len(clients) == 0:
#         return None
#     avg_model = copy.deepcopy(clients[0])
#
#     with torch.no_grad():
#         for param_name, param in avg_model.named_parameters():
#             param.data.zero_()
#             for model in clients:
#                 param.data += model.state_dict()[param_name] / len(clients)
#
#     return avg_model


def get_similarities(clients, target="m"):
    n_clients = len(clients)

    similarities = {"cos": [[None for _ in range(n_clients)] for _ in range(n_clients)],
                    "euc": [[None for _ in range(n_clients)] for _ in range(n_clients)],
                    "kl": [[None for _ in range(n_clients)] for _ in range(n_clients)]}

    for i, client1 in enumerate(clients):
        for j, client2 in enumerate(clients):
            if i <= j:
                cos_sim, euc_dis, kl_sim = 0, 0, 0
                if target == "m":
                    m1, m2 = client1.model, client2.model
                    cos_sim = cal_model_sim(m1, m2, "cos")
                    euc_dis = cal_model_sim(m1, m2, "euc")
                    kl_sim = cal_model_sim(m1, m2, "kl")
                elif target == "g":
                    g1, g2 = subtract_model(client1.model, client1.pre_model), subtract_model(client2.model,
                                                                                              client2.pre_model)
                    cos_sim = cal_model_sim(g1, g2, "cos")
                    euc_dis = cal_model_sim(g1, g2, "euc")
                    kl_sim = cal_model_sim(g1, g2, "kl")

                similarities["cos"][i][j] = cos_sim
                similarities["euc"][i][j] = euc_dis
                similarities["kl"][i][j] = kl_sim

                similarities["cos"][j][i] = cos_sim
                similarities["euc"][j][i] = euc_dis
                similarities["kl"][j][i] = kl_sim

                logging.info(f"Similarity between Client {client1.id} and Client {client2.id}:")
                logging.info(f"  Cosine Similarity: {cos_sim}")
                logging.info(f"  Euclidean Distance: {euc_dis}")
                logging.info(f"  KL Divergence: {kl_sim}")

    return similarities


def plot_similarity_matrices(similarities, client_ids):
    for metric, matrix in similarities.items():
        if metric == "kl":
            continue
        matrix = np.array(matrix)

        plt.figure(figsize=(8, 6))
        plt.imshow(matrix)

        plt.colorbar(label=f"{metric.capitalize()} Similarity")

        plt.xticks(ticks=np.arange(len(client_ids)), labels=client_ids, rotation=45)
        plt.yticks(ticks=np.arange(len(client_ids)), labels=client_ids)

        plt.title(f"{metric.capitalize()} Matrix")
        plt.xlabel("Client ID")
        plt.ylabel("Client ID")

        for i in range(len(client_ids)):
            for j in range(len(client_ids)):
                value = matrix[i][j]
                if value is not None:
                    plt.text(j, i, f"{value:.2f}", ha='center', va='center', color="white" if value < 0.5 else "black")

        plt.tight_layout()
        plt.show()


def plot_similarity_lines(like_data, dislike_data, title):
    plt.figure(figsize=(8, 6))
    plt.plot(like_data, marker='o', linestyle='-', color='b', label='like')
    plt.plot(dislike_data, marker='x', linestyle='-', color='r', label='dislike')
    plt.title(title, fontsize=16)
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


def train_and_show(clients, epochs, target="m"):
    like_cos_his = []
    like_euc_dis = []
    dislike_cos_his = []
    dislike_euc_dis = []

    center = 0
    like = 1
    dislike = 5

    for i in range(epochs):
        for client in clients:
            client.train_local_model(1)

        sim_res = get_similarities(clients, target)
        like_cos_his.append(sim_res['cos'][center][like])
        like_euc_dis.append(sim_res['euc'][center][like])
        dislike_cos_his.append(sim_res['cos'][center][dislike])
        dislike_euc_dis.append(sim_res['euc'][center][dislike])

        plot_similarity_matrices(sim_res, [0, 1, 2, 3, 4, 5])

    plot_similarity_lines(like_cos_his, dislike_cos_his,"cos_line")
    plot_similarity_lines(like_euc_dis, dislike_euc_dis,"euc_line")


model = CNN().to(device)
client_loaders = split_cifar100()

clients = [Client(loader, copy.deepcopy(model), i) for i, loader in enumerate(client_loaders)]

train_and_show(clients, epochs=EPOCHS,target="g")
