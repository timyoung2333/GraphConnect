# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import copy
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import csv

# argparse
import argparse


# CNN module
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=20,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.out1 = torch.zeros((batch_size, 20, 24, 24))

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.out2 = torch.zeros((batch_size, 20, 12, 12))

        self.conv2 = nn.Conv2d(
            in_channels=20,
            out_channels=50,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.out3 = torch.zeros((batch_size, 50, 8, 8))

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.out4 = torch.zeros((batch_size, 50, 4, 4))

        self.conv3 = nn.Conv2d(
            in_channels=50,
            out_channels=500,
            kernel_size=(4, 4),
            stride = (1, 1),
            padding = (0, 0)
        )
        self.out5 = torch.zeros((batch_size, 500, 1, 1))
        self.out6 = torch.zeros((batch_size, 500, 1, 1))

        self.conv4 = nn.Conv2d(
            in_channels = 500,
            out_channels = num_classes,
            kernel_size = (1, 1),
            stride = (1, 1),
            padding = (0, 0)
        )

    def forward(self, x):
        x = self.conv1(x)
        self.out1 = x

        x = self.pool1(x)
        self.out2 = x

        x = self.conv2(x)
        self.out3 = x

        x = self.pool2(x)
        self.out4 = x

        # x = x.reshape(x.shape[0], -1)
        x = self.conv3(x)
        self.out5 = x

        x = F.relu(x)
        self.out6 = x

        x = self.conv4(x)
        # reshape
        x = x.reshape(batch_size, num_classes)

        return x
    

# Load Data
def loadData(dataset=None):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data/", train=True, transform=transform, download=True
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(
        root="./data/", train=False, transform=transforms.ToTensor(), download=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataset, train_loader, test_dataset, test_loader

# get training data subset
def getSubset(dataset, num, seed):
    N = torch.unique(dataset.targets).shape[0]
    l = torch.arange(dataset.data.shape[0])
    C = {}
    for i in range(0, N):
        tmp = l[dataset.targets == i]
        t = torch.randperm(tmp.size(0))
        C[i] = tmp[t]

    indices = np.array([])
    for i in range(N):
        if num <= C[i].size(0):
            # trainIdx = torch.cat([trainIdx, classIndices[i][0: num]], dim=0)
            indices = np.concatenate((indices, C[i][0:num]), axis=None)
        else:
            # trainIdx = torch.cat([trainIdx, classIndices[i]], dim=0)
            indices = np.concatenate((indices, C[i]), axis=None)
    
    indices = indices.astype(int)
    mean = torch.mean(dataset.data[indices] / 255)
    std = torch.std(dataset.data[indices] / 255)

    subset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(subset, batch_size=len(indices), shuffle=True)
    data, targets = next(iter(loader))

    return DataLoader(subset, batch_size=batch_size, shuffle=True), data, targets, mean, std


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

# calculate pairwise distance
def pairwise_dist(n, x):
    x = x.cpu()
    pdist = nn.PairwiseDistance(p=2)
    x1 = tile(x, 0, n).cpu()
    x2 = torch.cat(n * [x]).cpu()
    res = pdist(x1, x2)
    res = res.reshape(n, n)
    return res

def calc_sigmas(num, train_dataset, value=None):
    sigmas = np.zeros(num_classes)
    if value != None:
        sigmas = np.ones(num_classes) * value
    else:
        x = train_dataset.data / 255.0
        x_vec = x.reshape(x.shape[0], -1)
        for c in range(num_classes):
            tmp_x = x_vec[C[c][0:num]]
            tmp_xdist = pairwise_dist(num, tmp_x)
            sigmas[c] = torch.sum(tmp_xdist) / (num * (num - 1))
    return sigmas

# calculate class matrix, if res[i,j] = 1, it means there exists an edge
def class_matrix(batch_size, targets):
    targets = targets.cpu()
    t1 = tile(targets, 0, batch_size)
    t2 = torch.cat(batch_size * [targets])
    res = (t1 == t2)
    res = res.reshape(batch_size, batch_size)
    return res

# calculate weights, here we use the same sigmas (bandwidth) for different class
def calc_weights(data, targets, sigmas):
    data = data.cpu().reshape(data.shape[0], -1)
    
    pd = pairwise_dist(batch_size, data)
    pd = pd.reshape(batch_size, batch_size)
    pd = torch.exp(-pd ** 2 / sigmas[0] ** 2)

    targets_matrix = class_matrix(batch_size, targets)
    edges_num = (torch.sum(targets_matrix) - batch_size) / 2
    W = pd * targets_matrix
    
    D = torch.zeros((batch_size, batch_size))
    tmp = torch.sum(W, dim=0)
    for i in range(batch_size):
        D[i, i] = tmp[i]
    L = D - W

    return W, L, edges_num

# data: original data (after norm), x: feature (layer output)
def calc_loss(W, e_num, x, targets, lam):
    x = x.to(device=device)
    x_vec = x.reshape(x.shape[0], -1)
    f_dist = pairwise_dist(batch_size, x_vec)

    W_vec = W.reshape(-1)
    f_dist_vec = f_dist.reshape(-1) ** 2
    J = lam * torch.dot(W_vec, f_dist_vec) / e_num

    return J

def test(model, loader, epoch, writer):
    model.eval()

    num_correct = 0
    num_samples = 0

    test_loss = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            test_loss += criterion(scores, y)

            _, predictions = scores.max(1)

            predictions = predictions.reshape(-1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        test_loss /= 100
        accuracy = num_correct / num_samples
        return test_loss, accuracy

def train(name_dataset, num, wd, lam, bw, seed, t, mode=None):
    train_dataset, train_loader, test_dataset, test_loader = loadData()

    if mode != None and bw != None:
        sigmas = calc_sigmas(num, train_dataset, bw)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=wd)

    model.train()
    
    for epoch in range(num_epochs):

        trainloader, init_data, targets, mean, std = getSubset(train_dataset, num, seed)

        train_loss = 0
        
        for idx, (data, targets) in enumerate(trainloader):
            data = data - mean
            # data = data / std
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            scores = scores.reshape(batch_size, num_classes)

            if mode != None:
                W, L, edges_num = calc_weights(data, targets, sigmas)

            ori_loss = criterion(scores, targets)
            
            if mode == "one":
                # layer 3 loss
                J3 = calc_loss(W, edges_num, model.out3, targets, lam)

                # layer 5 loss
                J5 = calc_loss(W, edges_num, model.out5, targets, lam)

                loss = ori_loss + J3 + J5


            elif mode == "all":
                # layer 6 loss
                J6 = calc_loss(W, edges_num, model.out6, targets, lam)

                loss = ori_loss + J6
            
            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            # for future print
            print(f"    batch_train_loss: {loss:.4f}, original={loss:.4f}, J3={J3:.4f}, J5={J5:.4f}")
            train_loss += loss


        train_loss /= len(trainloader)
        test_loss, acc = test(model, test_loader, epoch, writer)
        acc *= 100
        train_loss, test_loss, acc = train_loss.item(), test_loss.item(), acc.item()
        if mode == None:
            results.append([epoch, train_loss, test_loss, acc])
        if mode == "one":
            results.append([epoch, train_loss, test_loss, acc])
        print(f'(lam={lam},bw={bw},t={t})epoch={epoch}, trainloss={train_loss:.3f}, testloss={test_loss:.3f}, testacc={acc:.3f}')
        model.train()

    return model
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--num', type=int, default=50, help='number of each class')
    parser.add_argument('--wd', type=float, default=0, help='weight decay parameter')
    parser.add_argument('--lam', type=float, default=0, help='graph connect coefficient lambda')
    parser.add_argument('--bw', type=float, default=1e-5, help='bandwidth(sigma)')
    parser.add_argument('--T', type=int, default=1, help='number of running times')
    parser.add_argument('--seedtype', type=str, default='one', help='seed type, including 3 types: "one", "multiple", "random"')

    args = parser.parse_args()
    print(f'num={args.num}, wd={args.wd}, lam={args.lam}, bw={args.bw}, T={args.T}, seedtype={args.seedtype}')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Hyperparameters
    in_channel = 1
    num_classes = 10
    learning_rate = 0.001
    batch_size = 100
    num_epochs = 200
    momentum = 0.9

    num, wd, lam, bw, T, seedtype = args.num, args.wd, args.lam, args.bw, args.T, args.seedtype
    seeds = np.zeros(T)
    if seedtype == 'one': # means all the T times use the same seed that the user enters
        seed = input('enter one seed: ')
        seeds = int(seed) * np.ones(T)
    elif seedtype == 'multiple': # means different times use different seeds
        while True:
            seeds = [int(item) for item in input('enter the seeds: ').split()]
            if len(seeds) == T:
                break
    elif seedtype == 'random': # randomly generate T seeds
        seeds = np.random.randint(1, 100, size=T)
    
    # torch.cuda.empty_cache()
    written_results = [] # final epoch result
    filename = f"num{num}_lam{lam}_bw{bw}.csv"
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='excel')
        for t in range(T):
            results = [] # each epoch result
            train(name_dataset="MNIST", num=num, wd=wd, lam=lam, bw=bw, t=t, seed=seeds[t], mode="one")
            written_results.append(results[-1])
        writer.writerows(written_results)
        





