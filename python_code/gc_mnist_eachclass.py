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
import sys, os
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
        x = x.reshape(x.shape[0], num_classes)

        return x

# get training data subset
def getSubset(dataset, num, seed):

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

def getTestLoader(dataset, c=None):
    if c != None:
        indices = torch.arange(dataset.data.shape[0])[dataset.targets == c]
        subset = torch.utils.data.Subset(dataset, indices)
        return DataLoader(subset, batch_size=tc_nums[c], shuffle=True)
    else:
        return DataLoader(dataset, batch_size=dataset.data.shape[0], shuffle=True)


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

def calc_sigmas(num, dataset, value=None):
    sigmas = np.zeros(num_classes)
    if value != None:
        sigmas = np.ones(num_classes) * value
    else:
        x = dataset.data / 255.0
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

    tmp = np.zeros(len(targets))
    for i in range(len(targets)):
        tmp[i] = sigmas[targets[i]]
    sigmas_mat = np.tile(tmp, (batch_size, 1))

    tmp2 = np.divide(-pd ** 2, sigmas_mat ** 2)

    pd = torch.exp(tmp2)
    
    
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

def test(model, loaders):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        test_loss, acc = np.zeros(10), np.zeros(10)
        for idx, loader in enumerate(loaders):
            num_correct = 0
            num_samples = 0
            for x, y in loader:
                x = x.to(device=device)
                y = y.to(device=device)

                scores = model(x)
                test_loss[idx] += criterion(scores, y)

                _, predictions = scores.max(1)

                predictions = predictions.reshape(-1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
            
            acc[idx] = (num_correct / num_samples) * 100
        return test_loss, acc

def train(name_dataset, tc, seed, num, wd, lam, bw, mode=None):
    
    # if bw != None:
    #     sigmas = calc_sigmas(num, train_dataset, bw)
    # else:
    sigmas = calc_sigmas(num, train_dataset)

    if (tc != None):
        sigmas[tc] = bw

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
            print(f"    batch_train_loss: {loss:.4f}, original={loss:.4f}, J3={J3:.4f}, J5={J5:.4f}", flush=True)
            train_loss += loss

        train_loss /= len(trainloader)    # num: 50, 100, 200, 500, 1000
    # wd: [0 1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1:1e-1:5e-1]
    # lam: [0 1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1]
    # bw: np.logspace(-3, 3, 50)
        test_loss, acc = test(model, test_loaders)
        train_loss, test_loss, acc = train_loss.item(), [test_loss[c].item() for c in range(10)], [acc[c].item() for c in range(10)]
        if mode == None:
            results.append([epoch, train_loss, test_loss, acc])
        if mode == "one":
            results.append([epoch, train_loss, test_loss, acc])
        testloss_print = ["{:.3f}".format(test_loss[i]) for i in range(N)]
        acc_print = ["{:.3f}".format(acc[i]) for i in range(N)]
        print(f'(num={num},lam={lam},bw={bw})epoch={epoch}, trainloss={train_loss:.3f}, testloss={testloss_print}, testacc={acc_print}', flush=True)
        model.train()

    return model
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--seed', type=int, default=0, help='seed value')
    parser.add_argument('--tc', type=int, default=0, help='test class')
    parser.add_argument('--num', type=int, default=500, help='number of each class')
    parser.add_argument('--wd', type=float, default=0, help='weight decay parameter')
    parser.add_argument('--lam', type=float, default=0.0001, help='graph connect coefficient lambda')
    parser.add_argument('--bw', type=float, default=1e-5, help='bandwidth(sigma)')

    args = parser.parse_args()
    print(f'seed={args.seed}, testclass={args.tc}, num={args.num}, wd={args.wd}, lam={args.lam}, bw={args.bw}', flush=True)
    seed, tc, num, wd, lam, bw = args.seed, args.tc, args.num, args.wd, args.lam, args.bw
    bw_str = "{:.4f}".format(bw)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    # Hyperparameters
    in_channel = 1
    num_classes = 10
    learning_rate = 0.001
    batch_size = 100
    num_epochs = 100
    momentum = 0.9


    # load data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data/", train=True, transform=transform, download=True
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(
        root="./data/", train=False, transform=transforms.ToTensor(), download=True
    )

    # split each class
    N = torch.unique(train_dataset.targets).shape[0]
    l = torch.arange(train_dataset.data.shape[0])
    C = {}
    for i in range(0, N):
        tmp = l[train_dataset.targets == i]
        torch.manual_seed(seed)
        t = torch.randperm(tmp.size(0))
        C[i] = tmp[t]
    
    # get number of samples of each class in test dataset
    test_loaders = []
    tc_nums = []
    for c in range(10):
        tc_nums.append(test_dataset.targets[test_dataset.targets==c].shape[0])
        test_loaders.append(getTestLoader(test_dataset, c))
    
    # num: 50, 100, 200, 500, 1000
    # wd: [0 1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1:1e-1:5e-1]
    # lam: [0 1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1]
    # bw: np.logspace(-3, 3, 50)
    
    # open a log file
    cwd = os.getcwd()
    print(cwd)

    log_file = open(f"{cwd}/log_eachclass/seed{seed}_tc{tc}_num{num}_wd{wd}_lam{lam}_bw{bw_str}.log", 'w')
    sys.stdout = log_file
    
    # torch.cuda.empty_cache()
    written_results = [] # final epoch result
    
    
    filename = f"{cwd}/gc_mnist_result_eachclass/seed{seed}_tc{tc}_num{num}_wd{wd}_lam{lam}_bw{bw_str}.csv"
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='excel')
        results = [] # each epoch result
        train(name_dataset="MNIST", tc=tc, seed=seed, num=num, wd=wd, lam=lam, bw=bw, mode="one")
        written_results.append(results)
        writer.writerows(written_results)
    log_file.close()
        





