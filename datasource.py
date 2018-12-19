import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms 


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return image, label


class Mnist():
    IID = True
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100

    def __init__(self, batchsize):
        self.train_data = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=batchsize, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=10000, shuffle=True)

    def get_train_data(self):
        return self.train_loader

    def get_test_data(self):
        return self.test_loader



class Mnist_noniid():
    IID = False
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100

    def __init__(self, batchsize):
        self.train_data = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())

        idxs = np.arange(len(self.train_data))
        labels = self.train_data.train_labels.numpy()
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[  :  , idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        p1 = idxs[0:12000]
        p2 = idxs[12000:24000]
        p3 = idxs[24000:36000]
        p4 = idxs[36000:48000]
        p5 = idxs[48000:60000]

        self.train_loader1 = DataLoader(DatasetSplit(self.train_data, p1), batch_size=batchsize, shuffle=True)
        self.train_loader2 = DataLoader(DatasetSplit(self.train_data, p2), batch_size=batchsize, shuffle=True)
        self.train_loader3 = DataLoader(DatasetSplit(self.train_data, p3), batch_size=batchsize, shuffle=True)
        self.train_loader4 = DataLoader(DatasetSplit(self.train_data, p4), batch_size=batchsize, shuffle=True)
        self.train_loader5 = DataLoader(DatasetSplit(self.train_data, p5), batch_size=batchsize, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=10000, shuffle=True)


    def get_train_data1(self):  
        return self.train_loader1
    def get_train_data2(self):
        return self.train_loader2
    def get_train_data3(self):
        return self.train_loader3
    def get_train_data4(self):
        return self.train_loader4
    def get_train_data5(self):
        return self.train_loader5

    def get_test_data(self):
        return self.test_loader



class Cifar10():
    IID = True
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100

    def __init__(self, batchsize):

        self.train_data = datasets.CIFAR10(root='./cifar10/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.CIFAR10(root='./cifar10/', train=False, transform=transforms.ToTensor())

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=batchsize, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=10000, shuffle=True)

    def get_train_data(self):
        return self.train_loader

    def get_test_data(self):
        return self.test_loader




class Cifar10_noniid():
    IID = False
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100

    def __init__(self, batchsize):
        self.train_data = datasets.CIFAR10(root='./cifar10/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.CIFAR10(root='./cifar10/', train=False, transform=transforms.ToTensor())

        idxs = np.arange(len(self.train_data))
        labels = self.train_data.train_labels.numpy()
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[  :  , idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        p1 = idxs[0:10000]
        p2 = idxs[10000:20000]
        p3 = idxs[20000:30000]
        p4 = idxs[30000:40000]
        p5 = idxs[40000:50000]

        self.train_loader1 = DataLoader(DatasetSplit(self.train_data, p1), batch_size=batchsize, shuffle=True)
        self.train_loader2 = DataLoader(DatasetSplit(self.train_data, p2), batch_size=batchsize, shuffle=True)
        self.train_loader3 = DataLoader(DatasetSplit(self.train_data, p3), batch_size=batchsize, shuffle=True)
        self.train_loader4 = DataLoader(DatasetSplit(self.train_data, p4), batch_size=batchsize, shuffle=True)
        self.train_loader5 = DataLoader(DatasetSplit(self.train_data, p5), batch_size=batchsize, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=10000, shuffle=True)


    def get_train_data1(self):  
        return self.train_loader1
    def get_train_data2(self):
        return self.train_loader2
    def get_train_data3(self):
        return self.train_loader3
    def get_train_data4(self):
        return self.train_loader4
    def get_train_data5(self):
        return self.train_loader5

    def get_test_data(self):
        return self.test_loader
