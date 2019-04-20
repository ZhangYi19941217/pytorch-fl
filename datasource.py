import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return image, label

class KWSconstructor(Dataset):
    def __init__(self, root, transform=None):
        f = open(root, 'r')
        data = []
        for line in f:
            s = line.split('\n')
            info = s[0].split(' ')
            data.append( (info[0], int(info[1])) )
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        f, label = self.data[index]
        feature = np.loadtxt(f)
        feature = np.reshape(feature, (1, 50, 10))
        feature = feature.astype(np.float32)
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label
 
    def __len__(self):
        return len(self.data)

class KWS():
    IID = True
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100

    def __init__(self, rank, batchsize=BATCH_SIZE):
        random.seed(rank)
        torch.manual_seed(rank)
        self.train_data = KWSconstructor(root='./kws/index_train.txt', transform=None)
        self.test_data = KWSconstructor(root='./kws/index_test.txt', transform=None)

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=batchsize, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=60, shuffle=True)

    def get_train_data(self):
        return self.train_loader

    def get_test_data(self):
        return self.test_loader

class KWS_noniid():
    IID = False
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100

    def __init__(self, batchsize=BATCH_SIZE, total_part=5):
        self.train_data = KWSconstructor(root='./kws/index_train.txt', transform=None)
        self.test_data = KWSconstructor(root='./kws/index_test.txt', transform=None)

        idxs = np.arange(len(self.train_data))
        labels = np.array(self.train_data.data)[:,1].astype(np.int)
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[  :  , idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        each_part_size = len(self.train_data) // total_part

        partition = []
        for i in range(total_part):
            first = each_part_size * i
            last = each_part_size * (i+1)
            part = idxs[first:last]
            partition.append(part)
        
        self.train_loader = []
        for part in partition:
            loader = DataLoader(DatasetSplit(self.train_data, part), batch_size=batchsize, shuffle=True)
            self.train_loader.append(loader)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=60, shuffle=True)

    def get_train_data(self, part_id):
        return self.train_loader[part_id]

    def get_test_data(self):
        return self.test_loader
    
    
class Mnist():
    IID = True
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100

    def __init__(self, rank, batchsize=BATCH_SIZE):
        random.seed(rank)
        torch.manual_seed(rank)
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

    def __init__(self, batchsize=BATCH_SIZE, total_part=5):
        self.train_data = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())

        idxs = np.arange(len(self.train_data))
        print(type(self.train_data.train_labels))
        labels = self.train_data.train_labels.numpy()
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[  :  , idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        each_part_size = len(self.train_data) // total_part

        partition = []
        for i in range(total_part):
            first = each_part_size * i
            last = each_part_size * (i+1)
            part = idxs[first:last]
            partition.append(part)
        
        self.train_loader = []
        for part in partition:
            loader = DataLoader(DatasetSplit(self.train_data, part), batch_size=batchsize, shuffle=True)
            self.train_loader.append(loader)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=10000, shuffle=True)

    def get_train_data(self, part_id):
        return self.train_loader[part_id]

    def get_test_data(self):
        return self.test_loader



class Cifar10():
    IID = True
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100

    def __init__(self, rank, batchsize=BATCH_SIZE):
        random.seed(rank)
        torch.manual_seed(rank)
        self.train_data = datasets.CIFAR10(root='./cifar10/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.CIFAR10(root='./cifar10/', train=False, transform=transforms.ToTensor())

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=batchsize, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=100, shuffle=True)

    def get_train_data(self):
        return self.train_loader

    def get_test_data(self):
        return self.test_loader
    


class Cifar10_noniid(): 
    IID = False
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100

    def __init__(self, batchsize=BATCH_SIZE, total_part=5):
        self.train_data = datasets.CIFAR10(root='./cifar10/', train=True, transform=transforms.ToTensor(), download=True)
        self.train_data_mnist = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.CIFAR10(root='./cifar10/', train=False, transform=transforms.ToTensor())

        idxs = np.arange(len(self.train_data))
#        print(type(self.train_data.train_labels))
#        print(type(self.train_data_mnist.train_labels))
        labels = np.array(self.train_data.train_labels)
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[  :  , idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        each_part_size = len(self.train_data) // total_part

        partition = []
        for i in range(total_part):
            first = each_part_size * i
            last = each_part_size * (i+1)
            part = idxs[first:last]
            partition.append(part)
        
        self.train_loader = []
        for part in partition:
            loader = DataLoader(DatasetSplit(self.train_data, part), batch_size=batchsize, shuffle=True)
            self.train_loader.append(loader)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=10000, shuffle=True)

    def get_train_data(self, part_id):
        return self.train_loader[part_id]

    def get_test_data(self):
        return self.test_loader
