import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms 

class Mnist():
    IID = False
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100

    def __init__(self):

        self.train_data = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())


        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=Mnist.BATCH_SIZE, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=Mnist.BATCH_SIZE, shuffle=True)

    def get_train_data(self):
        return self.train_loader

    def get_test_data(self):
        return self.test_data


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        #error liuying
        #only integers, slices (`:`), ellipsis (`...`), None and long or byte Variables are valid indices (got numpy.float64)
        #print("item",item)
        #print("self.idxs[item]",self.idxs[item])
        image, label = self.dataset[int(self.idxs[item])]
        return image, label


class Mnist_noniid():
    IID = True
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100
    NUM_USERS = 3

    def __init__(self):

        self.train_data = datasets.MNIST('./mnist/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        idxs = np.arange(len(self.train_data))
        labels = self.train_data.train_labels.numpy()

        idxs_labels = np.vstack((idxs, labels))

        idxs_labels = idxs_labels[  :  , idxs_labels[1, :].argsort()]

        idxs = idxs_labels[0, :]
        #idxs is the rank arrange according to class
        p1 = idxs[0:12000]
        p2 = idxs[12000:24000]
        p3 = idxs[24000:36000]
        p4 = idxs[36000:48000]
        p5 = idxs[48000:60000]


        self.part1 = DataLoader(DatasetSplit(self.train_data, p1), batch_size=Mnist.BATCH_SIZE, shuffle=True)
        self.part2 = DataLoader(DatasetSplit(self.train_data, p2), batch_size=Mnist.BATCH_SIZE, shuffle=True)
        self.part3 = DataLoader(DatasetSplit(self.train_data, p3), batch_size=Mnist.BATCH_SIZE, shuffle=True)
        self.part4 = DataLoader(DatasetSplit(self.train_data, p4), batch_size=Mnist.BATCH_SIZE, shuffle=True)
        self.part5 = DataLoader(DatasetSplit(self.train_data, p5), batch_size=Mnist.BATCH_SIZE, shuffle=True)



        self.test_data = datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
        idxs = np.arange(len(self.test_data))
        labels = self.test_data.test_labels.numpy()

        idxs_labels = np.vstack((idxs, labels))

        #idxs_labels = idxs_labels[  :  , idxs_labels[1, :].argsort()]

        idxs = idxs_labels[0, :]
        p1 = idxs[0:2000]
        p2 = idxs[2000:4000]
        p3 = idxs[4000:6000]
        p4 = idxs[6000:8000]
        p5 = idxs[8000:10000]

        self.testpart1 = DataLoader(DatasetSplit(self.test_data, p1), batch_size=2000, shuffle=True)
        self.testpart2 = DataLoader(DatasetSplit(self.test_data, p2), batch_size=2000, shuffle=True)
        self.testpart3 = DataLoader(DatasetSplit(self.test_data, p3), batch_size=2000, shuffle=True)
        self.testpart4 = DataLoader(DatasetSplit(self.test_data, p4), batch_size=2000, shuffle=True)
        self.testpart5 = DataLoader(DatasetSplit(self.test_data, p5), batch_size=2000, shuffle=True)

        #init
        #self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=Mnist.BATCH_SIZE, shuffle=True)
        #self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=Mnist.BATCH_SIZE, shuffle=True)

    def get_train_data1(self):
        return self.part1
    def get_train_data2(self):
        return self.part2
    def get_train_data3(self):
        return self.part3
    def get_train_data4(self):
        return self.part4
    def get_train_data5(self):
        return self.part5

    def get_test_data1(self):
        return self.testpart1
    def get_test_data2(self):
        return self.testpart2
    def get_test_data3(self):
        return self.testpart3
    def get_test_data4(self):
        return self.testpart4
    def get_test_data5(self):
        return self.testpart5


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    m = Mnist_noniid()
    #m.mnist_noniid(dataset_train, num)
    m.get_train_data1()
    print("good")