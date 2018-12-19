import torch
import torch.distributed.deprecated as dist
from datasource import Mnist, Mnist_noniid, Cifar10, Cifar10_noniid
from model import CNNMnist, CNNCifar
import copy
from torch.multiprocessing import Process
import argparse
import time

LR = 0.001
IID = False

def get_global_model(model, group):
    for param in model.parameters():
        dist.broadcast(param.data, src=0, group=group)
    return model

def get_local_data(rank, batchsize):
    if IID == True:
        train_loader = Mnist(batchsize).get_train_data()
    else:
        if rank == 0:
            train_loader = Mnist_noniid(batchsize).get_train_data1()
        if rank == 1:
            train_loader = Mnist_noniid(batchsize).get_train_data2()
        if rank == 2:
            train_loader = Mnist_noniid(batchsize).get_train_data3()
        if rank == 3:
            train_loader = Mnist_noniid(batchsize).get_train_data4()
        if rank == 4:
            train_loader = Mnist_noniid(batchsize).get_train_data5()
    return train_loader

def get_testset(batchsize=100):
    if IID == True:
        test_loader = Mnist(batchsize).get_test_data()
    else:
        test_loader = Mnist_noniid(batchsize).get_test_data()
    for step, (b_x, b_y) in enumerate(test_loader):
        test_x = b_x
        test_y = b_y
    return test_x, test_y

def init_param(model, src, group):
    for param in model.parameters():
        dist.broadcast(param.data, src=src, group=group)


def run(size, rank, epoch, batchsize):
    MAX_EPOCH = epoch
    model = CNNMnist()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    train_loader = get_local_data(rank, batchsize)
    if rank == 0 :
        test_x, test_y = get_testset()
        fo = open("file"+str(rank)+".txt", 'w')

    group_list = [i for i in range(size)]
    group = dist.new_group(group_list)

    for epoch in range(MAX_EPOCH):
        if epoch == 0:
            init_param(model, 0, group)
        
        if rank == 0:
            test_output = model(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, ' Rank: ', rank, '| test accuracy: %.2f' % accuracy)
            fo.write(str(epoch) + "    " + str(rank) + "    " + str(accuracy) + "\n")

        for step, (b_x, b_y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(b_x)
            loss = loss_func(output, b_y)
            loss.backward()   
            optimizer.step()

        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.reduce_op.SUM, group=group)
            param.data /= size
    
    fo.close()

def init_processes(size, rank, epoch, batchsize, run):
    dist.init_process_group(backend='tcp', init_method='tcp://127.0.0.1:5000', world_size=size, rank=rank)
    run(size, rank, epoch, batchsize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', '-s', type=int, default=5)
    parser.add_argument('--epoch', '-e', type=int, default=500)
    parser.add_argument('--batchsize', '-b', type=int, default=100)
    args = parser.parse_args()
    
    size = args.size
    epoch = args.epoch
    batchsize = args.batchsize

    processes = []
    for rank in range(0, size):
        p = Process(target=init_processes, args=(size, rank, epoch, batchsize, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

