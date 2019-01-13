import torch
import torch.distributed.deprecated as dist
from datasource import Mnist, Mnist_noniid, Cifar10, Cifar10_noniid
from model import CNNMnist, CNNCifar, ResNet18
import copy
from torch.multiprocessing import Process
import argparse
import time
import sys
import os
sys.stdout.flush()

LR = 0.001
MAX_ROUND = 3000
ROUND_NUMBER_FOR_SAVE = 50
ROUND_NUMBER_FOR_REDUCE = 5
IID = False
DATA_SET = 'Mnist'
#DATA_SET = 'Cifar10'
MODEL = 'CNN'
#MODEL = 'ResNet18'

def get_local_data(size, rank, batchsize):
    if IID == True:
        if DATA_SET == 'Mnist':
            train_loader = Mnist(rank, batchsize).get_train_data()
        if DATA_SET == 'Cifar10':
            train_loader = Cifar10(rank, batchsize).get_train_data()
    else:
        if DATA_SET == 'Mnist':
            train_loader = Mnist_noniid(batchsize, size).get_train_data(rank)
        if DATA_SET == 'Cifar10':
            train_loader = Cifar10_noniid(batchsize, size).get_train_data(rank)
    return train_loader

def get_testset(rank):
    if IID == True:
        if DATA_SET == 'Mnist':
            test_loader = Mnist(rank).get_test_data()
        if DATA_SET == 'Cifar10':
            test_loader = Cifar10(rank).get_test_data()
    else:
        if DATA_SET == 'Mnist':
            test_loader = Mnist_noniid().get_test_data()
        if DATA_SET == 'Cifar10':
            test_loader = Cifar10_noniid().get_test_data()
    return test_loader

def init_param(model, src, group):
    for param in model.parameters():
        #print(param)
        sys.stdout.flush()
        dist.broadcast(param.data, src=src, group=group)
        #print('done')
        sys.stdout.flush()
    
def save_model(model, round, rank):
    print('===> Saving models...')
    state = {
        'state': model.state_dict(),
        'round': round,
        }
    torch.save(state, 'autoencoder' + str(rank) + '.t7')

def load_model(model, group, rank):
    print('===> Try resume from checkpoint')
    if os.path.exists('autoencoder' + str(rank) + '.t7'):
        checkpoint = torch.load('autoencoder' + str(rank) + '.t7')
        model.load_state_dict(checkpoint['state'])
        round = checkpoint['round']
        print('===> Load last checkpoint data')
    else:
        round = 0
        init_param(model, 0, group)
    return model, round


def all_reduce(model, size, group):
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.reduce_op.SUM, group=group)
        param.data /= size
    return model

def exchange(model, size, rank):
    old_model = copy.deepcopy(model)
    for param in old_model.parameters():
        dist.isend( param.data, dst=(rank+1)%size )
    for param in model.parameters():
        dist.recv( param.data, src=(rank-1)%size )
    return model


def run(size, rank, epoch, batchsize):
    #print('run')
    if MODEL == 'CNN' and DATA_SET == 'Mnist':
        model = CNNMnist()
    if MODEL == 'CNN' and DATA_SET == 'Cifar10':
        model = CNNCifar()
    if MODEL == 'ResNet18' and DATA_SET == 'Cifar10':
        model = ResNet18()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_func = torch.nn.CrossEntropyLoss()

    train_loader = get_local_data(size, rank, batchsize)
    if rank == 0 :
        test_loader = get_testset(rank)
        #fo = open("file_multi"+str(rank)+".txt", 'w')

    group_list = [i for i in range(size)]
    group = dist.new_group(group_list)

    model, round = load_model(model, group, rank)
    while round < MAX_ROUND:
        if rank == 0:
            accuracy = 0
            for step, (test_x, test_y) in enumerate(test_loader):
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                test_output = model(test_x)
                pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                accuracy += float((pred_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
            accuracy /= 100
            print('Round: ', round, 'Rank: ', rank, '| test accuracy: %.4f' % accuracy)
            #fo.write(str(round) + "    " + str(rank) + "    " + str(accuracy) + "\n")

        for epoch_cnt in range(epoch):
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                optimizer.zero_grad()
                output = model(b_x)
                loss = loss_func(output, b_y)
                loss.backward()   
                optimizer.step()

        model = all_reduce(model, size, group)
        #if (round+1) % ROUND_NUMBER_FOR_REDUCE == 0:
            #model = all_reduce(model, size, group)

        if (round+1) % ROUND_NUMBER_FOR_SAVE == 0:
            save_model(model, round+1, rank)
        round += 1
    #fo.close()

def init_processes(size, rank, epoch, batchsize, run):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:22222', world_size=size, rank=rank)
    run(size, rank, epoch, batchsize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', '-s', type=int, default=5)
    parser.add_argument('--epoch', '-e', type=int, default=1)
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
