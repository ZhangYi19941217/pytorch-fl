import torch
import torch.distributed.deprecated as dist
from datasource import Mnist, Mnist_noniid, Cifar10, Cifar10_noniid
from model import CNNMnist, CNNCifar, ResNet18
import copy
from torch.multiprocessing import Process
import argparse
import time
from datetime import datetime
import sys
import os
sys.stdout.flush()

LR = 0.001
MAX_ROUND = 3000
MAX_ROUND = 1
ROUND_NUMBER_FOR_SAVE = 50
ROUND_NUMBER_FOR_REDUCE = 5
IID = False
#IID = True
DATA_SET = 'Mnist'
#DATA_SET = 'Cifar10'
MODEL = 'CNN'
#MODEL = 'ResNet18'
EXCHANGE = False
SAVE = True

def logging(string):
    print(str(datetime.now())+' '+str(string))
    sys.stdout.flush()

def get_local_data(world_size, rank, batch_size):
    logging('enter get local data')
    if IID == True:
        if DATA_SET == 'Mnist':
            train_loader = Mnist(rank, batch_size).get_train_data()
        if DATA_SET == 'Cifar10':
            train_loader = Cifar10(rank, batch_size).get_train_data()
    else:
        if DATA_SET == 'Mnist':
            train_loader = Mnist_noniid(batch_size, world_size).get_train_data(rank)
        if DATA_SET == 'Cifar10':
            train_loader = Cifar10_noniid(batch_size, world_size).get_train_data(rank)
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
        dist.broadcast(param.data, src=src, group=group)
    
def save_model(model, round, rank):
    logging('===> Saving models...')
    state = {
        'state': model.state_dict(),
        'round': round,
        }
    torch.save(state, 'autoencoder' + str(rank) + '.t7')

def load_model(group, rank):
    if MODEL == 'CNN' and DATA_SET == 'Mnist':
        model = CNNMnist()
    if MODEL == 'CNN' and DATA_SET == 'Cifar10':
        model = CNNCifar()
    if MODEL == 'ResNet18' and DATA_SET == 'Cifar10':
        model = ResNet18()
    if SAVE and os.path.exists('autoencoder'+str(rank)+'.t7'):
        logging('===> Try resume from checkpoint')
        checkpoint = torch.load('autoencoder'+str(rank)+'.t7')
        model.load_state_dict(checkpoint['state'])
        round = checkpoint['round']
        print('===> Load last checkpoint data')
    else:
        round = 0
        init_param(model, 0, group)
    return model, round


def all_reduce(model, world_size, group):
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.reduce_op.SUM, group=group)
        param.data /= world_size
    return model

def exchange(model, world_size, rank):
    old_model = copy.deepcopy(model)
    for param in old_model.parameters():
        dist.isend( param.data, dst=(rank+1)%world_size )
    for param in model.parameters():
        dist.recv( param.data, src=(rank-1)%world_size )
    return model

def test(test_loader, model):
    accuracy = 0
    positive_test_number = 0
    total_test_number = 0
    for step, (test_x, test_y) in enumerate(test_loader):
        test_output = model(test_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        positive_test_number += (pred_y == test_y.data.numpy()).astype(int).sum()
        total_test_number += float(test_y.size(0))
    accuracy = positive_test_number / total_test_number
    return accuracy


def run(world_size, rank, group, epoch_per_round, batch_size):
    train_loader = get_local_data(world_size, rank, batch_size)
    test_loader = get_testset(rank)

    logging('start load')
    model, round = load_model(group, rank)
    logging('finish load'+str(rank))
    initial_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_func = torch.nn.CrossEntropyLoss()
    logging('prepare enter'+str(round)+'; max:'+str(MAX_ROUND))

#    print(list(model.parameters()))
    while round < MAX_ROUND:
        logging(' Start round: '+ str(round))
        if SAVE and round == 0 and not os.path.exists('autoencoder'+str(rank)+'.t7'):
            save_model(model, round, rank)
            logging(' Model Saved')

        accuracy = test(test_loader, model)
        print('Before round: ', round, 'Rank: ', rank, '| test accuracy: '+str(accuracy))

        for epoch_cnt in range(epoch_per_round):
            logging(epoch_cnt)
            for step, (b_x, b_y) in enumerate(train_loader):
#                print(list(b_y))
#                print(len(list(b_y)))
                print step,
                optimizer.zero_grad()
                output = model(b_x)
                loss = loss_func(output, b_y)
                loss.backward()   
                optimizer.step()

        accuracy = test(test_loader, model)
        print('Before All_reduce ', round, 'Rank: ', rank, '| test accuracy: '+str(accuracy))

        gradients = []
        for param1, param2 in zip(initial_model.parameters(), model.parameters()):
            print(param1.shape)
            gradients.append(param2 - param1)
        print('local gradient:')
        print(gradients[0][0])
#        print(gradients)

        if EXCHANGE:
            model = exchange(model, world_size, rank)
            if (round+1) % ROUND_NUMBER_FOR_REDUCE == 0:
                model = all_reduce(model, world_size, group)
        else:
            model = all_reduce(model, world_size, group)

        gradients = []
        for param1, param2 in zip(initial_model.parameters(), model.parameters()):
            gradients.append(param2 - param1)
        print('glocal gradient:')
        print(gradients[0][0])
#        print(gradients)
        print()

        accuracy = test(test_loader, model)
        print('After All_reduce ', round, 'Rank: ', rank, '| test accuracy: '+str(accuracy))

        logging(' Finish round: '+str(round)+'\n')
        round += 1

def init_processes(master_address, world_size, rank, epoch_per_round, batch_size, run):
    # change 'tcp' to 'nccl' if running on GPU worker
    dist.init_process_group(backend='tcp', init_method=master_address, world_size=world_size, rank=rank)
    group = dist.new_group([i for i in range(world_size)])
    run(world_size, rank, group, epoch_per_round, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_address', '-m', type=str, default='127.0.0.1')
    parser.add_argument('--world_size', '-w', type=int, default=5)
    parser.add_argument('--rank', '-r', type=int, default=0)
    parser.add_argument('--epoch_per_round', '-e', type=int, default=1)
    parser.add_argument('--batch_size', '-b', type=int, default=100)
    args = parser.parse_args()
    
    master_address = args.master_address
    world_size = args.world_size
    rank = args.rank
    epoch_per_round = args.epoch_per_round
    batch_size = args.batch_size
    batch_size = 30000
    batch_size = 128
    logging('Initialization:\n\t master_address: ' + str(master_address) + '; world_size: '+str(world_size) + ';\n\t rank: '+ str(rank) + '; epoch: '+str(epoch_per_round) + '; batch size: '+str(batch_size))
    init_processes(master_address, world_size, rank, epoch_per_round, batch_size, run)
