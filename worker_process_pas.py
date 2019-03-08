import torch
import numpy as np
#import torch.distributed.deprecated as dist
import torch.distributed as dist
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

#LR = 1
LR = 0.01
MAX_ROUND = 10000
ROUND_NUMBER_FOR_SAVE = 50
ROUND_NUMBER_FOR_REDUCE = 5
IID = True
#DATA_SET = 'Mnist'
DATA_SET = 'Cifar10'
MODEL = 'CNN'
#MODEL = 'ResNet18'
SAVE = True
CUDA = torch.cuda.is_available()
WINDOW_SIZE = 10
INITIAL_FREQ = 16 * 500  
INITIAL_FREQ = 128  
RATIO = 0.8  

def logging(string):
    print(str(datetime.now())+' '+str(string))
    sys.stdout.flush()

def get_train_loader(world_size, rank, batch_size):
    if IID == True:
        if DATASET == 'Mnist':
            train_loader = Mnist(rank, batch_size).get_train_data()
        if DATASET == 'Cifar10':
            train_loader = Cifar10(rank, batch_size).get_train_data()
    else:
        if DATASET == 'Mnist':
            train_loader = Mnist_noniid(batch_size, world_size).get_train_data(rank)
        if DATASET == 'Cifar10':
            train_loader = Cifar10_noniid(batch_size, world_size).get_train_data(rank)
    return train_loader

def get_test_loader(rank):
    if DATASET == 'Mnist':
        test_loader = Mnist(rank).get_test_data()
    if DATASET == 'Cifar10':
        test_loader = Cifar10(rank).get_test_data()
    return test_loader

def init_param(model, src, group):
    for param in model.parameters():
        dist.broadcast(param.data, src=src, group=group)
    
def save_model(model, round, rank):
#    logging('===> Saving models...')
    state = {
        'state': model.state_dict(),
        'round': round,
        }
    torch.save(state, 'autoencoder-' + DATASET + '-' + MODEL + '-' + str(rank) + '.t7')

def load_model(group, rank):
    if MODEL == 'CNN' and DATASET == 'Mnist':
        model = CNNMnist()
    if MODEL == 'CNN' and DATASET == 'Cifar10':
        model = CNNCifar()
    if MODEL == 'ResNet18' and DATASET == 'Cifar10':
        model = ResNet18()
    if CUDA:
        model.cuda()
    if SAVE and os.path.exists('autoencoder'+str(rank)+'.t7'):
        logging('===> Try resume from checkpoint')
        checkpoint = torch.load('autoencoder'+str(rank)+'.t7')
        model.load_state_dict(checkpoint['state'])
        round = checkpoint['round']
        logging('model loaded')
    else:
        round = 0
        init_param(model, 0, group)
        logging('model created')
    return model, round

def test(test_loader, model):
    accuracy = 0
    positive_test_number = 0
    total_test_number = 0
    for step, (test_x, test_y) in enumerate(test_loader):
        if CUDA:
            test_x = test_x.cuda()
            test_y = test_y.cuda()
        test_output = model(test_x)
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
        total_test_number += float(test_y.size(0))
    accuracy = positive_test_number / total_test_number
    return accuracy

class PAS_Manager:
    def __init__(self, model, sync_frequency, world_size, group):
        self.last_model = copy.deepcopy(model) # record the model checkpoint after the last synchronization
        print(list(self.last_model.parameters())[0][0][0])
        self.layer_num = len(list(model.parameters())) # get the number of layers in the model
        self.sync_frequency_list = [sync_frequency] * self.layer_num # record the synchronization frequency of each layer
        self.layer_acgrad_window = [[0.0] * WINDOW_SIZE for i in range(self.layer_num)] # record the recent accumulated gradient direction history for each layer
        self.next_sync_iter_ids = [i for i in self.sync_frequency_list] # record the iteration id that shall incur synchronization of each layer
        self.world_size = world_size
        self.group = group

    def update_layer_acgrad_window(self, layer_id, model):
        layer_gradient = list(model.parameters())[layer_id] - list(self.last_model.parameters())[layer_id]
        self.layer_acgrad_window[layer_id].pop(0)
        self.layer_acgrad_window[layer_id].append(layer_gradient.detach().cpu().numpy().reshape(-1)[0])
        print('- update window for layer: '+str(layer_id))
        print(self.layer_acgrad_window[layer_id])
        sys.stdout.flush()
#        print(type(layer_gradient))
#        self.layer_window[layer_id].append(layer_gradient.numpy().reshape(-1)[0])

    def check_layer_bifurcated(self, layer_id, model):
        self.update_layer_acgrad_window(layer_id, model)
        positive_acgrad_cnt = len([acgrad for acgrad in self.layer_acgrad_window[layer_id] if acgrad > 0])
        negative_acgrad_cnt = len([acgrad for acgrad in self.layer_acgrad_window[layer_id] if acgrad < 0])
        flag = torch.zeros(1)
        if positive_acgrad_cnt >= WINDOW_SIZE * RATIO:
            flag = torch.ones(1)
        if negative_acgrad_cnt >= WINDOW_SIZE * RATIO:
            flag = -1 * torch.ones(1)
        MAX_flag = copy.deepcopy(flag)
        MIN_flag = copy.deepcopy(flag)
        if CUDA:
            MAX_flag = MAX_flag.cuda()
            MIN_flag = MIN_flag.cuda()
        dist.all_reduce(MAX_flag, op=dist.reduce_op.MAX, group=self.group)
        dist.all_reduce(MIN_flag, op=dist.reduce_op.MIN, group=self.group)
        if MAX_flag.item() == 1 and MIN_flag.item() == -1:
            return True
        return False

    def layer_all_reduce(self, layer_id, model):
        dist.all_reduce(list(model.parameters())[layer_id].data, op=dist.reduce_op.SUM, group=self.group)
        list(model.parameters())[layer_id].data /= world_size
        list(self.last_model.parameters())[layer_id].data = copy.deepcopy(list(model.parameters())[layer_id].data)
    #    if layer_id == 0:
    #        print(list(self.last_model.parameters())[layer_id][0][0])
        # update the model checkpoint after the layer synchronization

    def sync(self, model, iter_id):
        for layer_id in range(self.layer_num):
            if iter_id == self.next_sync_iter_ids[layer_id]: # if this layer shall be synchronized at the current iteration
                # manage synchronization frequency
                if self.check_layer_bifurcated(layer_id, model):
                    if self.sync_frequency_list[layer_id] > 1:
                        self.sync_frequency_list[layer_id] /= 2
                    print('bifurcation detected! - layer ' + str(layer_id) + '; new frequency: ' + str(self.sync_frequency_list[layer_id]))
                self.next_sync_iter_ids[layer_id] = iter_id + self.sync_frequency_list[layer_id]  
                # conduct layer sync
                self.layer_all_reduce(layer_id, model)
                logging('layer ' + str(layer_id) + ' synchronized')
        # print(list(self.last_model.parameters())[0][0][0])
        
def run(world_size, rank, group, epoch_per_round, batch_size):

    train_loader = get_train_loader(world_size, rank, batch_size)
    test_loader = get_test_loader(rank)

    model, round = load_model(group, rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()

    print('initial model parameters: ')
    print(list(model.parameters())[0][0][0])
    print('\n\n ----- start training -----')
    sys.stdout.flush()

    pas_manager = PAS_Manager(model, INITIAL_FREQ, world_size, group)
    iter_id = 0
    epoch_id = 0
    while epoch_id < MAX_ROUND:            
        print('\n\n--- start epoch '+ str(epoch_id) + ' ---')
        if SAVE and epoch_id == 0 and not os.path.exists('autoencoder'+str(rank)+'.t7'):
            save_model(model, epoch_id, rank)
            logging('\t## Model Saved')
#        accuracy = test(test_loader, model)
#        print(' - Before epoch: ', epoch_id, 'Rank: ', rank, '| test accuracy: '+str(accuracy))
        
        for step, (b_x, b_y) in enumerate(train_loader):
#            print('--- start batch '+ str(batch_id) + ' ---')
            if CUDA:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            optimizer.zero_grad()
            output = model(b_x)
            loss = loss_func(output, b_y)
            loss.backward()
            optimizer.step()
            logging(' iteration ' + str(iter_id) + ' finished')

            iter_id += 1
            pas_manager.sync(model, iter_id)

            if iter_id % INITIAL_FREQ == 0:
                accuracy = test(test_loader, model)
                logging(' -- Finish epoch: '+str(iter_id/INITIAL_FREQ) + ' -- | test accuracy: '+str(accuracy))
                print(pas_manager.sync_frequency_list)

        epoch_id += 1

        #accuracy = test(test_loader, model)
        #print(' - After round ', round, 'Rank: ', rank, '| test accuracy: '+str(accuracy))
        #print('$ model parameters:')
        #print(list(model.parameters())[0][0][0])
        #logging(' -- Finish batch: '+str(batch_id) + ' -- | test accuracy: '+str(accuracy))
        #round += 1

def init_processes(master_address, world_size, rank, epoch_per_round, batch_size, run):
    # change 'tcp' to 'nccl' if running on GPU worker
    # master_address = "tcp://" + master_address + ":22222"
    if CUDA:
        dist.init_process_group(backend='nccl', init_method=master_address, world_size=world_size, rank=rank)
    else:
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
    parser.add_argument('--model', '-M', type=str, default='CNN')       
    parser.add_argument('--dataset', '-D', type=str, default='Cifar10')
    parser.add_argument('--iid', '-I', type=int, default=1)
    args = parser.parse_args()
    
    master_address = args.master_address
    world_size = args.world_size
    rank = args.rank
    epoch_per_round = args.epoch_per_round
    batch_size = args.batch_size
    # batch_size = 60000
    # batch_size = 128

    # global MODEL, DATASET, IID 
    MODEL = args.model
    DATASET = args.dataset
    if args.iid == 1:
        IID = True
    else:
        IID = False

    logging('Initialization:\n\t model: ' + MODEL + '; dataset: ' + DATASET + '; iid: ' + str(IID)
            + '\n\t master_address: ' + str(master_address) + '; world_size: '+str(world_size) 
            + ';\n\t rank: '+ str(rank) + '; epoch: '+str(epoch_per_round) + '; batch size: '+str(batch_size))
    init_processes(master_address, world_size, rank, epoch_per_round, batch_size, run)
