import torch
import numpy as np
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

#LR = 1
LR = 0.01
MAX_ROUND = 1000
ROUND_NUMBER_FOR_SAVE = 50
ROUND_NUMBER_FOR_REDUCE = 5
IID = True
#DATA_SET = 'Mnist'
DATA_SET = 'Cifar10'
MODEL = 'CNN'
#MODEL = 'ResNet18'
EXCHANGE = False
SAVE = True
CUDA = torch.cuda.is_available()
WINDOW_SIZE = 10
INITIAL_FREQ = 16 * 500  # batchsize设为100，因此每个round有500个batch
RATIO = 0.8  # 判断窗口的稳定状态为正或是为负


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

def all_reduce(model, world_size, group):
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.reduce_op.SUM, group=group)
        param.data /= world_size
    return model

'''
def all_reduce_layer_v2(model, index_to_allReduce, world_size, group):
    para_to_allReduce = [list(model.parameters())[i] for i in index_to_allReduce]
    dist.all_reduce(torch.Tensor(para_to_allReduce), op=dist.reduce_op.SUM, group=group)
    para_to_allReduce = [gradient / world_size for gradient in para_to_allReduce]
    i = 0
    for index in index_to_allReduce:
        list(model.parameters())[index] = para_to_allReduce[i]
        i += 1
    return model
'''

def all_reduce_layer(model, index_to_allReduce, world_size, group):
    for layer_id in index_to_allReduce:
        dist.all_reduce(list(model.parameters())[layer_id].data, op=dist.reduce_op.SUM, group=group)
        list(model.parameters())[layer_id].data /= world_size
    return model

def init_setup(layer_num):
    freq = []
    cur_freq = []
    window = []
    for layer_id in range(layer_num):
        freq.append(INITIAL_FREQ)
        cur_freq.append(INITIAL_FREQ)
        window_i = []
        window.append(window_i)
    return freq, cur_freq, window
    

def get_gradient(old_model, new_model):
    gradients = []
    for param1, param2 in zip(old_model.parameters(), new_model.parameters()):
        gradients.append((param2 - param1))
    return gradients


def choose_value(array):
    array = array.reshape(-1)
    return array[0]  # 每个layer选取第一个参数

def update_window(window, gradients):
    for layer_id in range(len(window)):
        KeyValue = choose_value(gradients[layer_id].detach().cpu().numpy())
        if len(window[layer_id]) < 10:
            window[layer_id].append(KeyValue)
        else:
            window[layer_id].pop(0)
            window[layer_id].append(KeyValue)
    return window

def update_freq(freq):
    new_freq = freq // 2
    return new_freq

def check_separation(window, world_size, group):
    if len(window) < WINDOW_SIZE:
        return False
    positive_num_cnt = 0
    for num in window:
        if num >= 0:
            positive_num_cnt += 1
    negative_num_cnt = WINDOW_SIZE - positive_num_cnt
    flag = torch.zeros(1)
    if positive_num_cnt >= WINDOW_SIZE * RATIO:
        flag = torch.ones(1)
    if negative_num_cnt >= WINDOW_SIZE * RATIO:
        flag = -1 * torch.ones(1)
    MAX_flag = copy.deepcopy(flag)
    MIN_flag = copy.deepcopy(flag)
    if CUDA:
        MAX_flag = MAX_flag.cuda()
        MIN_flag = MIN_flag.cuda()
    dist.all_reduce(MAX_flag, op=dist.reduce_op.MAX, group=group)
    dist.all_reduce(MIN_flag, op=dist.reduce_op.MIN, group=group)
    print(MAX_flag, MIN_flag)
    if MAX_flag.item() == 1 and MIN_flag.item() == -1:
        return True
    else:
        return False
    


'''
def aggregation(old_gradients, gradients, group):
    aggre = False
    arr1 = old_gradients.detach().numpy().reshape(1, -1)
    arr2 = gradients.detach().numpy().reshape(1, -1)
    print(arr1)
    print(arr2)
    l1 = np.linalg.norm(arr1[0])
    l2 = np.linalg.norm(arr2[0])
    cosin = np.dot(arr1[0], arr2[0]) / (l1 * l2)
    angle = np.arccos(cosin) * 360 / 2 / np.pi
    print(angle)
    flag = torch.ones(1)
    if  angle > 20:
        print("This client need to all_reduce")
        flag = torch.zeros(1)

    dist.all_reduce(flag, op=dist.reduce_op.PRODUCT, group=group)
    if flag.numpy()[0] == 0:
        print("The whole model have reduced")
        aggre = True
    return aggre
'''



#def exchange(model, world_size, rank):
    #old_model = copy.deepcopy(model)
    #for param in old_model.parameters():
        #dist.isend( param.data, dst=(rank+1)%world_size )
    #for param in model.parameters():
        #dist.recv( param.data, src=(rank-1)%world_size )
    #return model


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

def run(world_size, rank, group, epoch_per_round, batch_size):
    train_loader = get_train_loader(world_size, rank, batch_size)
    test_loader = get_test_loader(rank)
    model, round = load_model(group, rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    print('initial model parameters: ')
    print(list(model.parameters())[0][0][0])
    print('\n\n ----- start training -----')
    #sys.stdout.flush()

    layer_num = len(list(model.parameters()))
    freq, cur_freq, window = init_setup(layer_num)

    #while round < MAX_ROUND:                                                                                                                                                                             
        #initial_model = copy.deepcopy(model)
        #print('\n')
        #sys.stdout.flush()
        #print('--- start round '+ str(round) + ' ---')
        #if SAVE and round == 0 and not os.path.exists('autoencoder'+str(rank)+'.t7'):
            #save_model(model, round, rank)
            #logging('\t## Model Saved')
                
        #accuracy = test(test_loader, model)
        #print(' - Before round: ', round, 'Rank: ', rank, '| test accuracy: '+str(accuracy))
        
    batch_id = 0
    while( True ):
        for step, (b_x, b_y) in enumerate(train_loader):
            initial_model = copy.deepcopy(model)
            #print('--- start batch '+ str(batch_id) + ' ---')
            if CUDA:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            optimizer.zero_grad()
            output = model(b_x)
            loss = loss_func(output, b_y)
            loss.backward()
            optimizer.step()

            batch_id += 1
            gradients = get_gradient(initial_model, model)
            
            window = update_window(window, gradients)
            if min(freq) >= 50000/batch_size:
                if batch_id % (50000/batch_size) == 0:
                    print('$ batch_id:  ' + str(batch_id) + '\n')
                    print('$ FREQ:')
                    print(freq)
                    print ('\n')
                    print('$ current freq:')
                    print(cur_freq) 
                    print('\n')
                    print('$ current window:')
                    for i in window:
                        print(i)
                    print('\n')

            index_to_allReduce = []
            for layer_id in range(layer_num):
                layer_freq = freq[layer_id]
                layer_cur_freq = cur_freq[layer_id]
                layer_window = window[layer_id]
                if layer_cur_freq == 1:
                    index_to_allReduce.append(layer_id)
                    need_to_update_freq = check_separation(layer_window, world_size, group)
                    if need_to_update_freq == True:
                        new_freq = update_freq(layer_freq)
                        freq[layer_id] = new_freq
                        cur_freq[layer_id] = new_freq
                    else:
                        cur_freq[layer_id] = freq[layer_id]
                else:
                    layer_cur_freq -= 1
                    cur_freq[layer_id] = layer_cur_freq

            if len(index_to_allReduce) > 0:
                print('$ layer id for all_reduce:')
                print(index_to_allReduce)

                model = all_reduce_layer(model, index_to_allReduce, world_size, group)
                accuracy = test(test_loader, model)
                logging(' -- Finish batch: '+str(batch_id) + ' -- | test accuracy: '+str(accuracy))
            
        
            #KeyValue = choose_value(gradients[0].detach().cpu().numpy())
            #print(KeyValue)


            #if EXCHANGE:
                #model = exchange(model, world_size, rank)
                #if (round+1) % ROUND_NUMBER_FOR_REDUCE == 0:
                    #model = all_reduce(model, world_size, group)
            #else:
                #model = all_reduce(model, world_size, group)

            #accuracy = test(test_loader, model)
            #print(' - After round ', round, 'Rank: ', rank, '| test accuracy: '+str(accuracy))
            #print('$ model parameters:')
            #print(list(model.parameters())[0][0][0])
            #logging(' -- Finish batch: '+str(batch_id) + ' -- | test accuracy: '+str(accuracy))
        #round += 1
        

def init_processes(master_address, world_size, rank, epoch_per_round, batch_size, run):
    # change 'tcp' to 'nccl' if running on GPU worker
    master_address = "tcp://" + master_address + ":22222"
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
    parser.add_argument('--exchange', '-E', type=int, default=0)
    args = parser.parse_args()
    
    master_address = args.master_address
    world_size = args.world_size
    rank = args.rank
    epoch_per_round = args.epoch_per_round
    batch_size = args.batch_size
    #batch_size = 60000
    #batch_size = 128

#    global MODEL, DATASET, IID, EXCHANGE
    MODEL = args.model
    DATASET = args.dataset
    if args.iid == 1:
        IID = True
    else:
        IID = False
    if args.exchange == 1:
        EXCHANGE = True
    else:
        EXCHANGE = False

    logging('Initialization:\n\t model: ' + MODEL + '; dataset: ' + DATASET + '; iid: ' + str(IID)  +'; exchange: ' + str(EXCHANGE)
            + '\n\t master_address: ' + str(master_address) + '; world_size: '+str(world_size) 
            + ';\n\t rank: '+ str(rank) + '; epoch: '+str(epoch_per_round) + '; batch size: '+str(batch_size))
    init_processes(master_address, world_size, rank, epoch_per_round, batch_size, run)
