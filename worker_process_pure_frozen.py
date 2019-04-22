import torch
import psutil
import numpy
try:
    import torch.distributed.deprecated as dist
except ImportError:
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
WEIGHT_DECAY=1e-2
MAX_ROUND = 10000
ROUND_NUMBER_FOR_SAVE = 50
ROUND_NUMBER_FOR_REDUCE = 5
IID = True
#DATA_SET = 'Mnist'
DATA_SET = 'Cifar10'
MODEL = 'CNN'
SAVE = True
CUDA = torch.cuda.is_available()

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
    if False and SAVE and os.path.exists('autoencoder'+str(rank)+'.t7'):
        logging('===> Try resume from checkpoint')
        checkpoint = torch.load('autoencoder'+str(rank)+'.t7')
        model.load_state_dict(checkpoint['state'])
        logging('model loaded')
    else:
        init_param(model, 0, group)
        logging('model created')
    return model

def all_reduce(model, world_size, group):
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.reduce_op.SUM, group=group)
        param.data /= world_size
    return model

def test(test_loader, model):
    accuracy = 0
    positive_test_number = 0
    total_test_number = 0
    with torch.no_grad():
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
    def __init__(self, model, epoch_per_round, world_size, group):
        self.frag_shape_list = []
        self.frag_index_list = []
        self.group = group
        self.world_size = world_size
        self.sync_frequency = 1
        self.evaluate_frequency = 50 
        self.last_model_copy = copy.deepcopy(model)
        self.model = model

        s_index = 0
        flattened_grad = torch.tensor([])
        for p in model.parameters():
            self.frag_shape_list.append(p.data.shape)
            frag = p.data.view(-1)
            e_index = s_index + len(frag)
            self.frag_index_list.append([s_index, e_index])
            s_index = e_index
            flattened_grad = torch.cat((flattened_grad, frag),0)

        self.model_size = len(flattened_grad)
        self.flattened_shape = flattened_grad.shape
        self.synchronization_mask = torch.ones(self.flattened_shape).byte()
        self.global_ac_grad = torch.zeros(self.flattened_shape)

        self.ema_alpha = 0.95 
        self.global_ac_grad_ema = torch.zeros(self.flattened_shape).float()
        self.global_abs_ac_grad_ema = torch.zeros(self.flattened_shape).float()
        self.global_ac_grad_ema_threshold = 0.1 # this shall be larger than 1-self.ema_alpha

        self.round_id = 0
        self.frozen_lengths = torch.zeros(self.flattened_shape).int()
        self.defrozen_round_ids = torch.zeros(self.flattened_shape).int()

    def update_frozen_lengths(self):
        # update the synchronization mask of each parameter based on global gradient stability

        # restore grad in flattened form
        flattened_grad = torch.tensor([])
        for param1, param2 in zip(self.last_model_copy.parameters(), self.model.parameters()):
            frag = (param1.data - param2.data).view(-1)
            flattened_grad = torch.cat((flattened_grad, frag),0)
        self.global_ac_grad = flattened_grad

        self.global_ac_grad_ema = self.global_ac_grad_ema * self.ema_alpha + self.global_ac_grad * (1.0 - self.ema_alpha)
        self.global_abs_ac_grad_ema = self.global_abs_ac_grad_ema * self.ema_alpha + torch.abs(self.global_ac_grad) * (1.0 - self.ema_alpha)
        self.frozen_lengths = torch.where(torch.abs(self.global_ac_grad_ema) / self.global_abs_ac_grad_ema < self.global_ac_grad_ema_threshold, self.frozen_lengths+1, self.frozen_lengths/2)
        self.defrozen_round_ids = self.round_id + self.frozen_lengths + 1


    def sync(self, iter_id):
        if iter_id % self.sync_frequency == 0:
            param = [p.data for p in self.model.parameters()]
            flattened_param = torch.tensor([])
            for p in param:
                frag = p.view(-1)
                flattened_param = torch.cat((flattened_param, frag),0)

            transmitted_param = torch.masked_select(flattened_param, self.synchronization_mask)
            logging('prepare transmitting in iter: '+str(iter_id))
            dist.all_reduce(transmitted_param, op=dist.reduce_op.SUM, group=self.group)
            transmitted_param /= self.world_size
            flattened_param[self.synchronization_mask] = transmitted_param

            for i, p in enumerate(self.model.parameters()):
                p.data = flattened_param[self.frag_index_list[i][0]:self.frag_index_list[i][1]].view(self.frag_shape_list[i])


            if iter_id % self.evaluate_frequency == 0:
                # current for simplicity the full model is transmitted, but it can be compressed later
                self.update_frozen_lengths()
                self.last_model_copy = copy.deepcopy(self.model)

                # update round id and defrozen corresponded parameters
                self.round_id += 1
                self.synchronization_mask = torch.where(self.defrozen_round_ids == self.round_id, torch.tensor(1).byte(), torch.tensor(0).byte())
                stable_ratio = 1 - float(sum(self.synchronization_mask.int())) / self.model_size

                # adjust the synchronization frequency when necessary
                print 'at iteration: ', iter_id, '; round id: ', self.round_id, '; stable ratio: ', stable_ratio
                return True
        return False

    def filter(self, iter_id):
        grad = [p.grad for p in self.model.parameters()]
        flattened_grad = torch.tensor([])
        for g in grad:
            frag = g.view(-1)
            flattened_grad = torch.cat((flattened_grad, frag),0)

        filtered_grad = torch.where(self.synchronization_mask > 0, flattened_grad, torch.zeros(self.flattened_shape))

        for i, p in enumerate(self.model.parameters()):
            p.grad.data = filtered_grad[self.frag_index_list[i][0]:self.frag_index_list[i][1]].view(self.frag_shape_list[i])

        return False
        
def run(world_size, rank, group, epoch_per_round, batch_size):

    train_loader = get_train_loader(world_size, rank, batch_size)
    test_loader = get_test_loader(rank)

    model = load_model(group, rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_func = torch.nn.CrossEntropyLoss()

    print('initial model parameters: ')
    print(list(model.parameters())[0][0][0])
    print('\n\n ----- start training -----')
    sys.stdout.flush()

    pas_manager = PAS_Manager(model, epoch_per_round, world_size, group)
    if rank == 0:
        numpy.save(str(pas_manager.round_id), numpy.asarray([i.detach().numpy() for i in list(model.parameters())]))
    iter_id = 0
    epoch_id = 0
    while epoch_id < MAX_ROUND:            
        if SAVE and epoch_id == 0 and not os.path.exists('autoencoder'+str(rank)+'.t7'):
            save_model(model, epoch_id, rank)
            logging('\t## Model Saved')
        
        for step, (b_x, b_y) in enumerate(train_loader):
            iter_id += 1
            if CUDA:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            optimizer.zero_grad()
            output = model(b_x)
            loss = loss_func(output, b_y)
            loss.backward()
            pas_manager.filter(iter_id)
            optimizer.step()
            is_synced = pas_manager.sync(iter_id) # whether there is true synchronization is hidden in pas_manager
            if is_synced:
                accuracy = test(test_loader, model)
#                accuracy = 0.1
                logging(' -- Finish round: '+str(pas_manager.round_id) + ' -- | test accuracy: '+str(accuracy))
                if rank == 0:
                    numpy.save(str(pas_manager.round_id), numpy.asarray([i.detach().numpy() for i in list(model.parameters())]))
            logging('iter '+str(iter_id)+' finish')
            logging(str(psutil.virtual_memory()[2]))
        epoch_id += 1 

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
    MODEL = 'ResNet18'
    DATASET = args.dataset
    IID = True
#    if args.iid == 1:
#        IID = True
#    else:
#        IID = False

    logging('Initialization:\n\t model: ' + MODEL + '; dataset: ' + DATASET + '; iid: ' + str(IID)
            + '\n\t master_address: ' + str(master_address) + '; world_size: '+str(world_size) 
            + ';\n\t rank: '+ str(rank) + '; epoch: '+str(epoch_per_round) + '; batch size: '+str(batch_size))
    init_processes(master_address, world_size, rank, epoch_per_round, batch_size, run)
