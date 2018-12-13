import torch
import torch.distributed.deprecated as dist
import model
import time
import sys


def run():

    modell = model.CNN()
    #modell = model.AlexNet()

    size = dist.get_world_size()
    rank = dist.get_rank()

    group_list = []
    for i in range(size):
        group_list.append(i)
    group = dist.new_group(group_list)

    while(1):

        for param in modell.parameters():
            #for dst in range(1, size):
                #dist.send(param.data, dst=dst)
            dist.broadcast(param.data, src=0, group=group)

        for param in modell.parameters():
            tensor_temp = torch.zeros_like(param.data)
            dist.reduce(tensor_temp, dst=0, op=dist.reduce_op.SUM, group=group)
            param.data = tensor_temp / (size-1)
    

# run " python master.py -size NUM ", NUM is the value of size. 
if __name__ == "__main__":
    
    rank = 0

    if len(sys.argv) != 3 or sys.argv[1] != "-size" or sys.argv[2].isdigit() == False:
        print('Parameter list error!')
        sys.exit(0)

    size = int(sys.argv[2])

    dist.init_process_group(backend='tcp', init_method='tcp://127.0.0.1:5000', world_size=size, rank=rank)

    run()
