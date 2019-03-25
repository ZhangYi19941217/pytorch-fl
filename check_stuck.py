import numpy
import copy
shape = numpy.load('0.npy').shape
a = numpy.load('0.npy')
print a[0].shape
print a[1].shape
length = len(numpy.load('0.npy').reshape(-1))
print shape
print length
last_gradient = numpy.load('1.npy') - numpy.load('0.npy')
last_swap = numpy.asarray([numpy.load('0.npy') - numpy.load('0.npy') for i in range(20)])
swap_num = numpy.load('0.npy') - numpy.load('0.npy') 
bifurcate_point = [5000]*length

for i in range(1,2000):
    current_gradient = numpy.load(str(i+1)+'.npy') - numpy.load(str(i)+'.npy')
    numpy.delete(last_swap, 0)
    numpy.append(last_swap, current_gradient * last_gradient)
    last_gradient = current_gradient
    print last_swap.shape
    print last_swap < 0
    swap_num = sum(last_swap < 0).reshape(-1)
    for k in range(length):
        if swap_num[k] > 4:
            bifurcate_point[k] = min(bifurcate_point[k],i)

print bifurcate_point.reshape(shape)
