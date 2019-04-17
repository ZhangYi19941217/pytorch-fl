echo 'master(coordinator): '$master
echo 'worker_hosts: '$workers

world_size=0
for i in $workers
do
		world_size=$((world_size+1))
done

model='CNN'
dataset='Cifar10'
iid=0

num=0
for i in $workers
do
	echo "python /Users/chen/Dropbox/NIPS19/code/pytorch-fl/worker_process_3_new.py --master_address='tcp://'$master':22222' --rank=$num --world_size=$world_size --model=$model --dataset=$dataset --iid=$iid"
	nohup ssh $i "python /Users/chen/Dropbox/NIPS19/code/pytorch-fl/worker_process_pure_frozen.py --master_address='tcp://'$master':22222' --rank=$num --world_size=$world_size --model=$model --dataset=$dataset --iid=$iid" > /Users/chen/Dropbox/NIPS19/code/Log/worker_$num.log 2>&1 &
	num=$((num+1))
done

