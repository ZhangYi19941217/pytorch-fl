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
exchange=0

num=0
for i in $workers
do
	echo "python /home/ubuntu/pytorch-fl/worker_process.py --master_address='tcp://'$master':22222' --rank=$num --world_size=$world_size --model=$model --dataset=$dataset --iid=$iid --exchange=$exchange"
	nohup ssh $i "python /home/ubuntu/pytorch-fl/worker_process.py --master_address='tcp://'$master':22222' --rank=$num --world_size=$world_size --model=$model --dataset=$dataset --iid=$iid --exchange=$exchange" >/home/ubuntu/worker_$num.log 2>&1 &
	num=$((num+1))
done

#if [ "$1" == "iid" ]; then
#	num=0
#	for i in $workers
#	do
#		nohup ssh $i "python /home/ubuntu/pytorch-fl/worker_process.py --master_address='tcp://'$master':22222' --rank=$num --world_size=$world_size" >/home/ubuntu/worker_$num.log 2>&1 &
#		num=$((num+1))
#	done
#else
#	num=0
#	for i in $workers
#	do
#		nohup ssh $i "python /home/ubuntu/pytorch-fl/worker_process"_"$1.py --master_address='tcp://'$master':22222' --rank=$num --world_size=$world_size" >/home/ubuntu/worker_$num.log 2>&1 &
#		num=$((num+1))
#	done
#fi
