echo 'master(coordinator): '$master
echo 'worker_hosts: '$workers

size=0
for i in $workers
do
		size=$((size+1))
done

num=0
for i in $workers
do
	nohup ssh $i "python /home/ubuntu/pytorch-fl/single_slave.py --address=$master --rank=$num --size=$size" >/home/ubuntu/worker_$num.log 2>&1 &
	num=$((num+1))
done
