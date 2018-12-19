echo 'master(coordinator): '$master
echo 'worker_hosts: '$workers
num=0
for i in $workers
do
	nohup ssh $i "python /home/ubuntu/pytorch-fl/single_slave.py --address=$master --rank=$num" >/home/ubuntu/worker_$num.log 2>&1 &
	num=$((num+1))
done
