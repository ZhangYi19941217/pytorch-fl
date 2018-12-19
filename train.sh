echo 'worker_hosts: '$workers
num=0
for i in $workers
do
	nohup ssh $i "python /home/ubuntu/yi-fl/single_slave.py --address='172.31.46.217' --rank=$num" >/home/ubuntu/worker_$num.log 2>&1 &
	num=$((num+1))
done
