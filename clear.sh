pkill python
for i in $workers
do 
	ssh $i "pkill python"
	echo "finish clear - "$i
done
