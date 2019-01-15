echo "sync"
if [ ! -n "$1" ]; then
exit
fi
echo "OK"
for i in $other_workers
do
	ssh $i "sudo rm -rf `pwd`/$1"
	scp -r $1 "$i:`pwd`/$1"
	echo $i 
done
