#watch -n 86400 /home/agupt013/scripts/checkNodes.sh

for a in $(seq 1 8); do ping -c 1  n0${a} >/dev/null ; [[ $? -ne 0 ]] && echo `date` Node 0${a} down; done
