IMAGE="lightning:latest"
NAME="lightning"
DATASET="/home/tungbx/Data/tung/Datasets/Classification"

sudo docker run -it --detach --rm --name $NAME --gpus all --shm-size=64G --memory=12G -v $PWD:/$NAME -v $DATASET:/Datasets $IMAGE
sudo docker start $NAME
sudo docker exec -it $NAME /bin/bash