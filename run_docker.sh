#!/bin/bash

docker rm -fv ignacio_cis

docker run -it \
  --gpus '"device=0,1,2,3,4,5,6,7"' \
  --name ignacio_cis \
  --shm-size=32g \
  --ipc=host \
  -v /home/ignacio.bugueno/cachefs/cis/input:/app/input \
  -v /home/ignacio.bugueno/cachefs/cis/output:/app/output \
  ignacio_cis
