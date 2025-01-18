#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  NUM_GPUS=1
else
  NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

torchrun --nnodes 1 --nproc_per_node $NUM_GPUS --master_port 20002 src/llavapool/run.py $1
