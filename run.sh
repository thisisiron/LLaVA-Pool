export PYTHONPATH=src:$PYTHONPATH

torchrun --nnodes 1 --nproc_per_node 8 --master_port 20001 src/llavapool/run.py $1
