export PYTHONPATH=src:$PYTHONPATH

torchrun --nnodes 1 --nproc_per_node 4 --master_port 20001 src/llavapool/run.py examples/qwen2vl_full_sft.yaml