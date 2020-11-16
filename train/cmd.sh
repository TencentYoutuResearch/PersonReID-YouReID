export PYTHONPATH=/raid/home/fufuyu/torch-reid/
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port=1991 --node_rank=0 trainer_dis.py