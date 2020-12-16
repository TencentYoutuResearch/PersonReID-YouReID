PORT=9998
export PYTHONPATH=/raid/home/fufuyu/torch-reid/
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port=${PORT} --node_rank=0 trainer.py