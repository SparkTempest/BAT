# Training BAT
#NCCL_P2P_LEVEL=NVL python tracking/train.py --script bat --config rgbd --save_dir ./output --mode multiple --nproc_per_node 4
NCCL_P2P_LEVEL=NVL python tracking/train.py --script bat --config rgbt --save_dir ./output --mode multiple --nproc_per_node 3 
#python tracking/train.py --script bat --config rgbt --save_dir ./output --mode multiple --nproc_per_node 1 --use_wandb 1
#NCCL_P2P_LEVEL=NVL python tracking/train.py --script bat --config rgbe --save_dir ./output --mode multiple --nproc_per_node 4
