# test lasher
#CUDA_VISIBLE_DEVICES=1 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name bat --dataset_name LasHeR --yaml_name rgbt

# test rgbt234
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name bat --dataset_name RGBT234 --yaml_name rgbt 



#CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name bat --dataset_name DroneT --yaml_name rgbt


#CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name bat --dataset_name VTUAVST --yaml_name rgbt