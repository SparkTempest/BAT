class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/root/BAT-main'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/root/BAT-main/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/root/BAT-main/pretrained_networks'
        self.got10k_val_dir = '/root/BAT-main/data/got10k/val'
        self.lasot_lmdb_dir = '/root/BAT-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/root/BAT-main/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/root/BAT-main/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/root/BAT-main/data/coco_lmdb'
        self.coco_dir = '/root/BAT-main/data/coco'
        self.lasot_dir = '/root/BAT-main/data/lasot'
        self.got10k_dir = '/root/BAT-main/data/got10k/train'
        self.trackingnet_dir = '/root/BAT-main/data/trackingnet'
        self.depthtrack_dir = '/root/BAT-main/data/depthtrack/train'
        self.lasher_dir = '/root/LasHeR/TrainingSet'
        self.visevent_dir = '/root/BAT-main/data/visevent/train'
