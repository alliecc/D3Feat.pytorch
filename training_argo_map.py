import os
import time
import shutil
import json
from config import get_config
from easydict import EasyDict as edict
# ThreeDMatchDataset, ThreeDMatchTestset
from datasets.mapdatasets import KITTIMapDataset, ArgoverseMapDataset
from trainer import Trainer
from models.architectures import KPFCNN
# from models.D3Feat import KPFCNN
from datasets.dataloader import get_dataloader
from utils.loss import ContrastiveLoss, CircleLoss, DetLoss
from torch import optim
from torch import nn
import torch
import sys
import importlib
sys.path.append("ext/benchmark")


class KITTIConfig:
    def __init__(self):
        """
        Override the parameters you want to modify for this dataset
        """

        ####################
        # Dataset parameters
        ####################
   #     is_test = False
        #gpu_id = 0
        self.depth_max = 50
        self.root = "/home/allie/dataset/kitti_odometry/dataset"
        self.path_cmrdata = "/home/allie/dataset/cmr_original"

        self.dataset = 'KITTIMap'

        # Number of CPU threads for the input pipeline
        self.input_threads = 8

        #########################
        # Architecture definition
        #########################

        self.architecture = ['simple',
                             'resnetb',
                             'resnetb_strided',
                             'resnetb',
                             'resnetb_strided',
                             'resnetb',
                             'resnetb_strided',
                             'resnetb',
                             'resnetb_strided',
                             'resnetb',
                             'nearest_upsample',
                             'unary',]
#
        # KPConv specific parameters
        self.num_kernel_points = 15
        self.first_subsampling_dl = 0.30

        # Density of neighborhoods for deformable convs (which need bigger radiuses). For normal conv we use KP_extent
        self.density_parameter = 5.0

        # Influence function of KPConv in ('constant', 'linear', gaussian)
        self.KP_influence = 'linear'
        self.KP_extent = 1.0

        # Aggregation function of KPConv in ('closest', 'sum')
        self.convolution_mode = 'sum'

        # Can the network learn modulations in addition to deformations
        self.modulated = False

        # detector loss weight
        self.det_loss_weight = 1

        # Offset loss
        # 'permissive' only constrains offsets inside the big radius
        # 'fitting' helps deformed kernels to adapt to the geometry by penalizing distance to input points
        self.offsets_loss = 'fitting'
        self.offsets_decay = 0.1

        # Choice of input features
        self.in_features_dim = 1

        # Batch normalization parameters
        self.use_batch_norm = True
        self.batch_norm_momentum = 0.98

        # batch hard loss safe radius
        self.safe_radius = 1

        #####################
        # Training parameters
        #####################

        # Maximal number of epochs
        self.max_epoch = 200

        # Learning rate management
        self.learning_rate = 1e-1
        self.momentum = 0.98
        self.lr_decays = {i: 0.1 ** (1 / 80) for i in range(1, self.max_epoch)}
        self.grad_clip_norm = 100.0

        # Number of batch
        self.batch_num = 1
        # Number of keypoints
        self.keypts_num = 1024

        # Number of steps per epochs (cannot be None for this dataset)
        self.epoch_steps = 1000

        # Number of validation examples per epoch
        self.validation_size = 100

        # Number of epoch between each snapshot
        self.snapshot_gap = 1

        # Augmentations
        self.augment_scale_anisotropic = True
        self.augment_symmetries = [False, False, False]
        self.augment_rotation = 1
        self.augment_scale_min = 0.8
        self.augment_scale_max = 1.2
        self.augment_noise = 0.01
        self.augment_occlusion = 'none'
        self.augment_shift_range = 2

        # Do we nee to save convergence
        self.saving = True
        self.saving_path = None

        self.num_min_map_points = 2e4
        #model = None
        #evaluation_metric = None
        #train_loader = None
        #val_loader = None


class ConfigObj(object):
    def __init__(self, d):
        self.__dict__ = d


if __name__ == '__main__':
    config_default = get_config()
    kitti_config = KITTIConfig()

    dconfig = vars(config_default)
    config_default = edict(dconfig)

    config = {}
    config.update(config_default)
    config.update(kitti_config.__dict__)

    for k in config.keys():
        print(f"    {k}: {config[k]}")

    config = ConfigObj(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    shutil.copy2(os.path.join('.', 'training_kitti_map.py'),
                 os.path.join(config.snapshot_dir, 'train.py'))
    shutil.copy2(os.path.join('.', 'trainer.py'),
                 os.path.join(config.snapshot_dir, 'trainer.py'))
    shutil.copy2(os.path.join('models', 'architectures.py'), os.path.join(
        config.snapshot_dir, 'model.py'))  # for the model setting.
    shutil.copy2(os.path.join('models', 'blocks.py'), os.path.join(
        config.snapshot_dir, 'conv.py'))  # for the conv implementation.
    shutil.copy2(os.path.join('utils', 'loss.py'),
                 os.path.join(config.snapshot_dir, 'loss.py'))
    shutil.copy2(os.path.join('datasets', 'mapdatasets.py'),
                 os.path.join(config.snapshot_dir, 'dataset.py'))
    json.dump(
        config.__dict__,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )
    # if config.gpu_mode:
    config.device = torch.device('cuda')
    # else:
    #    config.device = torch.device('cpu')

    # create model
    # config.architecture = [
    #    'simple',
    #    'resnetb',
    # ]
    # for i in range(config.num_layers-1):
    #    config.architecture.append('resnetb_strided')
    #    config.architecture.append('resnetb')
    #    config.architecture.append('resnetb')
    # for i in range(config.num_layers-2):
    #    config.architecture.append('nearest_upsample')
    #    config.architecture.append('unary')
    # config.architecture.append('nearest_upsample')
    # config.architecture.append('last_unary')
    print("Network Architecture:\n", "".join(
        [layer+'\n' for layer in config.architecture]))


    config.model = KPFCNN(config)

    # create optimizer
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            # momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )

    # create dataset and dataloader
    cfg = importlib.import_module("configs.config")
    train_set = ArgoverseMapDataset("train", cfg,
                                config_d3feat=config, root=config.root)  # , config=config,
    # downsample=config.downsample,
    # self_augment=config.self_augment,
    # num_node=config.num_node,
    # augment_noise=config.augment_noise,
    # augment_axis=config.augment_axis,
    # augment_rotation=config.augment_rotation,
    # augment_translation=config.augment_translation,)
    val_set = ArgoverseMapDataset("val", cfg, config_d3feat=config,
                              root=config.root)

    # (root=config.root,
    # split='val',
    # num_node=64,
    # downsample=config.downsample,
    # self_augment=config.self_augment,
    # augment_noise=config.augment_noise,
    # augment_axis=config.augment_axis,
    # augment_rotation=config.augment_rotation,
    # augment_translation=config.augment_translation,
    # config=config,
    # )

    config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
                                                              batch_size=config.batch_size,
                                                              shuffle=True,
                                                              num_workers=config.num_workers,
                                                              )
    config.val_loader, _ = get_dataloader(dataset=val_set,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          num_workers=config.num_workers,
                                          neighborhood_limits=neighborhood_limits
                                          )

    # create evaluation
    if config.desc_loss == 'contrastive':
        desc_loss = ContrastiveLoss(
            pos_margin=config.pos_margin,
            neg_margin=config.neg_margin,
            metric='euclidean',
            safe_radius=config.safe_radius
        )
    else:
        desc_loss = CircleLoss(
            m=config.m,
            log_scale=config.log_scale,
            safe_radius=config.safe_radius
        )

    config.evaluation_metric = {
        'desc_loss': desc_loss,
        'det_loss': DetLoss(metric='euclidean'),
    }
    config.metric_weight = {
        'desc_loss': config.desc_loss_weight,
        'det_loss': config.det_loss_weight,
    }

    trainer = Trainer(config)
    trainer.train()
