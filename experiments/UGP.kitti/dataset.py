from ugp.datasets.registration.kitti.dataset import OdometryKittiPairDataset
from ugp.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)
from ugp.utils.preprocess_utils import registration_collate_fn_stack_mode_full

def train_valid_data_loader(cfg, distributed, distance=10):
    train_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_min_scale=cfg.train.augmentation_min_scale,
        augmentation_max_scale=cfg.train.augmentation_max_scale,
        augmentation_shift=cfg.train.augmentation_shift,
        augmentation_rotation=cfg.train.augmentation_rotation,
        distance=distance,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode_full,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode_full,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
        phase='train',
        max_points=cfg.max_points
    )

    valid_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        distance=distance,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode_full,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        phase='val',
        max_points=cfg.max_points
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg, distance=10):
    train_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_min_scale=cfg.train.augmentation_min_scale,
        augmentation_max_scale=cfg.train.augmentation_max_scale,
        augmentation_shift=cfg.train.augmentation_shift,
        augmentation_rotation=cfg.train.augmentation_rotation,
        distance=distance,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode_full,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_root,
        'test',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        distance=distance,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode_full,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        phase='test',
        max_points=cfg.max_points
    )

    return test_loader, neighbor_limits
