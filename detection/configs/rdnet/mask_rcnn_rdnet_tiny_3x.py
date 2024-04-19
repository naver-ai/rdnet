# Modified from https://github.com/open-mmlab/mmdetection/blob/v2.25.3/configs/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco.py

_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    backbone=dict(
        _delete_=True,
        type='RDNetBackbone',
        model_name='rdnet_tiny',
        features_only=True,
        pretrained=True,
        drop_path_rate=0.2,
    ),
    neck=dict(
        in_channels=[256, 440, 744, 1040],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline), persistent_workers=True)

optimizer = dict(
    _delete_=True,
    constructor='RDNetLearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'stage_wise',
        'num_layers': 7,
        'model_name': 'rdnet_tiny',
    })

lr_config = dict(warmup_iters=1500, step=[27, 33])  # 3x
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(max_epochs=36)

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
