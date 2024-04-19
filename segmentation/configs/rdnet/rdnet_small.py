# modified from https://github.com/open-mmlab/mmsegmentation/blob/v0.22.1/configs/convnext/upernet_convnext_small_fp16_512x512_160k_ade20k.py

_base_ = [
    '../_base_/models/upernet_rdnet.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
model = dict(
    backbone=dict(
        type='RDNetBackbone',
        model_name='rdnet_small',
        features_only=True,
        pretrained=True,
        drop_path_rate=0.2,
    ),
    decode_head=dict(
        in_channels=[264, 512, 760, 1264],
        num_classes=150,
    ),
    auxiliary_head=dict(in_channels=760, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

optimizer = dict(
    constructor='RDNetLearningRateDecayOptimizerConstructor',
    _delete_=True,
    type='AdamW',
    lr=8e-5,
    betas=(0.9, 0.999),
    weight_decay=0.03,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 7,  # == num_stage
        'model_name': 'rdnet_tiny',
    })

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=6e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=4)
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()
