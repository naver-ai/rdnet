# Copied from https://github.com/open-mmlab/mmdetection/blob/v2.25.3/configs/_base_/default_runtime.py

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='WandbLoggerHook',
        #      init_kwargs={
        #         'project': 'mmdetection',
        #         'name': 'maskrcnn-r50-fpn-1x-coco'
        #      },
        #      interval=50,
        #      log_artifact=False)
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)