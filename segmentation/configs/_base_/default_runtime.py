# copied from https://github.com/open-mmlab/mmsegmentation/blob/v0.22.1/configs/_base_/default_runtime.py

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='WandbLoggerHook',
        #      init_kwargs={
        #         'project': 'mmsegmentation',
        #         'name': 'maskrcnn-r50-fpn-1x-coco'
        #      },
        #      interval=50,
        #      log_artifact=False)
    ])
# yapf:enable
custom_hooks = []
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
