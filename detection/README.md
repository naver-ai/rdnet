## RDNet for Object Detection & Instance Segmentation
`backbone/rdnet.py` for detection and segmentation is the exact same file as `rdnet.py` for classification.

### Tested Environment
- python 3.9.18
- torch 1.11.0
- timm 0.9.6
- mmcv-full 1.4.8
- mmdet 2.25.3
- mmcls 0.25.0
- mmsegmentation 0.22.1

### Pretrained Models & logs

Mask-RCNN Pretrained Models & logs (HFHub): https://huggingface.co/naver-ai/rdnet_mask_rcnn_coco_3x/tree/main

Cascade Mask-RCNN Pretrained Models & logs (HFHub): https://huggingface.co/naver-ai/rdnet_cascade_mask_rcnn_coco_3x/tree/main

|      $Model$      | $Backbone$ | $Params$ | $FLOPs$ | $AP_{box}$ | $AP^{50}_{box}$ | $AP^{75}_{box}$ | $AP_{mask}$ | $AP^{50}_{mask}$ | $AP^{75}_{mask}$ |                                                                                                                url                                                                                                                 |
|:-----------------:|:----------:|:--------:|:-------:|:----------:|:---------------:|:---------------:|:-----------:|:----------------:|:----------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     Mask-RCNN     |  RDNet-T   |   43M    |  278G   |    47.5    |      68.5       |      52.1       |    42.4     |       65.6       |       45.7       |          [ckpt](https://huggingface.co/naver-ai/rdnet_mask_rcnn_coco_3x/blob/main/rdnet_tiny/epoch_36.pth), [train_log](https://huggingface.co/naver-ai/rdnet_mask_rcnn_coco_3x/blob/main/rdnet_tiny/20240308_095003.log)          |
| Cascade Mask-RCNN |  RDNet-T   |   81M    |  757G   |    51.6    |      70.5       |      56.0       |    44.6     |       67.9       |       48.3       |  [ckpt](https://huggingface.co/naver-ai/rdnet_cascade_mask_rcnn_coco_3x/blob/main/rdnet_tiny/epoch_36.pth), [train_log](https://huggingface.co/naver-ai/rdnet_cascade_mask_rcnn_coco_3x/blob/main/rdnet_tiny/20240309_072408.log)  |
| Cascade Mask-RCNN |  RDNet-S   |   108M   |  832G   |    52.3    |      70.8       |      56.6       |    45.4     |       68.5       |       49.3       | [ckpt](https://huggingface.co/naver-ai/rdnet_cascade_mask_rcnn_coco_3x/blob/main/rdnet_small/epoch_36.pth), [train_log](https://huggingface.co/naver-ai/rdnet_cascade_mask_rcnn_coco_3x/blob/main/rdnet_small/20240309_082553.log) |
| Cascade Mask-RCNN |  RDNet-B   |   144M   |  971G   |    52.9    |      71.5       |      57.2       |    46.0     |       69.1       |       50.0       |  [ckpt](https://huggingface.co/naver-ai/rdnet_cascade_mask_rcnn_coco_3x/blob/main/rdnet_base/epoch_36.pth), [train_log](https://huggingface.co/naver-ai/rdnet_cascade_mask_rcnn_coco_3x/blob/main/rdnet_base/20240401_014441.log)  |


### Train
```
torchrun --nproc_per_node=4 train.py configs/rdnet/mask_rcnn_rdnet_tiny_3x.py --launcher pytorch
torchrun --nproc_per_node=4 train.py configs/rdnet/cascade_mask_rcnn_rdnet_tiny_3x.py --launcher pytorch
torchrun --nproc_per_node=4 train.py configs/rdnet/cascade_mask_rcnn_rdnet_small_3x.py --launcher pytorch
torchrun --nproc_per_node=4 train.py configs/rdnet/cascade_mask_rcnn_rdnet_base_3x.py --launcher pytorch
```
