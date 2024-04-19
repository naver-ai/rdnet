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
TBA


### Train
```
torchrun --nproc_per_node=4 train.py configs/rdnet/rdnet_tiny.py --launcher pytorch
```
