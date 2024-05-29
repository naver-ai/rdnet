## RDNet for Semantic Segmentation (UperNet)
`backbone/rdnet.py` for detection and segmentation is the exact same file as `rdnet.py` for classification.

### Tested Environment
- python 3.9.18
- torch 1.11.0
- timm 0.9.6
- mmcv-full 1.4.8
- mmdet 2.25.3
- mmcls 0.25.0
- mmsegmentation 0.22.1

### Model Zoo

Pretrained Models & logs (HFHub): https://huggingface.co/naver-clova-ocr/rdnet_upernet_ade20k_160k/tree/main

| $Backbone$ | $Params$ | $FLOPs$ | $mIOU^{SS}$ | $mIOU^{MS}$ |                                                                                                                                                                              url                                                                                                                                                                               |
|:----------:|:--------:|:-------:|:-----------:|:-----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  RDNet-T   |   81M    |  757G   |    47.6     |    48.6     |  [ckpt](https://huggingface.co/naver-ai/rdnet_upernet_ade20k_160k/blob/main/rdnet_tiny/iter_160000.pth), [train_log](https://huggingface.co/naver-ai/rdnet_upernet_ade20k_160k/raw/main/rdnet_tiny/20240305_184355.log), [ms_eval_log](https://huggingface.co/naver-ai/rdnet_upernet_ade20k_160k/blob/main/rdnet_tiny/eval_multi_scale_20240306_172501.json)   |
|  RDNet-S   |   50M    |  832G   |    48.7     |    49.8     | [ckpt](https://huggingface.co/naver-ai/rdnet_upernet_ade20k_160k/blob/main/rdnet_small/iter_160000.pth), [train_log](https://huggingface.co/naver-ai/rdnet_upernet_ade20k_160k/raw/main/rdnet_small/20240304_055221.log), [ms_eval_log](https://huggingface.co/naver-ai/rdnet_upernet_ade20k_160k/blob/main/rdnet_small/eval_multi_scale_20240305_094952.json) |
|  RDNet-B   |   87M    |  971G   |    49.6     |    50.5     |  [ckpt](https://huggingface.co/naver-ai/rdnet_upernet_ade20k_160k/blob/main/rdnet_base/iter_160000.pth), [train_log](https://huggingface.co/naver-ai/rdnet_upernet_ade20k_160k/raw/main/rdnet_base/20240304_092357.log), [ms_eval_log](https://huggingface.co/naver-ai/rdnet_upernet_ade20k_160k/blob/main/rdnet_base/eval_multi_scale_20240305_172057.json)   |

### Train
```
torchrun --nproc_per_node=4 train.py configs/rdnet/rdnet_tiny.py --launcher pytorch
```

### mutltiscale inference
```
sudo -H $(which torchrun) --nproc_per_node=4 test.py configs/rdnet/rdnet_tiny.py [ckpt-path] --aug-test --launcher pytorch --eval mIoU --tmpdir [your-tmpdir]
```