---
datasets:
- imagenet-1k
library_name: timm
tags:
- image-classification
- timm
- rdnet
---
# Model card for rdnet_large.nv_in1k

A RDNet image classification model. Trained on ImageNet-1k, original torchvision weights.

## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Imagenet-1k validation top-1 accuracy: 84.8% 
  - Params (M): 186
  - GMACs: 34.7
  - Image size: 224 x 224
- **Papers:**
  - DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs: https://arxiv.org/abs/2403.19588
- **Dataset:** ImageNet-1k

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm
import torch
import rdnet  # register rdnet models to timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('rdnet_large.nv_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Feature Map Extraction
```python
from urllib.request import urlopen
from PIL import Image
import timm
import rdnet  # register rdnet models to timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'rdnet_large.nv_in1k',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    # e.g.:
    # torch.Size([1, 528, 56, 56])
    # torch.Size([1, 840, 28, 28])
    # torch.Size([1, 1528, 14, 14])
    # torch.Size([1, 2000, 7, 7])

    print(o.shape)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm
import rdnet  # register rdnet models to timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'rdnet_large.nv_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 2000, 7, 7) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

### Citation
```
@misc{kim2024densenets,
    title={DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs}, 
    author={Donghyun Kim and Byeongho Heo and Dongyoon Han},
    year={2024},
    eprint={2403.19588},
    archivePrefix={arXiv},
}
```