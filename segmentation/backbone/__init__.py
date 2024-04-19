# Modified from https://github.com/open-mmlab/mmsegmentation/blob/v0.22.1/mmseg/core/utils/layer_decay_optimizer_constructor.py

import json

import timm
from mmcv.runner import BaseModule
from mmcv.runner import OPTIMIZER_BUILDERS, get_dist_info, DefaultOptimizerConstructor
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger

import rdnet


@BACKBONES.register_module()
class RDNetBackbone(BaseModule):
    def __init__(self,
                 model_name,
                 features_only=True,
                 pretrained=False,
                 checkpoint_path='',
                 in_channels=3,
                 init_cfg=None,
                 **kwargs):

        super().__init__(init_cfg)
        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs)

        # reset classifier
        if hasattr(self.timm_model, 'reset_classifier'):
            self.timm_model.reset_classifier(0, '')

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

    def forward(self, x):
        features = self.timm_model(x)

        if isinstance(features, (list, tuple)):
            features = tuple(features)
        else:
            features = (features, )

        return features


def get_stage_id_for_rdnet(var_name, max_stage_id):
    if var_name.startswith('backbone.timm_model.stem'):
        return 0
    elif var_name.startswith('backbone.timm_model.dense_stages_'):
        stage_id = int(var_name.split(".")[2].split("_")[-1])
        return stage_id + 1
    else:
        return max_stage_id + 1


@OPTIMIZER_BUILDERS.register_module()
class RDNetLearningRateDecayOptimizerConstructor(DefaultOptimizerConstructor):
    """Different learning rates are set for different layers of backbone."""

    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """

        parameter_groups = {}

        logger = get_root_logger()
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        model_name = self.paramwise_cfg.get('model_name')
        logger.info('Build LearningRateDecayOptimizerConstructor  '
                  f'{decay_type} {decay_rate} - {num_layers}')
        weight_decay = self.base_wd
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            if decay_type == 'stage_wise':
                if 'rdnet' in model_name:
                    layer_id = get_stage_id_for_rdnet(name, num_layers)
                    logger.info(f'set param {name} as id {layer_id}')
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())
