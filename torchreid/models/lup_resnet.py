# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import warnings
import math
import torch
from torch import nn
import torch.nn.functional as F

from .mgn import *
from .resnet import *

pretrained_urls = {
    'lup_moco_r50': 'https://drive.google.com/u/0/uc?id=1pFyAdt9BOZCtzaLiE-W3CsX_kgWABKK6&export=download',
    'lup_moco_r101': 'https://drive.google.com/u/0/uc?id=1Ckn0iVtx-IhGQackRECoMR7IVVr4FC5h&export=download',
    'lup_moco_r152': 'https://drive.google.com/u/0/uc?id=1nGGatER6--ZTHdcTryhWEqKRKYU-Mrl_&export=download'
}

def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown
    from collections import OrderedDict

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    filename = key + '.pth'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(cached_file)
        )
    else:
        print(
            'Successfully loaded moco pretrained weights from "{}"'.
            format(cached_file)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

def lup_r50(num_classes, loss='softmax', pretrained=True, **kwargs):
    backbone = resnet50(num_classes=num_classes, loss=loss, pretrained=True)

    if pretrained:
        init_pretrained_weights(model=backbone, key='lup_moco_r50')

    model = MGN(
        num_classes=num_classes,
        loss=loss,
        backbone=backbone
    )

    return model