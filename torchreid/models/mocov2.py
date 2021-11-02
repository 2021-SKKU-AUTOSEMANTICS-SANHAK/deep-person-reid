# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import division, absolute_import
import os
from shutil import Error
import warnings
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import torch.distributed as dist

import torchvision.models as models
from torchreid.models import moco_encoders

from torchreid.utils.torchtools import load_pretrained_weights
# from torchvision.models import resnet50
from torch.nn.parallel import DistributedDataParallel as DDP


pretrained_urls = {
    'lup_moco_r50': 'https://drive.google.com/u/0/uc?id=1pFyAdt9BOZCtzaLiE-W3CsX_kgWABKK6&export=download',
    'moco_v2_imagenet': 'https://drive.google.com/u/0/uc?id=1V4QRLNkcD22X3x_L_0JWzqJ4uWOnCeox&export=download'
}

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, num_classes, dim=128, loss='SupConLoss', K=65536, m=0.999, T=0.07, mlp=False,  **kwargs):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.num_cls=num_classes
        self.loss = loss
        
        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
       
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), 
                                              nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), 
                                              nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_v0(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = ptr + batch_size
        else:
            # p_str = f"prt: {ptr}; bs: {batch_size} "
            remain = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys[:self.K - ptr].T
            self.queue[:, :remain] = keys[self.K - ptr:].T
            ptr = remain
            # p_str += f"remain: {remain}, new ptr: {ptr}"
            # print(p_str)

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q=None, im_k=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if not self.training:
            return nn.functional.normalize(self.encoder_q(im_q), dim=1)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        if self.loss == 'softmax':
            return logits
        elif self.loss == 'SupConvLoss':
            return logits, labels
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import errno
    import gdown

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
    if key == 'lup_moco_r50':
        filename = key + '.pth'
    elif key == 'moco_v2_imagenet':
        filename = key + '.pth.tar'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)
    
    try:
        print(f'Loading pre-model from {cached_file}')
        try:
            state_dict = torch.load(cached_file, map_location=torch.device('cuda'))
        except Error as e:
            print(e)
            state_dict = torch.load(cached_file, map_location=torch.device('cpu'))
            
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            print(f'Loading pre-model from {cached_file} successively')
        msg = model.load_state_dict(state_dict, strict=False)
        print(f'Load pre-model with MSG: {msg}')
        
    except AttributeError as e:
        print(e)


def DDPsetup():
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:13701', world_size=1, rank=0)

def mocov2(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    backbone = models.__dict__['resnet50']

    # use resnet50 as encoder of moco
    base_encoder = backbone
    DDPsetup()
    
    model = MoCo(
        base_encoder=base_encoder,
        num_classes=num_classes,
        dim=128,
        loss=loss,
        K=65536,
        m=0.999, 
        T=0.1, 
        mlp=1,
        **kwargs
        )
    if pretrained:
        #init_pretrained_weights(model=model, key='lup_moco_r50')
        init_pretrained_weights(model=model, key='moco_v2_imagenet')
    return model
