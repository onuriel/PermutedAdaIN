import random

import torch
import torch.nn as nn

DEFAULT_P = 0.01

class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class PermuteAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, p=None, **kwargs):
        if 'permute' in kwargs:
            self.permute = kwargs['permute']
        else:
            self.permute = True
        super(PermuteAdaptiveInstanceNorm2d, self).__init__()
        # self.perm_indices = None
        if p is None:
            self.p = DEFAULT_P
        else:
            self.p = p

    def forward(self, x):
        permute = random.random() < self.p
        if permute and self.training:
            perm_indices = torch.randperm(x.size()[0])
        else:
            return x
        size = x.size()
        N, C, H, W = size
        if (H, W) == (1, 1):
            print('encountered bad dims')
            return x
        return adaptive_instance_normalization(x, x[perm_indices], self.permute)

    def extra_repr(self) -> str:
        return 'p={} permute={}'.format(
            self.p, self.permute
        )


def calc_mean_std(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_std = feat.view(N, C, -1).std(dim=2).view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat, permute=False):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat.detach())
    content_mean, content_std = calc_mean_std(content_feat)
    content_std = content_std + 1e-5  # to avoid division by 0
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    if permute:
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    return normalized_feat
