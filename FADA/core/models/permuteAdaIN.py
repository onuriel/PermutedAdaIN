import random

import torch
import torch.nn as nn

DEFAULT_P = 0.01
class PermuteAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, p=None, adv=False, **kwargs):
        if 'permute' in kwargs:
            self.permute = kwargs['permute']
        else: self.permute = True
        super(PermuteAdaptiveInstanceNorm2d, self).__init__()
        if p is None:
            self.p = DEFAULT_P
        else:
            self.p = p
        self.adv = adv

    def forward(self, x):
        permute = random.random() < self.p
        if permute and self.training:
            perm_indices = torch.randperm(x.size()[0])
        else:
            return x


        size = x.size()
        N, C, H, W = size
        if N == 1:
            return x
        if (H,W) == (1,1):
            print('encountered bad dims')
            return x

        if self.adv:
            x = torch.cat([adaptive_instance_normalization(x[:N//2], x[N//2:], permute=True), x[N//2:]], dim=0)
        else:
            x = adaptive_instance_normalization(x, x[perm_indices], True)
        return x

    def extra_repr(self) -> str:
        return 'p={} permute={}, adv={}'.format(
            self.p, self.permute, self.adv
        )



def calc_mean_std(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_std = feat.view(N, C, -1).std(dim=2).view(N,C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N,C,1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat, permute=False):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()

    style_mean, style_std = calc_mean_std(style_feat.detach())
    content_mean, content_std = calc_mean_std(content_feat)
    content_std = content_std + 1e-4  # to avoid division by 0
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    if permute:
        normalized_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return normalized_feat


