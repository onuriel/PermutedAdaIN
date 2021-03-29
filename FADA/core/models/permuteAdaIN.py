import random

import torch
import torch.nn as nn

class PermuteAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, p=0.01, adv=False, **kwargs):
        super(PermuteAdaptiveInstanceNorm2d, self).__init__()

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
            x = torch.cat([adaptive_instance_normalization(x[:N//2], x[N//2:]), x[N//2:]], dim=0)
        else:
            x = adaptive_instance_normalization(x, x[perm_indices])
        return x

    def extra_repr(self) -> str:
        return 'p={}, adv={}'.format(
            self.p, self.adv
        )



def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_std = torch.sqrt(feat.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat, eps=1e-5):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat.detach(), eps)
    content_mean, content_std = calc_mean_std(content_feat, eps)
    content_std = content_std + eps  # to avoid division by 0
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

