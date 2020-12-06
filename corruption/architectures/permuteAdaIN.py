import random

import torch
import torch.nn as nn


#
# class PermuteAdaptiveInstanceNorm2d(nn.Module):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, **kwargs):
#         if 'permute' in kwargs:
#             self.permute = kwargs['permute']
#         else: self.permute = False
#         super(PermuteAdaptiveInstanceNorm2d, self).__init__()
#         self.perm_indices = None
#
#     def forward(self, x, train=True):
#         if self.perm_indices is None:
#             perm_indices = [i for i in range(x.size(0))]
#         else: perm_indices = self.perm_indices
#
#         return adaptive_instance_normalization(x, x[perm_indices], self.permute)
#

DEFAULT_P = 0.01
class PermuteAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, p=None, **kwargs):
        if 'permute' in kwargs:
            self.permute = kwargs['permute']
        else: self.permute = True
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
        if (H,W) == (1,1):
            print('encountered bad dims')
            return x
        return adaptive_instance_normalization(x, x[perm_indices], self.permute)

    def extra_repr(self) -> str:
        return 'p={} permute={}'.format(
            self.p, self.permute
        )
class InstanceNorm2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super(InstanceNorm2d, self).__init__()

    def forward(self, x):
        size = x.size()
        content_mean, content_std = calc_mean_std(x.detach())
        normalized_feat = (x - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat


def calc_mean_std(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_std = feat.view(N, C, -1).std(dim=2).view(N,C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N,C,1, 1)
    return feat_mean, feat_std

def calc_spatial_mean_std(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_std = feat.std(dim=1).view(N, 1, H, W)
    feat_mean = feat.mean(dim=1).view(N, 1, H, W)
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
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    # print(content_mean.mean(), content_mean.std(), content_std.mean(), content_std.std())
    # if train:
    #
    #     # return normalized_feat
    #     shift_mean = torch.normal(0, content_mean.std().cpu().item(),  content_mean.size())
    #     shift_std = torch.normal(0, content_std.std().cpu().item(),  content_mean.size())
    #     # shift_mean = np.random.normal(0, content_mean.std().cpu().item()*0.2, content_mean.size())
    #     # shift_std = np.random.normal(1, content_std.std().cpu().item()*1, content_mean.size())
    #     shift_mean = shift_mean.to(normalized_feat.device)
    #     shift_std = shift_std.to(normalized_feat.device)
    #     return normalized_feat * shift_std.expand(size) + shift_mean.expand(size)

    return normalized_feat


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

