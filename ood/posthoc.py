# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn

dataset = 'im1000'


thresh_dict = {
    'im1000': {0.05: [0.5640, -0.5430], 0.10: [0.4189, -0.4080]},
    'im100':  {0.05: [0.5625, -0.5415], 0.10: [0.4180, -0.4075]},
}
def applyReAct(feature, p=0.10):
    pos_thresh, neg_thresh = thresh_dict[dataset][p]
    feature[feature < neg_thresh] = neg_thresh
    # feature[feature > pos_thresh] = pos_thresh

    return feature


try:
    feat_mean = torch.load(f'restore/{dataset}-feat_mean.pt').cuda()
    feat_std  = torch.load(f'restore/{dataset}-feat_std.pt').cuda()
except:
    feat_mean, feat_std = None, None
    print('Warning: feat_mean and feat_std not found for BATS.')
def applyBATS(feature, lambd=2):
    feature = torch.where(feature<(feat_std*lambd+feat_mean),feature,feat_std*lambd+feat_mean)
    feature = torch.where(feature>(-feat_std*lambd+feat_mean),feature,-feat_std*lambd+feat_mean)

    return feature


def applyASH(feature, prune_ratio=0.05, method='S'):  # 0.05, S
    s = feature.abs().sum(dim=1, keepdim=True)
    n = feature.shape[1]
    k = int(round(n * prune_ratio))
    v, i = torch.topk(feature, k, dim=1, largest=False) # .abs() for imagenet-100

    feature.scatter_(dim=1, index=i, src=v.detach()*0)
    if method == 'B':
        raise NotImplementedError('ASH-B leads to poor performance')
        feature = (feature != 0).type_as(v) * s / (n - k)
    elif method == 'S':
        s2 = feature.abs().sum(dim=1, keepdim=True)
        feature *= s / s2

    return feature


if __name__ == '__main__':
    pass
