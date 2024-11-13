# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import faiss
from time import time

from .KNN import generate_outliers, generate_outliers_ours

class IDFeatPool(object):
    def __init__(self, class_num, sample_num=1500, feat_dim=512, mode='npos', device='cuda:0'):
        self.class_num = class_num
        self.sample_num = sample_num
        self.feat_dim = feat_dim
        self.device = device
        
        self.class_ptr = torch.zeros((class_num,))
        self.queue = torch.zeros((class_num, sample_num, feat_dim)).to(device)

        self.mode = mode
        if mode == 'npos':
            # Standard Gaussian distribution
            self.new_dis = MultivariateNormal(torch.zeros(self.feat_dim).to(self.queue.device), 
                                              torch.eye(self.feat_dim).to(self.queue.device))
            assert faiss.StandardGpuResources
            res = faiss.StandardGpuResources()
            self.KNN_index = faiss.GpuIndexFlatL2(res, self.feat_dim)
            self.K = 400
            self.select = 300
            self.pick_nums = 1  # 3
            self.sample_from = 1000
            self.ID_points_num = 1  # 2
        elif mode == 'vos':
            self.sample_from = 10000
            self.select = 300
            self.pick_nums = 1
        elif mode == 'ours':
            self.new_dis = MultivariateNormal(torch.zeros(self.feat_dim).to(self.queue.device), 
                                              torch.eye(self.feat_dim).to(self.queue.device))
            res = faiss.StandardGpuResources()
            self.KNN_index = faiss.GpuIndexFlatL2(res, self.feat_dim)
            self.sample_from = 150
            self.K = 25
            self.select = 25
            self.ID_points_num = 1
            self.pick_nums = 1
        else:
            raise NotImplementedError(mode)
    
    def update(self, features, labels):
        if self.queue.device != features.device:
            # self.queue = self.queue.to(features.device)
            features = features.to(self.device)
        if self.queue.dtype != features.dtype:
            self.queue = self.queue.type_as(features)

        # for ci, feat in zip(labels, features):
        #     self.queue[ci] = torch.cat((self.queue[ci][1:], feat.detach().view(1, -1)), 0)
        #     self.class_ptr[ci] = (self.class_ptr[ci] + 1).clamp(max=self.sample_num)
        
        unique_labels = torch.unique(labels)
        unique_indices = (unique_labels.view(-1, 1) == labels.view(1, -1)).int().argmax(dim=1)
        self.queue[unique_labels] = torch.cat((self.queue[unique_labels, 1:, :], features[unique_indices][:, None, :]), 1)
        self.class_ptr[unique_labels] = (self.class_ptr[unique_labels] + 1).clamp(max=self.sample_num)
    
    def ready(self):
        return (self.class_ptr >= self.sample_num).all()

    def gen_ood(self, anchors=None, normed=True, device='cuda:0', cls_mask=None, ret_cand=False):
        ood_samples, ood_labels = [], []

        if self.mode == 'vos':
            ood_samples, ood_labels = [], []
            mean_embed_id = self.queue.mean(dim=1, keepdim=True)
            X = self.queue - mean_embed_id
            covariance = torch.bmm(X.permute(0,2,1), X).mean(dim=0) / self.sample_number
            covariance += 0.0001 * torch.eye(len(covariance), device=X.device)

            new_dis = MultivariateNormal(torch.zeros(512).cuda(), covariance_matrix=covariance)
            negative_samples = new_dis.rsample((self.sample_from,))
            prob_density = new_dis.log_prob(negative_samples)
            cur_samples, index_prob = torch.topk(- prob_density, self.select)
            negative_samples = negative_samples[index_prob]

            for ci, miu in enumerate(mean_embed_id):
                rand_ind = torch.randperm(self.select)[:self.pick_nums]
                ood_samples.append(miu + negative_samples[rand_ind])
                ood_labels.extend([ci] * self.pick_nums)
            ood_samples = torch.cat(ood_samples)

        elif self.mode == 'npos':
            negative_samples = self.new_dis.rsample((self.sample_from,)).half().to(self.device)

            text_anchors = anchors.to(self.device) if anchors is not None else None
            ood_samples, ood_labels = generate_outliers(self.queue, input_index=self.KNN_index, negative_samples=negative_samples, 
                                                        ID_points_num=self.ID_points_num, K=self.K, select=self.select, 
                                                        sampling_ratio=1.0, pic_nums=self.pick_nums, depth=self.feat_dim,
                                                        text_anchors=text_anchors, cls_mask=cls_mask)
        
        elif self.mode == 'ours':
            negative_samples = self.new_dis.rsample((self.sample_from,)).to(self.device)
            ood_samples, ood_labels, candidates = \
                generate_outliers_ours(self.queue.float(), input_index=self.KNN_index, negative_samples=negative_samples, 
                                       ID_points_num=self.ID_points_num, K=self.K, select=self.select, 
                                       sampling_ratio=1.0, pic_nums=self.pick_nums, depth=self.feat_dim,
                                       text_anchors=anchors.float())

        ood_samples = torch.cat(ood_samples).to(device)
        if normed:
            ood_samples = F.normalize(ood_samples, p=2, dim=1)
        ood_labels = torch.tensor(ood_labels).to(device)

        if ret_cand:
            return ood_samples, ood_labels, candidates
        return ood_samples, ood_labels
    