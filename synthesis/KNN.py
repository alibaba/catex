# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import faiss
# import umap
# import time
#import matplotlib.pyplot as plt
import faiss.contrib.torch_utils
# from sklearn import manifold, datasets
# from torch.distributions import MultivariateNormal
import torch.nn.functional as F

def KNN_dis_search_decrease(target, index, K=50, select=1,):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    #Normalize the features

    target_norm = torch.norm(target, p=2, dim=1,  keepdim=True)
    normed_target = target / target_norm
    #start_time = time.time()

    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    #k_th_output_index = output_index[:, -1]
    k_th_distance, minD_idx = torch.topk(k_th_distance, select)
    #k_th_index = k_th_output_index[minD_idx]
    return minD_idx, k_th_distance

def KNN_dis_search_distance(target, index, K=50, num_points=10, length=2000,depth=342):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    #Normalize the features

    target_norm = torch.norm(target, p=2, dim=1,  keepdim=True)
    normed_target = target / target_norm
    #start_time = time.time()

    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    k_th = k_th_distance.view(length, -1)
    target_new = target.view(length, -1, depth)
    #k_th_output_index = output_index[:, -1]
    k_th_distance, minD_idx = torch.topk(k_th, num_points, dim=0)
    # minD_idx = minD_idx.squeeze()
    point_list = []
    for i in range(minD_idx.shape[1]):
        point_list.append(i*length + minD_idx[:,i])
    #return torch.cat(point_list, dim=0)
    return target[torch.cat(point_list)]


def generate_outliers(ID, input_index, negative_samples, ID_points_num=2, K=20, select=1, cov_mat=0.1, sampling_ratio=1.0, pic_nums=30, depth=342,
                      text_anchors=None, cls_mask=None):
    ncls, nsample, ndim = ID.shape
    length, _ = negative_samples.shape
    normed_data = ID / torch.norm(ID, p=2, dim=-1, keepdim=True)
    if cls_mask is not None:
        normed_data = normed_data[cls_mask] #.float()
        text_anchors = text_anchors[cls_mask]

    distance = torch.cdist(normed_data, normed_data.detach()).half()  # shape(ncls, nsample, nsample)
    k_th_distance = -torch.topk(-distance, K, dim=-1)[0][..., -1]  # k-th nearset (smallest distance), shape(ncls, nsample)
    minD_idx = torch.topk(k_th_distance, select, dim=1)[1]  # top-k largest distance, shape(ncls, select)
    minD_idx = minD_idx[:, np.random.choice(select, int(pic_nums), replace=False)]  #shape(ncls, pic_nums)
    cls_idx = torch.arange(ncls).view(ncls, 1)
    if cls_mask is not None:
        cls_idx = cls_idx[cls_mask]
    data_point_list = ID[cls_idx.repeat(1, pic_nums).view(-1), minD_idx.view(-1)].view(-1, pic_nums, 1, ndim)

    negative_sample_cov = cov_mat*negative_samples.view(1, 1, length, ndim)
    negative_sample_list = (negative_sample_cov + data_point_list).view(-1, pic_nums*length, ndim)

    normed_ood_feat = F.normalize(negative_sample_list, p=2, dim=-1)  #shape(cls, pic_nums*length, 512)
    distance = torch.cdist(normed_ood_feat, normed_data.half())  # shape(ncls, pic_nums*length, nsample)
    k_th_distance = -torch.topk(-distance, K, dim=-1)[0][..., -1]  # k-th nearset (smallest distance), shape(ncls, pic_nums*length)
    
    if text_anchors is not None:  # shape(cls,2,ndim)
        intra_similarity = torch.bmm(normed_ood_feat, text_anchors.permute(0, 2, 1))  #shape(cls,pic_nums*length,2)
        # only perserve samples with higher similarity to the perturbed text-feature
        intra_candidate = intra_similarity[..., 0] < intra_similarity[..., 1]
        
        # inter_similarity = normed_ood_feat.float() @ text_anchors[:, 0, :].float().T  #shape(cls, pic_nums*length,ncls)
        # # only perserve samples with highest similarity among in-distribution text-features
        # inter_candidate = inter_similarity.argmax(dim=-1) == torch.arange(ncls).view(ncls,1).to(inter_similarity.device)

        candidate = intra_candidate #& inter_candidate

        k_th_distance *= candidate.float()
    
    k_distance, minD_idx = torch.topk(k_th_distance, ID_points_num, dim=1)  # top-k largest distance, shape(ncls, ID_points_num)
    OOD_labels = torch.arange(normed_data.size(0)).view(-1, 1).repeat(1, ID_points_num).view(-1)
    OOD_syntheses = negative_sample_list[OOD_labels, minD_idx.view(-1)]    #shape(ncls*ID_points_num, 512)

    if text_anchors is not None:
        valid = k_distance.view(-1) > 0
        OOD_syntheses, OOD_labels = OOD_syntheses[valid], OOD_labels[valid]
    
    if OOD_syntheses.shape[0]:
        # concatenate ood_samples outside
        OOD_syntheses = torch.chunk(OOD_syntheses, OOD_syntheses.shape[0])
        OOD_labels = OOD_labels.numpy()

    return OOD_syntheses, OOD_labels


def generate_outliers_ours(ID, input_index, negative_samples, ID_points_num=2, K=20, select=1, cov_mat=0.1, sampling_ratio=1.0, pic_nums=30, depth=342,
                           text_anchors=None):
    assert text_anchors is not None
    length = negative_samples.shape[0]
    ncls, nsample, ndim = ID.shape

    if True:
        rand_ind = np.random.choice(nsample, select, replace=False)
        id_data_points = ID[:, rand_ind, :].detach().view(ncls, select, 1, ndim)
    else:
        # rand_ind = []
        # for ci in range(ncls):
        #     rand_ind.append(torch.randperm(nsample) + ci * nsample)
        # rand_ind = torch.cat(rand_ind)
        # id_data_points = ID.view(ncls*nsample, ndim)[rand_ind].view(ncls, nsample, ndim)
        # id_data_points = id_data_points[:, :select, :].view(ncls, select, 1, ndim)

        id_data_points = []
        normed_data = ID / torch.norm(ID, p=2, dim=-1, keepdim=True)
        for ci in range(ncls):
            input_index.add(normed_data[ci])
            minD_idx, k_th = KNN_dis_search_decrease(ID[ci], input_index, K, select)
            id_data_points.append(ID[ci][minD_idx])  # shape(select,ndim)
        id_data_points = torch.stack(id_data_points).view(ncls, select, 1, ndim)

    negative_sample_cov = cov_mat * negative_samples.view(1, 1, length, ndim)
    negative_samples = (id_data_points + negative_sample_cov).view(ncls, select*length, ndim)
    normed_ood_feat = F.normalize(negative_samples, p=2, dim=1)   #shape(select*length,512)

    inter_similarity = normed_ood_feat @ text_anchors[:, 0, :].T  #shape(ncls,select*length,ncls)
    # only perserve samples with highest similarity among in-distribution text-features
    inter_candidate = inter_similarity.argmax(dim=-1) == torch.arange(ncls).cuda().view(ncls, 1)

    intra_similarity = torch.bmm(normed_ood_feat, text_anchors.transpose(1,2))  #shape(ncls,select*length,2)
    # only perserve samples with higher similarity to the perturbed text-feature
    intra_candidate = intra_similarity[..., 0] < intra_similarity[..., 1]

    candidate = inter_candidate & intra_candidate

    ood_samples, ood_labels = [], []
    for ci in range(ncls):
        syntheses = negative_samples[ci][candidate[ci]]
        valid_num = len(syntheses)
        labels = np.full((valid_num,), ci, dtype=np.int64)
        if valid_num > ID_points_num:
            rand_ind = np.random.choice(valid_num, ID_points_num, replace=False, p=None)
            syntheses, labels = syntheses[rand_ind], labels[rand_ind]

        ood_samples.append(syntheses)
        ood_labels.append(labels)
    
    # concatenate ood_samples outside
    ood_labels = np.concatenate(ood_labels)

    return ood_samples, ood_labels, candidate


def generate_outliers_OOD(ID, input_index, negative_samples, K=100, select=100, sampling_ratio=1.0):
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[1], int(normed_data.shape[1] * sampling_ratio), replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th = KNN_dis_search_decrease(negative_samples, index, K, select)

    return negative_samples[minD_idx]



def generate_outliers_rand(ID, input_index,
                           negative_samples, ID_points_num=2, K=20, select=1,
                           cov_mat=0.1, sampling_ratio=1.0, pic_nums=10,
                           repeat_times=30, depth=342):
    length = negative_samples.shape[0]
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[1], int(normed_data.shape[1] * sampling_ratio), replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th = KNN_dis_search_decrease(ID, index, K, select)
    ID_boundary = ID[minD_idx]
    negative_sample_list = []
    for i in range(repeat_times):
        select_idx = np.random.choice(select, int(pic_nums), replace=False)
        sample_list = ID_boundary[select_idx]
        mean = sample_list.mean(0)
        var = torch.cov(sample_list.T)
        var = torch.mm(negative_samples, var)
        trans_samples = mean + var
        negative_sample_list.append(trans_samples)
    negative_sample_list = torch.cat(negative_sample_list, dim=0)
    point = KNN_dis_search_distance(negative_sample_list, index, K, ID_points_num, length,depth)

    index.reset()

    #return ID[minD_idx]
    return point

