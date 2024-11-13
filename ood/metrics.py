# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import numpy as np


def get_msp_scores(logits, ood_logits=None, method='MCM', ret_near_ood=False):
    assert method in ['MCM', 'Energy', 'MaxScore',
                      'MCM_Full', 'MCM_Full_Hard', 'MCM_Full_Soft', 'MCM_Full_Neg', 'MCM_Pair_Hard', 'MCM_Pair_Soft',
                      'MCM_Pair_Scale', 'MCM_Full_Scale'], \
        'OOD inference method %s has not been implemented.' % method
    
    probs = F.softmax(logits, dim=1)
    msp = probs.max(dim=1).values
    inconsistent = None
    
    if method == 'Energy':
        tau = 100.
        scores = -torch.logsumexp(logits * tau, dim=1)
    
    elif method == 'MCM':
        # assert ood_logits is None

        scores = - msp  # higher score means OOD

    else:
        assert ood_logits is not None

        pred = logits.argmax(dim=1)
        xrange = torch.arange(logits.shape[0])

        if 'MCM_Full' in method:
            full_logits = torch.cat((logits, ood_logits), dim=1)
            full_probs = F.softmax(full_logits, dim=1)
            full_pred = full_logits.argmax(dim=1)
            inconsistent = pred != full_pred  # probs < full_probs/ood_probs
            
            if 'Neg' in method:
                cls_num = logits.shape[1]
                scores = full_probs[:, cls_num:].sum(dim=1)   # higher score means OOD
            else:
                scores = - full_probs[xrange, pred]
                if 'Hard' in method:
                    scores[inconsistent] = 0.   # higher score means OOD
                elif 'Soft' in method:
                    # negative score adding a positive delta brings a higer score
                    scores += full_probs[xrange, full_pred] - full_probs[xrange, pred]  # higher score means OOD
                elif 'Scale' in method:
                    max_id_sim, max_ood_sim = logits.max(dim=1)[0], ood_logits.max(dim=1)[0]
                    pair_logits = torch.stack((max_id_sim, max_ood_sim), dim=1)
                    scale = F.softmax(pair_logits, dim=1)[:, :1].clamp(min=0.5)
                    # scale = F.softmax(pair_logits * 8., dim=1)[:, :1].clamp(min=0.5) / 8.
                    full_probs = F.softmax(full_logits * scale, dim=1)
                    scores = - full_probs[xrange, pred]
                else:
                    assert method == 'MCM_Full'

        elif 'MCM_Pair' in method:
            scores = - msp  # higher score means OOD

            pair_logits = torch.stack((logits[xrange, pred], ood_logits[xrange, pred]), dim=1)  # shape(nb,2)
            inconsistent = pair_logits[:, 0] < pair_logits[:, 1]  # id_sim < ood_sim
            pair_probs = F.softmax(pair_logits, dim=1)

            if 'Hard' in method:
                scores[inconsistent] = 0. # higher score means OOD
            elif 'Soft' in method:
                # negative score multiplying a smaller value brings a higher score
                scores *= pair_probs[:, 0].clamp(min=0.5)   # higher score means OOD
            elif 'Scale' in method:
                # print(F.softmax(pair_logits, dim=1)[:, :1].detach())
                scale = F.softmax(pair_logits, dim=1)[:, :1].clamp(min=0.5)  # 498
                # scale = F.softmax(pair_logits * 8., dim=1)[:, :1].clamp(min=0.) / 8.
                probs = F.softmax(logits * scale, dim=1)
                scores = - probs[xrange, pred]
            else:
                raise NotImplementedError

        elif method == 'MaxScore':
            pair_logits = torch.stack((logits[xrange, pred], ood_logits[xrange, pred]), dim=1)  # shape(nb,2)
            pair_probs = F.softmax(pair_logits*10., dim=1)

            scores = -pair_probs[:, 0] * logits[xrange, pred]
    
        
    if ret_near_ood:
        return scores, inconsistent
    else:
        return scores


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr, thresh = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr, thresh

