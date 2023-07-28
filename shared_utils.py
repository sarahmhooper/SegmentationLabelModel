
import numpy as np
import scipy

def compute_dice(mask1, mask2):
    assert list(np.unique(mask1)) == [0,1] or list(np.unique(mask1)) == [0] or list(np.unique(mask1)) == [1]
    assert list(np.unique(mask2)) == [0,1] or list(np.unique(mask2)) == [0] or list(np.unique(mask2)) == [1]
    area_overlap = np.sum(np.logical_and(mask1,mask2))
    total_pix = np.sum(mask1) + np.sum(mask2)
    if total_pix==0: return 1
    return 2 * area_overlap / float(total_pix)

def binarize_conditionals(conds):
    result = np.ones_like(conds)
    result[conds<0.5] = -1
    return result

def compute_baselines(samples):
    indicator_samples = np.zeros_like(samples)
    indicator_samples[samples==1] = 1
    indicator_samples = np.sum(indicator_samples,1)
    mv = binarize_conditionals(indicator_samples/samples.shape[1])
    intersection = binarize_conditionals(1.0*(indicator_samples>0))
    union = binarize_conditionals(1.0*(indicator_samples==samples.shape[1]))
    return mv, intersection, union

