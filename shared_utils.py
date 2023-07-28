
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

def create_example_data(n_train=150,n_dev=100,xy_size=112,z_size=6):

    X_train = []
    L_train = []
    X_dev = []
    L_dev = []
    Y_dev = []

    for nimg in range(n_train):

        x = np.zeros((xy_size,xy_size,z_size))
        startx = np.random.randint(xy_size-30)
        starty = np.random.randint(xy_size-30)
        startz = np.random.randint(z_size-2)
        x[startx:startx+30,starty:starty+30,startz:startz+3] = 1
        
        gt = x.copy()

        for slind in range(x.shape[-1]): x[:,:,slind] = scipy.ndimage.gaussian_filter(x[:,:,slind], 4)
        x -= np.min(x)
        x = x/np.max(x)
        
        l1 = np.zeros((xy_size,xy_size,z_size))
        l1[50:100,50:100,:] = 1
        l2 = gt.copy()
        l3 = np.pad(gt[10:,10:,:],((0,10),(0,10),(0,0)))
        l4 = np.pad(gt[:-5,:-5,:],((5,0),(5,0),(0,0)))
        l5 = np.zeros((xy_size,xy_size,z_size))
        l5[startx-10:startx+50,starty-10:starty+50,startz:startz+4] = 1

        X_train += [x]
        L_train += [np.stack([l1,l2,l3,l4,l5],0)]

    for nimg in range(n_dev):

        x = np.zeros((xy_size,xy_size,z_size))
        startx = np.random.randint(xy_size-30)
        starty = np.random.randint(xy_size-30)
        startz = np.random.randint(z_size-2)
        x[startx:startx+30,starty:starty+30,startz:startz+3] = 1
        
        gt = x.copy()

        for slind in range(x.shape[-1]): x[:,:,slind] = scipy.ndimage.gaussian_filter(x[:,:,slind], 4)
        x -= np.min(x)
        x = x/np.max(x)
        
        l1 = np.zeros((xy_size,xy_size,z_size))
        l1[50:100,50:100,:] = 1
        l2 = gt.copy()
        l3 = np.pad(gt[10:,10:,:],((0,10),(0,10),(0,0)))
        l4 = np.pad(gt[:-5,:-5,:],((5,0),(5,0),(0,0)))
        l5 = np.zeros((xy_size,xy_size,z_size))
        l5[startx-10:startx+50,starty-10:starty+50,startz:startz+4] = 1

        X_dev += [x]
        Y_dev += [gt]
        L_dev += [np.stack([l1,l2,l3,l4,l5],0)]

    return X_train, L_train, X_dev, L_dev, Y_dev



