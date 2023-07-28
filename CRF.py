import denseCRF, denseCRF3D, maxflow
import numpy as np
import random
from tqdm import tqdm 
from shared_utils import compute_dice

def run_maxflow3d(img_data,prob_data,l,s):
    
    bP = 1.0 - prob_data
    Prob = np.asarray([bP, prob_data])
    Prob = np.transpose(Prob, [1, 2, 3, 0])

    lamda = l
    sigma = s
    param = (lamda, sigma)
    lab = maxflow.maxflow3d(img_data, Prob, param)
    
    return lab

def run_crf(train_images, train_weak_labels, dev_images, dev_weak_labels, dev_masks, seed=1):
        
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    
    # Get std of pos region from val set --- can alternatively est this from train set w/ weak labels if you dont have a val set
    pixpos = []
    for dev_image, dev_mask in zip(dev_images,dev_masks):
        pixpos += [dev_image[dev_mask>.5].flatten()]
    pixpos = np.hstack(pixpos)
    stdval = np.std(pixpos)

    # Tune CRF params on dev set --- can alternatively just set these by hand if you dont have a dev set
    print('Tuning CRF parameters...')
    dice_dict = {}
    for dev_image, dev_weak_label, dev_mask in tqdm(zip(dev_images,dev_weak_labels,dev_masks),total=len(dev_images)):
        for w in [0,.1,.5,1,2,5,10,50]:
            in_img = np.moveaxis(dev_image,2,0)
            in_seg = np.moveaxis(dev_weak_label,2,0)

            out_seg = run_maxflow3d(in_img.astype('float32'),in_seg.astype('float32'),w,stdval)
            out_seg = np.moveaxis(out_seg,0,2)

            out_seg[out_seg>.5] = 1
            out_seg[out_seg<=.5] = 0

            local_dice = [compute_dice(dev_mask,out_seg)]

            if w not in dice_dict.keys(): 
                dice_dict[w] = local_dice
            else: 
                dice_dict[w] += local_dice

    dice_scores = [np.nanmean(d) for d in dice_dict.values()]
    best_crf_dice = np.max(dice_scores)
    sorted_params = [x for _,x in sorted(zip(dice_scores,dice_dict.keys()))]
    print('All possible mean Dice scores after CRF:',[np.around(d,2) for d in dice_scores])
    print('Selected dice and corresponding parameters:',np.around(np.nanmean(dice_dict[sorted_params[-1]]),2),sorted_params[-1])
    
    train_final_labels = []
    print('Applying CRF to train images...')
    for train_image, train_weak_label in tqdm(zip(train_images, train_weak_labels),total=len(train_images)):

        in_img = np.moveaxis(train_image,2,0)
        in_seg = np.moveaxis(train_weak_label,2,0)

        out_seg = run_maxflow3d(in_img.astype('float32'),in_seg.astype('float32'),sorted_params[-1],stdval)
        out_seg = np.moveaxis(out_seg,0,2)
        train_final_labels += [out_seg]
        
    dev_final_labels = []
    print('Applying CRF to dev images...')
    for dev_image, dev_weak_label in tqdm(zip(dev_images, dev_weak_labels),total=len(dev_images)):

        in_img = np.moveaxis(dev_image,2,0)
        in_seg = np.moveaxis(dev_weak_label,2,0)

        out_seg = run_maxflow3d(in_img.astype('float32'),in_seg.astype('float32'),sorted_params[-1],stdval)
        out_seg = np.moveaxis(out_seg,0,2)
        dev_final_labels += [out_seg]
    
    return train_final_labels, dev_final_labels


