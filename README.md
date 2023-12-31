# SegmentationLabelModel

## Overview
In this repo we provide code for the method presented in our 2021 ICLR paper, "Cut out the annotator, keep the cutout: better segmentation with weak supervision." This work is inspired by past work in weak supervision where weak labels are aggregated for classification problems, and we show how to use weak supervision for segmentation.  

To train a weakly supervised segmentation model, we progressed through three steps: (1) we trained 5 one-shot segmentation models to produce weak labels for the training set; (2) we aggregated the weak labels using the code provided here; then (3) we trained a final segmentation model with the aggregated labels. 

Any segmentation network can be used for step (1), for example training a one-shot model with a standard segmentation architecture or using one of the more recent off-the-shelf segmentation models. Similarly, any segmentation network can be used for step (3), where a segmentation network is trained with the aggregated labels. In this repo, we provide code for step (2), aggregating weak labels from five segmentation weak labeling sources, which is the primary technical contribution of the ICLR paper.

![plot](end_to_end_pipeline.png)

## Usage
To use this code, clone the repo and install the required pacakges using the ``requirements.txt`` file. The ``run_conditional_WS.ipynb`` notebook will walk you through how to use the proposed label aggregation method on synthetic data as well as custom data.
