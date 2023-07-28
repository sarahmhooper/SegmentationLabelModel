# SegmentationLabelModel

In this repo we provide code for the method presented in our 2021 ICLR paper, "Cut out the annotator, keep the cutout: better segmentation with weak supervision." This work is inspired by past work in weak supervision where weak labels are aggregated for classification problems, and we show how to use weak supervision for segmentation.  

To train a weakly supervised segmentation model with limited hand-labeled data, we progress through three steps: (1) we train 5 one-shot segmentation models to produce weak labels for a training set; (2) we aggregate the weak labels using the code provided here; then (3) we train a final segmentation model with the aggregated labels. 

Any segmentation network can be used for step (1), for example training a one-shot model with a standard segmentation architecture or using one of the more recent off-the-shelf segmentation models. Similarly, any segmentation network can be used for step (3), where a segmentation network is trained with the aggregated labels. In this repo, we provide code for step (2), aggregating weak labels from five segmentation weak labeling sources.
