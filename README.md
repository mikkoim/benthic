# benthic

Code for my Bachelor's thesis: "On imbalanced classification of benthic macroinvertebrates: Metrics and loss-functions"

This thesis has two main themes:  analyzing the suitability of different performance metrics used to evaluate imbalanced domain classification models, as well as testing methods that could be used to improve the performance of these models.  Performance metrics are analyzed from the standpoint of experts with no machine learning expertise, focusing on understandability and visualizations of the metrics.  Focus is given on metrics that can be derived from a multi-class confusion  matrix,  due  to  the  intuitive  derivation  of  these  metrics.   These  metrics  are  used  toproduce both single-score and class-wise metrics, that describe the model performance either as whole, or separately for each class. As for classification improvement methods, experiments with different loss functions, rebalancing and augmentation methods are conducted.

Thesis:
https://trepo.tuni.fi/bitstream/handle/10024/122275/Impi%C3%B6Mikko.pdf

This repo is still a work in progress and needs to be cleaned up further. Information on models can be found from ```benthic/notebooks/models.csv```, and the accompanying notebooks from the same folder. Most of the notebooks are self-contained, and can be run for example in Google Colab.

A good starting point might be the reference model notebook ```benthic/notebooks/18_01_2020_colab_splitN.ipynb ```
 
The data used in the experiments can be downloaded from https://etsin.fairdata.fi/dataset/a11cdc26-b9d0-4af1-9285-803d65a696a3
