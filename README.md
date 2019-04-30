# mono-semantic-occupancy
Codes and data of paper ''Monocular Semantic Occupancy Grid Mapping with Convolutional Variational Encoder-Decoder Networks'', IEEE Robotics and Automation Letters (also presented in IEEE International Conference on Robotics and Automation 2019).

Inplementation of the main experiment on the Cityscapes dataset. For KITTI, manual annotations are provided.

## Datasets
1. Two manually annotated datasets for testing are released:

    dataset/Cityscapes/Cityscapes_frankfurt_GT.zip

    dataset/KITTI/KITTI_semantics_GT.zip

2. The coarse top-view semantic occupancy grid map data for training (the predictions are also included), unzip to the same directory first:

    dataset/Cityscapes/Semantic_Occupancy_Grid_Multi_64.zip


## Usage
1. ```pip install -r requirements.txt```

2. Unzip the ground truth and training data into the same folder. Download the trained model if you don't want to re-train (link is in the checkpoint folder). Download the ```leftImg8bit_trainvaltest.zip``` files from [Cityscapes](https://cityscapes-dataset.com) website and unzip it into ```dataset/Cityscapes/leftImg8bit```

3. run 

    ```python vae_train.py```
 
    or 

    ```python vae_test.py```    

4. evaluate the results using jupyter notebook.

## Citation

```
@article{Lu2019icra-ral,
author = {Lu, Chenyang and van de Molengraft, Marinus Jacobus Gerardus and Dubbelman, Gijs},
journal = {IEEE Robotics and Automation Letters},
number = {2},
pages = {445--452},
title = {{Monocular Semantic Occupancy Grid Mapping With Convolutional Variational Encoder-Decoder Networks}},
volume = {4},
year = {2019}
}
```