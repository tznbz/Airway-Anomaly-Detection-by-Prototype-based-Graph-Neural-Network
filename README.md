# Airway-Anomaly-Detection-by-Prototype-based-Graph-Neural-Network

This is the code associated with the paper "Airway Anomaly Detection by Prototype-based Graph Neural Network"

## Abstract
Detecting the airway anomaly can be an essential part to aid the lung disease diagnosis. Since normal human airways share an anatomical structure, we design a graph prototype whose structure follows the normal airway anatomy. Then, we learn  the prototype and a graph neural network from a weakly-supervised airway dataset, i.e., only the holistic label is available, indicating if the airway has anomaly or not, but which bronchus node has the anomaly is unknown. During inference, the graph neural network predicts the anomaly score at both the holistic level and node-level of an airway. We initialize the airway anomaly detection problem by creating a large airway dataset with 2589 samples, and our prototype-based graph neural network shows high sensitivity and specificity on this new benchmark dataset. 


## Reference
    @paper{zhao2018pyramid,
    title={Airway Anomaly Detection by Prototype-based Graph Neural Network},
    author={Zhao, Tianyi and Yin, Zhaozheng},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention}, 
    year={2021},
    organization={Springer}
    }

## Website
 (http://web.mst.edu/~yinz/
 
 
## Data
The segmented airway mask is given in the data folder, in the NifTi format
The processed feature vectors are saved in the data folder, in the npy format

## Command
To get the feature form the airway mask, run:


Run our pre-trained model:

 
 
 
