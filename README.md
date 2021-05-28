# Airway-Anomaly-Detection-by-Prototype-based-Graph-Neural-Network

This is the code associated with the paper "Airway Anomaly Detection by Prototype-based Graph Neural Network"

##Abstract
Detecting the airway anomaly can be an essential part to aid the lung disease diagnosis. Since normal human airways share an anatomical structure, we design a graph prototype whose structure follows the normal airway anatomy. Then, we learn  the prototype and a graph neural network from a weakly-supervised airway dataset, i.e., only the holistic label is available, indicating if the airway has anomaly or not, but which bronchus node has the anomaly is unknown. During inference, the graph neural network predicts the anomaly score at both the holistic level and node-level of an airway. We initialize the airway anomaly detection problem by creating a large airway dataset with 2589 samples, and our prototype-based graph neural network shows high sensitivity and specificity on this new benchmark dataset. 


##Reference
@paper{zhao2021airwayanomaly,
title={Airway Anomaly Detection by Prototype-based Graph Neural Network},
author={Zhao, Tianyi and Yin, Zhaozheng},
booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention}, 
year={2021},
organization={Springer}
}
