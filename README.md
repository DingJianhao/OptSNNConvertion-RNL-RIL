# Optimal ANN-SNN Conversion for Fast and Accurate Inferencein Deep Spiking Neural Networks
This directory contains the code of this paper.

To sum up, this paper proposes a method for training ANN, which makes ANN closer to SNN. 
The method is based on the equivalent conversion theory of ANN-SNN and establish an analysis of firing rate approximation.

We propose an optimal fit curve to quantify the fit between the activation value of source ANN and the actual firing rate of target SNN, and derive one upper bound of this convergent curve. We show that based on the Squeeze Theorem, the inference time can be reduced by optimizing the coefficient in the upper bound. These results can not only systematically explain previous findings that reasonable scaling of the threshold can speed up inference, but also give a proper theoretical basis for fast inference research.

We suggest two techniques for ANN training:

- Rate Norm Layer

- Rate Inference Loss

See ``tutorial.py`` for a fast startup. Note that the ANN accuracy actually highly dependent with the data augmentation method. 
User can use your better data augmentation method to get better accuracy as the primary goal of our work is conversion.

