# ELU Network for CIFAR-100 classification
This repository contains a partial TensorFlow implementation of the first ELU (Exponential Linear Units) Neural Netowrk as presented in the [ELU Network Article](https://arxiv.org/pdf/1511.07289v5.pdf)

## Disclaimer
This implementation is not fully compatible with the one presented in the artice. First, it only impelements the smaller of the two networks (section 4.2 in the article), also, it does not apply padding with random crops taken (as explained in section 4.2). The code does do a pre-processing of ZCA and whitening.
Creating the full network and adding random cropping is pretty straight forward given the code here.
With this system you should still reach an accuracy of about 64% (comparing to 75.7% in the original article)

## Requirements
Code is written in python (2.7) and requires:
- TensorFlow (0.7 or above), 
- Numpy, 
- MatplotLib
