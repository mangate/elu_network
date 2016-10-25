# ELU Network for CIFAR-100 classification
This repository contains a partial TensorFlow implementation of the first ELU (Exponential Linear Units) Neural Netowrk as presented in Clevert's at el. [ELU Network Article](https://arxiv.org/pdf/1511.07289v5.pdf)

## Disclaimer
This implementation is not fully compatible with the one presented in the artice. First, it only impelements the smaller of the two networks (section 4.2 in the article), also, it does not apply padding with random crops taken (as explained in section 4.2). The code does do a pre-processing of ZCA and whitening.
Creating the full network and adding random cropping is pretty straight forward given the code here.
With this system you should still reach an accuracy of about 64% (comparing to 75.7% in the original article)

## Requirements
- python (2.7)
- TensorFlow (0.7 or above)
- Numpy 
- MatplotLib
- PyLearn2
- cPickle

## Usage
1) Download CIFAR-100 data from [here](https://www.cs.toronto.edu/~kriz/cifar.html)

2) Data files (meta,test,train,and file.txt) should be placed under `<some_dir>/cifar-100/cifar-100-python/`

3) set `$PYLEARN2_DATA_PATH` to `<some_dir>` from step 2

4) Run once: `python process_cifar_100_data.py` to whiten and do ZCA on the data

5) On `open_cifar_100_data.py` change `ROOT_FOLDER` to the folder where the whitenend (and ZCA'd) data was saved

5) Now you can use the model by `python elu_network.py`

## References
[Fast and Accurate Deep Network Learning By Exponential Linear Units (ELU's)](https://arxiv.org/pdf/1511.07289v5.pdf)

[Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
