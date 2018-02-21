## PyTorch Implementation of  Deformable Convolution  
This repository implements the defromable convolution architecture proposed in this paper:  
[*Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu and Yichen Wei. Deformable Convolutional Networks. arXiv preprint arXiv:1703.06211, 2017.*](https://arxiv.org/abs/1703.06211)  

### Usage
* The defromable convolution module, i.e., *DeformConv2D*, is defined in `deform_conv.py`.  
* A simple demo is shown in `demo.py`, it's easy to interpolate the *DeformConv2D* module into your own networks.  

### TODO
 - [x] Memory effeicent implementation.
 - [x] Test against MXNet's official implementation.
 - [ ] Visualize offsets
 - [ ] Demo for RFCN implemantation

### Notes
* Although there has already been some implementations, such as [PyTorch](https://github.com/oeway/pytorch-deform-conv)/[TensorFlow](https://github.com/felixlaumon/deform-conv), they seem to have some problems as discussed [here](https://github.com/felixlaumon/deform-conv/issues/4).  
* In my opinion, the *DeformConv2D* module is better added to top of higher-level features for the sake of better learning the offsets. More experiments are needed to validate this conjecture.
* This repo has been verified by comparing with the official MXNet implementation, as showed in `test_against_mxnet.ipynb`.

### Requirements
* [PyTorch-v0.3.0](http://pytorch.org/docs/0.3.0/)
