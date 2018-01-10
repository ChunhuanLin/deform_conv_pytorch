## PyTorch Implementation of  Deformable Convolution  
This repository implements the defromable convolution architecture proposed in this paper.  
[Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)  

### Usage
* The defromable convolution module, i.e., *DeformConv2D*, is defined in `deform_conv.py`.  
* A simple demo is shown in `demo.py`, it's easy to interpolate the *DeformConv2D* module into your own networks.  

### Statement
* Previous [PyTorch](https://github.com/oeway/pytorch-deform-conv)/[TensorFlow](https://github.com/felixlaumon/deform-conv) implementation are different from the original paper as discussed in this [issue](https://github.com/felixlaumon/deform-conv/issues/4), which motivates me to do a new implementation in this repo.  
* In my opinion, the *DeformConv2D* module is better added to top of higher-level features for the sake of better learning the offsets. More experiments are needed to validate this conjecture.

