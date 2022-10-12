# galaxy_spin_classifier

Morphology classifiers for galaxy spin research.


## Installation

Just install it like usual python packages. If you use pip, you can do

```
pip install -e .
```


## Usage

Please check the example notebooks for the usage of the CE-ResNet classifier.
Basically, it expects a 3x160x160 image with FOV roughly equals to 3x galaxy half light size, and
returns the emperical probability scores for the galaxy to be a Z-Spiral, S-Spiral and Non-Spiral,
respectively.


## References

* He Jia, Hong-Ming Zhu and Ue-Li Pen,
*Galaxy Spin Classification I: Z-wise vs S-wise Spirals With Chirality Equivariant Residual Network*,
[2210.04168](https://arxiv.org/abs/2210.04168)
