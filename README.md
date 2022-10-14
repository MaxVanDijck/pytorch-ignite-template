# PyTorch-Ignite-Template
### An extensible deep learning template for research and production

## Introduction
Without a good starting point, research code often becomes unorganized and difficult to reproduce. This template provides a foundation for code that is configurable, understandable and most importantly, reduces long term [suffering](http://karpathy.github.io/2019/04/25/recipe/).

## Getting Started
```bash
# Create a Conda environment
# (You can use other Virtual Environments of your choosing)
conda create --name my-awesome-project python=3.10 -y
conda activate my-awesome-project

# install requirements
pip install -r requirements.txt

# test by running the CIFAR-10 example
python train.py
```

![](https://github.com/Lune-AI/pytorch-ignite-template/blob/resources/ignite-diagram.svg)

## Citing this Template
If you use this template in your research please use the following BibTeX entries to recognise not only us, but the brilliant minds at Hydra, PyTorch & Ignite

##### NN Template
```
@Misc{nn-template
  author =       {M. van Dijck},
  title =        {NN Template: An extensible deep learning template for research and production},
  year =         {2022},
  publisher =    {Github},
  journal =      {Github repository},
  howpublished = {\url{https://github.com/Lune-AI/ignite-hydra-template}}
}
```
##### Hydra
```
@Misc{Yadan2019Hydra,
  author =       {Omry Yadan},
  title =        {Hydra - A framework for elegantly configuring complex applications},
  howpublished = {Github},
  year =         {2019},
  url =          {https://github.com/facebookresearch/hydra}
}
```
##### PyTorch
```
@incollection{NEURIPS2019_9015,
title =     {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
author =    {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
booktitle = {Advances in Neural Information Processing Systems 32},
editor =    {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages =     {8024--8035},
year =      {2019},
publisher = {Curran Associates, Inc.},
url =       {http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf}
}
```
##### Ignite
```
@misc{pytorch-ignite,
  author =       {V. Fomin and J. Anmol and S. Desroziers and J. Kriss and A. Tejani},
  title =        {High-level library to help with training neural networks in PyTorch},
  year =         {2020},
  publisher =    {GitHub},
  journal =      {GitHub repository},
  howpublished = {\url{https://github.com/pytorch/ignite}},
}
```
