# frog <img src="https://github.com/kevbuh/frog/actions/workflows/test.yml/badge.svg" alt="unit test badge" >
<div align="center" >
  <img src="https://raw.githubusercontent.com/kevbuh/frog/main/assets/froog.jpeg" alt="froog the frog" height="300">
  <br/>
  frog: fast real-time optimization of gradients 
  <br/>
  a beautifully compact machine-learning library
  <br/>
  <a href="https://github.com/kevbuh/frog">homepage</a> | <a href="https://github.com/kevbuh/frog/tree/main/docs">documentation</a> | <a href="https://github.com/kevbuh/frog/tree/main/examples">examples</a>| <a href="https://pypi.org/project/froog/">pip</a>
  <br/>
  <br/>
</div>

<!-- modern ml development is unintuitive, time consuming, and unaccessible. why not make it possible for anyone to build? -->
# Installation
```bash
pip install froog
```

### Overview of Features
- Tensors
- Automatic Differentiation
    - Forward and backward passes
- Input/gradient shape-tracking
- MNIST example
- 2D Convolutions (im2col)
- Gradient checking
- The most common optimizers (SGD, Adam, RMSProp)

### Math Operations
- Scalar-Matrix Multiplication
- Dot Product
- Sum
- ReLU
- Log Softmax
- 2D Convolution

# Bounties
We really want to get a useful model working right out of the box! Our top bounty is to get EfficientNet v2 model working inside of the <a href="https://github.com/kevbuh/frog/tree/main/examples">examples</a>  folder.

- EfficientNet v2 (**top priority**)

#### Easy
- built in MLP model
- binary cross entropy
- dropout layer
- flatten

#### Medium
- simplify how context and gradients are handled

#### Hard
- Transformers
- Stable Diffusion
- Winograd Convs
- MPS support
- CUDA support

# Contributing

Here are some basic guidelines for contributing:

* Reduce code, currently at 585 lines
* Increase speed
* Add features
* In that order

Bug fixes are the best and always welcome
Conceptual cleanups are great
All features must include <a href="https://github.com/kevbuh/frog/tree/main/tests">tests</a>