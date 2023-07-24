# frog <img src="https://github.com/kevbuh/frog/actions/workflows/test.yml/badge.svg" alt="unit test badge" >
<div align="center" >
  <img src="https://raw.githubusercontent.com/kevbuh/frog/main/assets/froog.jpeg" alt="froog the frog" height="300">
  <br/>
  frog: fast real-time optimization of gradients 
  <br/>
  a beautifully compact machine-learning library
  <br/>
  <a href="https://github.com/kevbuh/frog">homepage</a> | <a href="https://github.com/kevbuh/frog/tree/main/docs">documentation</a> | <a href="https://github.com/kevbuh/frog/tree/main/examples">examples</a> | <a href="https://pypi.org/project/froog/">pip</a>
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
- Numerical gradient checking
- The most common optimizers (SGD, Adam, RMSProp)

### Math Operations
- Scalar-Matrix Multiplication
- Dot Product
- Sum
- ReLU
- Log Softmax
- 2D Convolutions
- Avg & Max pooling
- <a href="https://github.com/kevbuh/frog/blob/main/frog/ops.py">More</a> 

# Bounties
Want to help but don't know where to start? 

Our top bounty is to get EfficientNet v2 model working inside of the <a href="https://github.com/kevbuh/frog/tree/main/examples">examples</a> folder.

#### Easy
- built in MLP model
- binary cross entropy
- dropout layer
- flatten
#### Medium
- simplify how context and gradients are handled
#### Hard
- efficientNet
- transformers
- stable Diffusion
- winograd Convs
- MPS support
- CUDA support

# Contributing
here are some basic guidelines for contributing:
* reduce complexity (currently at 585 lines of code)
* increase speed
* add features, must include <a href="https://github.com/kevbuh/frog/tree/main/tests">tests</a>
* in that order

more info on <a href="https://github.com/kevbuh/frog/blob/main/docs/contributing.md">contributing</a>