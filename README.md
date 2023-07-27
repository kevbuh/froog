# froog <img src="https://github.com/kevbuh/froog/actions/workflows/test.yml/badge.svg" alt="unit test badge" >
<div align="center" >
  <img src="https://raw.githubusercontent.com/kevbuh/froog/main/assets/froog.jpeg" alt="froog the froog" height="300">
  <br/>
  froog: fast real-time optimization of gradients 
  <br/>
  a beautifully compact machine-learning library
  <br/>
  <a href="https://github.com/kevbuh/froog">homepage</a> | <a href="https://github.com/kevbuh/froog/tree/main/docs">documentation</a> | <a href="https://pypi.org/project/froog/">pip</a>
  <br/>
  <br/>
</div>

<!-- modern ml development is unintuitive, time consuming, and unaccessible. why not make it possible for anyone to build? -->
machine learning is like making a lego. combine pieces, of all shapes and colors, to make anything you could possibly imagine. 

froog is making those essential building blocks. 

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
- <a href="https://github.com/kevbuh/froog/blob/main/froog/ops.py">More</a> 

# Bounties
Want to help but don't know where to start? Here are some bounties for you to claim

#### $   <!-- ez money  -->
- binary cross entropy
- flatten
- batch_norm
- pad
- swish
- dropout 
#### $$  <!-- mid tier -->
- simplify how context and gradients are handled
#### $$$ <!-- EXPERT LEVEL!!!  -->
- efficient net
- transformers
- stable Diffusion
- winograd Convs
- MPS support
- CUDA support

# Contributing
Here are some basic guidelines for contributing:
* reduce complexity 
* increase speed
* reduce lines of code (currently at 584 lines of code)
* add features

more info on <a href="https://github.com/kevbuh/froog/blob/main/docs/contributing.md">contributing</a>