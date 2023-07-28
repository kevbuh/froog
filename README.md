# froog <img src="https://github.com/kevbuh/froog/actions/workflows/test.yml/badge.svg" alt="unit test badge" >
<div align="center" >
  <img src="https://raw.githubusercontent.com/kevbuh/froog/main/assets/froog.png" alt="froog the frog" height="300">
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
<!-- the goal of froog is to make a neural network libary to power any type of device from enterprise to small home robotics -->
<!-- machine learning is like making a lego. you combine standardized pieces, of all shapes and sizes, to create anything you imagine -->
<!-- froog is making those essential building blocks. -->

froog is extremely simple, with a goal of running ml on any device, by any human, easily and efficiently

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
#### Small   <!-- ez money  -->
- binary cross entropy
- flatten
- batch_norm
- div
- pow
- dropout 
#### Medium  <!-- mid tier -->
- start doing ops with opencl
- simplify how context and gradients are handled
- efficient net
#### Large <!-- EXPERT LEVEL!!!  -->
- transformers
- stable diffusion
- winograd convs
- MPS support
- CUDA support

# Contributing
Here are the rules for contributing:
* increase simplicity
* increase efficiency
* increase functionality

more info on <a href="https://github.com/kevbuh/froog/blob/main/docs/contributing.md">contributing</a>