# frog <img src="https://github.com/kevbuh/frog/actions/workflows/test.yml/badge.svg" alt="unit test badge" >
<div align="center" >
  <img src="https://github.com/kevbuh/frog/blob/main/assets/froog.jpeg" alt="froog the frog" height="300">
  
  <br/>
  frog: fast real-time optimization of gradients 
  <br/>
  <a href="https://github.com/kevbuh/frog/tree/main/docs">documentation</a>
  <br/>
</div>

why does modern ml development have to be so hard? a beautifully compact machine-learning library

### Overview of Features
- Tensors
- Automatic Differentiation
    - Forward and Backward passes
- Input/Grad shape-tracking
- MNIST example
- JIT 2D Convolutions
- Gradient checking
- The most common optimizers (SGD, Adam, RMSProp)

### Math Operations
- Scalar-Matrix Multiplication
- Dot Product
- Sum
- ReLU
- Log Softmax
- 2D Convolution

# Contributing

Pull requests are always welcome.

Here are some basic guidelines for contributing:

Bug fixes are the best and always welcome!
Conceptual cleanups are great.
Features are welcome. Though if you are adding a feature, you need to include tests.

# Bounties

We really want to get a useful model working right out of the box! Our top bounty is to get EfficientNet v2 model working inside of the examples folder.

- EfficientNet v2 **top priority**

### Other bounties

#### Easy
- built in MLP model
- binary cross entropy

#### Medium
- Simplify how context and gradients are handled

#### Hard
- Transformers
- Stable Diffusion
- Winograd Convs
- MPS support
- CUDA support