# Froog Operations

## Basic Math Operations
- `add` - Addition
- `sub` - Subtraction
- `mul` - Multiplication
- `div` - Division (implemented directly in tensor.py)
- `pow` - Power function
- `sum` - Sum all elements
- `mean` - Mean of all elements (implemented directly in tensor.py)
- `sqrt` - Square root (implemented directly in tensor.py)

## Linear Algebra Operations
- `dot` - Matrix multiplication
- `matmul` - Alias for dot

## Neural Network Operations
- `relu` - Rectified Linear Unit
- `sigmoid` - Sigmoid activation function
- `dropout` - Dropout regularization
- `logsoftmax` - Log of softmax function

## Tensor Manipulation
- `reshape` - Change tensor shape
- `pad2d` - Pad 2D tensors

## Convolution Operations
- `conv2d` - 2D convolution
- `im2col2dconv` - Image to column for convolution

## Pooling Operations
- `max_pool2d` - 2D max pooling
- `avg_pool2d` - 2D average pooling

## Tensor Creation Methods
- `zeros` - Create tensor of zeros
- `ones` - Create tensor of ones
- `randn` - Create tensor with random normal values
- `eye` - Create identity matrix
- `arange` - Create tensor with evenly spaced values

## GPU Support
All core operations have GPU implementations. 