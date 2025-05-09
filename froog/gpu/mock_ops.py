"""
Mock GPU operations for testing.
This module provides dummy GPU implementations that fall back to CPU operations.
"""

import os
import numpy as np
from froog.tensor import Function, register, Tensor

print("Loading mock GPU operations for testing")

# *** Basic Operations ***

class FakeGPUAdd(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x + y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output, grad_output

class FakeGPUSub(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x - y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output, -grad_output

class FakeGPUMul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y * grad_output, x * grad_output

class FakeGPUPow(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return np.power(x, y)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y * np.power(x, y-1) * grad_output, np.log(x) * np.power(x, y) * grad_output

class FakeGPUSum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([np.sum(input)])
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return np.ones_like(input) * grad_output[0]

class FakeGPUReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.maximum(input, 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input > 0)

class FakeGPUDot(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return np.matmul(input, weight)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = np.matmul(grad_output, weight.T)
        grad_weight = np.matmul(input.T, grad_output)
        return grad_input, grad_weight

class FakeGPUMaxPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size=(2,2), stride=None):
        if stride is None:
            stride = kernel_size
        
        N, C, H, W = x.shape
        kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        sH, sW = stride if isinstance(stride, tuple) else (stride, stride)
        
        outH = (H - kH) // sH + 1
        outW = (W - kW) // sW + 1
        output = np.zeros((N, C, outH, outW))
        
        for n in range(N):
            for c in range(C):
                for h in range(outH):
                    for w in range(outW):
                        h_start = h * sH
                        h_end = h_start + kH
                        w_start = w * sW
                        w_end = w_start + kW
                        window = x[n, c, h_start:h_end, w_start:w_end]
                        output[n, c, h, w] = np.max(window)
        
        ctx.save_for_backward(x, np.array(kernel_size), np.array(stride))
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified backward pass
        x, kernel_size, stride = ctx.saved_tensors
        N, C, H, W = x.shape
        kH, kW = kernel_size
        sH, sW = stride
        
        outH = (H - kH) // sH + 1
        outW = (W - kW) // sW + 1
        
        grad_input = np.zeros_like(x)
        
        for n in range(N):
            for c in range(C):
                for h in range(outH):
                    for w in range(outW):
                        h_start = h * sH
                        h_end = h_start + kH
                        w_start = w * sW
                        w_end = w_start + kW
                        
                        window = x[n, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        grad_input[n, c, h_start + max_idx[0], w_start + max_idx[1]] += grad_output[n, c, h, w]
        
        return grad_input

class FakeGPUAvgPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size=(2,2), stride=None):
        if stride is None:
            stride = kernel_size
        
        N, C, H, W = x.shape
        kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        sH, sW = stride if isinstance(stride, tuple) else (stride, stride)
        
        outH = (H - kH) // sH + 1
        outW = (W - kW) // sW + 1
        output = np.zeros((N, C, outH, outW))
        
        for n in range(N):
            for c in range(C):
                for h in range(outH):
                    for w in range(outW):
                        h_start = h * sH
                        h_end = h_start + kH
                        w_start = w * sW
                        w_end = w_start + kW
                        window = x[n, c, h_start:h_end, w_start:w_end]
                        output[n, c, h, w] = np.mean(window)
        
        ctx.save_for_backward(x, np.array(kernel_size), np.array(stride))
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, kernel_size, stride = ctx.saved_tensors
        N, C, H, W = x.shape
        kH, kW = kernel_size
        sH, sW = stride
        
        outH = (H - kH) // sH + 1
        outW = (W - kW) // sW + 1
        
        grad_input = np.zeros_like(x)
        
        for n in range(N):
            for c in range(C):
                for h in range(outH):
                    for w in range(outW):
                        h_start = h * sH
                        h_end = h_start + kH
                        w_start = w * sW
                        w_end = w_start + kW
                        
                        grad_val = grad_output[n, c, h, w] / (kH * kW)
                        grad_input[n, c, h_start:h_end, w_start:w_end] += grad_val
        
        return grad_input

class FakeGPUPad2d(Function):
    @staticmethod
    def forward(ctx, x, padding):
        left, right, top, bottom = padding
        N, C, H, W = x.shape
        new_H = H + top + bottom
        new_W = W + left + right
        output = np.zeros((N, C, new_H, new_W))
        output[:, :, top:top+H, left:left+W] = x
        
        ctx.save_for_backward(np.array(padding))
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        padding, = ctx.saved_tensors
        left, right, top, bottom = padding
        
        return grad_output[:, :, top:grad_output.shape[2]-bottom, left:grad_output.shape[3]-right]

class FakeGPUReshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(np.array(x.shape))
        return x.reshape(shape)
    
    @staticmethod
    def backward(ctx, grad_output):
        orig_shape, = ctx.saved_tensors
        return grad_output.reshape(orig_shape)

class FakeGPUDropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, training=True):
        ctx.save_for_backward(x)
        ctx.p = p
        ctx.training = training
        
        if training:
            mask = np.random.random(x.shape) > p
            output = x * mask / (1 - p)
            ctx.mask = mask
        else:
            output = x
            
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        
        if ctx.training:
            return grad_output * ctx.mask / (1 - ctx.p)
        else:
            return grad_output

# Register operations
print("Registering mock GPU operations")
register("add", FakeGPUAdd, gpu=True)
register("sub", FakeGPUSub, gpu=True)
register("mul", FakeGPUMul, gpu=True)
register("pow", FakeGPUPow, gpu=True)
register("sum", FakeGPUSum, gpu=True)
register("relu", FakeGPUReLU, gpu=True)
register("dot", FakeGPUDot, gpu=True)
register("max_pool2d", FakeGPUMaxPool2d, gpu=True)
register("avg_pool2d", FakeGPUAvgPool2d, gpu=True)
register("pad2d", FakeGPUPad2d, gpu=True)
register("reshape", FakeGPUReshape, gpu=True)
register("dropout", FakeGPUDropout, gpu=True)

# Run this immediately when imported
print(f"Tensor.ops_gpu keys: {list(Tensor.ops_gpu.keys())}") 