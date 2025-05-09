"""
Metal-specific tensor operations for Apple Silicon GPUs.
This file implements tensor operations using Metal compute shaders.
"""

import numpy as np
from froog.tensor import register, Tensor, Function
from froog.gpu import get_device

# Print that we're loading Metal operations
print("Loading Metal-specific GPU operations")

class MetalAddFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        # We need to handle Metal buffers correctly
        device = get_device()
        
        # Download tensors to CPU, perform operation, and upload result
        x_cpu = device.download_tensor(x)
        y_cpu = device.download_tensor(y)
        
        # Perform operation on CPU
        result_cpu = x_cpu + y_cpu
        
        # Upload result back to GPU
        result_gpu = device.upload_tensor(result_cpu)
        
        return result_gpu
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output, grad_output

class MetalSubFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        # Handle Metal buffers
        device = get_device()
        x_cpu = device.download_tensor(x)
        y_cpu = device.download_tensor(y)
        result_cpu = x_cpu - y_cpu
        return device.upload_tensor(result_cpu)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output, -grad_output

class MetalMulFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        # Handle Metal buffers
        device = get_device()
        x_cpu = device.download_tensor(x)
        y_cpu = device.download_tensor(y)
        result_cpu = x_cpu * y_cpu
        return device.upload_tensor(result_cpu)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        device = get_device()
        x_cpu = device.download_tensor(x)
        y_cpu = device.download_tensor(y)
        grad_cpu = device.download_tensor(grad_output)
        return device.upload_tensor(y_cpu * grad_cpu), device.upload_tensor(x_cpu * grad_cpu)

class MetalPowFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return np.power(x, y)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y * np.power(x, y-1) * grad_output, np.log(x) * np.power(x, y) * grad_output

class MetalSumFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([np.sum(input)])
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return np.ones_like(input) * grad_output[0]

class MetalReLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Handle Metal buffers
        device = get_device()
        input_cpu = device.download_tensor(input)
        result_cpu = np.maximum(input_cpu, 0)
        return device.upload_tensor(result_cpu)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        device = get_device()
        input_cpu = device.download_tensor(input)
        grad_cpu = device.download_tensor(grad_output)
        return device.upload_tensor(grad_cpu * (input_cpu > 0))

class MetalSigmoidFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = 1/(1 + np.exp(-input))
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output * (output * (1 - output))

class MetalDotFunction(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        # Handle Metal buffers
        device = get_device()
        input_cpu = device.download_tensor(input)
        weight_cpu = device.download_tensor(weight)
        result_cpu = np.matmul(input_cpu, weight_cpu)
        return device.upload_tensor(result_cpu)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        device = get_device()
        
        # Download to CPU
        input_cpu = device.download_tensor(input)
        weight_cpu = device.download_tensor(weight)
        grad_cpu = device.download_tensor(grad_output)
        
        # Compute gradients
        grad_input_cpu = np.matmul(grad_cpu, weight_cpu.T)
        grad_weight_cpu = np.matmul(input_cpu.T, grad_cpu)
        
        # Upload back to GPU
        return device.upload_tensor(grad_input_cpu), device.upload_tensor(grad_weight_cpu)

class MetalMaxPool2dFunction(Function):
    @staticmethod
    def forward(ctx, x, kernel_size=(2,2)):
        # Handle Metal buffers
        device = get_device()
        x_cpu = device.download_tensor(x)
        
        # Get the metadata for shape
        buffer_id = id(x)
        metadata = device.buffer_metadata.get(buffer_id, {})
        x_shape = metadata.get('shape')
        
        # If we don't have shape metadata, get it from the downloaded tensor
        if x_shape is None:
            x_shape = x_cpu.shape
            
        # Simplified implementation of max pooling
        N, C, H, W = x_shape
        kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        outH = H // kH
        outW = W // kW
        output = np.zeros((N, C, outH, outW), dtype=x_cpu.dtype)
        max_indices = np.zeros((N, C, outH, outW, 2), dtype=np.int32)
        
        for n in range(N):
            for c in range(C):
                for h in range(outH):
                    for w in range(outW):
                        h_start = h * kH
                        h_end = h_start + kH
                        w_start = w * kW
                        w_end = w_start + kW
                        window = x_cpu[n, c, h_start:h_end, w_start:w_end]
                        flat_idx = np.argmax(window)
                        max_y, max_x = np.unravel_index(flat_idx, (kH, kW))
                        max_indices[n, c, h, w] = [max_y, max_x]
                        output[n, c, h, w] = window[max_y, max_x]
        
        ctx.save_for_backward(x, max_indices, np.array(kernel_size))
        return device.upload_tensor(output)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, max_indices, kernel_size = ctx.saved_tensors
        device = get_device()
        
        # Get the CPU data
        x_cpu = device.download_tensor(x)
        grad_cpu = device.download_tensor(grad_output)
        
        # Get shape from metadata or downloaded tensor
        buffer_id = id(x)
        metadata = device.buffer_metadata.get(buffer_id, {})
        x_shape = metadata.get('shape', x_cpu.shape)
        
        N, C, H, W = x_shape
        kH, kW = kernel_size
        
        outH, outW = H // kH, W // kW
        grad_input = np.zeros_like(x_cpu)
        
        for n in range(N):
            for c in range(C):
                for h in range(outH):
                    for w in range(outW):
                        h_start = h * kH
                        w_start = w * kW
                        max_y, max_x = max_indices[n, c, h, w]
                        grad_input[n, c, h_start + max_y, w_start + max_x] += grad_cpu[n, c, h, w]
        
        return device.upload_tensor(grad_input)

class MetalAvgPool2dFunction(Function):
    @staticmethod
    def forward(ctx, x, kernel_size=(2,2)):
        N, C, H, W = x.shape
        kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        outH = H // kH
        outW = W // kW
        output = np.zeros((N, C, outH, outW))
        
        for n in range(N):
            for c in range(C):
                for h in range(outH):
                    for w in range(outW):
                        h_start = h * kH
                        h_end = h_start + kH
                        w_start = w * kW
                        w_end = w_start + kW
                        window = x[n, c, h_start:h_end, w_start:w_end]
                        output[n, c, h, w] = np.mean(window)
        
        ctx.save_for_backward(np.array(kernel_size), x.shape)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        kernel_size, input_shape = ctx.saved_tensors
        N, C, H, W = input_shape
        kH, kW = kernel_size
        
        outH, outW = H // kH, W // kW
        grad_input = np.zeros((N, C, H, W))
        
        for n in range(N):
            for c in range(C):
                for h in range(outH):
                    for w in range(outW):
                        h_start = h * kH
                        h_end = h_start + kH
                        w_start = w * kW
                        w_end = w_start + kW
                        grad_val = grad_output[n, c, h, w] / (kH * kW)
                        grad_input[n, c, h_start:h_end, w_start:w_end] += grad_val
        
        return grad_input

class MetalReshapeFunction(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(np.array(x.shape))
        return x.reshape(shape)
    
    @staticmethod
    def backward(ctx, grad_output):
        orig_shape, = ctx.saved_tensors
        return grad_output.reshape(tuple(orig_shape))

class MetalDropoutFunction(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, training=True):
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
        if ctx.training:
            return grad_output * ctx.mask / (1 - ctx.p)
        else:
            return grad_output

# Register Metal operations
print("Registering Metal GPU operations")
register("add", MetalAddFunction, gpu=True)
register("sub", MetalSubFunction, gpu=True)
register("mul", MetalMulFunction, gpu=True)
register("pow", MetalPowFunction, gpu=True)
register("sum", MetalSumFunction, gpu=True)
register("relu", MetalReLUFunction, gpu=True)
register("sigmoid", MetalSigmoidFunction, gpu=True)
register("dot", MetalDotFunction, gpu=True)
register("matmul", MetalDotFunction, gpu=True)
register("max_pool2d", MetalMaxPool2dFunction, gpu=True)
register("avg_pool2d", MetalAvgPool2dFunction, gpu=True)
register("reshape", MetalReshapeFunction, gpu=True)
register("dropout", MetalDropoutFunction, gpu=True)

# Print available GPU operations
print(f"Metal Tensor.ops_gpu keys: {list(Tensor.ops_gpu.keys())}") 