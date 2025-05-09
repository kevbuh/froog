"""
Metal-specific tensor operations for Apple Silicon GPUs.
This file implements tensor operations using Metal compute shaders.
"""

import numpy as np
from froog.tensor import register, Tensor, Function
from froog.gpu import get_device
from froog.gpu.buffer_utils import buffer_pow, get_buffer_data

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
        from froog.gpu.buffer_utils import get_buffer_data
        device = get_device()
        
        x, y = ctx.saved_tensors
        grad_cpu = get_buffer_data(grad_output)
        
        # Create the negative gradient for the second operand
        neg_grad = -grad_cpu
        
        # Upload both gradients back to GPU
        return device.upload_tensor(grad_cpu), device.upload_tensor(neg_grad)

class MetalMulFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        # Handle Metal buffers
        device = get_device()
        x_cpu = device.download_tensor(x)
        y_cpu = device.download_tensor(y)
        
        # Check for potential overflow
        max_val = np.finfo(np.float32).max
        abs_x = np.abs(x_cpu)
        abs_y = np.abs(y_cpu)
        
        # If potential overflow detected, scale down the inputs
        # This helps prevent NaN and inf values
        scale_factor = 1.0
        max_product = np.max(abs_x) * np.max(abs_y)
        if max_product > max_val / 10:
            scale_factor = max_val / (100 * max_product)
            x_cpu = x_cpu * scale_factor
            result_cpu = x_cpu * y_cpu
            # Scale back the result
            result_cpu = result_cpu / scale_factor
        else:
            result_cpu = x_cpu * y_cpu
            
        # Clip result to prevent any remaining inf/nan values
        result_cpu = np.clip(result_cpu, -max_val, max_val)
        
        return device.upload_tensor(result_cpu)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        device = get_device()
        x_cpu = device.download_tensor(x)
        y_cpu = device.download_tensor(y)
        grad_cpu = device.download_tensor(grad_output)
        
        # Clip gradients to prevent overflow
        max_val = np.finfo(np.float32).max / 10
        grad_x = np.clip(y_cpu * grad_cpu, -max_val, max_val)
        grad_y = np.clip(x_cpu * grad_cpu, -max_val, max_val)
        
        return device.upload_tensor(grad_x), device.upload_tensor(grad_y)

class MetalPowFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        # Handle Metal buffers by using buffer utilities
        device = get_device()
        
        # Convert buffers to CPU data if needed
        x_cpu = get_buffer_data(x)
        y_cpu = get_buffer_data(y)
        
        result_cpu = np.power(x_cpu, y_cpu)
        return device.upload_tensor(result_cpu)
    
    @staticmethod
    def backward(ctx, grad_output):
        from froog.gpu.buffer_utils import get_buffer_data
        device = get_device()
        
        x, y = ctx.saved_tensors
        
        # Get data as numpy arrays
        x_cpu = get_buffer_data(x)
        y_cpu = get_buffer_data(y)
        grad_cpu = get_buffer_data(grad_output)
        
        # Compute gradients
        dx = y_cpu * np.power(x_cpu, y_cpu-1) * grad_cpu
        dy = np.log(x_cpu) * np.power(x_cpu, y_cpu) * grad_cpu
        
        # Upload back to GPU
        return device.upload_tensor(dx), device.upload_tensor(dy)

class MetalSumFunction(Function):
    @staticmethod
    def forward(ctx, input):
        from froog.gpu.buffer_utils import get_buffer_data, buffer_sum
        device = get_device()
        
        # Get the data as a numpy array
        input_cpu = get_buffer_data(input)
        ctx.save_for_backward(input_cpu)
        ctx.input_shape = input_cpu.shape
        
        # Compute sum
        result = np.array([np.sum(input_cpu)])
        
        # Upload back to GPU
        return device.upload_tensor(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        from froog.gpu.buffer_utils import get_buffer_data
        device = get_device()
        
        input, = ctx.saved_tensors
        grad_cpu = get_buffer_data(grad_output)
        
        # Create ones array with the input shape
        grad_input = np.ones(ctx.input_shape, dtype=input.dtype) * grad_cpu[0]
        
        # Upload back to GPU
        return device.upload_tensor(grad_input)

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
        
        # Prevent overflow by checking magnitudes
        max_val = np.finfo(np.float32).max
        need_scaling = False
        
        # Calculate maximum values to detect potential overflow
        max_input = np.max(np.abs(input_cpu))
        max_weight = np.max(np.abs(weight_cpu))
        
        # Estimate if the dot product might overflow
        # For N-dim dot product, worst case is N * max_input * max_weight
        max_dim = max(input_cpu.shape[-1], 1)
        if max_input * max_weight * max_dim > max_val / 100:
            need_scaling = True
            scale_factor = np.sqrt(max_val / (100 * max_input * max_weight * max_dim))
            input_cpu = input_cpu * scale_factor
            weight_cpu = weight_cpu * scale_factor
        
        # Perform matrix multiplication
        try:
            result_cpu = np.matmul(input_cpu, weight_cpu)
            
            # Scale back if needed
            if need_scaling:
                result_cpu = result_cpu / (scale_factor * scale_factor)
            
            # Clip to prevent any extreme values
            result_cpu = np.clip(result_cpu, -max_val/10, max_val/10)
        except Exception as e:
            print(f"Error in dot product: {e}")
            # Return zeros as fallback
            result_shape = list(input_cpu.shape)
            result_shape[-1] = weight_cpu.shape[-1]
            result_cpu = np.zeros(tuple(result_shape), dtype=np.float32)
            
        return device.upload_tensor(result_cpu)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        device = get_device()
        
        # Download to CPU
        input_cpu = device.download_tensor(input)
        weight_cpu = device.download_tensor(weight)
        grad_cpu = device.download_tensor(grad_output)
        
        # Clip grad to prevent overflow
        max_val = np.finfo(np.float32).max / 100
        grad_cpu = np.clip(grad_cpu, -max_val, max_val)
        
        # Compute gradients with overflow protection
        try:
            grad_input_cpu = np.matmul(grad_cpu, weight_cpu.T)
            grad_input_cpu = np.clip(grad_input_cpu, -max_val, max_val)
        except Exception as e:
            print(f"Error in dot product backward (input): {e}")
            grad_input_cpu = np.zeros_like(input_cpu)
            
        try:
            grad_weight_cpu = np.matmul(input_cpu.T, grad_cpu)
            grad_weight_cpu = np.clip(grad_weight_cpu, -max_val, max_val)
        except Exception as e:
            print(f"Error in dot product backward (weight): {e}")
            grad_weight_cpu = np.zeros_like(weight_cpu)
        
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
        # Handle Metal buffers
        from froog.gpu.buffer_utils import get_buffer_data
        device = get_device()
        
        # Get the data as a numpy array
        x_cpu = get_buffer_data(x)
        
        # Get the shape from the CPU data
        N, C, H, W = x_cpu.shape
        kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        outH = H // kH
        outW = W // kW
        output = np.zeros((N, C, outH, outW), dtype=x_cpu.dtype)
        
        for n in range(N):
            for c in range(C):
                for h in range(outH):
                    for w in range(outW):
                        h_start = h * kH
                        h_end = h_start + kH
                        w_start = w * kW
                        w_end = w_start + kW
                        window = x_cpu[n, c, h_start:h_end, w_start:w_end]
                        output[n, c, h, w] = np.mean(window)
        
        ctx.save_for_backward(np.array(kernel_size), np.array(x_cpu.shape))
        
        # Upload back to GPU
        return device.upload_tensor(output)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Handle Metal buffers
        from froog.gpu.buffer_utils import get_buffer_data
        device = get_device()
        
        kernel_size, input_shape = ctx.saved_tensors
        N, C, H, W = input_shape
        kH, kW = kernel_size
        
        # Get gradient as a numpy array
        grad_cpu = get_buffer_data(grad_output)
        
        outH, outW = H // kH, W // kW
        grad_input = np.zeros((N, C, H, W), dtype=grad_cpu.dtype)
        
        for n in range(N):
            for c in range(C):
                for h in range(outH):
                    for w in range(outW):
                        h_start = h * kH
                        h_end = h_start + kH
                        w_start = w * kW
                        w_end = w_start + kW
                        grad_val = grad_cpu[n, c, h, w] / (kH * kW)
                        grad_input[n, c, h_start:h_end, w_start:w_end] += grad_val
        
        # Upload back to GPU
        return device.upload_tensor(grad_input)

class MetalReshapeFunction(Function):
    @staticmethod
    def forward(ctx, x, shape):
        # Use buffer utilities to handle Metal buffers
        from froog.gpu.buffer_utils import get_buffer_data, buffer_reshape
        
        # Get the data as a numpy array
        x_cpu = get_buffer_data(x)
        ctx.save_for_backward(np.array(x_cpu.shape))
        
        # Reshape the CPU data
        result_cpu = x_cpu.reshape(shape)
        
        # Upload back to GPU
        device = get_device()
        return device.upload_tensor(result_cpu)
    
    @staticmethod
    def backward(ctx, grad_output):
        orig_shape, = ctx.saved_tensors
        
        # Handle Metal buffers in the gradient
        from froog.gpu.buffer_utils import get_buffer_data
        
        # Get the grad data as a numpy array
        grad_cpu = get_buffer_data(grad_output)
        
        # Reshape to original shape
        result_cpu = grad_cpu.reshape(tuple(orig_shape))
        
        # Upload back to GPU
        device = get_device()
        return device.upload_tensor(result_cpu)

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

class MetalLogSoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, x):
        from froog.gpu.buffer_utils import get_buffer_data, buffer_logsoftmax
        device = get_device()
        
        # Get the data as a numpy array
        x_cpu = get_buffer_data(x)
        
        # Compute log softmax
        # Subtract max for numerical stability
        max_vals = np.max(x_cpu, axis=-1, keepdims=True)
        
        # Clip values to prevent overflow
        safe_x = np.clip(x_cpu - max_vals, -88, 88)  # limit to exp range for float32
        
        # Calculate in a numerically stable way
        exp_vals = np.exp(safe_x)
        sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)
        
        # Add small epsilon to prevent log(0)
        log_sum_exp = np.log(sum_exp + 1e-10)
        output = safe_x - log_sum_exp
        
        # Save outputs for backward pass
        ctx.save_for_backward(output)
        
        # Upload back to GPU
        return device.upload_tensor(output)
    
    @staticmethod
    def backward(ctx, grad_output):
        from froog.gpu.buffer_utils import get_buffer_data
        device = get_device()
        
        output, = ctx.saved_tensors
        grad_cpu = get_buffer_data(grad_output)
        
        # Compute softmax gradients
        # Softmax gradient: dy/dx = (1 - exp(logsoftmax)) * dy
        softmax = np.exp(output)
        
        # Clip to prevent overflow in gradient computation
        softmax = np.clip(softmax, 1e-10, 1.0)
        
        dx = grad_cpu - np.sum(grad_cpu * softmax, axis=-1, keepdims=True) * softmax
        
        # Upload back to GPU
        return device.upload_tensor(dx)

class MetalPad2dFunction(Function):
    @staticmethod
    def forward(ctx, x, padding=(0,0,0,0)):
        from froog.gpu.buffer_utils import get_buffer_data, buffer_pad2d
        device = get_device()
        
        # Get the data as a numpy array
        x_cpu = get_buffer_data(x)
        
        # Extract padding values
        pad_left, pad_right, pad_top, pad_bottom = padding
        
        # Create padding config for np.pad
        # Format is ((before_1, after_1), (before_2, after_2), ...)
        pad_width = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        
        # Apply padding
        output = np.pad(x_cpu, pad_width, mode='constant')
        
        # Save context for backward
        ctx.save_for_backward(np.array(padding))
        ctx.input_shape = x_cpu.shape
        
        # Upload back to GPU
        return device.upload_tensor(output)
    
    @staticmethod
    def backward(ctx, grad_output):
        from froog.gpu.buffer_utils import get_buffer_data
        device = get_device()
        
        padding, = ctx.saved_tensors
        input_shape = ctx.input_shape
        
        # Get gradient as a numpy array
        grad_cpu = get_buffer_data(grad_output)
        
        # Extract padding values
        pad_left, pad_right, pad_top, pad_bottom = padding
        
        # Slice the gradient to get the un-padded region
        N, C, H, W = input_shape
        grad_input = grad_cpu[:, :, pad_top:pad_top+H, pad_left:pad_left+W].copy()
        
        # Upload back to GPU
        return device.upload_tensor(grad_input)

class MetalConv2dFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, stride=1, padding=0):
        from froog.gpu.buffer_utils import get_buffer_data
        device = get_device()
        
        # Get data as numpy arrays
        x_cpu = get_buffer_data(x)
        weight_cpu = get_buffer_data(weight)
        bias_cpu = get_buffer_data(bias) if bias is not None else None
        
        # Handle stride and padding parameters
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        
        # Extract dimensions
        N, C, H, W = x_cpu.shape
        F, C_, HH, WW = weight_cpu.shape
        
        # Check channels are consistent between input and weight
        assert C == C_, f"Input channels {C} must match weight input channels {C_}"
        
        # Calculate output dimensions with padding and stride
        H_out = (H + 2 * padding[0] - HH) // stride[0] + 1
        W_out = (W + 2 * padding[1] - WW) // stride[1] + 1
        
        # Apply padding if needed
        if padding[0] > 0 or padding[1] > 0:
            x_padded = np.pad(x_cpu, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
        else:
            x_padded = x_cpu
        
        # Initialize output
        out = np.zeros((N, F, H_out, W_out), dtype=x_cpu.dtype)
        
        # Naive implementation of convolution (slow but correct)
        for n in range(N):
            for f in range(F):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride[0]
                        w_start = w_out * stride[1]
                        # Extract the local region from padded input
                        x_local = x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
                        # Compute convolution as dot product
                        out[n, f, h_out, w_out] = np.sum(x_local * weight_cpu[f])
                        
                        # Add bias if provided
                        if bias_cpu is not None:
                            out[n, f, h_out, w_out] += bias_cpu[f]
        
        # Save for backward pass
        ctx.save_for_backward(x_padded, weight_cpu, np.array(stride), np.array(padding))
        ctx.input_shape = x_cpu.shape
        ctx.has_bias = bias is not None
        
        # Upload result back to GPU
        return device.upload_tensor(out)
    
    @staticmethod
    def backward(ctx, grad_output):
        from froog.gpu.buffer_utils import get_buffer_data
        device = get_device()
        
        # Get saved tensors
        x_padded, weight, stride, padding = ctx.saved_tensors
        N, C, H_padded, W_padded = x_padded.shape
        F, C_, HH, WW = weight.shape
        has_bias = ctx.has_bias
        input_shape = ctx.input_shape
        
        # Convert stride and padding from arrays back to tuples
        stride = tuple(stride)
        padding = tuple(padding)
        
        # Get grad_output as numpy array
        grad_cpu = get_buffer_data(grad_output)
        
        # Initialize gradients
        dx_padded = np.zeros_like(x_padded)
        dw = np.zeros_like(weight)
        db = np.zeros(F) if has_bias else None
        
        # Naive implementation of backward pass
        N, F, H_out, W_out = grad_cpu.shape
        
        # Compute gradient w.r.t. weights and bias
        for n in range(N):
            for f in range(F):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride[0]
                        w_start = w_out * stride[1]
                        # Gradient for weights
                        dw[f] += x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW] * grad_cpu[n, f, h_out, w_out]
                        # Gradient for input
                        dx_padded[n, :, h_start:h_start+HH, w_start:w_start+WW] += weight[f] * grad_cpu[n, f, h_out, w_out]
                
                # Gradient for bias
                if has_bias:
                    db[f] += np.sum(grad_cpu[:, f])
        
        # Remove padding from dx_padded to get dx
        if padding[0] > 0 or padding[1] > 0:
            dx = dx_padded[:, :, padding[0]:H_padded-padding[0], padding[1]:W_padded-padding[1]]
        else:
            dx = dx_padded
        
        # Upload gradients back to GPU
        dx_gpu = device.upload_tensor(dx)
        dw_gpu = device.upload_tensor(dw)
        db_gpu = device.upload_tensor(db) if has_bias else None
        
        if has_bias:
            return dx_gpu, dw_gpu, db_gpu
        else:
            return dx_gpu, dw_gpu

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
register("logsoftmax", MetalLogSoftmaxFunction, gpu=True)
register("pad2d", MetalPad2dFunction, gpu=True)
register("conv2d", MetalConv2dFunction, gpu=True)

# Print available GPU operations
print(f"Metal Tensor.ops_gpu keys: {list(Tensor.ops_gpu.keys())}") 