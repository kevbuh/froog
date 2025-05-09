import numpy as np
from typing import Any, Tuple, Optional, Union, List, Dict, Callable
import functools

from ..device import Device

# check if OpenCL is available
try:
    import pyopencl as cl
    CL_AVAILABLE = True
except ImportError:
    cl = None
    CL_AVAILABLE = False

class OpenCLDevice(Device):
    """OpenCL device implementation."""
    
    def __init__(self):
        """Initialize the OpenCL device."""
        self.cl_ctx = None
        self.cl_queue = None
        if CL_AVAILABLE:
            try:
                self.cl_ctx = cl.create_some_context(answers=[0]) 
            except (cl._cl.RuntimeError, TypeError):
                self.cl_ctx = cl.create_some_context(interactive=False)
            self.cl_queue = cl.CommandQueue(self.cl_ctx)
    
    @property
    def name(self) -> str:
        """Return the device name."""
        return "OpenCL"
    
    def is_available(self) -> bool:
        """Check if the OpenCL device is available."""
        return CL_AVAILABLE and self.cl_ctx is not None and self.cl_queue is not None
    
    # Device class abstract method implementations
    
    def allocate_memory(self, size: int):
        """Allocate memory on the device."""
        if not self.is_available():
            raise Exception("No OpenCL support! Install pyopencl")
        buffer = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_WRITE, size * 4)  # Assuming float32 (4 bytes)
        return buffer
    
    def free_memory(self, buffer):
        """Free device memory."""
        # OpenCL buffers are automatically garbage collected
        pass
    
    def upload_tensor(self, host_array) -> object:
        """Copy data from host to device."""
        return self.tensor_to_device(host_array)
    
    def download_tensor(self, buffer) -> object:
        """Copy data from device to host."""
        return self.tensor_to_cpu(buffer)
    
    def compile_kernel(self, source: str, kernel_name: str) -> object:
        """Compile a kernel from source code."""
        program = cl.Program(self.cl_ctx, source).build()
        return getattr(program, kernel_name)
    
    def execute_kernel(self, compiled_kernel, grid_size: tuple, threadgroup_size: tuple, buffers: list):
        """Execute a kernel on the device."""
        global_size = grid_size
        local_size = threadgroup_size
        event = compiled_kernel(self.cl_queue, global_size, local_size, *buffers)
        return event
    
    def synchronize(self):
        """Wait for all pending operations to complete."""
        self.cl_queue.finish()
    
    def get_capabilities(self) -> dict:
        """Query device capabilities."""
        if not self.is_available():
            return {"name": "None", "available": False}
        
        device = self.cl_ctx.devices[0]
        return {
            "name": device.name,
            "available": True,
            "max_work_group_size": device.max_work_group_size,
            "max_compute_units": device.max_compute_units
        }
    
    # OpenCL-specific methods
    
    def tensor_to_device(self, data: np.ndarray) -> Any:
        """Convert CPU data to OpenCL buffer."""
        if not self.is_available(): raise Exception("No OpenCL support! Install pyopencl")
        
        assert data.dtype == np.float32  # GPU only allows float32
        gpu_buffer = cl.Buffer(
            self.cl_ctx, 
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
            hostbuf=data.ravel()
        )
        gpu_buffer.shape = data.shape
        gpu_buffer.dtype = data.dtype
        return gpu_buffer
    
    def tensor_to_cpu(self, tensor: Any) -> np.ndarray:
        """Convert an OpenCL buffer to CPU."""
        if hasattr(tensor, "data"): tensor = tensor.data
        data = np.empty(tensor.shape, dtype=np.float32)
        cl.enqueue_copy(self.cl_queue, data, tensor)
        return data
    
    def is_buffer(self, data: Any) -> bool:
        """Check if data is an OpenCL buffer."""
        return CL_AVAILABLE and isinstance(data, cl._cl.Buffer)
    
    def get_size(self, x: Any) -> int:
        """Return the total number of elements in x."""
        return int(np.prod(x.shape))
    
    def buffer_new(self, shape: Tuple[int, ...]) -> Any:
        """Create a new empty OpenCL buffer with the given shape."""
        res_g = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, 4*self.get_size(shape))
        res_g.shape = shape
        res_g.dtype = np.float32
        return res_g
    
    def buffer_zeros(self, shape: Tuple[int, ...]) -> Any:
        """Create a new OpenCL buffer filled with zeros."""
        res_g = cl.Buffer(
            self.cl_ctx, 
            cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, 
            hostbuf=np.zeros(shape)
        )
        res_g.shape = shape
        res_g.dtype = np.float32
        return res_g
    
    def buffer_like(self, x: Any) -> Any:
        """Create a new OpenCL buffer with the same shape as x."""
        return self.buffer_new(x.shape)
    
    @functools.lru_cache
    def build_program(self, code: str) -> Any:
        """Build an OpenCL program."""
        if self.is_available():
            return cl.Program(self.cl_ctx, code).build()
        return None
    
    def binary_op(self, code: str, x: Any, y: Any) -> Any:
        """Apply a binary operation to two OpenCL buffers."""
        xdiv = 1
        ydiv = 1
        if x.shape != y.shape:
            # special case broadcasting
            if len(y.shape) == 4 and x.shape[0:2] == y.shape[0:2] and y.shape[2] == 1 and y.shape[3] == 1:
                ydiv = x.shape[2] * x.shape[3]
            elif len(y.shape) == 4 and x.shape[0:2] == y.shape[0:2] and x.shape[2] == 1 and x.shape[3] == 1:
                xdiv = y.shape[2] * y.shape[3]
            elif self.get_size(y) == 1:
                ydiv = self.get_size(x)
            else:
                raise Exception(f"binary op shape mismatch: {x.shape} != {y.shape}")
        
        ret = self.buffer_like(x if self.get_size(x) >= self.get_size(y) else y)
        prg = self.build_program("""
        __kernel void binop(  __global const float *a_g, __global const float *b_g, __global float *res_g, int xdiv, int ydiv) {
          int gid = get_global_id(0);
          float a = a_g[gid/xdiv];
          float b = b_g[gid/ydiv];
          res_g[gid] = """+code+""";
        }
        """)
        prg.binop(self.cl_queue, [self.get_size(ret)], None, x, y, ret, np.int32(xdiv), np.int32(ydiv))
        return ret
    
    def unary_op(self, code: str, x: Any) -> Any:
        """Apply a unary operation to an OpenCL buffer."""
        ret = self.buffer_like(x)
        prg = self.build_program("""
        __kernel void unop(
            __global const float *a_g, __global float *res_g)
        {
          int gid = get_global_id(0);
          float a = a_g[gid];
          res_g[gid] = """+code+""";
        }
        """)
        prg.unop(self.cl_queue, [self.get_size(ret)], None, x, ret)
        return ret
    
    @functools.lru_cache
    def _build_pooling_kernel(self, iter_op: str, result_op: str, init_val: Union[int, str] = 0) -> Any:
        """Build a pooling kernel."""
        code = f"""
        __kernel void pool2d(
            __global const float *input, 
            __global float *output,
            int isize, int xsize, int ysize, 
            int kernsize, int stride) 
        {{
            int x = get_global_id(0); int y = get_global_id(1); int b = get_global_id(2);
            if (x >= xsize || y >= ysize) return;
            
            int start_idx = b * isize * isize;
            int out_idx = (b * ysize * xsize) + (y * xsize) + x;
            
            int y_start = y * stride;
            int x_start = x * stride;
            int y_end = min(y_start + kernsize, isize);
            int x_end = min(x_start + kernsize, isize);
            
            float result = {init_val};
            for (int j = y_start; j < y_end; j++) {{
                for (int i = x_start; i < x_end; i++) {{
                    int in_idx = start_idx + j * isize + i;
                    result = {iter_op};
                }}
            }}
            output[out_idx] = {result_op};
        }}
        """
        return self.build_program(code)
    
    def pooling_op(self, input: Any, kernel_size: Tuple[int, int], 
                  iter_op: str, result_op: str, init_val: Union[int, str] = 0) -> Any:
        """Apply a pooling operation to an OpenCL buffer."""
        N, C, Y, X = input.shape
        py, px = kernel_size
        ret = self.buffer_new((N, C, Y//py, X//px))
        osize = np.array((X//px, Y//py), dtype=cl.cltypes.uint2)
        isize = np.array((X, Y), dtype=cl.cltypes.uint2)
        ksize = np.array((px, py), dtype=cl.cltypes.uint2)
        prg = self._build_pooling_kernel(iter_op, result_op, init_val=init_val)
        prg.subsample(self.cl_queue, (N*C, Y//py, X//px), None, ret, input, osize, isize, ksize, np.int32(input.size))
        return ret 