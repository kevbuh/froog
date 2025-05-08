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
    
    def tensor_to_device(self, data: np.ndarray) -> Any:
        """Convert CPU data to OpenCL buffer."""
        if not self.is_available():
            raise Exception("No OpenCL support! Install pyopencl")
        
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
        data = np.empty(tensor.shape, dtype=np.float32)
        cl.enqueue_copy(self.cl_queue, data, tensor)
        return data
    
    def is_device_tensor(self, data: Any) -> bool:
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
        """Build an OpenCL kernel for pooling operations."""
        prg = """
        __kernel void subsample(
          __global float *output, __global const float *input, uint2 osize, uint2 isize, uint2 kernel_size, int nelem
        ) {
          int3 gid = (int3)(get_global_id(2), get_global_id(1), get_global_id(0));
          int oid = gid.x + osize.x*(gid.y + osize.y*gid.z);
          float group_res = """+str(init_val)+""";
          for (uint j=0; j<kernel_size.y; ++j) {
            for (uint i=0; i<kernel_size.x; ++i) {
              int iid  = (gid.x*kernel_size.x+i) + isize.x*((gid.y*kernel_size.y+j) + isize.y*gid.z);
              if (iid < nelem)
                """+iter_op+""";
            }
          }
          output[oid] = """+result_op+""";
        }
        """
        return self.build_program(prg)
    
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