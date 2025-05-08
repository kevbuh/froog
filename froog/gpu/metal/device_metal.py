import numpy as np
from typing import Any, Tuple, Optional, Union, List, Dict, Callable
import functools

from ..device import Device

# Check if Metal is available
try:
    import objc
    import Metal
    import MetalPerformanceShaders
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

class MetalBuffer:
    """Wrapper class for Metal buffers with shape information."""
    
    def __init__(self, buffer, shape, dtype=np.float32):
        self.buffer = buffer
        self.shape = shape
        self.dtype = dtype
    
    def contents(self):
        return self.buffer.contents()
    
    def length(self):
        return self.buffer.length()

class MetalDevice(Device):
    """Metal device implementation for Apple platforms."""
    
    def __init__(self):
        """Initialize the Metal device."""
        self.device = None
        self.command_queue = None
        self.library = None
        
        if METAL_AVAILABLE:
            try:
                self.device = Metal.MTLCreateSystemDefaultDevice()
                self.command_queue = self.device.newCommandQueue()
                # Create a default library from the bundle
                self.library = self.device.newDefaultLibrary()
            except Exception:
                # Metal initialization failed
                pass
    
    @property
    def name(self) -> str:
        """Return the device name."""
        return "Metal"
    
    def is_available(self) -> bool:
        """Check if the Metal device is available."""
        return METAL_AVAILABLE and self.device is not None and self.command_queue is not None
    
    def tensor_to_device(self, data: np.ndarray) -> Any:
        """Convert CPU data to Metal buffer."""
        if not self.is_available():
            raise Exception("No Metal support! Install pyobjc, pyobjc-framework-Metal, and pyobjc-framework-MetalPerformanceShaders")
        
        # Make sure data is float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Create a buffer with the data
        data_bytes = data.tobytes()
        buffer = self.device.newBufferWithBytes_length_options_(
            data_bytes, 
            len(data_bytes), 
            Metal.MTLResourceStorageModeShared
        )
        
        # Create a wrapper that stores shape information
        return MetalBuffer(buffer, data.shape, data.dtype)
    
    def tensor_to_cpu(self, tensor: Any) -> np.ndarray:
        """Convert a Metal buffer to CPU."""
        if not isinstance(tensor, MetalBuffer):
            raise TypeError(f"Expected MetalBuffer, got {type(tensor)}")
            
        # Get the buffer contents as bytes
        buffer_length = tensor.buffer.length()
        buffer_ptr = tensor.buffer.contents()
        
        # Create a bytes object from the buffer
        buffer_bytes = bytes(buffer_ptr[:buffer_length])
        
        # Convert bytes to numpy array
        result = np.frombuffer(buffer_bytes, dtype=np.float32)
        return result.reshape(tensor.shape)
    
    def is_device_tensor(self, data: Any) -> bool:
        """Check if data is a Metal buffer."""
        return METAL_AVAILABLE and isinstance(data, MetalBuffer)
    
    def get_size(self, x: Any) -> int:
        """Return the total number of elements in x."""
        return int(np.prod(x.shape))
    
    def buffer_new(self, shape: Tuple[int, ...]) -> Any:
        """Create a new empty Metal buffer with the given shape."""
        size = int(np.prod(shape)) * 4  # 4 bytes per float32
        buffer = self.device.newBufferWithLength_options_(size, Metal.MTLResourceStorageModeShared)
        return MetalBuffer(buffer, shape, np.float32)
    
    def buffer_zeros(self, shape: Tuple[int, ...]) -> Any:
        """Create a new Metal buffer filled with zeros."""
        data = np.zeros(shape, dtype=np.float32)
        return self.tensor_to_device(data)
    
    def buffer_like(self, x: Any) -> Any:
        """Create a new Metal buffer with the same shape as x."""
        return self.buffer_new(x.shape)
    
    @functools.lru_cache
    def build_program(self, code: str) -> Any:
        """Build a Metal program/kernel from source code."""
        if not self.is_available():
            return None
            
        # Create a new library from the metal shader code
        options = Metal.MTLCompileOptions.alloc().init()
        library, error = self.device.newLibraryWithSource_options_error_(code, options, None)
        
        if library is None:
            raise Exception(f"Failed to compile Metal library: {error}")
            
        return library
    
    def binary_op(self, code: str, x: Any, y: Any) -> Any:
        """Apply a binary operation to two Metal buffers."""
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
        
        # Metal shader code
        metal_code = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void binop(device const float *a_g [[buffer(0)]],
                          device const float *b_g [[buffer(1)]],
                          device float *res_g [[buffer(2)]],
                          constant int &xdiv [[buffer(3)]],
                          constant int &ydiv [[buffer(4)]],
                          uint gid [[thread_position_in_grid]])
        {
          float a = a_g[gid/xdiv];
          float b = b_g[gid/ydiv];
          res_g[gid] = """ + code + """;
        }
        """
        
        library = self.build_program(metal_code)
        function = library.newFunctionWithName_("binop")
        pipeline_state = self.device.newComputePipelineStateWithFunction_error_(function, None)[0]
        
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        
        compute_encoder.setComputePipelineState_(pipeline_state)
        compute_encoder.setBuffer_offset_atIndex_(x.buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(y.buffer, 0, 1)
        compute_encoder.setBuffer_offset_atIndex_(ret.buffer, 0, 2)
        
        # Set constants
        xdiv_buffer = np.array([xdiv], dtype=np.int32)
        ydiv_buffer = np.array([ydiv], dtype=np.int32)
        compute_encoder.setBytes_length_atIndex_(xdiv_buffer, 4, 3)
        compute_encoder.setBytes_length_atIndex_(ydiv_buffer, 4, 4)
        
        # Calculate threads and threadgroups
        threads_per_group = min(pipeline_state.maxTotalThreadsPerThreadgroup(), 256)
        threadgroups = (self.get_size(ret) + threads_per_group - 1) // threads_per_group
        
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(threadgroups, 1, 1),
            Metal.MTLSizeMake(threads_per_group, 1, 1)
        )
        
        compute_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        return ret
    
    def unary_op(self, code: str, x: Any) -> Any:
        """Apply a unary operation to a Metal buffer."""
        ret = self.buffer_like(x)
        
        # Metal shader code
        metal_code = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void unop(device const float *a_g [[buffer(0)]],
                         device float *res_g [[buffer(1)]],
                         uint gid [[thread_position_in_grid]])
        {
          float a = a_g[gid];
          res_g[gid] = """ + code + """;
        }
        """
        
        library = self.build_program(metal_code)
        function = library.newFunctionWithName_("unop")
        pipeline_state = self.device.newComputePipelineStateWithFunction_error_(function, None)[0]
        
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        
        compute_encoder.setComputePipelineState_(pipeline_state)
        compute_encoder.setBuffer_offset_atIndex_(x.buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(ret.buffer, 0, 1)
        
        # Calculate threads and threadgroups
        threads_per_group = min(pipeline_state.maxTotalThreadsPerThreadgroup(), 256)
        threadgroups = (self.get_size(ret) + threads_per_group - 1) // threads_per_group
        
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(threadgroups, 1, 1),
            Metal.MTLSizeMake(threads_per_group, 1, 1)
        )
        
        compute_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        return ret
    
    @functools.lru_cache
    def _build_pooling_kernel(self, iter_op: str, result_op: str, init_val: Union[int, str] = 0) -> dict:
        """Build a Metal kernel for pooling operations."""
        # Metal shader code for pooling
        metal_code = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void subsample(
            device float *output [[buffer(0)]],
            device const float *input [[buffer(1)]],
            constant uint2 &osize [[buffer(2)]],
            constant uint2 &isize [[buffer(3)]],
            constant uint2 &kernel_size [[buffer(4)]],
            constant int &nelem [[buffer(5)]],
            uint3 gid [[thread_position_in_grid]])
        {
            int oid = gid.x + osize.x*(gid.y + osize.y*gid.z);
            float group_res = """ + str(init_val) + """;
            
            for (uint j=0; j<kernel_size.y; ++j) {
                for (uint i=0; i<kernel_size.x; ++i) {
                    int iid = (gid.x*kernel_size.x+i) + isize.x*((gid.y*kernel_size.y+j) + isize.y*gid.z);
                    if (iid < nelem)
                        """ + iter_op + """;
                }
            }
            output[oid] = """ + result_op + """;
        }
        """
        
        library = self.build_program(metal_code)
        function = library.newFunctionWithName_("subsample")
        pipeline_state = self.device.newComputePipelineStateWithFunction_error_(function, None)[0]
        
        return {
            "pipeline_state": pipeline_state,
            "library": library,
            "function": function
        }
    
    def pooling_op(self, input: Any, kernel_size: Tuple[int, int], 
                  iter_op: str, result_op: str, init_val: Union[int, str] = 0) -> Any:
        """Apply a pooling operation to a Metal buffer."""
        N, C, Y, X = input.shape
        py, px = kernel_size
        ret = self.buffer_new((N, C, Y//py, X//px))
        
        kernel = self._build_pooling_kernel(iter_op, result_op, init_val=init_val)
        pipeline_state = kernel["pipeline_state"]
        
        # Setup buffers for constants
        osize = np.array([X//px, Y//py], dtype=np.uint32)
        isize = np.array([X, Y], dtype=np.uint32)
        ksize = np.array([px, py], dtype=np.uint32)
        nelem = np.array([input.size], dtype=np.int32)
        
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        
        compute_encoder.setComputePipelineState_(pipeline_state)
        compute_encoder.setBuffer_offset_atIndex_(ret.buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(input.buffer, 0, 1)
        compute_encoder.setBytes_length_atIndex_(osize, 8, 2)  # uint2 = 8 bytes
        compute_encoder.setBytes_length_atIndex_(isize, 8, 3)
        compute_encoder.setBytes_length_atIndex_(ksize, 8, 4)
        compute_encoder.setBytes_length_atIndex_(nelem, 4, 5)
        
        # Calculate threads and threadgroups
        compute_encoder.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(X//px, Y//py, N*C),
            Metal.MTLSizeMake(1, 1, 1)
        )
        
        compute_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        return ret 