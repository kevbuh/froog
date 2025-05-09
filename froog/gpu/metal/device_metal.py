import time 
from froog.gpu.device import Device

class MetalDevice(Device):
    """GPU device implementation for Apple Metal."""
    
    def __init__(self):
        """Initialize the Metal device and command queue."""
        self.device = None
        self.command_queue = None
        self.buffer_metadata = {}
        
        try:
            import Metal
            import objc
            self.device = Metal.MTLCreateSystemDefaultDevice()
            if self.device is None: raise RuntimeError("No Metal-supported GPU found")
            self.command_queue = self.device.newCommandQueue()
            if self.command_queue is None: raise RuntimeError("Failed to create Metal command queue")
                
        except Exception as e:
            print(f"Metal device initialization failed: {str(e)}")
            self.device = None
            self.command_queue = None
            raise
    
    @property
    def name(self):
        """Return the name of the Metal device."""
        return str(self.device.name()) if self.device else "Unknown Metal Device"
    
    def is_available(self):
        """Check if the Metal device is available and initialized."""
        return self.device is not None and self.command_queue is not None
    
    def allocate_memory(self, size: int):
        """Allocate a Metal buffer of the given size (in bytes) on the GPU."""
        import Metal
        options = Metal.MTLResourceStorageModeShared 
        buffer = self.device.newBufferWithLength_options_(size, options)
        return buffer
    
    def free_memory(self, buffer):
        """Free a Metal buffer."""
        buffer_id = id(buffer)
        if buffer_id in self.buffer_metadata:
            del self.buffer_metadata[buffer_id]
        buffer = None
    
    def upload_tensor(self, host_array) -> object:
        """Upload a NumPy array from host to a new Metal buffer on the device."""
        import numpy as np
        
        host_array = np.ascontiguousarray(host_array, dtype=np.float32)
        
        if np.isnan(host_array).any() or np.isinf(host_array).any():
            print(f"Warning: Array contains NaN or Inf values before uploading to Metal")
            max_val = np.finfo(np.float32).max / 100
            host_array = np.nan_to_num(host_array, nan=0.0, posinf=max_val, neginf=-max_val)
        
        byte_data = host_array.tobytes()
        buffer = self.device.newBufferWithBytes_length_options_(
            byte_data,
            len(byte_data),
            1
        )
        
        if buffer is None: raise RuntimeError("Failed to create Metal buffer")
        
        buffer_id = id(buffer)
        self.buffer_metadata[buffer_id] = {
            'shape': host_array.shape,
            'dtype': host_array.dtype,
            'numpy_array': host_array.copy(),
            'size': host_array.size,
            'upload_time': time.time()
        }
        return buffer
    
    def download_tensor(self, buffer) -> object:
        """Download data from a Metal buffer back to host memory as a NumPy array."""
        import numpy as np
        
        # If Tensor object, get the data
        if hasattr(buffer, "data"):
            buffer = buffer.data
        
        if buffer is None: raise ValueError("Cannot download from a None buffer")
            
        try:
            buffer_id = id(buffer)
            metadata = self.buffer_metadata.get(buffer_id, {})
            
            if 'numpy_array' in metadata:
                return metadata['numpy_array'].copy()
            
            if hasattr(buffer, 'contents'):
                buffer_length = buffer.length()
                float_count = buffer_length // 4
                
                shape = metadata.get('shape', (float_count,))
                dtype = metadata.get('dtype', np.float32)
                
                contents = buffer.contents()
                if contents:
                    import ctypes
                    ptr = ctypes.cast(contents, ctypes.POINTER(ctypes.c_float))
                    result = np.ctypeslib.as_array(ptr, shape=(float_count,))
                    
                    if np.isnan(result).any() or np.isinf(result).any():
                        print(f"Warning: NaN or Inf values detected in buffer {buffer_id}")
                        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if shape != (float_count,):
                        result = result.reshape(shape)
                    
                    # Cache the result for future use
                    if buffer_id not in self.buffer_metadata:
                        self.buffer_metadata[buffer_id] = {}
                    self.buffer_metadata[buffer_id]['numpy_array'] = result.copy()
                    
                    return result
                else:
                    raise ValueError("Buffer contents pointer is null")
            else:
                raise ValueError(f"Buffer does not have 'contents' attribute: {type(buffer)}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to download tensor from Metal: {e}")
    
    def compile_kernel(self, source: str, kernel_name: str):
        """Compile a Metal compute kernel from source code."""
        compile_opts = Metal.MTLCompileOptions.alloc().init()
        library, error = self.device.newLibraryWithSource_options_error_(source, compile_opts, None)
        if library is None:
            err_msg = str(error.localizedDescription()) if error else "Unknown error"
            raise RuntimeError(f"Metal kernel compilation failed: {err_msg}")
        kernel_func = library.newFunctionWithName_(kernel_name)
        if kernel_func is None:
            raise RuntimeError(f"Kernel function '{kernel_name}' not found in compiled library")
        pipeline_state, err2 = self.device.newComputePipelineStateWithFunction_error_(kernel_func, None)
        if pipeline_state is None:
            err_msg = str(err2.localizedDescription()) if err2 else "Unknown error"
            raise RuntimeError(f"Failed to create pipeline state: {err_msg}")
        return pipeline_state
    
    def execute_kernel(self, compiled_kernel, grid_size: tuple, threadgroup_size: tuple, buffers: list):
        """Execute a compiled Metal kernel with the given grid and threadgroup sizes."""
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(compiled_kernel)
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
        global_size = Metal.MTLSizeMake(*grid_size)
        local_size  = Metal.MTLSizeMake(*threadgroup_size)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(global_size, local_size)
        encoder.endEncoding()
        command_buffer.commit()
        self._last_command_buffer = command_buffer
    
    def synchronize(self):
        """Wait for all queued commands to finish execution on the GPU."""
        if hasattr(self, '_last_command_buffer'):
            self._last_command_buffer.waitUntilCompleted()
            del self._last_command_buffer
    
    def get_capabilities(self) -> dict:
        """Return basic information about the Metal device."""
        name = str(self.device.name())
        max_threads = self.device.maxThreadsPerThreadgroup().width
        max_buffer_size = self.device.maxBufferLength()
        return {
            "name": name,
            "max_threads_per_threadgroup": max_threads,
            "max_buffer_size": max_buffer_size
        }
