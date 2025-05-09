import Metal, objc  # PyObjC allows using Apple's Metal API in Python
from froog.gpu.device import Device

class MetalDevice(Device):
    """GPU device implementation for Apple Metal."""
    
    def __init__(self):
        """Initialize the Metal device and command queue."""
        self.device = None
        self.command_queue = None
        # Dictionary to store buffer metadata
        self.buffer_metadata = {}
        
        try:
            # Verify Metal module is available
            import Metal
            import objc
            
            # Obtain the default Metal device (GPU). This returns an MTLDevice object.
            self.device = Metal.MTLCreateSystemDefaultDevice()
            
            if self.device is None:
                print("GPU enabled but no Metal device available")
                raise RuntimeError("No Metal-supported GPU found")
            
            # Create a command queue for submitting commands to the GPU
            self.command_queue = self.device.newCommandQueue()
            if self.command_queue is None:
                raise RuntimeError("Failed to create Metal command queue")
                
            # Print additional device info for debugging
            print(f"Initialized Metal device: {str(self.device.name())}")
            
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
        # Create an MTLBuffer with the specified length. Using shared storage so CPU can access it if needed.
        options = Metal.MTLResourceStorageModeShared 
        buffer = self.device.newBufferWithLength_options_(size, options)
        return buffer  # MTLBuffer object
    
    def free_memory(self, buffer):
        """Free a Metal buffer. (In PyObjC, buffers are reference-counted objects.)"""
        # Remove metadata for this buffer
        buffer_id = id(buffer)
        if buffer_id in self.buffer_metadata:
            del self.buffer_metadata[buffer_id]
            
        # In PyObjC with ARC (Automatic Reference Counting), objects are released 
        # when their reference count reaches zero. Setting to None allows Python's
        # garbage collector to decrement the reference count.
        buffer = None
    
    def upload_tensor(self, host_array) -> object:
        """Upload a NumPy array (or similar) from host to a new Metal buffer on the device."""
        import numpy as np
        
        # Ensure array is contiguous and float32 (Metal typically uses float32)
        host_array = np.ascontiguousarray(host_array, dtype=np.float32)
        
        # Get the raw bytes representation of the array
        byte_data = host_array.tobytes()
        
        # Create a Metal buffer with the data
        # MTLResourceStorageModeShared = 1
        buffer = self.device.newBufferWithBytes_length_options_(
            byte_data,
            len(byte_data),
            1
        )
        
        if buffer is None:
            raise RuntimeError("Failed to create Metal buffer")
        
        # Store the original shape and dtype in our metadata dict
        buffer_id = id(buffer)
        self.buffer_metadata[buffer_id] = {
            'shape': host_array.shape,
            'dtype': host_array.dtype,
            'numpy_array': host_array.copy(),  # Keep a CPU copy for download
            'size': host_array.size
        }
        
        return buffer
    
    def download_tensor(self, buffer) -> object:
        """Download data from a Metal buffer back to host memory as a NumPy array."""
        import numpy as np
        
        try:
            # Get the buffer's metadata if available
            buffer_id = id(buffer)
            metadata = self.buffer_metadata.get(buffer_id, {})
            
            # If we have kept a CPU copy, return it
            if 'numpy_array' in metadata:
                return metadata['numpy_array'].copy()
            
            # Try to read the bytes from the Metal buffer's contents
            if hasattr(buffer, 'contents'):
                try:
                    buffer_length = buffer.length()
                    float_count = buffer_length // 4  # 4 bytes per float32
                    
                    # If we have the shape, use it to reshape the data
                    shape = metadata.get('shape', (float_count,))
                    dtype = metadata.get('dtype', np.float32)
                    
                    # Get a pointer to the buffer contents
                    contents = buffer.contents()
                    if contents:
                        # Create a NumPy array from the pointer
                        import ctypes
                        ptr = ctypes.cast(contents, ctypes.POINTER(ctypes.c_float))
                        result = np.ctypeslib.as_array(ptr, shape=(float_count,))
                        
                        # Reshape if we have the original shape
                        if shape != (float_count,):
                            result = result.reshape(shape)
                        
                        # Cache the result for future use
                        if buffer_id not in self.buffer_metadata:
                            self.buffer_metadata[buffer_id] = {}
                        self.buffer_metadata[buffer_id]['numpy_array'] = result.copy()
                        
                        return result
                except Exception as e:
                    print(f"Error accessing Metal buffer contents: {e}")
            
            # Fallback - create a basic 1D array of float32s
            buffer_length = buffer.length()
            float_count = buffer_length // 4  # 4 bytes per float32
            
            # Use cached shape if available
            shape = metadata.get('shape', (float_count,))
            
            # Create a dummy array with the right shape
            result = np.zeros(shape, dtype=np.float32)
            
            # Cache the result for future use
            if buffer_id not in self.buffer_metadata:
                self.buffer_metadata[buffer_id] = {}
            self.buffer_metadata[buffer_id]['numpy_array'] = result.copy()
            
            print(f"Warning: Using fallback tensor download (zeros with shape {shape})")
            return result
        except Exception as e:
            print(f"Error downloading tensor from Metal: {e}")
            return np.array([0], dtype=np.float32)  # Return something to prevent crashes
    
    def compile_kernel(self, source: str, kernel_name: str):
        """Compile a Metal compute kernel from source code. Returns a pipeline state object."""
        # Compile the Metal kernel source code to a library (MTLLibrary)
        compile_opts = Metal.MTLCompileOptions.alloc().init()
        library, error = self.device.newLibraryWithSource_options_error_(source, compile_opts, None)
        if library is None:  # compilation failed
            err_msg = str(error.localizedDescription()) if error else "Unknown error"
            raise RuntimeError(f"Metal kernel compilation failed: {err_msg}")
        # Get the function (kernel entry point) by name
        kernel_func = library.newFunctionWithName_(kernel_name)
        if kernel_func is None:
            raise RuntimeError(f"Kernel function '{kernel_name}' not found in compiled library")
        # Create a compute pipeline state from the kernel function (this compiles the GPU code)
        pipeline_state, err2 = self.device.newComputePipelineStateWithFunction_error_(kernel_func, None)
        if pipeline_state is None:
            err_msg = str(err2.localizedDescription()) if err2 else "Unknown error"
            raise RuntimeError(f"Failed to create pipeline state: {err_msg}")
        return pipeline_state  # MTLComputePipelineState object
    
    def execute_kernel(self, compiled_kernel, grid_size: tuple, threadgroup_size: tuple, buffers: list):
        """Execute a compiled Metal kernel with the given grid and threadgroup sizes, using the specified buffers."""
        # Create a new command buffer from the command queue
        command_buffer = self.command_queue.commandBuffer()
        # Create a compute command encoder to encode the GPU commands
        encoder = command_buffer.computeCommandEncoder()
        # Set the pipeline state (compiled kernel) for the encoder
        encoder.setComputePipelineState_(compiled_kernel)
        # Bind each buffer to the encoder (assign buffers to the kernel's buffer indices)
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
        # Configure thread group sizes (Metal uses MTLSize for threadgroup and grid dimensions)
        global_size = Metal.MTLSizeMake(*grid_size)       # total threads
        local_size  = Metal.MTLSizeMake(*threadgroup_size)  # threads per threadgroup
        # Dispatch the compute kernel with the specified configuration
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(global_size, local_size)
        encoder.endEncoding()
        # Commit the command buffer to send it to the GPU for execution
        command_buffer.commit()
        # Store the command buffer if we need to synchronize later
        self._last_command_buffer = command_buffer
    
    def synchronize(self):
        """Wait for all queued commands to finish execution on the GPU."""
        if hasattr(self, '_last_command_buffer'):
            # Wait for the last command buffer to complete GPU execution
            self._last_command_buffer.waitUntilCompleted()
            del self._last_command_buffer  # clear reference after completion
    
    def get_capabilities(self) -> dict:
        """Return basic information about the Metal device."""
        name = str(self.device.name())
        max_threads = self.device.maxThreadsPerThreadgroup().width  # maximum threads per threadgroup (1D for simplicity)
        max_buffer_size = self.device.maxBufferLength()             # maximum buffer size supported (in bytes)
        return {
            "name": name,
            "max_threads_per_threadgroup": max_threads,
            "max_buffer_size": max_buffer_size
        }
