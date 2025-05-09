import Metal, objc  # PyObjC allows using Apple's Metal API in Python
from froog.gpu.device import Device

class MetalDevice(Device):
    """GPU device implementation for Apple Metal."""
    
    def __init__(self):
        """Initialize the Metal device and command queue."""
        # Obtain the default Metal device (GPU). This returns an MTLDevice object.
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None: raise RuntimeError("No Metal-supported GPU found")
        # Create a command queue for submitting commands to the GPU
        self.command_queue = self.device.newCommandQueue()
    
    def allocate_memory(self, size: int):
        """Allocate a Metal buffer of the given size (in bytes) on the GPU."""
        # Create an MTLBuffer with the specified length. Using shared storage so CPU can access it if needed.
        options = Metal.MTLResourceStorageModeShared 
        buffer = self.device.newBufferWithLength_options_(size, options)
        return buffer  # MTLBuffer object
    
    def free_memory(self, buffer):
        """Free a Metal buffer. (In PyObjC, buffers are reference-counted objects.)"""
        # To free the buffer, release its reference (PyObjC manages ref counting; use objc.release for safety).
        objc.release(buffer)
    
    def upload_tensor(self, host_array) -> object:
        """Upload a NumPy array (or similar) from host to a new Metal buffer on the device."""
        data_bytes = host_array.tobytes()  # raw bytes of the host array
        buffer = self.allocate_memory(len(data_bytes))
        # Copy data into the GPU buffer by writing to its contents
        ptr = buffer.contents() # void* pointer to the buffer's memory
        import ctypes
        ctypes.memmove(ptr, data_bytes, len(data_bytes))
        return buffer
    
    def download_tensor(self, buffer) -> object:
        """Download data from a Metal buffer back to host memory as a NumPy array."""
        length = buffer.length() # size of the buffer in bytes
        import ctypes, numpy as np
        # Allocate a ctypes buffer to receive the data
        dest = (ctypes.c_byte * length)()
        ctypes.memmove(dest, buffer.contents(), length)
        # Convert to NumPy array (assuming float32 data for this example)
        result = np.frombuffer(dest, dtype=np.float32).copy()  # copy to detach from ctypes buffer
        return result
    
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
