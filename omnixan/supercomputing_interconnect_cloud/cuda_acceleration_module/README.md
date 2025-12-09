# CUDA Acceleration Module for Omnixan

## Overview

Production-ready CUDA acceleration module for the Omnixan Supercomputing Interconnect Cloud Block. This module provides high-performance GPU computing capabilities with support for multiple GPU backends, multi-device operations, and advanced memory management.

## Features

### Core Capabilities
- **Multi-Backend Support**: Automatic selection between CuPy and PyCUDA backends
- **Multi-GPU Operations**: Parallel execution across multiple GPU devices
- **Stream Processing**: Concurrent kernel execution using CUDA streams
- **Memory Management**: Efficient GPU memory allocation and pooling
- **Performance Profiling**: Built-in performance monitoring and metrics
- **Tensor Core Integration**: Interface for tensor core module integration
- **Error Handling**: Comprehensive error handling and logging

### Methods

#### Initialization
```python
from cuda_acceleration_module import CUDAAccelerationModule, GPUBackend

# Auto-detect backend
module = CUDAAccelerationModule(
    backend=GPUBackend.AUTO,
    enable_profiling=True
)

# Initialize
module.initialize()
```

#### Memory Operations
```python
# Allocate GPU memory
gpu_array = module.allocate_gpu_memory(
    size=1000000,
    device_id=0,
    dtype=np.float32
)

# Copy data (host to device)
gpu_data = module.copy_data(
    src=cpu_array,
    async_copy=True,
    stream=stream
)

# Copy data (device to host)
cpu_data = module.copy_data(src=gpu_array)
```

#### Kernel Execution
```python
# CUDA kernel code
kernel_code = '''
extern "C" __global__ void vector_add(
    const float* a, const float* b, float* c, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
'''

# Launch kernel
module.launch_kernel(
    kernel_code=kernel_code,
    kernel_name="vector_add",
    grid=(grid_size,),
    block=(block_size,),
    args=(a_gpu, b_gpu, c_gpu, n),
    shared_mem=0
)
```

#### Multi-GPU Execution
```python
def process_chunk(data_chunk, device_id):
    # Process data on specific GPU
    module.select_device(device_id)
    result = module.launch_kernel(...)
    return result

# Execute across multiple GPUs
results = module.execute_multi_gpu(
    func=process_chunk,
    data=data_chunks,
    device_ids=[0, 1, 2, 3]
)
```

#### Stream Operations
```python
# Use stream context for concurrent operations
with module.stream_context(device_id=0, stream_id=1) as stream:
    result = module.copy_data(
        src=data,
        stream=stream,
        async_copy=True
    )
```

#### GPU Statistics
```python
# Get GPU utilization and memory stats
stats = module.get_gpu_stats(device_id=0)
print(f"Memory Used: {stats['used_memory_gb']:.2f} GB")
print(f"Memory Utilization: {stats['memory_utilization_%']:.1f}%")
```

#### Performance Profiling
```python
# Profile operations
with module.profiler("matrix_multiply"):
    result = module.launch_kernel(...)

# View metrics
for op_name, metrics in module.metrics.items():
    print(f"{op_name}: {metrics.total_time_ms:.3f} ms")
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA Toolkit 11.0+
- NVIDIA GPU with compute capability 3.5+

### Install Dependencies

#### Option 1: CuPy (Recommended)
```bash
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

#### Option 2: PyCUDA
```bash
pip install pycuda
```

#### Additional Requirements
```bash
pip install numpy
```

## Integration with Omnixan

### Directory Structure
```
omnixan/
└── supercomputing_interconnect_cloud/
    ├── cuda_acceleration_module/
    │   ├── __init__.py
    │   ├── cuda_acceleration_module.py
    │   ├── requirements.txt
    │   └── examples/
    │       ├── vector_operations.py
    │       ├── matrix_multiplication.py
    │       └── multi_gpu_example.py
    └── tensor_core_module/
        └── (tensor core integration)
```

### Usage in Omnixan Framework
```python
from omnixan.supercomputing_interconnect_cloud.cuda_acceleration_module import (
    CUDAAccelerationModule
)

# Initialize within Omnixan infrastructure
class SupercomputingInterconnect:
    def __init__(self):
        self.cuda_module = CUDAAccelerationModule(
            enable_profiling=True
        )
        self.cuda_module.initialize()

    def execute_quantum_circuit(self, circuit):
        # Accelerate quantum circuit execution on GPU
        with self.cuda_module.profiler("quantum_circuit"):
            result = self.cuda_module.launch_kernel(...)
        return result
```

## Advanced Usage

### Custom Kernel Compilation Options
```python
module.launch_kernel(
    kernel_code=kernel_code,
    kernel_name="custom_kernel",
    grid=(grid_x, grid_y),
    block=(block_x, block_y),
    args=kernel_args,
    compile_options=[
        '--use_fast_math',
        '--maxrregcount=32'
    ]
)
```

### Device Management
```python
# List available devices
for device in module.devices:
    print(device)

# Select specific device
module.select_device(device_id=1)

# Get current device info
current_device = module.devices[module.current_device]
print(f"Using: {current_device}")
```

### Context Manager
```python
# Automatic resource cleanup
with CUDAAccelerationModule(enable_profiling=True) as cuda_module:
    # Perform GPU operations
    result = cuda_module.launch_kernel(...)

# Resources automatically released
```

## Performance Optimization

### Best Practices
1. **Memory Management**
   - Reuse allocated memory when possible
   - Use memory pools for frequent allocations
   - Minimize host-device transfers

2. **Kernel Optimization**
   - Optimize grid/block dimensions for your GPU
   - Use shared memory for frequently accessed data
   - Minimize thread divergence

3. **Multi-GPU**
   - Balance workload across devices
   - Use peer-to-peer transfers when available
   - Minimize inter-device synchronization

4. **Streaming**
   - Overlap data transfers with computation
   - Use multiple streams for concurrent operations
   - Pipeline operations for maximum throughput

## Troubleshooting

### Common Issues

**Issue**: No CUDA devices found
```
Solution: Ensure NVIDIA drivers and CUDA toolkit are installed
Check: nvidia-smi
```

**Issue**: Out of memory errors
```
Solution: Reduce batch size or use memory-efficient operations
Monitor: module.get_gpu_stats()
```

**Issue**: Kernel launch failures
```
Solution: Check kernel syntax and compilation options
Enable: Detailed logging with logging.DEBUG
```

## API Reference

### Classes

#### `CUDAAccelerationModule`
Main class for CUDA acceleration operations.

**Constructor Parameters:**
- `backend` (GPUBackend): GPU backend selection
- `device_ids` (List[int]): GPU devices to use
- `enable_profiling` (bool): Enable performance profiling
- `memory_pool_size` (int): Memory pool size per device

#### `GPUDeviceInfo`
Container for GPU device information.

#### `PerformanceMetrics`
Performance profiling metrics container.

### Methods Reference

See inline documentation in source code for complete API reference.

## License

Part of the Omnixan project. See main repository for license information.

## Contributing

Contributions welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All methods include type hints and docstrings
- Unit tests cover new functionality
- Performance benchmarks included for optimizations

## Contact

For issues and questions:
- GitHub: https://github.com/Andrei-Barwood/Omnixan
- Project: Omnixan - Quantum Cloud Computing Platform

## Acknowledgments

- Snocomm - Project sponsor
- The Amarr Imperial Academy - Resource provider
- NVIDIA CUDA Team - GPU computing platform
