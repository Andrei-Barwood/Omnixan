"""
CUDA Acceleration Module for Omnixan Supercomputing Interconnect Cloud
Author: Omnixan Development Team
Description: Production-ready CUDA acceleration with multi-GPU support,
             stream processing, and tensor core integration
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import contextmanager
import numpy as np

try:
    import cupy as cp
    import cupyx
    from cupyx.profiler import benchmark
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    cuda = None


class GPUBackend(Enum):
    """Supported GPU computation backends."""
    CUPY = "cupy"
    PYCUDA = "pycuda"
    AUTO = "auto"


@dataclass
class GPUDeviceInfo:
    """GPU device information container."""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory: int
    available_memory: int
    multiprocessor_count: int
    clock_rate: int
    cuda_cores: Optional[int] = None

    def __str__(self) -> str:
        return (f"GPU {self.device_id}: {self.name} "
                f"(CC {self.compute_capability[0]}.{self.compute_capability[1]}, "
                f"{self.total_memory / (1024**3):.2f} GB)")


@dataclass
class KernelConfig:
    """Configuration for CUDA kernel execution."""
    grid_dim: Tuple[int, int, int]
    block_dim: Tuple[int, int, int]
    shared_mem_bytes: int = 0
    stream: Optional[Any] = None


@dataclass
class PerformanceMetrics:
    """Performance profiling metrics."""
    kernel_time_ms: float = 0.0
    memory_transfer_time_ms: float = 0.0
    total_time_ms: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    throughput_gflops: float = 0.0


class CUDAAccelerationModule:
    """
    CUDA Acceleration Module for high-performance GPU computing.

    Provides unified interface for CUDA operations with support for:
    - Multiple GPU backends (CuPy, PyCUDA)
    - Multi-GPU operations with device management
    - Stream-based concurrent execution
    - Memory management and optimization
    - Performance profiling and monitoring
    - Integration with tensor core operations

    Attributes:
        backend: Selected GPU backend (CuPy or PyCUDA)
        devices: List of available GPU devices
        current_device: Currently selected GPU device
        streams: Dictionary of CUDA streams per device
        memory_pools: Memory pool management per device
        profiling_enabled: Enable performance profiling
    """

    def __init__(
        self,
        backend: GPUBackend = GPUBackend.AUTO,
        device_ids: Optional[List[int]] = None,
        enable_profiling: bool = False,
        memory_pool_size: Optional[int] = None
    ):
        """
        Initialize CUDA Acceleration Module.

        Args:
            backend: GPU backend to use (CuPy, PyCUDA, or auto-detect)
            device_ids: List of GPU device IDs to use (None = all available)
            enable_profiling: Enable performance profiling
            memory_pool_size: Size of memory pool per device in bytes

        Raises:
            RuntimeError: If no compatible GPU backend is available
        """
        self.logger = logging.getLogger(__name__)
        self.profiling_enabled = enable_profiling
        self._lock = threading.Lock()

        # Select backend
        self.backend = self._select_backend(backend)
        self.logger.info(f"Using GPU backend: {self.backend.value}")

        # Initialize devices
        self.devices: List[GPUDeviceInfo] = []
        self.current_device: int = 0
        self.device_ids = device_ids

        # Initialize streams and memory management
        self.streams: Dict[int, List[Any]] = {}
        self.memory_pools: Dict[int, Any] = {}
        self._kernel_cache: Dict[str, Any] = {}

        # Performance metrics
        self.metrics: Dict[str, PerformanceMetrics] = {}

        # Tensor core module integration placeholder
        self.tensor_core_module = None

        self.initialize()

    def _select_backend(self, backend: GPUBackend) -> GPUBackend:
        """Select appropriate GPU backend."""
        if backend == GPUBackend.AUTO:
            if CUPY_AVAILABLE:
                return GPUBackend.CUPY
            elif PYCUDA_AVAILABLE:
                return GPUBackend.PYCUDA
            else:
                raise RuntimeError(
                    "No compatible GPU backend found. "
                    "Please install CuPy or PyCUDA."
                )
        elif backend == GPUBackend.CUPY and not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available. Please install cupy.")
        elif backend == GPUBackend.PYCUDA and not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA not available. Please install pycuda.")

        return backend

    def initialize(self) -> bool:
        """
        Initialize CUDA devices and resources.

        Returns:
            True if initialization successful

        Raises:
            RuntimeError: If GPU initialization fails
        """
        try:
            self.logger.info("Initializing CUDA Acceleration Module...")

            # Detect and initialize GPU devices
            self._detect_devices()

            if not self.devices:
                raise RuntimeError("No CUDA-capable GPU devices found")

            # Initialize streams and memory pools for each device
            for device_info in self.devices:
                device_id = device_info.device_id
                self._initialize_device(device_id)

            self.logger.info(
                f"Successfully initialized {len(self.devices)} GPU device(s)"
            )
            return True

        except Exception as e:
            self.logger.error(f"GPU initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize CUDA module: {e}")

    def _detect_devices(self) -> None:
        """Detect available CUDA devices."""
        if self.backend == GPUBackend.CUPY:
            num_devices = cp.cuda.runtime.getDeviceCount()
            device_range = (self.device_ids if self.device_ids 
                          else range(num_devices))

            for device_id in device_range:
                with cp.cuda.Device(device_id):
                    props = cp.cuda.runtime.getDeviceProperties(device_id)

                    device_info = GPUDeviceInfo(
                        device_id=device_id,
                        name=props["name"].decode(),
                        compute_capability=(
                            props["major"],
                            props["minor"]
                        ),
                        total_memory=props["totalGlobalMem"],
                        available_memory=cp.cuda.Device().mem_info[0],
                        multiprocessor_count=props["multiProcessorCount"],
                        clock_rate=props["clockRate"],
                        cuda_cores=self._estimate_cuda_cores(props)
                    )
                    self.devices.append(device_info)
                    self.logger.info(f"Detected: {device_info}")

        elif self.backend == GPUBackend.PYCUDA:
            cuda.init()
            num_devices = cuda.Device.count()
            device_range = (self.device_ids if self.device_ids 
                          else range(num_devices))

            for device_id in device_range:
                device = cuda.Device(device_id)

                device_info = GPUDeviceInfo(
                    device_id=device_id,
                    name=device.name(),
                    compute_capability=device.compute_capability(),
                    total_memory=device.total_memory(),
                    available_memory=device.total_memory(),
                    multiprocessor_count=device.get_attribute(
                        cuda.device_attribute.MULTIPROCESSOR_COUNT
                    ),
                    clock_rate=device.get_attribute(
                        cuda.device_attribute.CLOCK_RATE
                    )
                )
                self.devices.append(device_info)
                self.logger.info(f"Detected: {device_info}")

    def _estimate_cuda_cores(self, props: Dict) -> int:
        """Estimate number of CUDA cores based on architecture."""
        mp_count = props["multiProcessorCount"]
        major, minor = props["major"], props["minor"]

        # CUDA cores per SM by compute capability
        cores_per_sm = {
            (3, 0): 192, (3, 5): 192, (3, 7): 192,
            (5, 0): 128, (5, 2): 128,
            (6, 0): 64, (6, 1): 128,
            (7, 0): 64, (7, 5): 64,
            (8, 0): 64, (8, 6): 128, (8, 9): 128,
            (9, 0): 128
        }

        return mp_count * cores_per_sm.get((major, minor), 128)

    def _initialize_device(self, device_id: int) -> None:
        """Initialize streams and memory pools for a device."""
        if self.backend == GPUBackend.CUPY:
            with cp.cuda.Device(device_id):
                # Create multiple streams for concurrent operations
                self.streams[device_id] = [
                    cp.cuda.Stream(non_blocking=True) for _ in range(4)
                ]

                # Initialize memory pool
                mempool = cp.get_default_memory_pool()
                self.memory_pools[device_id] = mempool

        elif self.backend == GPUBackend.PYCUDA:
            # PyCUDA stream initialization
            context = cuda.Device(device_id).make_context()
            self.streams[device_id] = [
                cuda.Stream() for _ in range(4)
            ]
            context.pop()

    def select_device(self, device_id: int) -> None:
        """
        Select GPU device for subsequent operations.

        Args:
            device_id: GPU device ID to select

        Raises:
            ValueError: If device_id is invalid
        """
        if device_id >= len(self.devices):
            raise ValueError(
                f"Invalid device_id {device_id}. "
                f"Available devices: 0-{len(self.devices)-1}"
            )

        self.current_device = device_id

        if self.backend == GPUBackend.CUPY:
            cp.cuda.Device(device_id).use()

        self.logger.debug(f"Selected GPU device {device_id}")

    def allocate_gpu_memory(
        self,
        size: int,
        device_id: Optional[int] = None,
        dtype: type = np.float32
    ) -> Union[cp.ndarray, gpuarray.GPUArray]:
        """
        Allocate memory on GPU device.

        Args:
            size: Number of elements to allocate
            device_id: Target GPU device (None = current device)
            dtype: Data type for allocation

        Returns:
            GPU array handle

        Raises:
            MemoryError: If allocation fails
        """
        device_id = device_id if device_id is not None else self.current_device

        try:
            if self.backend == GPUBackend.CUPY:
                with cp.cuda.Device(device_id):
                    gpu_array = cp.empty(size, dtype=dtype)
                    self.logger.debug(
                        f"Allocated {size * np.dtype(dtype).itemsize / (1024**2):.2f} MB "
                        f"on GPU {device_id}"
                    )
                    return gpu_array

            elif self.backend == GPUBackend.PYCUDA:
                gpu_array = gpuarray.empty(size, dtype=dtype)
                self.logger.debug(
                    f"Allocated {size * np.dtype(dtype).itemsize / (1024**2):.2f} MB "
                    f"on GPU {device_id}"
                )
                return gpu_array

        except Exception as e:
            raise MemoryError(f"Failed to allocate GPU memory: {e}")

    def copy_data(
        self,
        src: Union[np.ndarray, cp.ndarray, gpuarray.GPUArray],
        dst: Optional[Union[np.ndarray, cp.ndarray, gpuarray.GPUArray]] = None,
        stream: Optional[Any] = None,
        async_copy: bool = False
    ) -> Union[np.ndarray, cp.ndarray, gpuarray.GPUArray]:
        """
        Copy data between host and device or between devices.

        Args:
            src: Source array (host or device)
            dst: Destination array (None = allocate automatically)
            stream: CUDA stream for async copy
            async_copy: Enable asynchronous copying

        Returns:
            Destination array with copied data
        """
        try:
            if self.backend == GPUBackend.CUPY:
                # Host to Device
                if isinstance(src, np.ndarray):
                    if dst is None:
                        dst = cp.empty_like(src)
                    if async_copy and stream:
                        with stream:
                            cp.copyto(dst, cp.asarray(src))
                    else:
                        cp.copyto(dst, cp.asarray(src))

                # Device to Host
                elif isinstance(src, cp.ndarray):
                    if dst is None:
                        dst = cp.asnumpy(src)
                    else:
                        cp.copyto(dst, src)

                return dst

            elif self.backend == GPUBackend.PYCUDA:
                # Host to Device
                if isinstance(src, np.ndarray):
                    if dst is None:
                        dst = gpuarray.to_gpu(src)
                    else:
                        dst.set(src)

                # Device to Host
                elif isinstance(src, gpuarray.GPUArray):
                    if dst is None:
                        dst = src.get()
                    else:
                        src.get(dst)

                return dst

        except Exception as e:
            self.logger.error(f"Data copy failed: {e}")
            raise

    def launch_kernel(
        self,
        kernel_code: str,
        kernel_name: str,
        grid: Tuple[int, ...],
        block: Tuple[int, ...],
        args: Tuple,
        shared_mem: int = 0,
        stream: Optional[Any] = None,
        compile_options: Optional[List[str]] = None
    ) -> Any:
        """
        Launch CUDA kernel with specified configuration.

        Args:
            kernel_code: CUDA C/C++ kernel source code
            kernel_name: Name of kernel function to execute
            grid: Grid dimensions (x, y, z)
            block: Block dimensions (x, y, z)
            args: Tuple of kernel arguments
            shared_mem: Shared memory size in bytes
            stream: CUDA stream for execution
            compile_options: Compiler options

        Returns:
            Kernel execution result

        Example:
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
            module.launch_kernel(
                kernel_code, "vector_add",
                grid=(n//256 + 1,), block=(256,),
                args=(a_gpu, b_gpu, c_gpu, n)
            )
        """
        cache_key = f"{kernel_name}_{hash(kernel_code)}"

        try:
            if self.backend == GPUBackend.CUPY:
                # Check kernel cache
                if cache_key not in self._kernel_cache:
                    # Compile kernel
                    module = cp.RawModule(
                        code=kernel_code,
                        options=compile_options or []
                    )
                    kernel = module.get_function(kernel_name)
                    self._kernel_cache[cache_key] = kernel
                else:
                    kernel = self._kernel_cache[cache_key]

                # Launch kernel
                if stream:
                    with stream:
                        kernel(grid, block, args, shared_mem=shared_mem)
                else:
                    kernel(grid, block, args, shared_mem=shared_mem)

                # Synchronize if profiling
                if self.profiling_enabled:
                    cp.cuda.Stream.null.synchronize()

            elif self.backend == GPUBackend.PYCUDA:
                # Compile kernel
                if cache_key not in self._kernel_cache:
                    module = SourceModule(kernel_code)
                    kernel = module.get_function(kernel_name)
                    self._kernel_cache[cache_key] = kernel
                else:
                    kernel = self._kernel_cache[cache_key]

                # Launch kernel
                kernel(
                    *args,
                    block=block,
                    grid=grid,
                    shared=shared_mem,
                    stream=stream
                )

            self.logger.debug(
                f"Launched kernel '{kernel_name}' with grid={grid}, block={block}"
            )

        except Exception as e:
            self.logger.error(f"Kernel launch failed: {e}")
            raise RuntimeError(f"Failed to launch kernel '{kernel_name}': {e}")

    def get_gpu_stats(
        self,
        device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get current GPU statistics and utilization.

        Args:
            device_id: GPU device ID (None = current device)

        Returns:
            Dictionary containing GPU statistics
        """
        device_id = device_id if device_id is not None else self.current_device

        stats = {
            "device_id": device_id,
            "device_name": self.devices[device_id].name,
        }

        try:
            if self.backend == GPUBackend.CUPY:
                with cp.cuda.Device(device_id):
                    mempool = self.memory_pools[device_id]
                    free_mem, total_mem = cp.cuda.Device().mem_info

                    stats.update({
                        "total_memory_gb": total_mem / (1024**3),
                        "free_memory_gb": free_mem / (1024**3),
                        "used_memory_gb": (total_mem - free_mem) / (1024**3),
                        "memory_utilization_%": (
                            (total_mem - free_mem) / total_mem * 100
                        ),
                        "memory_pool_used_bytes": mempool.used_bytes(),
                        "memory_pool_total_bytes": mempool.total_bytes(),
                    })

            elif self.backend == GPUBackend.PYCUDA:
                free_mem, total_mem = cuda.mem_get_info()

                stats.update({
                    "total_memory_gb": total_mem / (1024**3),
                    "free_memory_gb": free_mem / (1024**3),
                    "used_memory_gb": (total_mem - free_mem) / (1024**3),
                    "memory_utilization_%": (
                        (total_mem - free_mem) / total_mem * 100
                    ),
                })

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get GPU stats: {e}")
            return stats

    @contextmanager
    def stream_context(self, device_id: Optional[int] = None, stream_id: int = 0):
        """
        Context manager for stream-based operations.

        Args:
            device_id: GPU device ID
            stream_id: Stream index (0-3)

        Yields:
            CUDA stream object

        Example:
            with module.stream_context(device_id=0, stream_id=1) as stream:
                # Operations in this block use the specified stream
                result = module.copy_data(data, stream=stream, async_copy=True)
        """
        device_id = device_id if device_id is not None else self.current_device
        stream = self.streams[device_id][stream_id]

        try:
            yield stream
        finally:
            if self.backend == GPUBackend.CUPY:
                stream.synchronize()
            elif self.backend == GPUBackend.PYCUDA:
                stream.synchronize()

    def execute_multi_gpu(
        self,
        func: callable,
        data: List[Any],
        device_ids: Optional[List[int]] = None
    ) -> List[Any]:
        """
        Execute function across multiple GPUs in parallel.

        Args:
            func: Function to execute on each GPU
            data: List of data chunks (one per GPU)
            device_ids: List of GPU devices to use

        Returns:
            List of results from each GPU

        Example:
            def process_chunk(data_chunk, device_id):
                # Process data on specific GPU
                return result

            results = module.execute_multi_gpu(
                process_chunk,
                data_chunks,
                device_ids=[0, 1, 2, 3]
            )
        """
        device_ids = device_ids or [d.device_id for d in self.devices]

        if len(data) != len(device_ids):
            raise ValueError(
                f"Data chunks ({len(data)}) must match "
                f"number of devices ({len(device_ids)})"
            )

        results = [None] * len(device_ids)
        threads = []

        def worker(device_id, data_chunk, result_idx):
            try:
                self.select_device(device_id)
                result = func(data_chunk, device_id)
                results[result_idx] = result
            except Exception as e:
                self.logger.error(
                    f"Error on GPU {device_id}: {e}"
                )
                results[result_idx] = None

        # Launch parallel execution
        for idx, (device_id, data_chunk) in enumerate(zip(device_ids, data)):
            thread = threading.Thread(
                target=worker,
                args=(device_id, data_chunk, idx)
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        return results

    def integrate_tensor_core(self, tensor_core_module: Any) -> None:
        """
        Integrate with tensor core module for optimized operations.

        Args:
            tensor_core_module: Tensor core module instance
        """
        self.tensor_core_module = tensor_core_module
        self.logger.info("Integrated with Tensor Core Module")

    @contextmanager
    def profiler(self, operation_name: str):
        """
        Context manager for performance profiling.

        Args:
            operation_name: Name of operation to profile

        Example:
            with module.profiler("matrix_multiply"):
                result = module.launch_kernel(...)
        """
        if not self.profiling_enabled:
            yield
            return

        metrics = PerformanceMetrics()

        if self.backend == GPUBackend.CUPY:
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()

            start_event.record()
            try:
                yield metrics
            finally:
                end_event.record()
                end_event.synchronize()
                metrics.total_time_ms = cp.cuda.get_elapsed_time(
                    start_event, end_event
                )
                self.metrics[operation_name] = metrics

                self.logger.info(
                    f"Profile [{operation_name}]: {metrics.total_time_ms:.3f} ms"
                )
        else:
            import time
            start_time = time.perf_counter()
            try:
                yield metrics
            finally:
                end_time = time.perf_counter()
                metrics.total_time_ms = (end_time - start_time) * 1000
                self.metrics[operation_name] = metrics

    def execute(self, *args, **kwargs) -> Any:
        """
        Generic execution method for compatibility.

        This method provides a unified interface for executing
        GPU operations across different modules.
        """
        operation_type = kwargs.get("operation_type", "default")

        if operation_type == "kernel":
            return self.launch_kernel(*args, **kwargs)
        elif operation_type == "copy":
            return self.copy_data(*args, **kwargs)
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")

    def shutdown(self) -> None:
        """
        Cleanup and release GPU resources.

        Releases memory pools, destroys streams, and performs
        final synchronization.
        """
        try:
            self.logger.info("Shutting down CUDA Acceleration Module...")

            # Clear kernel cache
            self._kernel_cache.clear()

            # Cleanup device resources
            for device_id in self.streams.keys():
                if self.backend == GPUBackend.CUPY:
                    with cp.cuda.Device(device_id):
                        # Free memory pool
                        if device_id in self.memory_pools:
                            self.memory_pools[device_id].free_all_blocks()

                        # Synchronize all streams
                        for stream in self.streams[device_id]:
                            stream.synchronize()

                elif self.backend == GPUBackend.PYCUDA:
                    # Synchronize streams
                    for stream in self.streams[device_id]:
                        stream.synchronize()

            # Clear data structures
            self.streams.clear()
            self.memory_pools.clear()
            self.metrics.clear()

            self.logger.info("CUDA Acceleration Module shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
        return False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CUDAAccelerationModule("
            f"backend={self.backend.value}, "
            f"devices={len(self.devices)}, "
            f"current_device={self.current_device})"
        )
