from __future__ import annotations

import asyncio
import importlib

import pytest


def test_cuda_acceleration_module_imports_without_gpu_backends() -> None:
    module = importlib.import_module(
        "omnixan.supercomputing_interconnect_cloud.cuda_acceleration_module.module"
    )
    status = module.get_optional_backend_status()

    assert "cupy" in status
    assert "pycuda" in status


def test_cuda_acceleration_module_fails_at_runtime_with_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        "omnixan.supercomputing_interconnect_cloud.cuda_acceleration_module.module"
    )

    monkeypatch.setattr(module, "_load_cupy", lambda: False)
    monkeypatch.setattr(module, "_load_pycuda", lambda: False)
    monkeypatch.setattr(
        module,
        "get_optional_backend_status",
        lambda: {
            "cupy": {"available": False, "error": "simulated missing cupy"},
            "pycuda": {"available": False, "error": "simulated missing pycuda"},
        },
    )

    with pytest.raises(RuntimeError, match="GPU backends are optional"):
        module.CUDAAccelerationModule()


@pytest.mark.parametrize(
    ("model_format", "expected_message"),
    [
        ("TENSORFLOW", "TensorFlow model support is optional"),
        ("PYTORCH", "PyTorch model support is optional"),
    ],
)
def test_edge_ai_module_rejects_missing_model_runtime(
    monkeypatch: pytest.MonkeyPatch,
    model_format: str,
    expected_message: str,
) -> None:
    from omnixan.in_memory_computing_cloud.edge_ai_module.module import (
        EdgeAIModule,
        ModelFormat,
        EdgeAIError,
    )

    async def scenario() -> None:
        module = EdgeAIModule()
        await module.initialize()
        try:
            with pytest.raises(EdgeAIError, match=expected_message):
                await module.deploy_model(
                    name=f"{model_format.lower()}-model",
                    version="1.0",
                    format=getattr(ModelFormat, model_format),
                    input_shape=[1, 2],
                    output_shape=[1, 2],
                    weights=[[1.0, 2.0], [3.0, 4.0]],
                )
        finally:
            await module.shutdown()

    monkeypatch.setattr(
        "omnixan.in_memory_computing_cloud.edge_ai_module.module._runtime_available",
        lambda *names: False,
    )
    asyncio.run(scenario())


def test_edge_ai_module_rejects_missing_gpu_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnixan.in_memory_computing_cloud.edge_ai_module.module import (
        AcceleratorType,
        EdgeAIConfig,
        EdgeAIError,
        EdgeAIModule,
    )

    async def scenario() -> None:
        module = EdgeAIModule(
            EdgeAIConfig(default_accelerator=AcceleratorType.GPU)
        )
        with pytest.raises(EdgeAIError, match="GPU acceleration is optional"):
            await module.initialize()

    monkeypatch.setattr(
        "omnixan.in_memory_computing_cloud.edge_ai_module.module._runtime_available",
        lambda *names: False,
    )
    asyncio.run(scenario())
