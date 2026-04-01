from __future__ import annotations

import asyncio

import pytest

from omnixan.carbon_based_quantum_cloud.load_balancing_module.module import (
    LoadBalancingModule,
)
from omnixan.carbon_based_quantum_cloud.redundant_deployment_module.module import (
    RedundantDeploymentModule,
)
from omnixan.edge_computing_network.cache_coherence_module.module import (
    CacheCoherenceModule,
)
from omnixan.heterogenous_computing_group.non_blocking_module.module import (
    NonBlockingModule,
)
from omnixan.in_memory_computing_cloud.edge_ai_module.module import EdgeAIModule
from omnixan.in_memory_computing_cloud.fog_computing_module.module import (
    FogComputingModule,
)
from omnixan.supercomputing_interconnect_cloud.tensor_core_module.module import (
    TensorCoreModule,
)
from omnixan.virtualized_cluster.fault_mitigation_module.module import (
    FaultMitigationModule,
)


MODULE_FACTORIES = {
    "load_balancing": LoadBalancingModule,
    "redundant_deployment": RedundantDeploymentModule,
    "cache_coherence": CacheCoherenceModule,
    "non_blocking": NonBlockingModule,
    "fog_computing": FogComputingModule,
    "tensor_core": TensorCoreModule,
    "fault_mitigation": FaultMitigationModule,
    "edge_ai": EdgeAIModule,
}


@pytest.mark.parametrize("factory", MODULE_FACTORIES.values(), ids=MODULE_FACTORIES.keys())
def test_core_modules_expose_consistent_public_methods(factory) -> None:
    module = factory()

    for method_name in ("initialize", "execute", "shutdown", "get_status", "get_metrics"):
        assert callable(getattr(module, method_name))


@pytest.mark.parametrize("factory", MODULE_FACTORIES.values(), ids=MODULE_FACTORIES.keys())
def test_execute_requires_operation_string(factory) -> None:
    async def scenario() -> None:
        module = factory()
        await module.initialize()
        try:
            with pytest.raises(ValueError, match="Missing required 'operation' string in params"):
                await module.execute({})
        finally:
            await module.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("factory", MODULE_FACTORIES.values(), ids=MODULE_FACTORIES.keys())
def test_execute_get_status_and_get_metrics_use_standard_envelope(factory) -> None:
    async def scenario() -> None:
        module = factory()
        await module.initialize()
        try:
            direct_status = module.get_status()
            direct_metrics = module.get_metrics()
            status_response = await module.execute({"operation": "get_status"})
            metrics_response = await module.execute({"operation": "get_metrics"})

            assert isinstance(direct_status, dict)
            assert isinstance(direct_metrics, dict)
            assert status_response["status"] == "success"
            assert status_response["operation"] == "get_status"
            assert metrics_response["status"] == "success"
            assert metrics_response["operation"] == "get_metrics"
            assert status_response["initialized"] is True
        finally:
            await module.shutdown()

    asyncio.run(scenario())


def test_error_responses_use_standard_envelope_for_missing_resources() -> None:
    async def scenario() -> None:
        non_blocking = NonBlockingModule()
        fog = FogComputingModule()
        edge_ai = EdgeAIModule()

        await non_blocking.initialize()
        await fog.initialize()
        await edge_ai.initialize()
        try:
            poll_response = await non_blocking.execute(
                {"operation": "poll", "op_id": "missing"}
            )
            task_response = await fog.execute(
                {"operation": "get_task_status", "task_id": "missing"}
            )
            model_response = await edge_ai.execute(
                {"operation": "get_model_info", "model_id": "missing"}
            )

            assert poll_response["status"] == "error"
            assert poll_response["operation"] == "poll"
            assert task_response["status"] == "error"
            assert task_response["operation"] == "get_task_status"
            assert model_response["status"] == "error"
            assert model_response["operation"] == "get_model_info"
        finally:
            await non_blocking.shutdown()
            await fog.shutdown()
            await edge_ai.shutdown()

    asyncio.run(scenario())
