from __future__ import annotations

import asyncio
import importlib

import numpy as np


def _import_block(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert getattr(module, "__file__", None)


def test_carbon_based_quantum_cloud_smoke() -> None:
    _import_block("omnixan.carbon_based_quantum_cloud")

    from omnixan.carbon_based_quantum_cloud.load_balancing_module.module import (
        BackendConfig,
        HealthCheckConfig,
        LoadBalancingAlgorithm,
        LoadBalancingAlgorithmType,
        LoadBalancingModule,
        LoadBalancingModuleConfig,
        Request,
    )

    async def scenario() -> None:
        module = LoadBalancingModule(
            LoadBalancingModuleConfig(
                algorithm=LoadBalancingAlgorithm(
                    algorithm_type=LoadBalancingAlgorithmType.ROUND_ROBIN
                ),
                health_check=HealthCheckConfig(healthy_threshold=1),
            )
        )
        await module.initialize()
        try:
            backend_id = await module.add_backend(
                BackendConfig(host="127.0.0.1", port=8080)
            )
            result = await module.route_request(Request(client_ip="127.0.0.1"))
            distribution = await module.get_load_distribution()

            assert result.backend_id == backend_id
            assert distribution.total_requests == 1
            assert len(distribution.backends) == 1
        finally:
            await module.shutdown()

    asyncio.run(scenario())


def test_edge_computing_network_smoke() -> None:
    _import_block("omnixan.edge_computing_network")

    from omnixan.edge_computing_network.cache_coherence_module.module import (
        CacheCoherenceModule,
    )

    async def scenario() -> None:
        module = CacheCoherenceModule()
        await module.initialize()
        try:
            module.register_node("node-a")
            module.register_node("node-b")
            await module.write("node-a", "shared-key", {"value": 42})
            value, hit = await module.read("node-b", "shared-key")
            directory = module.get_directory_state()

            assert value == {"value": 42}
            assert hit is False
            assert directory["total_entries"] == 1
        finally:
            await module.shutdown()

    asyncio.run(scenario())


def test_heterogenous_computing_group_smoke() -> None:
    _import_block("omnixan.heterogenous_computing_group")

    from omnixan.heterogenous_computing_group.non_blocking_module.module import (
        NonBlockingModule,
        OperationType,
    )

    async def scenario() -> None:
        module = NonBlockingModule()
        await module.initialize()
        try:
            operation = await module.submit(
                OperationType.COMPUTE,
                data={"payload": "hello"},
            )
            result = await module.wait(operation.op_id, timeout=1.0)
            metrics = module.get_metrics()

            assert result is not None
            assert result.result["processed"] is True
            assert metrics["completed_operations"] == 1
        finally:
            await module.shutdown()

    asyncio.run(scenario())


def test_in_memory_computing_cloud_smoke() -> None:
    _import_block("omnixan.in_memory_computing_cloud")

    from omnixan.in_memory_computing_cloud.fog_computing_module.module import (
        FogComputingModule,
        FogConfig,
        NodeType,
    )

    async def scenario() -> None:
        module = FogComputingModule(FogConfig(resource_check_interval=60.0))
        await module.initialize()
        try:
            await module.register_node(
                name="edge-1",
                node_type=NodeType.EDGE,
                location=(0.0, 0.0),
                cpu_cores=4,
                memory_mb=4096,
                bandwidth_mbps=100.0,
                latency_ms=5.0,
            )
            await module.submit_task(
                name="core-smoke-task",
                compute_units=1,
                memory_mb=128,
            )
            await asyncio.sleep(0.2)
            metrics = module.get_metrics()

            assert metrics["total_nodes"] == 1
            assert metrics["total_tasks"] == 1
            assert metrics["completed_tasks"] + metrics["failed_tasks"] == 1
        finally:
            await module.shutdown()

    asyncio.run(scenario())


def test_supercomputing_interconnect_cloud_smoke() -> None:
    _import_block("omnixan.supercomputing_interconnect_cloud")

    from omnixan.supercomputing_interconnect_cloud.tensor_core_module.module import (
        TensorCoreModule,
    )

    async def scenario() -> None:
        module = TensorCoreModule()
        await module.initialize()
        try:
            result = await module.execute(
                {
                    "operation": "gemm",
                    "A": [[1, 2], [3, 4]],
                    "B": [[5, 6], [7, 8]],
                }
            )
            metrics = module.get_metrics()

            assert result["shape"] == [2, 2]
            assert np.allclose(
                np.array(result["result"]),
                np.array([[19.0, 22.0], [43.0, 50.0]]),
            )
            assert metrics["total_operations"] == 1
        finally:
            await module.shutdown()

    asyncio.run(scenario())


def test_virtualized_cluster_smoke() -> None:
    _import_block("omnixan.virtualized_cluster")

    from omnixan.virtualized_cluster.fault_mitigation_module.module import (
        FaultMitigationModule,
    )

    async def scenario() -> None:
        module = FaultMitigationModule()
        await module.initialize()
        try:
            component = await module.register_component("worker-a")
            checkpoint = await module.create_checkpoint(
                component.component_id,
                {"mode": "warm", "sequence": 1},
            )
            restored = await module.restore_checkpoint(
                component.component_id,
                checkpoint.checkpoint_id,
            )
            status = module.get_status()

            assert restored == {"mode": "warm", "sequence": 1}
            assert len(status["components"]) == 1
            assert status["components"][0]["name"] == "worker-a"
        finally:
            await module.shutdown()

    asyncio.run(scenario())
