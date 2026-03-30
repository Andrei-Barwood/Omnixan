from __future__ import annotations

import asyncio
import importlib.util

import pytest


RAY_AVAILABLE = importlib.util.find_spec("ray") is not None
DASK_AVAILABLE = importlib.util.find_spec("dask") is not None

try:
    from dask.distributed import Client, LocalCluster

    DASK_DISTRIBUTED_AVAILABLE = True
except Exception:
    Client = None
    LocalCluster = None
    DASK_DISTRIBUTED_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not (RAY_AVAILABLE and DASK_AVAILABLE and DASK_DISTRIBUTED_AVAILABLE),
    reason="Distributed smoke tests require ray and dask[distributed]",
)


def _dask_inc(value: int) -> int:
    return value + 2


def test_ray_runtime_smoke() -> None:
    import ray

    ray.shutdown()
    info = ray.init(
        num_cpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        logging_level="ERROR",
    )
    try:
        remote_inc = ray.remote(lambda value: value + 1)
        assert info.address_info.get("address")
        assert ray.get(remote_inc.remote(41)) == 42
    finally:
        ray.shutdown()


def test_dask_distributed_runtime_smoke() -> None:
    cluster = LocalCluster(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=None,
    )
    client = Client(cluster)
    try:
        future = client.submit(_dask_inc, 40)
        assert future.result() == 42
    finally:
        client.close()
        cluster.close()


def test_fog_computing_module_smoke() -> None:
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
                name="smoke-task",
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


def test_cache_coherence_module_smoke() -> None:
    from omnixan.edge_computing_network.cache_coherence_module.module import (
        CacheCoherenceModule,
    )

    async def scenario() -> None:
        module = CacheCoherenceModule()
        await module.initialize()
        try:
            module.register_node("node1")
            module.register_node("node2")
            await module.write("node1", "key", "value")
            value, hit = await module.read("node2", "key")
            directory = module.get_directory_state()

            assert value == "value"
            assert hit is False
            assert "node1" in directory["owners"].values()
        finally:
            await module.shutdown()

    asyncio.run(scenario())


def test_fault_mitigation_module_smoke() -> None:
    from omnixan.virtualized_cluster.fault_mitigation_module.module import (
        FaultMitigationModule,
        FaultType,
    )

    async def scenario() -> None:
        module = FaultMitigationModule()
        await module.initialize()
        try:
            component = await module.register_component("worker-a")
            await module.report_fault(
                component.component_id,
                FaultType.TRANSIENT,
                "smoke fault",
            )
            await asyncio.sleep(0.3)
            metrics = module.get_metrics()
            assert metrics["total_faults_detected"] == 1
            assert metrics["faults_mitigated"] == 1
        finally:
            await module.shutdown()

    asyncio.run(scenario())
