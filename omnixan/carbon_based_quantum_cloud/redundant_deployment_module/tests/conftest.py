"""
Shared pytest fixtures for redundant_deployment_module tests

Author: Kirtan Teg Singh
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, List

import pytest

from redundant_deployment_module import (
    RedundantDeploymentModule,
    ServiceConfig,
    RegionConfig,
    DeploymentMode,
    ReplicationConfig,
    ReplicationStrategy,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def module() -> AsyncGenerator[RedundantDeploymentModule, None]:
    """
    Create and initialize a module instance for testing.
    Automatically cleans up after test completes.
    """
    mod = RedundantDeploymentModule(log_level="DEBUG")
    await mod.initialize()
    yield mod
    await mod.shutdown()


@pytest.fixture
def sample_regions() -> List[RegionConfig]:
    """Sample region configurations for testing."""
    return [
        RegionConfig(
            region_id="us-east-1",
            endpoint="https://us-east-1.test.omnixan.io",
            priority=90,
            latency_ms=10.0,
            capacity=1000,
        ),
        RegionConfig(
            region_id="eu-west-1",
            endpoint="https://eu-west-1.test.omnixan.io",
            priority=85,
            latency_ms=50.0,
            capacity=800,
        ),
        RegionConfig(
            region_id="ap-south-1",
            endpoint="https://ap-south-1.test.omnixan.io",
            priority=80,
            latency_ms=100.0,
            capacity=600,
        ),
    ]


@pytest.fixture
def sample_service(sample_regions: List[RegionConfig]) -> ServiceConfig:
    """Sample service configuration for testing."""
    return ServiceConfig(
        name="test-service",
        version="1.0.0",
        regions=sample_regions,
        deployment_mode=DeploymentMode.ACTIVE_PASSIVE,
        health_check_interval=5,
        health_check_timeout=2,
        min_healthy_instances=1,
    )


@pytest.fixture
def sample_replication_config() -> ReplicationConfig:
    """Sample replication configuration for testing."""
    return ReplicationConfig(
        service_id="test-service-id",
        strategy=ReplicationStrategy.EVENTUAL_CONSISTENCY,
        batch_size=100,
        sync_interval=60,
        max_lag_seconds=300,
        enable_compression=True,
        enable_encryption=True,
        conflict_resolution="last_write_wins",
    )


@pytest.fixture
def mock_cold_migration_module():
    """Mock cold migration module for testing integration."""
    class MockColdMigration:
        async def migrate(self, *args, **kwargs):
            return {"success": True}
    
    return MockColdMigration()


@pytest.fixture
def mock_load_balancing_module():
    """Mock load balancing module for testing integration."""
    class MockLoadBalancer:
        async def configure(self, *args, **kwargs):
            return {"success": True}
        
        async def update_weights(self, *args, **kwargs):
            return {"success": True}
    
    return MockLoadBalancer()
