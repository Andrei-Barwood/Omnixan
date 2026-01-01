"""
OMNIXAN Near-Data Processing Module
edge_computing_network/near_data_processing_module

Production-ready near-data processing implementation that pushes computation
closer to storage, reducing data movement and improving performance for
analytical and ML workloads.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4
import json

import numpy as np

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingLocation(str, Enum):
    """Where processing occurs"""
    STORAGE = "storage"  # At storage layer
    COMPUTE = "compute"  # At compute layer
    HYBRID = "hybrid"  # Split between both


class OperationType(str, Enum):
    """Types of near-data operations"""
    FILTER = "filter"
    PROJECT = "project"
    AGGREGATE = "aggregate"
    TRANSFORM = "transform"
    SCAN = "scan"
    JOIN = "join"


class AggregateFunction(str, Enum):
    """Supported aggregate functions"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    VARIANCE = "variance"
    STDDEV = "stddev"


@dataclass
class ProcessingTask:
    """A processing task to execute near data"""
    task_id: str
    operation: OperationType
    source: str
    parameters: Dict[str, Any]
    priority: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class ProcessingResult:
    """Result of near-data processing"""
    task_id: str
    success: bool
    data: Any = None
    rows_processed: int = 0
    rows_returned: int = 0
    bytes_scanned: int = 0
    bytes_returned: int = 0
    execution_time_ms: float = 0.0
    location: ProcessingLocation = ProcessingLocation.STORAGE
    error: Optional[str] = None


@dataclass
class ProcessingMetrics:
    """Metrics for near-data processing"""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_rows_processed: int = 0
    total_bytes_scanned: int = 0
    total_bytes_returned: int = 0
    data_reduction_ratio: float = 0.0
    avg_execution_time_ms: float = 0.0


class NDPConfig(BaseModel):
    """Configuration for near-data processing"""
    max_concurrent_tasks: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Maximum concurrent processing tasks"
    )
    push_down_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Selectivity threshold for pushdown"
    )
    enable_predicate_pushdown: bool = Field(
        default=True,
        description="Enable predicate pushdown"
    )
    enable_projection_pushdown: bool = Field(
        default=True,
        description="Enable projection pushdown"
    )
    enable_aggregate_pushdown: bool = Field(
        default=True,
        description="Enable aggregate pushdown"
    )
    batch_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Processing batch size"
    )
    vectorized: bool = Field(
        default=True,
        description="Use vectorized operations"
    )


class NDPError(Exception):
    """Base exception for near-data processing errors"""
    pass


class TaskExecutionError(NDPError):
    """Raised when task execution fails"""
    pass


# ============================================================================
# Operation Implementations
# ============================================================================

class StorageOperator(ABC):
    """Abstract base class for storage-side operators"""
    
    @abstractmethod
    def execute(self, data: Any, params: Dict[str, Any]) -> Any:
        """Execute operation on data"""
        pass
    
    @abstractmethod
    def estimate_selectivity(self, params: Dict[str, Any]) -> float:
        """Estimate output/input ratio"""
        pass


class FilterOperator(StorageOperator):
    """Filter rows based on predicate"""
    
    def execute(self, data: Dict[str, List], params: Dict[str, Any]) -> Dict[str, List]:
        """Execute filter operation"""
        column = params.get("column")
        operator = params.get("operator", "eq")
        value = params.get("value")
        
        if column not in data:
            return data
        
        col_data = data[column]
        mask = self._create_mask(col_data, operator, value)
        
        # Apply mask to all columns
        result = {}
        for col_name, col_values in data.items():
            result[col_name] = [v for v, m in zip(col_values, mask) if m]
        
        return result
    
    def _create_mask(
        self,
        col_data: List,
        operator: str,
        value: Any
    ) -> List[bool]:
        """Create boolean mask for filter"""
        ops = {
            "eq": lambda x: x == value,
            "ne": lambda x: x != value,
            "lt": lambda x: x < value,
            "le": lambda x: x <= value,
            "gt": lambda x: x > value,
            "ge": lambda x: x >= value,
            "in": lambda x: x in value,
            "like": lambda x: value in str(x) if x else False,
            "is_null": lambda x: x is None,
            "is_not_null": lambda x: x is not None,
        }
        
        op_func = ops.get(operator, lambda x: True)
        return [op_func(v) for v in col_data]
    
    def estimate_selectivity(self, params: Dict[str, Any]) -> float:
        """Estimate filter selectivity"""
        operator = params.get("operator", "eq")
        
        # Rough estimates
        selectivity_map = {
            "eq": 0.1,
            "ne": 0.9,
            "lt": 0.3,
            "le": 0.4,
            "gt": 0.3,
            "ge": 0.4,
            "in": 0.2,
            "like": 0.2,
            "is_null": 0.05,
            "is_not_null": 0.95,
        }
        
        return selectivity_map.get(operator, 0.5)


class ProjectOperator(StorageOperator):
    """Project (select) specific columns"""
    
    def execute(self, data: Dict[str, List], params: Dict[str, Any]) -> Dict[str, List]:
        """Execute projection"""
        columns = params.get("columns", [])
        
        if not columns:
            return data
        
        return {col: data[col] for col in columns if col in data}
    
    def estimate_selectivity(self, params: Dict[str, Any]) -> float:
        """Projection reduces data by column ratio"""
        return 1.0  # Row count unchanged


class AggregateOperator(StorageOperator):
    """Compute aggregates on data"""
    
    def execute(self, data: Dict[str, List], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute aggregation"""
        aggregates = params.get("aggregates", [])
        group_by = params.get("group_by", [])
        
        if group_by:
            return self._grouped_aggregate(data, aggregates, group_by)
        else:
            return self._simple_aggregate(data, aggregates)
    
    def _simple_aggregate(
        self,
        data: Dict[str, List],
        aggregates: List[Dict]
    ) -> Dict[str, Any]:
        """Simple aggregate without grouping"""
        result = {}
        
        for agg in aggregates:
            col = agg.get("column")
            func = AggregateFunction(agg.get("function", "count"))
            alias = agg.get("alias", f"{func.value}_{col}")
            
            if col and col in data:
                values = [v for v in data[col] if v is not None]
                result[alias] = self._compute_aggregate(values, func)
            elif func == AggregateFunction.COUNT:
                result[alias] = len(data[list(data.keys())[0]]) if data else 0
        
        return result
    
    def _grouped_aggregate(
        self,
        data: Dict[str, List],
        aggregates: List[Dict],
        group_by: List[str]
    ) -> Dict[str, List]:
        """Grouped aggregation"""
        if not data or not group_by:
            return {}
        
        # Build groups
        groups: Dict[tuple, Dict[str, List]] = {}
        num_rows = len(data[list(data.keys())[0]])
        
        for i in range(num_rows):
            # Create group key
            key = tuple(data[col][i] for col in group_by if col in data)
            
            if key not in groups:
                groups[key] = {col: [] for col in data}
            
            for col, values in data.items():
                groups[key][col].append(values[i])
        
        # Compute aggregates per group
        result = {col: [] for col in group_by}
        for agg in aggregates:
            alias = agg.get("alias", f"{agg.get('function')}_{agg.get('column')}")
            result[alias] = []
        
        for key, group_data in groups.items():
            # Add group key columns
            for idx, col in enumerate(group_by):
                result[col].append(key[idx])
            
            # Compute aggregates
            for agg in aggregates:
                col = agg.get("column")
                func = AggregateFunction(agg.get("function", "count"))
                alias = agg.get("alias", f"{func.value}_{col}")
                
                if col and col in group_data:
                    values = [v for v in group_data[col] if v is not None]
                    result[alias].append(self._compute_aggregate(values, func))
                elif func == AggregateFunction.COUNT:
                    result[alias].append(len(group_data[list(group_data.keys())[0]]))
        
        return result
    
    def _compute_aggregate(self, values: List, func: AggregateFunction) -> Any:
        """Compute single aggregate value"""
        if not values:
            return None
        
        if func == AggregateFunction.SUM:
            return sum(values)
        elif func == AggregateFunction.AVG:
            return sum(values) / len(values)
        elif func == AggregateFunction.MIN:
            return min(values)
        elif func == AggregateFunction.MAX:
            return max(values)
        elif func == AggregateFunction.COUNT:
            return len(values)
        elif func == AggregateFunction.COUNT_DISTINCT:
            return len(set(values))
        elif func == AggregateFunction.VARIANCE:
            mean = sum(values) / len(values)
            return sum((x - mean) ** 2 for x in values) / len(values)
        elif func == AggregateFunction.STDDEV:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
        return None
    
    def estimate_selectivity(self, params: Dict[str, Any]) -> float:
        """Aggregation significantly reduces data"""
        group_by = params.get("group_by", [])
        if group_by:
            return 0.1  # Grouped aggregation
        return 0.001  # Single result


class TransformOperator(StorageOperator):
    """Transform column values"""
    
    def execute(self, data: Dict[str, List], params: Dict[str, Any]) -> Dict[str, List]:
        """Execute transformation"""
        transforms = params.get("transforms", [])
        result = dict(data)
        
        for transform in transforms:
            col = transform.get("column")
            func = transform.get("function")
            new_col = transform.get("alias", col)
            
            if col in data:
                result[new_col] = self._apply_transform(data[col], func, transform)
        
        return result
    
    def _apply_transform(
        self,
        values: List,
        func: str,
        params: Dict
    ) -> List:
        """Apply transformation function"""
        transforms = {
            "upper": lambda x: str(x).upper() if x else x,
            "lower": lambda x: str(x).lower() if x else x,
            "abs": lambda x: abs(x) if x is not None else x,
            "round": lambda x: round(x, params.get("decimals", 0)) if x is not None else x,
            "sqrt": lambda x: x ** 0.5 if x is not None and x >= 0 else x,
            "log": lambda x: np.log(x) if x is not None and x > 0 else x,
            "exp": lambda x: np.exp(x) if x is not None else x,
        }
        
        transform_func = transforms.get(func, lambda x: x)
        return [transform_func(v) for v in values]
    
    def estimate_selectivity(self, params: Dict[str, Any]) -> float:
        """Transforms don't reduce rows"""
        return 1.0


# ============================================================================
# Storage Engine Simulation
# ============================================================================

class StorageEngine:
    """Simulated storage engine with near-data processing capability"""
    
    def __init__(self):
        self.tables: Dict[str, Dict[str, List]] = {}
        self.operators = {
            OperationType.FILTER: FilterOperator(),
            OperationType.PROJECT: ProjectOperator(),
            OperationType.AGGREGATE: AggregateOperator(),
            OperationType.TRANSFORM: TransformOperator(),
        }
    
    def create_table(self, name: str, data: Dict[str, List]) -> None:
        """Create table with data"""
        self.tables[name] = data
    
    def execute_near_data(
        self,
        table: str,
        operations: List[Tuple[OperationType, Dict[str, Any]]]
    ) -> Tuple[Any, int, int]:
        """Execute operations near data"""
        if table not in self.tables:
            raise NDPError(f"Table {table} not found")
        
        data = self.tables[table]
        initial_size = self._estimate_size(data)
        rows_processed = len(data[list(data.keys())[0]]) if data else 0
        
        # Execute operations in order
        for op_type, params in operations:
            if op_type in self.operators:
                data = self.operators[op_type].execute(data, params)
        
        final_size = self._estimate_size(data)
        rows_returned = len(data[list(data.keys())[0]]) if data and data.keys() else 0
        
        return data, rows_processed, rows_returned
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate data size in bytes"""
        if isinstance(data, dict):
            return sum(
                len(str(v).encode()) for values in data.values() 
                for v in values
            )
        return len(str(data).encode())


# ============================================================================
# Main Module Implementation
# ============================================================================

class NearDataProcessingModule:
    """
    Production-ready near-data processing module for OMNIXAN.
    
    Pushes computation to storage layer to reduce data movement:
    - Predicate pushdown (filtering)
    - Projection pushdown (column selection)
    - Aggregate pushdown (SUM, AVG, etc.)
    - Transform pushdown
    """
    
    def __init__(self, config: Optional[NDPConfig] = None):
        """Initialize the Near-Data Processing Module"""
        self.config = config or NDPConfig()
        self.storage = StorageEngine()
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.metrics = ProcessingMetrics()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._initialized = False
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the NDP module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing NearDataProcessingModule...")
            
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
            
            self._initialized = True
            self._logger.info("NearDataProcessingModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise NDPError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NDP operation"""
        if not self._initialized:
            raise NDPError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "register_table":
            name = params["name"]
            data = params["data"]
            self.register_table(name, data)
            return {"success": True}
        
        elif operation == "process":
            table = params["table"]
            operations = [
                (OperationType(op["type"]), op.get("params", {}))
                for op in params.get("operations", [])
            ]
            result = await self.process(table, operations)
            return {
                "success": result.success,
                "data": result.data,
                "rows_processed": result.rows_processed,
                "rows_returned": result.rows_returned,
                "execution_time_ms": result.execution_time_ms,
                "error": result.error
            }
        
        elif operation == "query":
            # High-level query interface
            table = params["table"]
            columns = params.get("columns")
            filters = params.get("filters", [])
            aggregates = params.get("aggregates", [])
            group_by = params.get("group_by", [])
            
            result = await self.query(table, columns, filters, aggregates, group_by)
            return {
                "success": result.success,
                "data": result.data,
                "rows_processed": result.rows_processed,
                "rows_returned": result.rows_returned,
                "execution_time_ms": result.execution_time_ms
            }
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def register_table(self, name: str, data: Dict[str, List]) -> None:
        """Register a table in storage"""
        self.storage.create_table(name, data)
        self._logger.info(f"Registered table: {name}")
    
    async def process(
        self,
        table: str,
        operations: List[Tuple[OperationType, Dict[str, Any]]]
    ) -> ProcessingResult:
        """Process data with pushdown operations"""
        task_id = str(uuid4())
        start_time = time.time()
        
        async with self._semaphore:
            try:
                # Decide processing location
                location = self._decide_location(operations)
                
                # Execute near-data
                data, rows_processed, rows_returned = self.storage.execute_near_data(
                    table, operations
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                # Update metrics
                self.metrics.total_tasks += 1
                self.metrics.successful_tasks += 1
                self.metrics.total_rows_processed += rows_processed
                
                result = ProcessingResult(
                    task_id=task_id,
                    success=True,
                    data=data,
                    rows_processed=rows_processed,
                    rows_returned=rows_returned,
                    execution_time_ms=execution_time,
                    location=location
                )
                
                # Update average execution time
                self._update_avg_time(execution_time)
                
                return result
            
            except Exception as e:
                self.metrics.total_tasks += 1
                self.metrics.failed_tasks += 1
                
                return ProcessingResult(
                    task_id=task_id,
                    success=False,
                    error=str(e)
                )
    
    async def query(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[Dict]] = None,
        aggregates: Optional[List[Dict]] = None,
        group_by: Optional[List[str]] = None
    ) -> ProcessingResult:
        """High-level query interface with automatic pushdown"""
        operations = []
        
        # Add filters (predicate pushdown)
        if filters and self.config.enable_predicate_pushdown:
            for f in filters:
                operations.append((
                    OperationType.FILTER,
                    {
                        "column": f.get("column"),
                        "operator": f.get("operator", "eq"),
                        "value": f.get("value")
                    }
                ))
        
        # Add aggregates (aggregate pushdown)
        if aggregates and self.config.enable_aggregate_pushdown:
            operations.append((
                OperationType.AGGREGATE,
                {
                    "aggregates": aggregates,
                    "group_by": group_by or []
                }
            ))
        
        # Add projection (projection pushdown)
        elif columns and self.config.enable_projection_pushdown:
            operations.append((
                OperationType.PROJECT,
                {"columns": columns}
            ))
        
        return await self.process(table, operations)
    
    def _decide_location(
        self,
        operations: List[Tuple[OperationType, Dict[str, Any]]]
    ) -> ProcessingLocation:
        """Decide where to execute operations"""
        # Calculate estimated selectivity
        total_selectivity = 1.0
        
        for op_type, params in operations:
            if op_type in self.storage.operators:
                selectivity = self.storage.operators[op_type].estimate_selectivity(params)
                total_selectivity *= selectivity
        
        # If high reduction, push to storage
        if total_selectivity < self.config.push_down_threshold:
            return ProcessingLocation.STORAGE
        else:
            return ProcessingLocation.HYBRID
    
    def _update_avg_time(self, new_time: float) -> None:
        """Update average execution time"""
        n = self.metrics.successful_tasks
        if n > 0:
            self.metrics.avg_execution_time_ms = (
                (self.metrics.avg_execution_time_ms * (n - 1) + new_time) / n
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return {
            "total_tasks": self.metrics.total_tasks,
            "successful_tasks": self.metrics.successful_tasks,
            "failed_tasks": self.metrics.failed_tasks,
            "success_rate": (
                self.metrics.successful_tasks / max(self.metrics.total_tasks, 1)
            ),
            "total_rows_processed": self.metrics.total_rows_processed,
            "avg_execution_time_ms": self.metrics.avg_execution_time_ms
        }
    
    async def shutdown(self) -> None:
        """Shutdown the NDP module"""
        self._logger.info("Shutting down NearDataProcessingModule...")
        
        self.storage.tables.clear()
        self._initialized = False
        
        self._logger.info("NearDataProcessingModule shutdown complete")


# Example usage
async def main():
    """Example usage of NearDataProcessingModule"""
    
    config = NDPConfig(
        enable_predicate_pushdown=True,
        enable_projection_pushdown=True,
        enable_aggregate_pushdown=True
    )
    
    module = NearDataProcessingModule(config)
    await module.initialize()
    
    try:
        # Register sample data
        data = {
            "id": list(range(100000)),
            "category": ["A", "B", "C", "D"] * 25000,
            "value": [float(i * 1.5) for i in range(100000)],
            "status": ["active", "inactive"] * 50000
        }
        module.register_table("sales", data)
        
        print("Processing with pushdown...")
        
        # Query with filters (predicate pushdown)
        result = await module.query(
            "sales",
            columns=["id", "category", "value"],
            filters=[
                {"column": "category", "operator": "eq", "value": "A"},
                {"column": "value", "operator": "gt", "value": 1000.0}
            ]
        )
        
        print(f"Filter result:")
        print(f"  Rows processed: {result.rows_processed}")
        print(f"  Rows returned: {result.rows_returned}")
        print(f"  Execution time: {result.execution_time_ms:.2f}ms")
        print(f"  Data reduction: {(1 - result.rows_returned/result.rows_processed)*100:.1f}%")
        
        # Aggregation with pushdown
        result = await module.query(
            "sales",
            aggregates=[
                {"column": "value", "function": "sum", "alias": "total_value"},
                {"column": "value", "function": "avg", "alias": "avg_value"},
                {"column": "id", "function": "count", "alias": "count"}
            ],
            group_by=["category"]
        )
        
        print(f"\nAggregate result:")
        print(f"  Rows processed: {result.rows_processed}")
        print(f"  Groups returned: {result.rows_returned}")
        print(f"  Data: {json.dumps(result.data, indent=2)}")
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        print(f"  Total tasks: {metrics['total_tasks']}")
        print(f"  Avg execution time: {metrics['avg_execution_time_ms']:.2f}ms")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

