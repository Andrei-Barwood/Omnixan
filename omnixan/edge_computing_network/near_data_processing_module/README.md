# Near-Data Processing Module

**Status: ✅ IMPLEMENTED**

Production-ready near-data processing (NDP) implementation that pushes computation closer to storage, reducing data movement and improving performance.

## Features

- **Pushdown Operations**
  - Predicate pushdown (filtering)
  - Projection pushdown (column selection)
  - Aggregate pushdown (SUM, AVG, COUNT, etc.)
  - Transform pushdown (data transformations)

- **Aggregate Functions**
  - SUM, AVG, MIN, MAX
  - COUNT, COUNT_DISTINCT
  - VARIANCE, STDDEV
  - Grouped aggregations

- **Performance**
  - Vectorized operations with NumPy
  - Concurrent task execution
  - Automatic processing location selection
  - Significant data reduction

## Quick Start

```python
from omnixan.edge_computing_network.near_data_processing_module.module import (
    NearDataProcessingModule,
    NDPConfig
)

# Initialize
config = NDPConfig(
    enable_predicate_pushdown=True,
    enable_projection_pushdown=True,
    enable_aggregate_pushdown=True
)

module = NearDataProcessingModule(config)
await module.initialize()

# Register data
data = {
    "id": list(range(1000000)),
    "category": ["A", "B", "C"] * 333334,
    "value": [float(i) for i in range(1000000)],
    "status": ["active", "inactive"] * 500000
}
module.register_table("sales", data)

# Query with filters (pushdown)
result = await module.query(
    "sales",
    columns=["id", "value"],
    filters=[
        {"column": "category", "operator": "eq", "value": "A"},
        {"column": "value", "operator": "gt", "value": 1000}
    ]
)

print(f"Rows processed: {result.rows_processed}")
print(f"Rows returned: {result.rows_returned}")
print(f"Data reduction: {(1 - result.rows_returned/result.rows_processed)*100:.1f}%")

# Aggregation query
result = await module.query(
    "sales",
    aggregates=[
        {"column": "value", "function": "sum", "alias": "total"},
        {"column": "value", "function": "avg", "alias": "average"}
    ],
    group_by=["category"]
)

await module.shutdown()
```

## Supported Operations

### Filter Operators
| Operator | Description |
|----------|-------------|
| `eq` | Equal to |
| `ne` | Not equal to |
| `lt` | Less than |
| `le` | Less than or equal |
| `gt` | Greater than |
| `ge` | Greater than or equal |
| `in` | In list |
| `like` | Contains string |
| `is_null` | Is NULL |
| `is_not_null` | Is not NULL |

### Aggregate Functions
| Function | Description |
|----------|-------------|
| `sum` | Sum of values |
| `avg` | Average |
| `min` | Minimum |
| `max` | Maximum |
| `count` | Count rows |
| `count_distinct` | Count unique values |
| `variance` | Population variance |
| `stddev` | Standard deviation |

### Transform Functions
| Function | Description |
|----------|-------------|
| `upper` | Uppercase string |
| `lower` | Lowercase string |
| `abs` | Absolute value |
| `round` | Round to decimals |
| `sqrt` | Square root |
| `log` | Natural logarithm |
| `exp` | Exponential |

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                      Query                             │
│   SELECT SUM(value) FROM sales WHERE category = 'A'   │
└───────────────────────┬────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│                  Query Optimizer                        │
│  • Predicate pushdown decision                         │
│  • Selectivity estimation                              │
│  • Processing location selection                       │
└───────────────────────┬────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│               Storage Engine (NDP)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Filter    │→ │  Project    │→ │  Aggregate  │    │
│  │category='A' │  │   value     │  │  SUM(value) │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│         ↓                                              │
│  Data scanned: 1,000,000 rows                          │
│  Data returned: 1 row (aggregated result)              │
└────────────────────────────────────────────────────────┘
```

## Benefits

- **Reduced Data Movement**: Filter at storage, not compute
- **Lower Network I/O**: Only relevant data traverses network
- **Better Latency**: Fewer bytes to transfer
- **CPU Efficiency**: Leverage storage processor

## Configuration

```python
NDPConfig(
    max_concurrent_tasks=8,       # Parallel task limit
    push_down_threshold=0.5,      # Selectivity threshold
    enable_predicate_pushdown=True,
    enable_projection_pushdown=True,
    enable_aggregate_pushdown=True,
    batch_size=10000,             # Processing batch size
    vectorized=True               # Use vectorized ops
)
```

## Integration

Part of OMNIXAN Edge Computing Network for efficient data processing at the edge.
