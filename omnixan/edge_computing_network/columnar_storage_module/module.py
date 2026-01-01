"""
OMNIXAN Columnar Storage Module
edge_computing_network/columnar_storage_module

Production-ready columnar storage implementation optimized for analytical
workloads with compression, encoding, predicate pushdown, and vectorized
query execution.
"""

import asyncio
import logging
import time
import struct
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from pathlib import Path
import json

import numpy as np

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataType(str, Enum):
    """Supported data types"""
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    BINARY = "binary"


class EncodingType(str, Enum):
    """Column encoding types"""
    PLAIN = "plain"
    DICTIONARY = "dictionary"
    RLE = "rle"  # Run-length encoding
    DELTA = "delta"  # Delta encoding for sorted data
    BIT_PACKED = "bit_packed"


class CompressionType(str, Enum):
    """Compression algorithms"""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    SNAPPY = "snappy"
    ZSTD = "zstd"


class PredicateOperator(str, Enum):
    """Predicate operators for filtering"""
    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    LT = "lt"  # Less than
    LE = "le"  # Less than or equal
    GT = "gt"  # Greater than
    GE = "ge"  # Greater than or equal
    IN = "in"  # In set
    BETWEEN = "between"
    LIKE = "like"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


@dataclass
class ColumnStatistics:
    """Statistics for a column chunk"""
    min_value: Any = None
    max_value: Any = None
    null_count: int = 0
    distinct_count: int = 0
    total_count: int = 0
    sum_value: Optional[float] = None
    avg_value: Optional[float] = None


@dataclass
class ColumnChunk:
    """A chunk of column data"""
    column_name: str
    data_type: DataType
    encoding: EncodingType
    compression: CompressionType
    data: bytes
    num_values: int
    statistics: ColumnStatistics
    dictionary: Optional[Dict[int, Any]] = None
    null_bitmap: Optional[bytes] = None


@dataclass
class RowGroup:
    """A row group containing multiple column chunks"""
    row_group_id: int
    num_rows: int
    columns: Dict[str, ColumnChunk]
    total_byte_size: int = 0


@dataclass
class TableSchema:
    """Schema definition for a table"""
    table_name: str
    columns: Dict[str, DataType]
    primary_key: Optional[List[str]] = None
    partition_columns: Optional[List[str]] = None
    sort_columns: Optional[List[str]] = None


@dataclass
class QueryResult:
    """Result of a query"""
    columns: Dict[str, List[Any]]
    num_rows: int
    execution_time_ms: float
    rows_scanned: int
    rows_filtered: int
    bytes_scanned: int


class ColumnarStorageConfig(BaseModel):
    """Configuration for columnar storage"""
    row_group_size: int = Field(
        default=100000,
        ge=1000,
        le=10000000,
        description="Number of rows per row group"
    )
    page_size: int = Field(
        default=8192,
        ge=1024,
        le=1048576,
        description="Page size in bytes"
    )
    default_compression: CompressionType = Field(
        default=CompressionType.ZLIB,
        description="Default compression algorithm"
    )
    default_encoding: EncodingType = Field(
        default=EncodingType.PLAIN,
        description="Default encoding type"
    )
    enable_statistics: bool = Field(
        default=True,
        description="Collect column statistics"
    )
    enable_dictionary: bool = Field(
        default=True,
        description="Enable dictionary encoding for low-cardinality columns"
    )
    dictionary_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Max cardinality ratio for dictionary encoding"
    )
    enable_bloom_filter: bool = Field(
        default=False,
        description="Enable bloom filters"
    )
    storage_path: str = Field(
        default="/tmp/omnixan_columnar",
        description="Base storage path"
    )


class ColumnarStorageError(Exception):
    """Base exception for columnar storage errors"""
    pass


class SchemaError(ColumnarStorageError):
    """Raised when schema operation fails"""
    pass


class DataError(ColumnarStorageError):
    """Raised when data operation fails"""
    pass


# ============================================================================
# Encoding Implementations
# ============================================================================

class Encoder(ABC):
    """Abstract base class for encoders"""
    
    @abstractmethod
    def encode(self, values: List[Any], data_type: DataType) -> Tuple[bytes, Optional[Dict]]:
        """Encode values to bytes"""
        pass
    
    @abstractmethod
    def decode(self, data: bytes, data_type: DataType, num_values: int, 
               dictionary: Optional[Dict] = None) -> List[Any]:
        """Decode bytes to values"""
        pass


class PlainEncoder(Encoder):
    """Plain encoding - values stored as-is"""
    
    def encode(self, values: List[Any], data_type: DataType) -> Tuple[bytes, Optional[Dict]]:
        """Encode values using plain encoding"""
        if data_type == DataType.INT32:
            return struct.pack(f'<{len(values)}i', *values), None
        elif data_type == DataType.INT64:
            return struct.pack(f'<{len(values)}q', *values), None
        elif data_type == DataType.FLOAT32:
            return struct.pack(f'<{len(values)}f', *values), None
        elif data_type == DataType.FLOAT64:
            return struct.pack(f'<{len(values)}d', *values), None
        elif data_type == DataType.BOOLEAN:
            return bytes([1 if v else 0 for v in values]), None
        elif data_type == DataType.STRING:
            # Length-prefixed strings
            result = bytearray()
            for s in values:
                s_bytes = s.encode('utf-8') if s else b''
                result.extend(struct.pack('<I', len(s_bytes)))
                result.extend(s_bytes)
            return bytes(result), None
        else:
            # Fallback to JSON
            return json.dumps(values).encode('utf-8'), None
    
    def decode(self, data: bytes, data_type: DataType, num_values: int,
               dictionary: Optional[Dict] = None) -> List[Any]:
        """Decode plain encoded data"""
        if data_type == DataType.INT32:
            return list(struct.unpack(f'<{num_values}i', data[:num_values * 4]))
        elif data_type == DataType.INT64:
            return list(struct.unpack(f'<{num_values}q', data[:num_values * 8]))
        elif data_type == DataType.FLOAT32:
            return list(struct.unpack(f'<{num_values}f', data[:num_values * 4]))
        elif data_type == DataType.FLOAT64:
            return list(struct.unpack(f'<{num_values}d', data[:num_values * 8]))
        elif data_type == DataType.BOOLEAN:
            return [bool(b) for b in data[:num_values]]
        elif data_type == DataType.STRING:
            result = []
            offset = 0
            for _ in range(num_values):
                length = struct.unpack('<I', data[offset:offset + 4])[0]
                offset += 4
                result.append(data[offset:offset + length].decode('utf-8'))
                offset += length
            return result
        else:
            return json.loads(data.decode('utf-8'))


class DictionaryEncoder(Encoder):
    """Dictionary encoding for low-cardinality columns"""
    
    def encode(self, values: List[Any], data_type: DataType) -> Tuple[bytes, Optional[Dict]]:
        """Encode using dictionary"""
        # Build dictionary
        unique_values = list(set(values))
        value_to_id = {v: i for i, v in enumerate(unique_values)}
        id_to_value = {i: v for i, v in enumerate(unique_values)}
        
        # Encode indices
        indices = [value_to_id[v] for v in values]
        
        # Use smallest integer type that fits
        max_id = len(unique_values) - 1
        if max_id < 256:
            fmt = 'B'  # uint8
        elif max_id < 65536:
            fmt = 'H'  # uint16
        else:
            fmt = 'I'  # uint32
        
        data = struct.pack(f'<{len(indices)}{fmt}', *indices)
        
        return data, id_to_value
    
    def decode(self, data: bytes, data_type: DataType, num_values: int,
               dictionary: Optional[Dict] = None) -> List[Any]:
        """Decode dictionary encoded data"""
        if dictionary is None:
            raise DataError("Dictionary required for decoding")
        
        # Determine index type from data size
        bytes_per_value = len(data) // num_values
        if bytes_per_value == 1:
            fmt = 'B'
        elif bytes_per_value == 2:
            fmt = 'H'
        else:
            fmt = 'I'
        
        indices = struct.unpack(f'<{num_values}{fmt}', data[:num_values * bytes_per_value])
        return [dictionary[i] for i in indices]


class RLEEncoder(Encoder):
    """Run-length encoding for columns with repeated values"""
    
    def encode(self, values: List[Any], data_type: DataType) -> Tuple[bytes, Optional[Dict]]:
        """Encode using RLE"""
        if not values:
            return b'', None
        
        # Encode runs
        runs = []
        current_value = values[0]
        count = 1
        
        for value in values[1:]:
            if value == current_value:
                count += 1
            else:
                runs.append((current_value, count))
                current_value = value
                count = 1
        runs.append((current_value, count))
        
        # Serialize runs
        result = bytearray()
        result.extend(struct.pack('<I', len(runs)))
        
        plain_encoder = PlainEncoder()
        for value, count in runs:
            result.extend(struct.pack('<I', count))
            value_bytes, _ = plain_encoder.encode([value], data_type)
            result.extend(struct.pack('<I', len(value_bytes)))
            result.extend(value_bytes)
        
        return bytes(result), None
    
    def decode(self, data: bytes, data_type: DataType, num_values: int,
               dictionary: Optional[Dict] = None) -> List[Any]:
        """Decode RLE encoded data"""
        result = []
        offset = 0
        
        num_runs = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        
        plain_encoder = PlainEncoder()
        
        for _ in range(num_runs):
            count = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            value_len = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            value = plain_encoder.decode(data[offset:offset + value_len], data_type, 1)[0]
            offset += value_len
            result.extend([value] * count)
        
        return result[:num_values]


# ============================================================================
# Compression
# ============================================================================

class Compressor:
    """Handles data compression and decompression"""
    
    @staticmethod
    def compress(data: bytes, compression: CompressionType) -> bytes:
        """Compress data"""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.ZLIB:
            return zlib.compress(data, level=6)
        else:
            # Fallback to zlib for unsupported types
            return zlib.compress(data, level=6)
    
    @staticmethod
    def decompress(data: bytes, compression: CompressionType) -> bytes:
        """Decompress data"""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.ZLIB:
            return zlib.decompress(data)
        else:
            return zlib.decompress(data)


# ============================================================================
# Column Writer/Reader
# ============================================================================

class ColumnWriter:
    """Writes column data with encoding and compression"""
    
    def __init__(self, config: ColumnarStorageConfig):
        self.config = config
        self.encoders = {
            EncodingType.PLAIN: PlainEncoder(),
            EncodingType.DICTIONARY: DictionaryEncoder(),
            EncodingType.RLE: RLEEncoder(),
        }
    
    def write_column(
        self,
        column_name: str,
        values: List[Any],
        data_type: DataType,
        encoding: Optional[EncodingType] = None,
        compression: Optional[CompressionType] = None
    ) -> ColumnChunk:
        """Write column data to a chunk"""
        encoding = encoding or self._select_encoding(values, data_type)
        compression = compression or self.config.default_compression
        
        # Calculate statistics
        statistics = self._calculate_statistics(values, data_type)
        
        # Encode
        encoder = self.encoders.get(encoding, self.encoders[EncodingType.PLAIN])
        encoded_data, dictionary = encoder.encode(values, data_type)
        
        # Compress
        compressed_data = Compressor.compress(encoded_data, compression)
        
        # Create null bitmap if needed
        null_bitmap = None
        if any(v is None for v in values):
            null_bitmap = bytes([1 if v is not None else 0 for v in values])
        
        return ColumnChunk(
            column_name=column_name,
            data_type=data_type,
            encoding=encoding,
            compression=compression,
            data=compressed_data,
            num_values=len(values),
            statistics=statistics,
            dictionary=dictionary,
            null_bitmap=null_bitmap
        )
    
    def _select_encoding(self, values: List[Any], data_type: DataType) -> EncodingType:
        """Select best encoding for data"""
        if not values:
            return EncodingType.PLAIN
        
        # Check for dictionary encoding
        if self.config.enable_dictionary:
            unique_count = len(set(values))
            cardinality_ratio = unique_count / len(values)
            
            if cardinality_ratio < self.config.dictionary_threshold:
                return EncodingType.DICTIONARY
        
        # Check for RLE (good for sorted data with repeats)
        if data_type in [DataType.INT32, DataType.INT64, DataType.STRING]:
            run_count = 1
            for i in range(1, len(values)):
                if values[i] != values[i - 1]:
                    run_count += 1
            
            compression_ratio = run_count / len(values)
            if compression_ratio < 0.3:  # More than 70% compression from RLE
                return EncodingType.RLE
        
        return EncodingType.PLAIN
    
    def _calculate_statistics(self, values: List[Any], data_type: DataType) -> ColumnStatistics:
        """Calculate column statistics"""
        if not self.config.enable_statistics or not values:
            return ColumnStatistics(total_count=len(values))
        
        non_null = [v for v in values if v is not None]
        
        stats = ColumnStatistics(
            null_count=len(values) - len(non_null),
            total_count=len(values),
            distinct_count=len(set(non_null)) if non_null else 0
        )
        
        if non_null:
            if data_type in [DataType.INT32, DataType.INT64, DataType.FLOAT32, DataType.FLOAT64]:
                stats.min_value = min(non_null)
                stats.max_value = max(non_null)
                stats.sum_value = sum(non_null)
                stats.avg_value = stats.sum_value / len(non_null)
            elif data_type == DataType.STRING:
                stats.min_value = min(non_null)
                stats.max_value = max(non_null)
        
        return stats


class ColumnReader:
    """Reads column data with decoding and decompression"""
    
    def __init__(self, config: ColumnarStorageConfig):
        self.config = config
        self.encoders = {
            EncodingType.PLAIN: PlainEncoder(),
            EncodingType.DICTIONARY: DictionaryEncoder(),
            EncodingType.RLE: RLEEncoder(),
        }
    
    def read_column(self, chunk: ColumnChunk) -> List[Any]:
        """Read column data from chunk"""
        # Decompress
        decompressed = Compressor.decompress(chunk.data, chunk.compression)
        
        # Decode
        encoder = self.encoders.get(chunk.encoding, self.encoders[EncodingType.PLAIN])
        values = encoder.decode(
            decompressed,
            chunk.data_type,
            chunk.num_values,
            chunk.dictionary
        )
        
        # Apply null bitmap
        if chunk.null_bitmap:
            values = [
                values[i] if chunk.null_bitmap[i] else None
                for i in range(len(values))
            ]
        
        return values


# ============================================================================
# Predicate Pushdown
# ============================================================================

@dataclass
class Predicate:
    """Query predicate for filtering"""
    column: str
    operator: PredicateOperator
    value: Any
    value2: Optional[Any] = None  # For BETWEEN


class PredicateEvaluator:
    """Evaluates predicates against data"""
    
    @staticmethod
    def can_skip_chunk(predicate: Predicate, stats: ColumnStatistics) -> bool:
        """Check if chunk can be skipped based on statistics"""
        if stats.min_value is None or stats.max_value is None:
            return False
        
        if predicate.operator == PredicateOperator.EQ:
            return (predicate.value < stats.min_value or 
                    predicate.value > stats.max_value)
        
        elif predicate.operator == PredicateOperator.LT:
            return predicate.value <= stats.min_value
        
        elif predicate.operator == PredicateOperator.LE:
            return predicate.value < stats.min_value
        
        elif predicate.operator == PredicateOperator.GT:
            return predicate.value >= stats.max_value
        
        elif predicate.operator == PredicateOperator.GE:
            return predicate.value > stats.max_value
        
        elif predicate.operator == PredicateOperator.BETWEEN:
            return (predicate.value2 < stats.min_value or
                    predicate.value > stats.max_value)
        
        return False
    
    @staticmethod
    def evaluate(predicate: Predicate, value: Any) -> bool:
        """Evaluate predicate against single value"""
        if value is None:
            if predicate.operator == PredicateOperator.IS_NULL:
                return True
            elif predicate.operator == PredicateOperator.IS_NOT_NULL:
                return False
            return False
        
        if predicate.operator == PredicateOperator.EQ:
            return value == predicate.value
        elif predicate.operator == PredicateOperator.NE:
            return value != predicate.value
        elif predicate.operator == PredicateOperator.LT:
            return value < predicate.value
        elif predicate.operator == PredicateOperator.LE:
            return value <= predicate.value
        elif predicate.operator == PredicateOperator.GT:
            return value > predicate.value
        elif predicate.operator == PredicateOperator.GE:
            return value >= predicate.value
        elif predicate.operator == PredicateOperator.IN:
            return value in predicate.value
        elif predicate.operator == PredicateOperator.BETWEEN:
            return predicate.value <= value <= predicate.value2
        elif predicate.operator == PredicateOperator.LIKE:
            import re
            pattern = predicate.value.replace('%', '.*').replace('_', '.')
            return bool(re.match(pattern, str(value)))
        elif predicate.operator == PredicateOperator.IS_NULL:
            return False
        elif predicate.operator == PredicateOperator.IS_NOT_NULL:
            return True
        
        return True


# ============================================================================
# Main Module Implementation
# ============================================================================

class ColumnarStorageModule:
    """
    Production-ready columnar storage module for OMNIXAN.
    
    Provides optimized columnar storage with:
    - Multiple encoding schemes (plain, dictionary, RLE)
    - Compression support
    - Predicate pushdown
    - Column statistics
    - Vectorized operations
    """
    
    def __init__(self, config: Optional[ColumnarStorageConfig] = None):
        """Initialize the Columnar Storage Module"""
        self.config = config or ColumnarStorageConfig()
        self.tables: Dict[str, TableSchema] = {}
        self.data: Dict[str, List[RowGroup]] = {}
        self.writer = ColumnWriter(self.config)
        self.reader = ColumnReader(self.config)
        self._initialized = False
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the columnar storage module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing ColumnarStorageModule...")
            
            # Create storage directory
            Path(self.config.storage_path).mkdir(parents=True, exist_ok=True)
            
            self._initialized = True
            self._logger.info("ColumnarStorageModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise ColumnarStorageError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute columnar storage operation"""
        if not self._initialized:
            raise ColumnarStorageError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "create_table":
            schema = TableSchema(
                table_name=params["table_name"],
                columns={k: DataType(v) for k, v in params["columns"].items()},
                primary_key=params.get("primary_key"),
                partition_columns=params.get("partition_columns"),
                sort_columns=params.get("sort_columns")
            )
            self.create_table(schema)
            return {"success": True}
        
        elif operation == "insert":
            table_name = params["table_name"]
            rows = params["rows"]
            await self.insert(table_name, rows)
            return {"success": True, "rows_inserted": len(rows)}
        
        elif operation == "select":
            table_name = params["table_name"]
            columns = params.get("columns")
            predicates = [
                Predicate(
                    column=p["column"],
                    operator=PredicateOperator(p["operator"]),
                    value=p["value"],
                    value2=p.get("value2")
                )
                for p in params.get("predicates", [])
            ]
            limit = params.get("limit")
            
            result = await self.select(table_name, columns, predicates, limit)
            return {
                "columns": result.columns,
                "num_rows": result.num_rows,
                "execution_time_ms": result.execution_time_ms,
                "rows_scanned": result.rows_scanned
            }
        
        elif operation == "get_statistics":
            table_name = params["table_name"]
            stats = self.get_statistics(table_name)
            return stats
        
        elif operation == "list_tables":
            return {"tables": list(self.tables.keys())}
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def create_table(self, schema: TableSchema) -> None:
        """Create a new table"""
        if schema.table_name in self.tables:
            raise SchemaError(f"Table {schema.table_name} already exists")
        
        self.tables[schema.table_name] = schema
        self.data[schema.table_name] = []
        
        self._logger.info(f"Created table: {schema.table_name}")
    
    async def insert(self, table_name: str, rows: List[Dict[str, Any]]) -> None:
        """Insert rows into table"""
        if table_name not in self.tables:
            raise SchemaError(f"Table {table_name} not found")
        
        schema = self.tables[table_name]
        
        # Group rows into row groups
        for i in range(0, len(rows), self.config.row_group_size):
            batch = rows[i:i + self.config.row_group_size]
            row_group = await self._create_row_group(schema, batch, len(self.data[table_name]))
            self.data[table_name].append(row_group)
        
        self._logger.info(f"Inserted {len(rows)} rows into {table_name}")
    
    async def _create_row_group(
        self,
        schema: TableSchema,
        rows: List[Dict[str, Any]],
        row_group_id: int
    ) -> RowGroup:
        """Create a row group from rows"""
        columns = {}
        total_size = 0
        
        for col_name, col_type in schema.columns.items():
            # Extract column values
            values = [row.get(col_name) for row in rows]
            
            # Write column chunk
            chunk = self.writer.write_column(col_name, values, col_type)
            columns[col_name] = chunk
            total_size += len(chunk.data)
        
        return RowGroup(
            row_group_id=row_group_id,
            num_rows=len(rows),
            columns=columns,
            total_byte_size=total_size
        )
    
    async def select(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        predicates: Optional[List[Predicate]] = None,
        limit: Optional[int] = None
    ) -> QueryResult:
        """Select data from table with predicate pushdown"""
        start_time = time.time()
        
        if table_name not in self.tables:
            raise SchemaError(f"Table {table_name} not found")
        
        schema = self.tables[table_name]
        columns = columns or list(schema.columns.keys())
        predicates = predicates or []
        
        result_columns = {col: [] for col in columns}
        rows_scanned = 0
        rows_filtered = 0
        bytes_scanned = 0
        
        for row_group in self.data[table_name]:
            # Check if row group can be skipped using statistics
            skip_row_group = False
            for predicate in predicates:
                if predicate.column in row_group.columns:
                    chunk = row_group.columns[predicate.column]
                    if PredicateEvaluator.can_skip_chunk(predicate, chunk.statistics):
                        skip_row_group = True
                        break
            
            if skip_row_group:
                continue
            
            # Read required columns
            column_data = {}
            for col in set(columns + [p.column for p in predicates]):
                if col in row_group.columns:
                    chunk = row_group.columns[col]
                    column_data[col] = self.reader.read_column(chunk)
                    bytes_scanned += len(chunk.data)
            
            # Apply predicates
            for i in range(row_group.num_rows):
                rows_scanned += 1
                
                # Check all predicates
                passes = True
                for predicate in predicates:
                    if predicate.column in column_data:
                        value = column_data[predicate.column][i]
                        if not PredicateEvaluator.evaluate(predicate, value):
                            passes = False
                            rows_filtered += 1
                            break
                
                if passes:
                    # Add row to result
                    for col in columns:
                        if col in column_data:
                            result_columns[col].append(column_data[col][i])
                    
                    # Check limit
                    if limit and len(result_columns[columns[0]]) >= limit:
                        break
            
            if limit and len(result_columns[columns[0]]) >= limit:
                break
        
        execution_time = (time.time() - start_time) * 1000
        
        return QueryResult(
            columns=result_columns,
            num_rows=len(result_columns[columns[0]]) if columns else 0,
            execution_time_ms=execution_time,
            rows_scanned=rows_scanned,
            rows_filtered=rows_filtered,
            bytes_scanned=bytes_scanned
        )
    
    def get_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get table statistics"""
        if table_name not in self.tables:
            raise SchemaError(f"Table {table_name} not found")
        
        schema = self.tables[table_name]
        row_groups = self.data[table_name]
        
        total_rows = sum(rg.num_rows for rg in row_groups)
        total_bytes = sum(rg.total_byte_size for rg in row_groups)
        
        column_stats = {}
        for col_name in schema.columns:
            col_chunks = [rg.columns[col_name] for rg in row_groups if col_name in rg.columns]
            
            if col_chunks:
                column_stats[col_name] = {
                    "data_type": col_chunks[0].data_type.value,
                    "encoding": col_chunks[0].encoding.value,
                    "compression": col_chunks[0].compression.value,
                    "total_values": sum(c.num_values for c in col_chunks),
                    "null_count": sum(c.statistics.null_count for c in col_chunks),
                    "distinct_count": col_chunks[0].statistics.distinct_count,
                    "min_value": col_chunks[0].statistics.min_value,
                    "max_value": col_chunks[-1].statistics.max_value
                }
        
        return {
            "table_name": table_name,
            "num_rows": total_rows,
            "num_row_groups": len(row_groups),
            "total_bytes": total_bytes,
            "columns": column_stats
        }
    
    async def shutdown(self) -> None:
        """Shutdown the columnar storage module"""
        self._logger.info("Shutting down ColumnarStorageModule...")
        
        self.tables.clear()
        self.data.clear()
        self._initialized = False
        
        self._logger.info("ColumnarStorageModule shutdown complete")


# Example usage
async def main():
    """Example usage of ColumnarStorageModule"""
    
    config = ColumnarStorageConfig(
        row_group_size=1000,
        default_compression=CompressionType.ZLIB,
        enable_dictionary=True
    )
    
    module = ColumnarStorageModule(config)
    await module.initialize()
    
    try:
        # Create table
        schema = TableSchema(
            table_name="events",
            columns={
                "id": DataType.INT64,
                "timestamp": DataType.INT64,
                "user_id": DataType.STRING,
                "event_type": DataType.STRING,
                "value": DataType.FLOAT64
            },
            sort_columns=["timestamp"]
        )
        module.create_table(schema)
        
        # Insert data
        rows = [
            {
                "id": i,
                "timestamp": 1700000000 + i * 1000,
                "user_id": f"user_{i % 100}",
                "event_type": ["click", "view", "purchase"][i % 3],
                "value": float(i * 1.5)
            }
            for i in range(10000)
        ]
        
        await module.insert("events", rows)
        print(f"Inserted {len(rows)} rows")
        
        # Query with predicate pushdown
        result = await module.select(
            "events",
            columns=["id", "user_id", "event_type", "value"],
            predicates=[
                Predicate(
                    column="event_type",
                    operator=PredicateOperator.EQ,
                    value="purchase"
                ),
                Predicate(
                    column="value",
                    operator=PredicateOperator.GT,
                    value=100.0
                )
            ],
            limit=10
        )
        
        print(f"\nQuery Result:")
        print(f"Rows returned: {result.num_rows}")
        print(f"Rows scanned: {result.rows_scanned}")
        print(f"Rows filtered: {result.rows_filtered}")
        print(f"Execution time: {result.execution_time_ms:.2f}ms")
        
        # Get statistics
        stats = module.get_statistics("events")
        print(f"\nTable Statistics:")
        print(json.dumps(stats, indent=2, default=str))
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

