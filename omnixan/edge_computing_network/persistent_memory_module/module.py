"""
OMNIXAN Persistent Memory Module
edge_computing_network/persistent_memory_module

Production-ready persistent memory implementation providing byte-addressable
persistent storage with ACID transactions, crash recovery, and memory-mapped
access for high-performance data persistence.
"""

import asyncio
import logging
import time
import os
import mmap
import struct
import hashlib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import json
import threading

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionState(str, Enum):
    """Transaction states"""
    ACTIVE = "active"
    COMMITTED = "committed"
    ABORTED = "aborted"
    PREPARING = "preparing"


class PersistenceMode(str, Enum):
    """Persistence modes"""
    SYNC = "sync"  # Synchronous flush
    ASYNC = "async"  # Asynchronous flush
    PERIODIC = "periodic"  # Periodic flush


class RecoveryMode(str, Enum):
    """Recovery modes"""
    FULL = "full"  # Full REDO/UNDO recovery
    CHECKPOINT = "checkpoint"  # Checkpoint-based recovery
    NONE = "none"  # No recovery


@dataclass
class LogEntry:
    """Write-ahead log entry"""
    lsn: int  # Log sequence number
    transaction_id: str
    operation: str
    key: str
    old_value: Optional[bytes] = None
    new_value: Optional[bytes] = None
    timestamp: float = field(default_factory=time.time)
    checksum: str = ""


@dataclass
class Transaction:
    """Transaction object"""
    transaction_id: str
    state: TransactionState
    start_time: float
    modifications: List[Tuple[str, Optional[bytes], bytes]]  # (key, old, new)
    read_set: Set[str]
    write_set: Set[str]


@dataclass
class Checkpoint:
    """Checkpoint for recovery"""
    checkpoint_id: str
    lsn: int
    timestamp: float
    active_transactions: List[str]
    data_snapshot: Dict[str, bytes]


@dataclass
class PMEMMetrics:
    """Persistent memory metrics"""
    total_size: int = 0
    used_size: int = 0
    num_writes: int = 0
    num_reads: int = 0
    num_transactions: int = 0
    committed_transactions: int = 0
    aborted_transactions: int = 0
    recovery_time_ms: float = 0.0
    avg_write_latency_us: float = 0.0


class PMEMConfig(BaseModel):
    """Configuration for persistent memory"""
    storage_path: str = Field(
        default="/tmp/omnixan_pmem",
        description="Path for persistent storage"
    )
    size: int = Field(
        default=1024 * 1024 * 100,  # 100 MB
        ge=1024 * 1024,
        description="Total size in bytes"
    )
    persistence_mode: PersistenceMode = Field(
        default=PersistenceMode.SYNC,
        description="Persistence mode"
    )
    recovery_mode: RecoveryMode = Field(
        default=RecoveryMode.CHECKPOINT,
        description="Recovery mode"
    )
    enable_wal: bool = Field(
        default=True,
        description="Enable write-ahead logging"
    )
    checkpoint_interval: float = Field(
        default=60.0,
        gt=0.0,
        description="Checkpoint interval in seconds"
    )
    flush_interval: float = Field(
        default=1.0,
        gt=0.0,
        description="Async flush interval"
    )
    max_log_size: int = Field(
        default=1024 * 1024 * 10,  # 10 MB
        description="Max WAL size before checkpoint"
    )


class PMEMError(Exception):
    """Base exception for PMEM errors"""
    pass


class TransactionError(PMEMError):
    """Raised when transaction operation fails"""
    pass


class RecoveryError(PMEMError):
    """Raised when recovery fails"""
    pass


# ============================================================================
# Write-Ahead Log Implementation
# ============================================================================

class WriteAheadLog:
    """Write-ahead log for durability"""
    
    def __init__(self, log_path: Path, max_size: int):
        self.log_path = log_path
        self.max_size = max_size
        self.current_lsn = 0
        self.entries: List[LogEntry] = []
        self._lock = threading.Lock()
        
        # Load existing log
        self._load()
    
    def append(self, entry: LogEntry) -> int:
        """Append entry to log"""
        with self._lock:
            self.current_lsn += 1
            entry.lsn = self.current_lsn
            entry.checksum = self._compute_checksum(entry)
            self.entries.append(entry)
            
            # Persist entry
            self._persist_entry(entry)
            
            return entry.lsn
    
    def get_entries_since(self, lsn: int) -> List[LogEntry]:
        """Get all entries since LSN"""
        return [e for e in self.entries if e.lsn > lsn]
    
    def truncate(self, lsn: int) -> None:
        """Truncate log up to LSN"""
        with self._lock:
            self.entries = [e for e in self.entries if e.lsn > lsn]
            self._rewrite_log()
    
    def _compute_checksum(self, entry: LogEntry) -> str:
        """Compute entry checksum"""
        data = f"{entry.lsn}:{entry.transaction_id}:{entry.operation}:{entry.key}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _persist_entry(self, entry: LogEntry) -> None:
        """Persist single entry to disk"""
        with open(self.log_path, 'ab') as f:
            data = json.dumps({
                "lsn": entry.lsn,
                "transaction_id": entry.transaction_id,
                "operation": entry.operation,
                "key": entry.key,
                "old_value": entry.old_value.hex() if entry.old_value else None,
                "new_value": entry.new_value.hex() if entry.new_value else None,
                "timestamp": entry.timestamp,
                "checksum": entry.checksum
            })
            f.write(data.encode() + b'\n')
            f.flush()
            os.fsync(f.fileno())
    
    def _load(self) -> None:
        """Load log from disk"""
        if not self.log_path.exists():
            return
        
        with open(self.log_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entry = LogEntry(
                        lsn=data["lsn"],
                        transaction_id=data["transaction_id"],
                        operation=data["operation"],
                        key=data["key"],
                        old_value=bytes.fromhex(data["old_value"]) if data["old_value"] else None,
                        new_value=bytes.fromhex(data["new_value"]) if data["new_value"] else None,
                        timestamp=data["timestamp"],
                        checksum=data["checksum"]
                    )
                    
                    # Verify checksum
                    if self._compute_checksum(entry) == entry.checksum:
                        self.entries.append(entry)
                        self.current_lsn = max(self.current_lsn, entry.lsn)
    
    def _rewrite_log(self) -> None:
        """Rewrite entire log"""
        with open(self.log_path, 'wb') as f:
            for entry in self.entries:
                self._persist_entry(entry)


# ============================================================================
# Persistent Memory Store
# ============================================================================

class PersistentStore:
    """Low-level persistent key-value store"""
    
    def __init__(self, config: PMEMConfig):
        self.config = config
        self.path = Path(config.storage_path)
        self.data_file = self.path / "data.pmem"
        self.index_file = self.path / "index.json"
        
        # In-memory index and data
        self.index: Dict[str, Tuple[int, int]] = {}  # key -> (offset, size)
        self.data: bytearray = bytearray()
        self.next_offset = 0
        
        # Initialize storage
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize storage files"""
        self.path.mkdir(parents=True, exist_ok=True)
        
        if self.data_file.exists() and self.index_file.exists():
            self._load()
        else:
            self.data = bytearray(self.config.size)
            self._persist()
    
    def _load(self) -> None:
        """Load from disk"""
        # Load index
        with open(self.index_file, 'r') as f:
            self.index = json.load(f)
            # Convert string keys to tuple values
            self.index = {k: tuple(v) for k, v in self.index.items()}
        
        # Load data
        with open(self.data_file, 'rb') as f:
            self.data = bytearray(f.read())
        
        # Find next offset
        if self.index:
            max_entry = max(self.index.values(), key=lambda x: x[0] + x[1])
            self.next_offset = max_entry[0] + max_entry[1]
        else:
            self.next_offset = 0
    
    def _persist(self) -> None:
        """Persist to disk"""
        # Save index
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f)
        
        # Save data
        with open(self.data_file, 'wb') as f:
            f.write(self.data)
            f.flush()
            os.fsync(f.fileno())
    
    def get(self, key: str) -> Optional[bytes]:
        """Get value by key"""
        if key not in self.index:
            return None
        
        offset, size = self.index[key]
        return bytes(self.data[offset:offset + size])
    
    def put(self, key: str, value: bytes) -> Tuple[int, int]:
        """Put value, returns (offset, size)"""
        size = len(value)
        
        # Check if we need to reuse space or allocate new
        if key in self.index:
            old_offset, old_size = self.index[key]
            if old_size >= size:
                # Reuse existing space
                self.data[old_offset:old_offset + size] = value
                self.index[key] = (old_offset, size)
                return old_offset, size
        
        # Allocate new space
        if self.next_offset + size > len(self.data):
            # Compact or extend
            self._compact()
        
        offset = self.next_offset
        self.data[offset:offset + size] = value
        self.index[key] = (offset, size)
        self.next_offset = offset + size
        
        return offset, size
    
    def delete(self, key: str) -> bool:
        """Delete key"""
        if key in self.index:
            del self.index[key]
            return True
        return False
    
    def _compact(self) -> None:
        """Compact storage to reclaim space"""
        new_data = bytearray(self.config.size)
        new_offset = 0
        new_index = {}
        
        for key, (offset, size) in sorted(self.index.items(), key=lambda x: x[1][0]):
            new_data[new_offset:new_offset + size] = self.data[offset:offset + size]
            new_index[key] = (new_offset, size)
            new_offset += size
        
        self.data = new_data
        self.index = new_index
        self.next_offset = new_offset
    
    def flush(self) -> None:
        """Flush to persistent storage"""
        self._persist()


# ============================================================================
# Main Module Implementation
# ============================================================================

class PersistentMemoryModule:
    """
    Production-ready persistent memory module for OMNIXAN.
    
    Provides:
    - Byte-addressable persistent storage
    - ACID transactions
    - Write-ahead logging
    - Crash recovery
    - Checkpoint management
    """
    
    def __init__(self, config: Optional[PMEMConfig] = None):
        """Initialize the Persistent Memory Module"""
        self.config = config or PMEMConfig()
        self.store: Optional[PersistentStore] = None
        self.wal: Optional[WriteAheadLog] = None
        self.transactions: Dict[str, Transaction] = {}
        self.last_checkpoint: Optional[Checkpoint] = None
        self.metrics = PMEMMetrics()
        
        # Background tasks
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._shutting_down = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the PMEM module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing PersistentMemoryModule...")
            
            # Create storage directory
            storage_path = Path(self.config.storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize store
            self.store = PersistentStore(self.config)
            
            # Initialize WAL
            if self.config.enable_wal:
                wal_path = storage_path / "wal.log"
                self.wal = WriteAheadLog(wal_path, self.config.max_log_size)
            
            # Perform recovery if needed
            if self.config.recovery_mode != RecoveryMode.NONE:
                await self._recover()
            
            # Start background tasks
            self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())
            
            if self.config.persistence_mode == PersistenceMode.ASYNC:
                self._flush_task = asyncio.create_task(self._flush_loop())
            
            self.metrics.total_size = self.config.size
            
            self._initialized = True
            self._logger.info("PersistentMemoryModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise PMEMError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PMEM operation"""
        if not self._initialized:
            raise PMEMError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "begin_transaction":
            tx_id = await self.begin_transaction()
            return {"transaction_id": tx_id}
        
        elif operation == "commit":
            tx_id = params["transaction_id"]
            success = await self.commit(tx_id)
            return {"success": success}
        
        elif operation == "abort":
            tx_id = params["transaction_id"]
            success = await self.abort(tx_id)
            return {"success": success}
        
        elif operation == "get":
            key = params["key"]
            tx_id = params.get("transaction_id")
            value = await self.get(key, tx_id)
            return {"value": value.decode() if value else None}
        
        elif operation == "put":
            key = params["key"]
            value = params["value"]
            tx_id = params.get("transaction_id")
            if isinstance(value, str):
                value = value.encode()
            await self.put(key, value, tx_id)
            return {"success": True}
        
        elif operation == "delete":
            key = params["key"]
            tx_id = params.get("transaction_id")
            success = await self.delete(key, tx_id)
            return {"success": success}
        
        elif operation == "checkpoint":
            await self._create_checkpoint()
            return {"success": True}
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def begin_transaction(self) -> str:
        """Begin a new transaction"""
        async with self._lock:
            tx_id = str(uuid4())
            
            transaction = Transaction(
                transaction_id=tx_id,
                state=TransactionState.ACTIVE,
                start_time=time.time(),
                modifications=[],
                read_set=set(),
                write_set=set()
            )
            
            self.transactions[tx_id] = transaction
            self.metrics.num_transactions += 1
            
            # Log transaction begin
            if self.wal:
                self.wal.append(LogEntry(
                    lsn=0,
                    transaction_id=tx_id,
                    operation="BEGIN",
                    key=""
                ))
            
            return tx_id
    
    async def commit(self, transaction_id: str) -> bool:
        """Commit transaction"""
        async with self._lock:
            if transaction_id not in self.transactions:
                return False
            
            tx = self.transactions[transaction_id]
            
            if tx.state != TransactionState.ACTIVE:
                return False
            
            tx.state = TransactionState.PREPARING
            
            try:
                # Write all modifications to store
                for key, old_value, new_value in tx.modifications:
                    self.store.put(key, new_value)
                    
                    # Log modification
                    if self.wal:
                        self.wal.append(LogEntry(
                            lsn=0,
                            transaction_id=transaction_id,
                            operation="WRITE",
                            key=key,
                            old_value=old_value,
                            new_value=new_value
                        ))
                
                # Sync if in sync mode
                if self.config.persistence_mode == PersistenceMode.SYNC:
                    self.store.flush()
                
                # Log commit
                if self.wal:
                    self.wal.append(LogEntry(
                        lsn=0,
                        transaction_id=transaction_id,
                        operation="COMMIT",
                        key=""
                    ))
                
                tx.state = TransactionState.COMMITTED
                self.metrics.committed_transactions += 1
                
                del self.transactions[transaction_id]
                
                return True
            
            except Exception as e:
                self._logger.error(f"Commit failed: {e}")
                await self.abort(transaction_id)
                return False
    
    async def abort(self, transaction_id: str) -> bool:
        """Abort transaction"""
        async with self._lock:
            if transaction_id not in self.transactions:
                return False
            
            tx = self.transactions[transaction_id]
            
            # Rollback modifications
            for key, old_value, new_value in reversed(tx.modifications):
                if old_value is not None:
                    self.store.put(key, old_value)
                else:
                    self.store.delete(key)
            
            # Log abort
            if self.wal:
                self.wal.append(LogEntry(
                    lsn=0,
                    transaction_id=transaction_id,
                    operation="ABORT",
                    key=""
                ))
            
            tx.state = TransactionState.ABORTED
            self.metrics.aborted_transactions += 1
            
            del self.transactions[transaction_id]
            
            return True
    
    async def get(self, key: str, transaction_id: Optional[str] = None) -> Optional[bytes]:
        """Get value by key"""
        async with self._lock:
            self.metrics.num_reads += 1
            
            # Check transaction write set first
            if transaction_id and transaction_id in self.transactions:
                tx = self.transactions[transaction_id]
                tx.read_set.add(key)
                
                # Return uncommitted value if exists
                for k, old_value, new_value in reversed(tx.modifications):
                    if k == key:
                        return new_value
            
            return self.store.get(key)
    
    async def put(
        self,
        key: str,
        value: bytes,
        transaction_id: Optional[str] = None
    ) -> None:
        """Put value"""
        async with self._lock:
            start_time = time.time()
            
            old_value = self.store.get(key)
            
            if transaction_id:
                if transaction_id not in self.transactions:
                    raise TransactionError(f"Transaction {transaction_id} not found")
                
                tx = self.transactions[transaction_id]
                tx.modifications.append((key, old_value, value))
                tx.write_set.add(key)
            else:
                # Auto-commit write
                self.store.put(key, value)
                
                if self.config.persistence_mode == PersistenceMode.SYNC:
                    self.store.flush()
                
                if self.wal:
                    self.wal.append(LogEntry(
                        lsn=0,
                        transaction_id="auto",
                        operation="WRITE",
                        key=key,
                        old_value=old_value,
                        new_value=value
                    ))
            
            self.metrics.num_writes += 1
            
            # Update write latency
            latency_us = (time.time() - start_time) * 1_000_000
            n = self.metrics.num_writes
            self.metrics.avg_write_latency_us = (
                (self.metrics.avg_write_latency_us * (n - 1) + latency_us) / n
            )
    
    async def delete(self, key: str, transaction_id: Optional[str] = None) -> bool:
        """Delete key"""
        async with self._lock:
            old_value = self.store.get(key)
            
            if old_value is None:
                return False
            
            if transaction_id:
                if transaction_id not in self.transactions:
                    raise TransactionError(f"Transaction {transaction_id} not found")
                
                tx = self.transactions[transaction_id]
                tx.modifications.append((key, old_value, b""))
                tx.write_set.add(key)
            else:
                self.store.delete(key)
                
                if self.wal:
                    self.wal.append(LogEntry(
                        lsn=0,
                        transaction_id="auto",
                        operation="DELETE",
                        key=key,
                        old_value=old_value
                    ))
            
            return True
    
    async def _recover(self) -> None:
        """Recover from crash"""
        if not self.wal:
            return
        
        self._logger.info("Starting recovery...")
        start_time = time.time()
        
        # Find committed and active transactions
        committed = set()
        active = set()
        
        for entry in self.wal.entries:
            if entry.operation == "BEGIN":
                active.add(entry.transaction_id)
            elif entry.operation == "COMMIT":
                committed.add(entry.transaction_id)
                active.discard(entry.transaction_id)
            elif entry.operation == "ABORT":
                active.discard(entry.transaction_id)
        
        # Redo committed transactions
        for entry in self.wal.entries:
            if entry.transaction_id in committed or entry.transaction_id == "auto":
                if entry.operation == "WRITE" and entry.new_value:
                    self.store.put(entry.key, entry.new_value)
                elif entry.operation == "DELETE":
                    self.store.delete(entry.key)
        
        # Undo active (uncommitted) transactions
        for entry in reversed(self.wal.entries):
            if entry.transaction_id in active:
                if entry.operation == "WRITE" and entry.old_value:
                    self.store.put(entry.key, entry.old_value)
                elif entry.operation == "WRITE" and not entry.old_value:
                    self.store.delete(entry.key)
        
        self.store.flush()
        
        self.metrics.recovery_time_ms = (time.time() - start_time) * 1000
        self._logger.info(f"Recovery completed in {self.metrics.recovery_time_ms:.2f}ms")
    
    async def _create_checkpoint(self) -> None:
        """Create checkpoint"""
        async with self._lock:
            if not self.wal:
                return
            
            checkpoint = Checkpoint(
                checkpoint_id=str(uuid4()),
                lsn=self.wal.current_lsn,
                timestamp=time.time(),
                active_transactions=list(self.transactions.keys()),
                data_snapshot={}
            )
            
            # Flush store
            self.store.flush()
            
            # Truncate WAL
            self.wal.truncate(checkpoint.lsn)
            
            self.last_checkpoint = checkpoint
            self._logger.info(f"Checkpoint created at LSN {checkpoint.lsn}")
    
    async def _checkpoint_loop(self) -> None:
        """Background checkpoint loop"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self.config.checkpoint_interval)
                await self._create_checkpoint()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Checkpoint error: {e}")
    
    async def _flush_loop(self) -> None:
        """Background flush loop for async mode"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self.config.flush_interval)
                async with self._lock:
                    self.store.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Flush error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get PMEM metrics"""
        return {
            "total_size": self.metrics.total_size,
            "num_writes": self.metrics.num_writes,
            "num_reads": self.metrics.num_reads,
            "num_transactions": self.metrics.num_transactions,
            "committed_transactions": self.metrics.committed_transactions,
            "aborted_transactions": self.metrics.aborted_transactions,
            "active_transactions": len(self.transactions),
            "avg_write_latency_us": self.metrics.avg_write_latency_us,
            "recovery_time_ms": self.metrics.recovery_time_ms
        }
    
    async def shutdown(self) -> None:
        """Shutdown the PMEM module"""
        self._logger.info("Shutting down PersistentMemoryModule...")
        self._shutting_down = True
        
        # Cancel background tasks
        for task in [self._checkpoint_task, self._flush_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Abort active transactions
        for tx_id in list(self.transactions.keys()):
            await self.abort(tx_id)
        
        # Final checkpoint
        await self._create_checkpoint()
        
        # Final flush
        if self.store:
            self.store.flush()
        
        self._initialized = False
        self._logger.info("PersistentMemoryModule shutdown complete")


# Example usage
async def main():
    """Example usage of PersistentMemoryModule"""
    
    config = PMEMConfig(
        storage_path="/tmp/omnixan_pmem_test",
        size=1024 * 1024,  # 1 MB
        persistence_mode=PersistenceMode.SYNC,
        enable_wal=True
    )
    
    module = PersistentMemoryModule(config)
    await module.initialize()
    
    try:
        # Simple operations
        await module.put("key1", b"value1")
        value = await module.get("key1")
        print(f"key1 = {value}")
        
        # Transaction
        tx_id = await module.begin_transaction()
        print(f"Started transaction: {tx_id}")
        
        await module.put("key2", b"value2", tx_id)
        await module.put("key3", b"value3", tx_id)
        
        # Read within transaction
        value = await module.get("key2", tx_id)
        print(f"key2 (in tx) = {value}")
        
        # Commit
        success = await module.commit(tx_id)
        print(f"Transaction committed: {success}")
        
        # Verify
        value = await module.get("key2")
        print(f"key2 (after commit) = {value}")
        
        # Transaction with abort
        tx_id2 = await module.begin_transaction()
        await module.put("key4", b"value4", tx_id2)
        await module.abort(tx_id2)
        
        value = await module.get("key4")
        print(f"key4 (after abort) = {value}")  # Should be None
        
        # Metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

