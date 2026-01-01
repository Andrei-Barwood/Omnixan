"""
OMNIXAN Cryogenic Control Module
virtualized_cluster/cryogenic_control_module

Production-ready cryogenic control system for quantum computing environments
with millikelvin temperature management, dilution refrigerator control,
and thermal stabilization for superconducting qubits.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import numpy as np

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoolingStage(str, Enum):
    """Dilution refrigerator cooling stages"""
    ROOM_TEMP = "300K"
    STAGE_50K = "50K"
    STAGE_4K = "4K"
    STAGE_1K = "1K"
    STAGE_100MK = "100mK"
    STAGE_20MK = "20mK"
    MIXING_CHAMBER = "10mK"


class SystemState(str, Enum):
    """Cryogenic system states"""
    IDLE = "idle"
    COOLING_DOWN = "cooling_down"
    STABLE = "stable"
    WARMING_UP = "warming_up"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class HeaterMode(str, Enum):
    """Heater operation modes"""
    OFF = "off"
    PID = "pid"
    MANUAL = "manual"
    RAMP = "ramp"


@dataclass
class TemperatureSensor:
    """Cryogenic temperature sensor"""
    sensor_id: str
    name: str
    stage: CoolingStage
    current_temp_k: float
    target_temp_k: float
    calibration_curve: str = "RuO2"  # Sensor type
    is_active: bool = True
    last_reading: float = field(default_factory=time.time)


@dataclass
class Heater:
    """Stage heater for temperature control"""
    heater_id: str
    name: str
    stage: CoolingStage
    max_power_w: float
    current_power_w: float = 0.0
    mode: HeaterMode = HeaterMode.OFF
    resistance_ohm: float = 100.0


@dataclass
class CryoStage:
    """A cooling stage in the dilution refrigerator"""
    stage_id: str
    stage: CoolingStage
    target_temp_k: float
    current_temp_k: float
    sensors: List[str] = field(default_factory=list)
    heaters: List[str] = field(default_factory=list)
    thermal_load_w: float = 0.0
    cooling_power_w: float = 0.0
    is_stable: bool = False


@dataclass
class CryoAlert:
    """Cryogenic system alert"""
    alert_id: str
    level: AlertLevel
    message: str
    stage: CoolingStage
    timestamp: float
    resolved: bool = False


@dataclass
class CryoMetrics:
    """Cryogenic system metrics"""
    base_temp_mk: float = 10.0
    stability_mk: float = 0.1
    cooldown_time_hours: float = 0.0
    uptime_hours: float = 0.0
    he3_circulation_rate: float = 0.0
    total_cooling_power_uw: float = 0.0
    alerts_count: int = 0


class CryogenicConfig(BaseModel):
    """Configuration for cryogenic control"""
    base_temperature_mk: float = Field(
        default=10.0,
        ge=5.0,
        le=100.0,
        description="Target base temperature in mK"
    )
    stability_threshold_mk: float = Field(
        default=0.5,
        ge=0.01,
        description="Temperature stability threshold"
    )
    cooldown_rate_k_per_hour: float = Field(
        default=10.0,
        ge=1.0,
        description="Cooldown rate"
    )
    enable_auto_pid: bool = Field(
        default=True,
        description="Enable automatic PID control"
    )
    poll_interval_s: float = Field(
        default=1.0,
        gt=0.0,
        description="Sensor polling interval"
    )


class CryogenicError(Exception):
    """Base exception for cryogenic errors"""
    pass


# ============================================================================
# PID Controller for Temperature
# ============================================================================

class CryoPIDController:
    """PID controller optimized for cryogenic temperatures"""
    
    def __init__(
        self,
        target_temp: float,
        kp: float = 0.5,
        ki: float = 0.01,
        kd: float = 0.1,
        max_output: float = 1.0
    ):
        self.target_temp = target_temp
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        
        self._integral = 0.0
        self._prev_error = 0.0
        self._last_time = time.time()
    
    def compute(self, current_temp: float) -> float:
        """Compute heater power (0 to max_output)"""
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        
        if dt <= 0:
            return 0.0
        
        error = current_temp - self.target_temp
        
        # Proportional
        p = self.kp * error
        
        # Integral with anti-windup
        self._integral += error * dt
        self._integral = max(-self.max_output, min(self.max_output, self._integral))
        i = self.ki * self._integral
        
        # Derivative
        derivative = (error - self._prev_error) / dt
        d = self.kd * derivative
        self._prev_error = error
        
        # Output (only heat if above target, cooling is passive)
        output = p + i + d
        return max(0, min(self.max_output, output))
    
    def set_target(self, target_temp: float) -> None:
        """Update target temperature"""
        self.target_temp = target_temp
        self._integral = 0.0


# ============================================================================
# Main Module Implementation
# ============================================================================

class CryogenicControlModule:
    """
    Production-ready Cryogenic Control module for OMNIXAN.
    
    Provides:
    - Dilution refrigerator control
    - Millikelvin temperature management
    - Multi-stage cooling control
    - PID temperature stabilization
    - He3 circulation monitoring
    - Alert and safety systems
    """
    
    def __init__(self, config: Optional[CryogenicConfig] = None):
        """Initialize the Cryogenic Control Module"""
        self.config = config or CryogenicConfig()
        
        self.stages: Dict[str, CryoStage] = {}
        self.sensors: Dict[str, TemperatureSensor] = {}
        self.heaters: Dict[str, Heater] = {}
        self.alerts: List[CryoAlert] = []
        
        self.pid_controllers: Dict[str, CryoPIDController] = {}
        self.state = SystemState.IDLE
        self.metrics = CryoMetrics()
        
        self._monitor_task: Optional[asyncio.Task] = None
        self._start_time = time.time()
        self._initialized = False
        self._shutting_down = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the cryogenic control module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing CryogenicControlModule...")
            
            # Create dilution refrigerator stages
            await self._create_dr_stages()
            
            # Start monitoring
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
            self._initialized = True
            self._logger.info("CryogenicControlModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise CryogenicError(f"Failed to initialize module: {str(e)}")
    
    async def _create_dr_stages(self) -> None:
        """Create dilution refrigerator stages"""
        stage_configs = [
            (CoolingStage.STAGE_50K, 50.0),
            (CoolingStage.STAGE_4K, 4.0),
            (CoolingStage.STAGE_1K, 1.0),
            (CoolingStage.STAGE_100MK, 0.1),
            (CoolingStage.MIXING_CHAMBER, self.config.base_temperature_mk / 1000),
        ]
        
        for cooling_stage, target_k in stage_configs:
            stage = CryoStage(
                stage_id=str(uuid4()),
                stage=cooling_stage,
                target_temp_k=target_k,
                current_temp_k=300.0,  # Start at room temp
                cooling_power_w=self._get_cooling_power(cooling_stage)
            )
            self.stages[stage.stage_id] = stage
            
            # Create sensor for stage
            sensor = TemperatureSensor(
                sensor_id=str(uuid4()),
                name=f"{cooling_stage.value}_sensor",
                stage=cooling_stage,
                current_temp_k=300.0,
                target_temp_k=target_k
            )
            self.sensors[sensor.sensor_id] = sensor
            stage.sensors.append(sensor.sensor_id)
            
            # Create heater for stage (except mixing chamber)
            if cooling_stage != CoolingStage.MIXING_CHAMBER:
                heater = Heater(
                    heater_id=str(uuid4()),
                    name=f"{cooling_stage.value}_heater",
                    stage=cooling_stage,
                    max_power_w=self._get_heater_power(cooling_stage)
                )
                self.heaters[heater.heater_id] = heater
                stage.heaters.append(heater.heater_id)
            
            # Create PID controller
            self.pid_controllers[stage.stage_id] = CryoPIDController(
                target_temp=target_k,
                max_output=self._get_heater_power(cooling_stage)
            )
    
    def _get_cooling_power(self, stage: CoolingStage) -> float:
        """Get cooling power for stage in Watts"""
        powers = {
            CoolingStage.STAGE_50K: 50.0,
            CoolingStage.STAGE_4K: 1.5,
            CoolingStage.STAGE_1K: 0.1,
            CoolingStage.STAGE_100MK: 0.0005,  # 500 µW
            CoolingStage.MIXING_CHAMBER: 0.00001,  # 10 µW
        }
        return powers.get(stage, 0.0)
    
    def _get_heater_power(self, stage: CoolingStage) -> float:
        """Get maximum heater power for stage in Watts"""
        powers = {
            CoolingStage.STAGE_50K: 10.0,
            CoolingStage.STAGE_4K: 0.5,
            CoolingStage.STAGE_1K: 0.01,
            CoolingStage.STAGE_100MK: 0.0001,  # 100 µW
        }
        return powers.get(stage, 0.0)
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cryogenic operation"""
        if not self._initialized:
            raise CryogenicError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "start_cooldown":
            await self.start_cooldown()
            return {"state": self.state.value}
        
        elif operation == "stop_cooldown":
            await self.stop_cooldown()
            return {"state": self.state.value}
        
        elif operation == "set_target_temp":
            stage_id = params["stage_id"]
            temp_k = params["temp_k"]
            await self.set_target_temperature(stage_id, temp_k)
            return {"success": True}
        
        elif operation == "set_heater":
            heater_id = params["heater_id"]
            power_w = params["power_w"]
            await self.set_heater_power(heater_id, power_w)
            return {"success": True}
        
        elif operation == "get_status":
            return self.get_status()
        
        elif operation == "get_temperatures":
            return {
                "temperatures": {
                    s.sensor_id: {
                        "name": s.name,
                        "stage": s.stage.value,
                        "current_k": round(s.current_temp_k, 6),
                        "target_k": s.target_temp_k
                    }
                    for s in self.sensors.values()
                }
            }
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def start_cooldown(self) -> None:
        """Start cooldown sequence"""
        async with self._lock:
            if self.state == SystemState.COOLING_DOWN:
                return
            
            self.state = SystemState.COOLING_DOWN
            self._logger.info("Starting cooldown sequence")
            
            # Enable PID control
            for heater in self.heaters.values():
                heater.mode = HeaterMode.PID
    
    async def stop_cooldown(self) -> None:
        """Stop cooldown and begin warmup"""
        async with self._lock:
            self.state = SystemState.WARMING_UP
            self._logger.info("Starting warmup sequence")
            
            # Disable PID, enable warmup heaters
            for heater in self.heaters.values():
                heater.mode = HeaterMode.OFF
    
    async def set_target_temperature(self, stage_id: str, temp_k: float) -> None:
        """Set target temperature for a stage"""
        async with self._lock:
            if stage_id not in self.stages:
                raise CryogenicError("Stage not found")
            
            stage = self.stages[stage_id]
            stage.target_temp_k = temp_k
            
            # Update sensors
            for sensor_id in stage.sensors:
                if sensor_id in self.sensors:
                    self.sensors[sensor_id].target_temp_k = temp_k
            
            # Update PID
            if stage_id in self.pid_controllers:
                self.pid_controllers[stage_id].set_target(temp_k)
    
    async def set_heater_power(self, heater_id: str, power_w: float) -> None:
        """Set heater power manually"""
        async with self._lock:
            if heater_id not in self.heaters:
                raise CryogenicError("Heater not found")
            
            heater = self.heaters[heater_id]
            heater.mode = HeaterMode.MANUAL
            heater.current_power_w = min(power_w, heater.max_power_w)
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self.config.poll_interval_s)
                
                async with self._lock:
                    await self._update_temperatures()
                    await self._run_pid_control()
                    await self._check_stability()
                    await self._update_metrics()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Monitor error: {e}")
    
    async def _update_temperatures(self) -> None:
        """Update temperature readings (simulation)"""
        for stage in self.stages.values():
            if self.state == SystemState.COOLING_DOWN:
                # Simulate cooling
                cooling_rate = self.config.cooldown_rate_k_per_hour / 3600
                thermal_load = stage.thermal_load_w
                
                # Get heater power
                heater_power = 0.0
                for heater_id in stage.heaters:
                    if heater_id in self.heaters:
                        heater_power += self.heaters[heater_id].current_power_w
                
                # Net cooling
                net_cooling = stage.cooling_power_w - thermal_load - heater_power
                
                if stage.current_temp_k > stage.target_temp_k:
                    # Cool down
                    temp_change = cooling_rate * self.config.poll_interval_s
                    stage.current_temp_k = max(
                        stage.target_temp_k,
                        stage.current_temp_k - temp_change
                    )
            
            elif self.state == SystemState.WARMING_UP:
                # Simulate warmup
                warmup_rate = 20.0 / 3600  # 20K/hour
                temp_change = warmup_rate * self.config.poll_interval_s
                stage.current_temp_k = min(300.0, stage.current_temp_k + temp_change)
            
            # Update sensors
            for sensor_id in stage.sensors:
                if sensor_id in self.sensors:
                    # Add small noise
                    noise = np.random.normal(0, 0.0001)
                    self.sensors[sensor_id].current_temp_k = stage.current_temp_k + noise
                    self.sensors[sensor_id].last_reading = time.time()
    
    async def _run_pid_control(self) -> None:
        """Run PID control for each stage"""
        if not self.config.enable_auto_pid:
            return
        
        for stage in self.stages.values():
            if stage.stage_id not in self.pid_controllers:
                continue
            
            pid = self.pid_controllers[stage.stage_id]
            power = pid.compute(stage.current_temp_k)
            
            # Apply to heaters
            for heater_id in stage.heaters:
                heater = self.heaters.get(heater_id)
                if heater and heater.mode == HeaterMode.PID:
                    heater.current_power_w = min(power, heater.max_power_w)
    
    async def _check_stability(self) -> None:
        """Check temperature stability"""
        all_stable = True
        
        for stage in self.stages.values():
            temp_diff = abs(stage.current_temp_k - stage.target_temp_k)
            threshold = self.config.stability_threshold_mk / 1000  # Convert mK to K
            
            stage.is_stable = temp_diff <= threshold
            
            if not stage.is_stable:
                all_stable = False
        
        # Update system state
        if self.state == SystemState.COOLING_DOWN and all_stable:
            self.state = SystemState.STABLE
            self._logger.info("All stages stable at target temperatures")
    
    async def _update_metrics(self) -> None:
        """Update system metrics"""
        # Find mixing chamber temperature
        for stage in self.stages.values():
            if stage.stage == CoolingStage.MIXING_CHAMBER:
                self.metrics.base_temp_mk = stage.current_temp_k * 1000
                break
        
        # Calculate total cooling power
        self.metrics.total_cooling_power_uw = sum(
            s.cooling_power_w * 1e6 for s in self.stages.values()
        )
        
        # Uptime
        self.metrics.uptime_hours = (time.time() - self._start_time) / 3600
        
        # He3 circulation (simulated)
        if self.state == SystemState.STABLE:
            self.metrics.he3_circulation_rate = 50.0  # µmol/s
        else:
            self.metrics.he3_circulation_rate = 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get cryogenic system status"""
        return {
            "state": self.state.value,
            "stages": [
                {
                    "stage": s.stage.value,
                    "current_temp_k": round(s.current_temp_k, 6),
                    "target_temp_k": s.target_temp_k,
                    "is_stable": s.is_stable,
                    "cooling_power_w": s.cooling_power_w
                }
                for s in self.stages.values()
            ],
            "heaters": [
                {
                    "name": h.name,
                    "mode": h.mode.value,
                    "power_w": round(h.current_power_w, 6)
                }
                for h in self.heaters.values()
            ],
            "active_alerts": sum(1 for a in self.alerts if not a.resolved)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cryogenic metrics"""
        return {
            "base_temp_mk": round(self.metrics.base_temp_mk, 3),
            "stability_mk": round(self.metrics.stability_mk, 3),
            "uptime_hours": round(self.metrics.uptime_hours, 2),
            "he3_circulation_rate_umol_s": round(self.metrics.he3_circulation_rate, 1),
            "total_cooling_power_uw": round(self.metrics.total_cooling_power_uw, 3),
            "alerts_count": self.metrics.alerts_count,
            "state": self.state.value
        }
    
    async def shutdown(self) -> None:
        """Shutdown the cryogenic control module"""
        self._logger.info("Shutting down CryogenicControlModule...")
        self._shutting_down = True
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.stages.clear()
        self.sensors.clear()
        self.heaters.clear()
        self._initialized = False
        
        self._logger.info("CryogenicControlModule shutdown complete")


# Example usage
async def main():
    """Example usage of CryogenicControlModule"""
    
    config = CryogenicConfig(
        base_temperature_mk=10.0,
        stability_threshold_mk=0.5
    )
    
    module = CryogenicControlModule(config)
    await module.initialize()
    
    try:
        # Start cooldown
        await module.start_cooldown()
        print(f"State: {module.state.value}")
        
        # Wait for cooling
        await asyncio.sleep(3)
        
        # Get status
        status = module.get_status()
        print(f"\nCryogenic Status:")
        for stage in status["stages"]:
            print(f"  {stage['stage']}: {stage['current_temp_k']:.6f} K (target: {stage['target_temp_k']} K)")
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

