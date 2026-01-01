"""
OMNIXAN Liquid Cooling Module
heterogenous_computing_group/liquid_cooling_module

Production-ready liquid cooling management and monitoring module for
high-performance computing systems with thermal management, flow control,
and predictive maintenance.
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


class CoolantType(str, Enum):
    """Types of coolant"""
    WATER = "water"
    GLYCOL = "glycol"  # Water-glycol mix
    DIELECTRIC = "dielectric"  # Non-conductive
    MINERAL_OIL = "mineral_oil"
    NOVEC = "novec"  # 3M Novec


class CoolingMode(str, Enum):
    """Cooling operation modes"""
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    ECO = "eco"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class ComponentType(str, Enum):
    """Component types for cooling"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    VRM = "vrm"
    CHIPSET = "chipset"
    STORAGE = "storage"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ThermalSensor:
    """Temperature sensor"""
    sensor_id: str
    name: str
    component: ComponentType
    current_temp_c: float = 25.0
    target_temp_c: float = 65.0
    max_temp_c: float = 85.0
    min_temp_c: float = 10.0
    is_active: bool = True


@dataclass
class CoolingLoop:
    """A cooling loop"""
    loop_id: str
    name: str
    coolant_type: CoolantType
    flow_rate_lpm: float  # Liters per minute
    inlet_temp_c: float
    outlet_temp_c: float
    pressure_bar: float
    pump_speed_rpm: int
    pump_power_w: float
    max_flow_rate_lpm: float = 20.0
    max_pressure_bar: float = 3.0
    is_active: bool = True
    sensors: List[str] = field(default_factory=list)  # sensor_ids


@dataclass
class ColdPlate:
    """Cold plate for direct cooling"""
    plate_id: str
    name: str
    component: ComponentType
    thermal_resistance_c_per_w: float  # °C/W
    max_tdp_w: float
    current_load_w: float = 0.0
    loop_id: Optional[str] = None


@dataclass
class CoolingAlert:
    """Cooling system alert"""
    alert_id: str
    level: AlertLevel
    message: str
    component: str
    timestamp: float
    resolved: bool = False


@dataclass
class CoolingMetrics:
    """Cooling system metrics"""
    avg_temp_c: float = 25.0
    max_temp_c: float = 25.0
    total_heat_dissipated_w: float = 0.0
    total_pump_power_w: float = 0.0
    cooling_efficiency: float = 1.0  # COP
    uptime_hours: float = 0.0
    alerts_count: int = 0


class CoolingConfig(BaseModel):
    """Configuration for liquid cooling"""
    target_temp_c: float = Field(
        default=65.0,
        ge=20.0,
        le=90.0,
        description="Target temperature"
    )
    max_temp_c: float = Field(
        default=85.0,
        ge=50.0,
        le=105.0,
        description="Maximum temperature"
    )
    min_flow_rate_lpm: float = Field(
        default=2.0,
        ge=0.5,
        description="Minimum flow rate"
    )
    default_mode: CoolingMode = Field(
        default=CoolingMode.NORMAL,
        description="Default cooling mode"
    )
    enable_predictive: bool = Field(
        default=True,
        description="Enable predictive thermal management"
    )
    poll_interval_s: float = Field(
        default=1.0,
        gt=0.0,
        description="Sensor polling interval"
    )


class CoolingError(Exception):
    """Base exception for cooling errors"""
    pass


# ============================================================================
# Thermal Controller
# ============================================================================

class ThermalController:
    """PID-based thermal controller"""
    
    def __init__(self, target_temp: float, kp: float = 2.0, ki: float = 0.1, kd: float = 0.5):
        self.target_temp = target_temp
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self._integral = 0.0
        self._prev_error = 0.0
        self._last_time = time.time()
    
    def compute(self, current_temp: float) -> float:
        """Compute control output (0-100%)"""
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        
        if dt <= 0:
            return 50.0
        
        error = current_temp - self.target_temp
        
        # Proportional
        p = self.kp * error
        
        # Integral
        self._integral += error * dt
        self._integral = max(-100, min(100, self._integral))  # Anti-windup
        i = self.ki * self._integral
        
        # Derivative
        derivative = (error - self._prev_error) / dt
        d = self.kd * derivative
        self._prev_error = error
        
        # Output (clamped to 0-100%)
        output = 50.0 + p + i + d
        return max(0, min(100, output))
    
    def set_target(self, target_temp: float) -> None:
        """Update target temperature"""
        self.target_temp = target_temp
        self._integral = 0.0  # Reset integral


# ============================================================================
# Main Module Implementation
# ============================================================================

class LiquidCoolingModule:
    """
    Production-ready Liquid Cooling module for OMNIXAN.
    
    Provides:
    - Thermal sensor monitoring
    - Cooling loop management
    - PID-based temperature control
    - Flow rate and pressure monitoring
    - Predictive maintenance alerts
    - Multi-zone cooling
    """
    
    def __init__(self, config: Optional[CoolingConfig] = None):
        """Initialize the Liquid Cooling Module"""
        self.config = config or CoolingConfig()
        
        self.sensors: Dict[str, ThermalSensor] = {}
        self.loops: Dict[str, CoolingLoop] = {}
        self.cold_plates: Dict[str, ColdPlate] = {}
        self.alerts: List[CoolingAlert] = []
        
        self.controllers: Dict[str, ThermalController] = {}
        self.mode = self.config.default_mode
        self.metrics = CoolingMetrics()
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._start_time = time.time()
        self._initialized = False
        self._shutting_down = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the liquid cooling module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing LiquidCoolingModule...")
            
            # Create default cooling loop
            await self._create_default_loop()
            
            # Start monitoring
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
            self._initialized = True
            self._logger.info("LiquidCoolingModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise CoolingError(f"Failed to initialize module: {str(e)}")
    
    async def _create_default_loop(self) -> None:
        """Create default cooling loop with sensors"""
        # Create main loop
        loop = CoolingLoop(
            loop_id=str(uuid4()),
            name="main_loop",
            coolant_type=CoolantType.WATER,
            flow_rate_lpm=10.0,
            inlet_temp_c=25.0,
            outlet_temp_c=35.0,
            pressure_bar=1.5,
            pump_speed_rpm=2000,
            pump_power_w=20.0
        )
        self.loops[loop.loop_id] = loop
        
        # Create default sensors
        for comp in [ComponentType.CPU, ComponentType.GPU]:
            sensor = ThermalSensor(
                sensor_id=str(uuid4()),
                name=f"{comp.value}_temp",
                component=comp,
                current_temp_c=45.0,
                target_temp_c=self.config.target_temp_c,
                max_temp_c=self.config.max_temp_c
            )
            self.sensors[sensor.sensor_id] = sensor
            loop.sensors.append(sensor.sensor_id)
            
            # Create controller for this sensor
            self.controllers[sensor.sensor_id] = ThermalController(
                target_temp=self.config.target_temp_c
            )
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cooling operation"""
        if not self._initialized:
            raise CoolingError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "add_sensor":
            sensor = await self.add_sensor(
                name=params["name"],
                component=ComponentType(params["component"]),
                loop_id=params.get("loop_id")
            )
            return {"sensor_id": sensor.sensor_id}
        
        elif operation == "add_loop":
            loop = await self.add_loop(
                name=params["name"],
                coolant_type=CoolantType(params.get("coolant", "water")),
                max_flow_rate=params.get("max_flow_rate", 20.0)
            )
            return {"loop_id": loop.loop_id}
        
        elif operation == "set_mode":
            mode = CoolingMode(params["mode"])
            await self.set_mode(mode)
            return {"mode": mode.value}
        
        elif operation == "update_temp":
            sensor_id = params["sensor_id"]
            temp = params["temp_c"]
            await self.update_temperature(sensor_id, temp)
            return {"success": True}
        
        elif operation == "get_status":
            return self.get_status()
        
        elif operation == "get_alerts":
            return {
                "alerts": [
                    {
                        "alert_id": a.alert_id,
                        "level": a.level.value,
                        "message": a.message,
                        "component": a.component,
                        "resolved": a.resolved
                    }
                    for a in self.alerts
                    if not a.resolved
                ]
            }
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def add_sensor(
        self,
        name: str,
        component: ComponentType,
        loop_id: Optional[str] = None
    ) -> ThermalSensor:
        """Add a thermal sensor"""
        async with self._lock:
            sensor = ThermalSensor(
                sensor_id=str(uuid4()),
                name=name,
                component=component,
                target_temp_c=self.config.target_temp_c,
                max_temp_c=self.config.max_temp_c
            )
            
            self.sensors[sensor.sensor_id] = sensor
            
            # Add to loop
            if loop_id and loop_id in self.loops:
                self.loops[loop_id].sensors.append(sensor.sensor_id)
            elif self.loops:
                # Add to first loop
                first_loop = next(iter(self.loops.values()))
                first_loop.sensors.append(sensor.sensor_id)
            
            # Create controller
            self.controllers[sensor.sensor_id] = ThermalController(
                target_temp=self.config.target_temp_c
            )
            
            return sensor
    
    async def add_loop(
        self,
        name: str,
        coolant_type: CoolantType = CoolantType.WATER,
        max_flow_rate: float = 20.0
    ) -> CoolingLoop:
        """Add a cooling loop"""
        async with self._lock:
            loop = CoolingLoop(
                loop_id=str(uuid4()),
                name=name,
                coolant_type=coolant_type,
                flow_rate_lpm=max_flow_rate / 2,
                inlet_temp_c=25.0,
                outlet_temp_c=35.0,
                pressure_bar=1.5,
                pump_speed_rpm=2000,
                pump_power_w=20.0,
                max_flow_rate_lpm=max_flow_rate
            )
            
            self.loops[loop.loop_id] = loop
            return loop
    
    async def set_mode(self, mode: CoolingMode) -> None:
        """Set cooling mode"""
        async with self._lock:
            self.mode = mode
            
            # Adjust targets based on mode
            target_adjustments = {
                CoolingMode.NORMAL: 0,
                CoolingMode.AGGRESSIVE: -10,
                CoolingMode.ECO: 5,
                CoolingMode.EMERGENCY: -20,
                CoolingMode.MAINTENANCE: 10,
            }
            
            adjustment = target_adjustments.get(mode, 0)
            
            for controller in self.controllers.values():
                controller.set_target(self.config.target_temp_c + adjustment)
            
            # Adjust pump speeds
            pump_multipliers = {
                CoolingMode.NORMAL: 1.0,
                CoolingMode.AGGRESSIVE: 1.5,
                CoolingMode.ECO: 0.7,
                CoolingMode.EMERGENCY: 2.0,
                CoolingMode.MAINTENANCE: 0.5,
            }
            
            multiplier = pump_multipliers.get(mode, 1.0)
            
            for loop in self.loops.values():
                loop.flow_rate_lpm = min(
                    loop.max_flow_rate_lpm,
                    (loop.max_flow_rate_lpm / 2) * multiplier
                )
                loop.pump_speed_rpm = int(2000 * multiplier)
            
            self._logger.info(f"Cooling mode set to {mode.value}")
    
    async def update_temperature(self, sensor_id: str, temp_c: float) -> None:
        """Update sensor temperature"""
        async with self._lock:
            if sensor_id not in self.sensors:
                return
            
            sensor = self.sensors[sensor_id]
            sensor.current_temp_c = temp_c
            
            # Check for alerts
            if temp_c >= sensor.max_temp_c:
                await self._create_alert(
                    AlertLevel.CRITICAL,
                    f"Temperature critical: {temp_c:.1f}°C",
                    sensor.name
                )
            elif temp_c >= sensor.target_temp_c + 10:
                await self._create_alert(
                    AlertLevel.WARNING,
                    f"Temperature elevated: {temp_c:.1f}°C",
                    sensor.name
                )
            
            # Adjust cooling
            if sensor_id in self.controllers:
                control_output = self.controllers[sensor_id].compute(temp_c)
                await self._adjust_cooling(sensor_id, control_output)
    
    async def _adjust_cooling(self, sensor_id: str, control_output: float) -> None:
        """Adjust cooling based on control output"""
        # Find which loop has this sensor
        for loop in self.loops.values():
            if sensor_id in loop.sensors:
                # Adjust flow rate
                target_flow = loop.max_flow_rate_lpm * (control_output / 100)
                loop.flow_rate_lpm = max(
                    self.config.min_flow_rate_lpm,
                    min(loop.max_flow_rate_lpm, target_flow)
                )
                
                # Adjust pump speed
                loop.pump_speed_rpm = int(1000 + (control_output / 100) * 2000)
                
                # Update pump power (simplified)
                loop.pump_power_w = loop.pump_speed_rpm / 100
                
                break
    
    async def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        component: str
    ) -> None:
        """Create an alert"""
        alert = CoolingAlert(
            alert_id=str(uuid4()),
            level=level,
            message=message,
            component=component,
            timestamp=time.time()
        )
        self.alerts.append(alert)
        self.metrics.alerts_count += 1
        
        if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            self._logger.warning(f"ALERT [{level.value}]: {message}")
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self.config.poll_interval_s)
                
                async with self._lock:
                    # Update metrics
                    await self._update_metrics()
                    
                    # Simulate temperature changes
                    await self._simulate_thermal()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Monitor error: {e}")
    
    async def _update_metrics(self) -> None:
        """Update cooling metrics"""
        if not self.sensors:
            return
        
        temps = [s.current_temp_c for s in self.sensors.values()]
        self.metrics.avg_temp_c = sum(temps) / len(temps)
        self.metrics.max_temp_c = max(temps)
        
        self.metrics.total_pump_power_w = sum(
            l.pump_power_w for l in self.loops.values()
        )
        
        # Calculate heat dissipated (simplified)
        for loop in self.loops.values():
            delta_t = loop.outlet_temp_c - loop.inlet_temp_c
            # Q = m * Cp * ΔT, Cp_water ≈ 4.186 kJ/kg·K
            heat_w = loop.flow_rate_lpm * 4.186 * delta_t / 60
            self.metrics.total_heat_dissipated_w = max(0, heat_w * 1000)
        
        # Cooling efficiency (COP)
        if self.metrics.total_pump_power_w > 0:
            self.metrics.cooling_efficiency = (
                self.metrics.total_heat_dissipated_w /
                self.metrics.total_pump_power_w
            )
        
        self.metrics.uptime_hours = (time.time() - self._start_time) / 3600
    
    async def _simulate_thermal(self) -> None:
        """Simulate temperature changes"""
        for sensor in self.sensors.values():
            # Find loop
            for loop in self.loops.values():
                if sensor.sensor_id in loop.sensors:
                    # Temperature tends toward equilibrium based on flow
                    flow_factor = loop.flow_rate_lpm / loop.max_flow_rate_lpm
                    equilibrium = 40 - (flow_factor * 15)  # Lower with more flow
                    
                    # Move toward equilibrium
                    diff = equilibrium - sensor.current_temp_c
                    sensor.current_temp_c += diff * 0.1  # Smooth transition
                    
                    break
    
    def get_status(self) -> Dict[str, Any]:
        """Get cooling system status"""
        return {
            "mode": self.mode.value,
            "sensors": [
                {
                    "sensor_id": s.sensor_id,
                    "name": s.name,
                    "component": s.component.value,
                    "current_temp_c": round(s.current_temp_c, 1),
                    "target_temp_c": s.target_temp_c,
                    "max_temp_c": s.max_temp_c
                }
                for s in self.sensors.values()
            ],
            "loops": [
                {
                    "loop_id": l.loop_id,
                    "name": l.name,
                    "coolant": l.coolant_type.value,
                    "flow_rate_lpm": round(l.flow_rate_lpm, 1),
                    "inlet_temp_c": round(l.inlet_temp_c, 1),
                    "outlet_temp_c": round(l.outlet_temp_c, 1),
                    "pump_rpm": l.pump_speed_rpm
                }
                for l in self.loops.values()
            ],
            "active_alerts": sum(1 for a in self.alerts if not a.resolved)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cooling metrics"""
        return {
            "avg_temp_c": round(self.metrics.avg_temp_c, 1),
            "max_temp_c": round(self.metrics.max_temp_c, 1),
            "total_heat_dissipated_w": round(self.metrics.total_heat_dissipated_w, 1),
            "total_pump_power_w": round(self.metrics.total_pump_power_w, 1),
            "cooling_efficiency": round(self.metrics.cooling_efficiency, 2),
            "uptime_hours": round(self.metrics.uptime_hours, 2),
            "alerts_count": self.metrics.alerts_count,
            "mode": self.mode.value
        }
    
    async def shutdown(self) -> None:
        """Shutdown the liquid cooling module"""
        self._logger.info("Shutting down LiquidCoolingModule...")
        self._shutting_down = True
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.sensors.clear()
        self.loops.clear()
        self.controllers.clear()
        self._initialized = False
        
        self._logger.info("LiquidCoolingModule shutdown complete")


# Example usage
async def main():
    """Example usage of LiquidCoolingModule"""
    
    config = CoolingConfig(
        target_temp_c=65.0,
        max_temp_c=85.0,
        default_mode=CoolingMode.NORMAL
    )
    
    module = LiquidCoolingModule(config)
    await module.initialize()
    
    try:
        # Get initial status
        status = module.get_status()
        print(f"Cooling mode: {status['mode']}")
        print(f"Sensors: {len(status['sensors'])}")
        print(f"Loops: {len(status['loops'])}")
        
        # Simulate temperature changes
        for sensor in module.sensors.values():
            await module.update_temperature(sensor.sensor_id, 70.0)
        
        print("\nAfter temperature update:")
        status = module.get_status()
        for s in status["sensors"]:
            print(f"  {s['name']}: {s['current_temp_c']}°C")
        
        # Change to aggressive mode
        await module.set_mode(CoolingMode.AGGRESSIVE)
        
        # Wait for cooling to adjust
        await asyncio.sleep(2)
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

