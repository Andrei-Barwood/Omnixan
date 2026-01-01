# Liquid Cooling Module

**Status: ✅ IMPLEMENTED**

Production-ready liquid cooling management for HPC systems with thermal monitoring and PID control.

## Features

- **Coolant Types**: Water, Glycol, Dielectric, Novec
- **Cooling Modes**: Normal, Aggressive, ECO, Emergency
- **PID Control**: Automatic temperature regulation
- **Multi-loop Support**: Independent cooling zones

## Quick Start

```python
from omnixan.heterogenous_computing_group.liquid_cooling_module.module import (
    LiquidCoolingModule, CoolingConfig, CoolingMode
)

module = LiquidCoolingModule(CoolingConfig(target_temp_c=65.0))
await module.initialize()

# Get status
status = module.get_status()
print(f"Mode: {status['mode']}")

# Change mode
await module.set_mode(CoolingMode.AGGRESSIVE)

# Monitor temperatures
for sensor in module.sensors.values():
    print(f"{sensor.name}: {sensor.current_temp_c}°C")

await module.shutdown()
```

## Cooling Modes

| Mode | Target Adj | Pump Speed |
|------|------------|------------|
| Normal | 0°C | 100% |
| Aggressive | -10°C | 150% |
| ECO | +5°C | 70% |
| Emergency | -20°C | 200% |

## Metrics

```python
{
    "avg_temp_c": 55.2,
    "total_heat_dissipated_w": 1500,
    "cooling_efficiency": 75.0,
    "alerts_count": 0
}
```
