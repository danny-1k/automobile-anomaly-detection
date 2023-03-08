import math
ALL_COLUMNS = ["AIR_INTAKE_TEMP", "AMBIENT_AIR_TEMP", "ENGINE_COOLANT_TEMP", "ENGINE_LOAD", "EQUIV_RATIO",
           "MAF", "SPEED", "THROTTLE_POS", "FUEL_LEVEL", "INTAKE_MANIFOLD_PRESSURE", "ENGINE_RPM"]

# https://en.wikipedia.org/wiki/OBD-II_PIDs
VALUE_LIMITS = {
    "AIR_INTAKE_TEMP": (-40, 215),
    "AMBIENT_AIR_TEMP": (-40, 215),
    "ENGINE_COOLANT_TEMP": (-40, 215),
    "ENGINE_LOAD": (0, 100),
    "EQUIV_RATIO": (0, 1),
    "MAF": (0, 655.35),
    "SPEED": (0, 255),
    "THROTTLE_POS": (0, 100),
    "FUEL_LEVEL": (0, 100),
    "INTAKE_MANIFOLD_PRESSURE": (0, 255),
    "ENGINE_RPM": (0, 16383.75),
    "LOG_ENGINE_RPM": (math.log(1e-6), math.log(16383.75)),
    "LOG_MAF": (math.log(1e-6), math.log(655.35)),
}