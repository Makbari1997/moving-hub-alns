import os

ORTOOLS_AVAILABLE = os.getenv("ORTOOLS_AVAILABLE", True)

USE_HAVERSINE = os.getenv("USE_HAVERSINE", False)

URL_MATRIX = os.getenv("URL_MATRIX")
APIKEY_MATRIX = os.getenv("APIKEY_MATRIX")

MIN_UTILIZATION_THRESHOLD = 0.01
OPTIMAL_UTILIZATION_RANGE = (0.3, 0.9)
VEHICLE_CAPACITIES = {"bike": 28, "big-box": 44, "carbox": 212, "van": 800}
VEHICLE_COST_PER_HOUR = {"bike": 160.0, "carbox": 210.0, "big-box": 175.0, "van": 262.0}
