import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class VehicleType(Enum):
    BIKE = "bike"
    CARBOX = "carbox"
    BIG_BOX = "big-box"


@dataclass
class Parcel:
    """Represents a parcel with pickup and delivery locations"""

    id: int
    pickup_location: Tuple[float, float]  # (lat, lon)
    delivery_location: Tuple[float, float]  # (lat, lon)
    size: int

    def __post_init__(self):
        # Validate parcel data
        if self.size <= 0:
            raise ValueError(f"Parcel {self.id} has invalid size: {self.size}")


@dataclass
class VehicleSpec:
    """Vehicle type specifications with new cost model"""

    vehicle_type: VehicleType
    capacity: int
    cost_per_hour: float  # Changed from cost_per_km
    fixed_cost_per_vehicle: float = (
        0.0  # Will be calculated as time_window * cost_per_hour
    )
    min_utilization_threshold: float = 0.3
    optimal_utilization_range: Tuple[float, float] = (0.6, 0.85)

    # New field for time window
    time_window_hours: float = 0.0  # Will be set during initialization

    def __post_init__(self):
        if self.capacity <= 0 or self.cost_per_hour <= 0:
            raise ValueError(
                f"Invalid vehicle specs: capacity={self.capacity}, cost_per_hour={self.cost_per_hour}"
            )
        if not 0 <= self.min_utilization_threshold <= 1:
            raise ValueError(
                f"Invalid utilization threshold: {self.min_utilization_threshold}"
            )

    def set_time_window(self, time_window_hours: float):
        """Set time window and calculate fixed cost"""
        self.time_window_hours = time_window_hours
        self.fixed_cost_per_vehicle = time_window_hours * self.cost_per_hour

    def calculate_utilization_score(self, used_capacity: int) -> float:
        """Calculate utilization efficiency score (higher is better)"""
        if used_capacity <= 0:
            return 0.0

        utilization = used_capacity / self.capacity

        # Penalty for under-utilization
        if utilization < self.min_utilization_threshold:
            return utilization * 0.5  # 50% penalty

        # Bonus for optimal range
        opt_min, opt_max = self.optimal_utilization_range
        if opt_min <= utilization <= opt_max:
            return 1.0 + (utilization - opt_min) * 0.5  # Up to 50% bonus

        # Gradual penalty for over-utilization
        if utilization > opt_max:
            return 1.0 - (utilization - opt_max) * 0.3

        return utilization

    def calculate_total_route_cost(
        self, used_capacity: int, include_utilization_penalty: bool = True
    ) -> float:
        """
        Calculate total cost for this vehicle with new cost model

        Args:
            used_capacity: Number of parcels/capacity used
            include_utilization_penalty: Whether to apply utilization penalties

        Returns:
            Total cost (fixed cost based on time window)
        """
        # Base cost is fixed per vehicle (time_window * cost_per_hour)
        base_cost = self.fixed_cost_per_vehicle

        if not include_utilization_penalty:
            return base_cost

        # Apply utilization modifier to encourage efficient packing
        utilization_score = self.calculate_utilization_score(used_capacity)
        utilization_penalty = max(0.5, utilization_score)  # Minimum 50% efficiency

        # Cost penalty for poor utilization
        return base_cost / utilization_penalty

    def calculate_cost_per_parcel(self, used_capacity: int) -> float:
        """Calculate cost per parcel for this vehicle configuration"""
        if used_capacity <= 0:
            return float("inf")

        total_cost = self.calculate_total_route_cost(used_capacity)
        return total_cost / used_capacity


@dataclass
class PickupTerminal:
    """Pickup terminal with its parcels"""

    pickup_id: int
    lat: float
    lon: float
    parcels: List[Parcel]

    def __post_init__(self):
        if not self.parcels:
            raise ValueError(f"Pickup terminal {self.pickup_id} has no parcels")

        # Validate all parcels have same pickup location
        for parcel in self.parcels:
            if parcel.pickup_location != (self.lat, self.lon):
                raise ValueError(f"Parcel {parcel.id} pickup location mismatch")


@dataclass
class Route:
    """Represents a vehicle route with enhanced cost tracking"""

    vehicle_id: int
    vehicle_type: VehicleType
    vehicle_spec: VehicleSpec
    parcels: List[Parcel]
    pickup_sequence: List[int]  # pickup_ids in optimized order
    delivery_sequence: List[
        Tuple[float, float]
    ]  # delivery locations in optimized order
    route_sequence: List[Tuple[str, Tuple[float, float]]] = None  # Mixed sequence
    total_distance: float = 0.0
    total_duration: int = 0  # in seconds
    is_optimized: bool = False

    # New cost tracking fields
    total_cost: float = 0.0
    cost_per_parcel: float = 0.0

    def __post_init__(self):
        self.total_size = sum(p.size for p in self.parcels)
        if self.total_size > self.vehicle_spec.capacity:
            raise ValueError(
                f"Route exceeds vehicle capacity: {self.total_size} > {self.vehicle_spec.capacity}"
            )

        # Initialize route sequence if not provided
        if self.route_sequence is None:
            self.route_sequence = []
            # Add pickups first
            for pickup_id in self.pickup_sequence:
                pickup_location = (
                    self.parcels[0].pickup_location if self.parcels else (0, 0)
                )
                self.route_sequence.append(("pickup", pickup_location))
            # Add deliveries
            for delivery_loc in self.delivery_sequence:
                self.route_sequence.append(("delivery", delivery_loc))

        # Calculate initial costs
        self.update_costs()

    def update_costs(self):
        """Update cost calculations based on current route state"""
        if len(self.parcels) == 0:
            self.total_cost = 0.0
            self.cost_per_parcel = 0.0
            return

        self.total_cost = self.vehicle_spec.calculate_total_route_cost(
            len(self.parcels), include_utilization_penalty=True
        )
        self.cost_per_parcel = self.total_cost / len(self.parcels)

    def is_time_feasible(self, time_window_seconds: int) -> bool:
        """Check if route meets time window constraint"""
        return self.total_duration <= time_window_seconds

    def get_locations_sequence(self) -> List[Tuple[float, float]]:
        """Get just the locations from route sequence"""
        return [loc for _, loc in self.route_sequence]

    def get_utilization_percentage(self) -> float:
        """Get capacity utilization as percentage"""
        if self.vehicle_spec.capacity == 0:
            return 0.0
        return (self.total_size / self.vehicle_spec.capacity) * 100

    def can_fit_parcels(self, additional_parcels: List[Parcel]) -> bool:
        """Check if additional parcels can fit in this route"""
        additional_size = sum(p.size for p in additional_parcels)
        return self.total_size + additional_size <= self.vehicle_spec.capacity

    def add_parcels(self, new_parcels: List[Parcel]):
        """Add parcels to route and update costs"""
        self.parcels.extend(new_parcels)
        self.total_size = sum(p.size for p in self.parcels)

        # Update delivery sequence
        for parcel in new_parcels:
            if parcel.delivery_location not in self.delivery_sequence:
                self.delivery_sequence.append(parcel.delivery_location)

        # Update costs
        self.update_costs()

        # Mark as needing optimization
        self.is_optimized = False

    def remove_parcels(self, parcels_to_remove: List[Parcel]):
        """Remove parcels from route and update costs"""
        for parcel in parcels_to_remove:
            if parcel in self.parcels:
                self.parcels.remove(parcel)
                # Remove from delivery sequence if no other parcels go there
                if not any(
                    p.delivery_location == parcel.delivery_location
                    for p in self.parcels
                ):
                    if parcel.delivery_location in self.delivery_sequence:
                        self.delivery_sequence.remove(parcel.delivery_location)

        self.total_size = sum(p.size for p in self.parcels)
        self.update_costs()
        self.is_optimized = False


class Solution:
    """Complete solution representation with enhanced cost tracking"""

    def __init__(
        self,
        pickup_terminals: List[PickupTerminal],
        vehicle_specs: Dict[VehicleType, VehicleSpec],
    ):
        self.pickup_terminals = pickup_terminals
        self.vehicle_specs = vehicle_specs
        self.routes: List[Route] = []
        self.pickup_assignments: Dict[int, List[int]] = (
            {}
        )  # pickup_id -> list of vehicle_ids
        self.unassigned_parcels: List[Parcel] = []

        # Enhanced cost tracking
        self.total_cost = float("inf")
        self.total_parcels = 0
        self.average_cost_per_parcel = float("inf")

        self.total_duration = 0
        self.is_feasible = False

    def add_route(self, route: Route):
        """Add a route to the solution"""
        self.routes.append(route)

        # Update pickup assignments
        for pickup_id in route.pickup_sequence:
            if pickup_id not in self.pickup_assignments:
                self.pickup_assignments[pickup_id] = []
            self.pickup_assignments[pickup_id].append(route.vehicle_id)

        # Update solution costs
        self.update_solution_costs()

    def remove_route(self, route: Route):
        """Remove a route from the solution"""
        if route in self.routes:
            self.routes.remove(route)

            # Update pickup assignments
            for pickup_id in route.pickup_sequence:
                if pickup_id in self.pickup_assignments:
                    if route.vehicle_id in self.pickup_assignments[pickup_id]:
                        self.pickup_assignments[pickup_id].remove(route.vehicle_id)
                    if not self.pickup_assignments[pickup_id]:
                        del self.pickup_assignments[pickup_id]

            # Update solution costs
            self.update_solution_costs()

    def update_solution_costs(self):
        """Update solution-level cost calculations"""
        if not self.routes:
            self.total_cost = 0.0
            self.total_parcels = 0
            self.average_cost_per_parcel = 0.0
            return

        # Calculate total cost (sum of all vehicle costs)
        self.total_cost = sum(route.total_cost for route in self.routes)

        # Calculate total parcels
        self.total_parcels = sum(len(route.parcels) for route in self.routes)

        # Calculate average cost per parcel
        if self.total_parcels > 0:
            self.average_cost_per_parcel = self.total_cost / self.total_parcels
        else:
            self.average_cost_per_parcel = float("inf")

    def calculate_cost_and_time(
        self, distance_matrix: Dict, eta_matrix: Dict
    ) -> Tuple[float, int]:
        """Calculate total solution cost and duration"""
        total_duration = 0

        for route in self.routes:
            route_distance, route_duration = self._calculate_route_metrics(
                route, distance_matrix, eta_matrix
            )
            total_duration += route_duration

            # Update route metrics
            route.total_distance = route_distance
            route.total_duration = route_duration
            route.update_costs()  # Recalculate costs after metrics update

        self.total_duration = total_duration
        self.update_solution_costs()  # Update solution costs after route updates

        return self.total_cost, total_duration

    def _calculate_route_metrics(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Tuple[float, int]:
        """Calculate distance and duration for a single route using optimized sequence"""
        total_distance = 0
        total_duration = 0

        if not route.route_sequence:
            return 0, 0

        # Get locations from optimized sequence
        route_locations = route.get_locations_sequence()

        # Calculate cumulative distance and time
        for i in range(len(route_locations) - 1):
            loc1 = route_locations[i]
            loc2 = route_locations[i + 1]

            # Add distance
            distance = distance_matrix.get(loc1, {}).get(loc2, 0)
            total_distance += distance

            # Add travel time
            travel_time = eta_matrix.get(loc1, {}).get(loc2, 0)
            total_duration += travel_time

        return total_distance, total_duration

    def get_cost_efficiency_metrics(self) -> Dict:
        """Get detailed cost efficiency metrics"""
        if not self.routes:
            return {
                "total_cost": 0.0,
                "total_parcels": 0,
                "average_cost_per_parcel": 0.0,
                "vehicle_efficiency": {},
                "utilization_stats": {},
            }

        vehicle_efficiency = {}
        utilization_stats = {
            "average_utilization": 0.0,
            "min_utilization": 100.0,
            "max_utilization": 0.0,
            "underutilized_routes": 0,  # < 50% utilization
        }

        total_utilization = 0
        for route in self.routes:
            vtype = route.vehicle_type.value
            utilization = route.get_utilization_percentage()

            if vtype not in vehicle_efficiency:
                vehicle_efficiency[vtype] = {
                    "count": 0,
                    "total_cost": 0.0,
                    "total_parcels": 0,
                    "average_cost_per_parcel": 0.0,
                    "average_utilization": 0.0,
                }

            vehicle_efficiency[vtype]["count"] += 1
            vehicle_efficiency[vtype]["total_cost"] += route.total_cost
            vehicle_efficiency[vtype]["total_parcels"] += len(route.parcels)
            vehicle_efficiency[vtype]["average_utilization"] += utilization

            total_utilization += utilization
            utilization_stats["min_utilization"] = min(
                utilization_stats["min_utilization"], utilization
            )
            utilization_stats["max_utilization"] = max(
                utilization_stats["max_utilization"], utilization
            )

            if utilization < 50.0:
                utilization_stats["underutilized_routes"] += 1

        # Calculate averages
        if self.routes:
            utilization_stats["average_utilization"] = total_utilization / len(
                self.routes
            )

        for vtype_data in vehicle_efficiency.values():
            if vtype_data["count"] > 0:
                vtype_data["average_cost_per_parcel"] = (
                    vtype_data["total_cost"] / vtype_data["total_parcels"]
                )
                vtype_data["average_utilization"] /= vtype_data["count"]

        return {
            "total_cost": self.total_cost,
            "total_parcels": self.total_parcels,
            "average_cost_per_parcel": self.average_cost_per_parcel,
            "vehicle_efficiency": vehicle_efficiency,
            "utilization_stats": utilization_stats,
        }

    def validate_knock_constraints(self, max_knock: int) -> bool:
        """Validate knock constraints"""
        for pickup_id, vehicle_ids in self.pickup_assignments.items():
            if len(set(vehicle_ids)) > max_knock:
                return False
        return True

    def validate_time_window(self, time_window_seconds: int) -> bool:
        """Validate that all routes meet time window constraints"""
        for route in self.routes:
            if (
                hasattr(route, "total_duration")
                and route.total_duration > time_window_seconds
            ):
                return False
        return True

    def get_capacity_utilization(self) -> Dict[VehicleType, List[Dict]]:
        """Calculate capacity utilization per vehicle type"""
        utilization = {}

        for route in self.routes:
            vehicle_type = route.vehicle_type
            if vehicle_type not in utilization:
                utilization[vehicle_type] = []

            util_percent = route.get_utilization_percentage()
            utilization[vehicle_type].append(
                {
                    "vehicle_id": route.vehicle_id,
                    "utilization_percent": round(util_percent, 1),
                    "cost_per_parcel": round(route.cost_per_parcel, 2),
                }
            )

        return utilization

    def _rebuild_assignments(self):
        """Rebuild pickup assignments after route modifications"""
        self.pickup_assignments = {}
        for route in self.routes:
            for pickup_id in route.pickup_sequence:
                if pickup_id not in self.pickup_assignments:
                    self.pickup_assignments[pickup_id] = []
                self.pickup_assignments[pickup_id].append(route.vehicle_id)

    def _get_pickup_location(self, pickup_id: int) -> Tuple[float, float]:
        """Get pickup location by ID"""
        for terminal in self.pickup_terminals:
            if terminal.pickup_id == pickup_id:
                return (terminal.lat, terminal.lon)
        raise ValueError(f"Pickup terminal {pickup_id} not found")

    def get_route_time_analysis(self) -> Dict:
        """Get detailed time analysis for each route"""
        analysis = {
            "routes": [],
            "total_duration": self.total_duration,
            "max_route_duration": 0,
            "time_window_violations": [],
        }

        for route in self.routes:
            route_duration = getattr(route, "total_duration", 0)
            analysis["max_route_duration"] = max(
                analysis["max_route_duration"], route_duration
            )

            route_info = {
                "vehicle_id": route.vehicle_id,
                "vehicle_type": route.vehicle_type.value,
                "duration_seconds": route_duration,
                "duration_hours": round(route_duration / 3600, 2),
                "num_stops": len(route.route_sequence),
                "parcels_count": len(route.parcels),
                "is_optimized": route.is_optimized,
                "total_cost": route.total_cost,
                "cost_per_parcel": route.cost_per_parcel,
                "utilization_percent": route.get_utilization_percentage(),
            }
            analysis["routes"].append(route_info)

        return analysis


class ProblemConfig:
    """Problem configuration with updated cost model"""

    def __init__(
        self,
        max_knock: int,
        time_window_hours: float,
        vehicle_specs: Dict[VehicleType, VehicleSpec],
    ):
        self.max_knock = max_knock
        self.time_window_hours = time_window_hours
        self.time_window_seconds = int(time_window_hours * 3600)
        self.vehicle_specs = vehicle_specs

        # Set time window for all vehicle specs
        for spec in self.vehicle_specs.values():
            spec.set_time_window(time_window_hours)

        if max_knock < 1:
            raise ValueError("max_knock must be >= 1")
        if time_window_hours <= 0:
            raise ValueError("time_window_hours must be > 0")

    def get_cost_summary(self) -> Dict:
        """Get summary of cost configuration"""
        return {
            "time_window_hours": self.time_window_hours,
            "vehicle_costs": {
                vtype.value: {
                    "cost_per_hour": spec.cost_per_hour,
                    "fixed_cost_per_vehicle": spec.fixed_cost_per_vehicle,
                    "capacity": spec.capacity,
                }
                for vtype, spec in self.vehicle_specs.items()
            },
        }
