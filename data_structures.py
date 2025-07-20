import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class VehicleType(Enum):
    VAN = "van"
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
    route_sequence: List[Tuple[str, Tuple[float, float]]] = None
    total_distance: float = 0.0
    total_duration: int = 0  # in seconds
    is_optimized: bool = False
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

            # Add pickups based on actual parcel pickup locations (not pickup_sequence)
            pickup_locations_added = set()
            for parcel in self.parcels:
                pickup_loc = parcel.pickup_location
                if pickup_loc not in pickup_locations_added:
                    self.route_sequence.append(("pickup", pickup_loc))
                    pickup_locations_added.add(pickup_loc)

            # Add deliveries
            for delivery_loc in self.delivery_sequence:
                self.route_sequence.append(("delivery", delivery_loc))

        # Calculate initial costs
        self.update_costs()

    def calculate_real_cost(self):
        """Route real cost without penalty"""
        return round(self.vehicle_spec.fixed_cost_per_vehicle, 2)

    def calculate_real_cost_per_parcel(self):
        return round(
            (
                self.vehicle_spec.fixed_cost_per_vehicle / len(self.parcels)
                if len(self.parcels) != 0
                else 0
            ),
            2,
        )

    def rebuild_route_sequence(self):
        """Rebuild route sequence based on current parcels"""
        self.route_sequence = []

        # Add pickups based on actual parcel pickup locations
        pickup_locations_added = set()
        for parcel in self.parcels:
            pickup_loc = parcel.pickup_location
            if pickup_loc not in pickup_locations_added:
                self.route_sequence.append(("pickup", pickup_loc))
                pickup_locations_added.add(pickup_loc)

        # Add deliveries
        for delivery_loc in self.delivery_sequence:
            self.route_sequence.append(("delivery", delivery_loc))

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

    def add_parcels(
        self, new_parcels: List[Parcel], pickup_terminals: List[PickupTerminal] = None
    ):
        """Add parcels to route and update pickup sequence consistently"""
        self.parcels.extend(new_parcels)
        self.total_size = sum(p.size for p in self.parcels)

        # Update delivery sequence
        for parcel in new_parcels:
            if parcel.delivery_location not in self.delivery_sequence:
                self.delivery_sequence.append(parcel.delivery_location)

        # Rebuild pickup sequence based on actual parcels if terminals provided
        if pickup_terminals:
            self.rebuild_pickup_sequence_from_parcels(pickup_terminals)

        self.rebuild_route_sequence()
        # Update costs
        self.update_costs()

        # Mark as needing optimization
        self.is_optimized = False

    def remove_parcels(
        self,
        parcels_to_remove: List[Parcel],
        pickup_terminals: List[PickupTerminal] = None,
    ):
        """Remove parcels from route and update pickup sequence consistently"""
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

        # Rebuild pickup sequence based on remaining parcels if terminals provided
        if pickup_terminals:
            self.rebuild_pickup_sequence_from_parcels(pickup_terminals)

        self.rebuild_route_sequence()

        self.update_costs()
        self.is_optimized = False

    def rebuild_pickup_sequence_from_parcels(
        self, pickup_terminals: List[PickupTerminal]
    ):
        """Rebuild pickup sequence based on actual parcels in the route"""
        current_pickups = set()

        # Find which pickup terminals are actually represented by parcels
        for parcel in self.parcels:
            for terminal in pickup_terminals:
                if parcel in terminal.parcels:
                    current_pickups.add(terminal.pickup_id)
                    break

        # Update pickup sequence to match actual parcels
        self.pickup_sequence = sorted(current_pickups)


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

    def _rebuild_assignments(self):
        """Rebuild pickup assignments after route modifications"""
        self.pickup_assignments = {}

        for route in self.routes:
            # Build pickup assignments based on actual parcels in the route
            parcels_by_pickup = {}

            # Group parcels by their pickup terminal
            for parcel in route.parcels:
                # Find which pickup terminal this parcel belongs to
                pickup_id = None
                for terminal in self.pickup_terminals:
                    if parcel in terminal.parcels:
                        pickup_id = terminal.pickup_id
                        break

                if pickup_id is not None:
                    if pickup_id not in parcels_by_pickup:
                        parcels_by_pickup[pickup_id] = []
                    parcels_by_pickup[pickup_id].append(parcel)

            # Update pickup assignments for each pickup terminal with parcels
            for pickup_id in parcels_by_pickup.keys():
                if pickup_id not in self.pickup_assignments:
                    self.pickup_assignments[pickup_id] = []

                # Add this vehicle to the pickup assignment if not already there
                if route.vehicle_id not in self.pickup_assignments[pickup_id]:
                    self.pickup_assignments[pickup_id].append(route.vehicle_id)

    def add_route(self, route: Route):
        """Add a route to the solution with proper pickup assignment tracking"""
        self.routes.append(route)

        # Build pickup assignments based on actual parcels in the route
        parcels_by_pickup = {}

        # Group parcels by their pickup terminal
        for parcel in route.parcels:
            # Find which pickup terminal this parcel belongs to
            pickup_id = None
            for terminal in self.pickup_terminals:
                if parcel in terminal.parcels:
                    pickup_id = terminal.pickup_id
                    break

            if pickup_id is not None:
                if pickup_id not in parcels_by_pickup:
                    parcels_by_pickup[pickup_id] = []
                parcels_by_pickup[pickup_id].append(parcel)

        # Update pickup assignments for each pickup terminal with parcels
        for pickup_id in parcels_by_pickup.keys():
            if pickup_id not in self.pickup_assignments:
                self.pickup_assignments[pickup_id] = []

            # Add this vehicle to the pickup assignment if not already there
            if route.vehicle_id not in self.pickup_assignments[pickup_id]:
                self.pickup_assignments[pickup_id].append(route.vehicle_id)

        # Update solution costs
        self.update_solution_costs()

    def remove_route(self, route: Route):
        """Remove a route from the solution with proper pickup assignment cleanup"""
        if route in self.routes:
            self.routes.remove(route)

            # Clean up pickup assignments
            parcels_by_pickup = {}

            # Group parcels by their pickup terminal for the removed route
            for parcel in route.parcels:
                pickup_id = None
                for terminal in self.pickup_terminals:
                    if parcel in terminal.parcels:
                        pickup_id = terminal.pickup_id
                        break

                if pickup_id is not None:
                    if pickup_id not in parcels_by_pickup:
                        parcels_by_pickup[pickup_id] = []
                    parcels_by_pickup[pickup_id].append(parcel)

            # Remove vehicle from pickup assignments for terminals it was serving
            for pickup_id in parcels_by_pickup.keys():
                if pickup_id in self.pickup_assignments:
                    if route.vehicle_id in self.pickup_assignments[pickup_id]:
                        self.pickup_assignments[pickup_id].remove(route.vehicle_id)

                    # Clean up empty pickup assignments
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
            vehicle_efficiency[vtype]["total_cost"] += route.calculate_real_cost()
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

    def validate_pickup_assignments_detailed(self, max_knock: int) -> Dict:
        """Enhanced validation with detailed pickup assignment analysis"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "pickup_analysis": {},
        }

        # Build expected assignments from actual route data
        expected_assignments = {}

        for route in self.routes:
            # Group parcels by pickup terminal
            parcels_by_pickup = {}

            for parcel in route.parcels:
                pickup_id = None
                for terminal in self.pickup_terminals:
                    if parcel in terminal.parcels:
                        pickup_id = terminal.pickup_id
                        break

                if pickup_id is not None:
                    if pickup_id not in parcels_by_pickup:
                        parcels_by_pickup[pickup_id] = []
                    parcels_by_pickup[pickup_id].append(parcel)

            # Update expected assignments
            for pickup_id in parcels_by_pickup.keys():
                if pickup_id not in expected_assignments:
                    expected_assignments[pickup_id] = []
                if route.vehicle_id not in expected_assignments[pickup_id]:
                    expected_assignments[pickup_id].append(route.vehicle_id)

        # Compare expected vs actual assignments
        for pickup_id, expected_vehicles in expected_assignments.items():
            actual_vehicles = self.pickup_assignments.get(pickup_id, [])

            validation_result["pickup_analysis"][pickup_id] = {
                "expected_vehicles": sorted(expected_vehicles),
                "actual_vehicles": sorted(actual_vehicles),
                "is_consistent": set(expected_vehicles) == set(actual_vehicles),
                "missing_vehicles": list(set(expected_vehicles) - set(actual_vehicles)),
                "extra_vehicles": list(set(actual_vehicles) - set(expected_vehicles)),
                "knock_count": len(set(expected_vehicles)),
                "knock_violation": len(set(expected_vehicles)) > max_knock,
            }

            if set(expected_vehicles) != set(actual_vehicles):
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Pickup {pickup_id}: Expected vehicles {sorted(expected_vehicles)}, "
                    f"got {sorted(actual_vehicles)}"
                )

            if len(set(expected_vehicles)) > max_knock:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Pickup {pickup_id}: Knock constraint violated "
                    f"({len(set(expected_vehicles))} > {max_knock})"
                )

        # Check for orphaned assignments
        for pickup_id, vehicles in self.pickup_assignments.items():
            if pickup_id not in expected_assignments:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Pickup {pickup_id}: Has assignment {vehicles} but no routes serve this pickup"
                )

        return validation_result


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
