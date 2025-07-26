import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class VehicleType(Enum):
    VAN = "van"
    BIKE = "bike"
    CARBOX = "carbox"
    BIG_BOX = "big-box"


class ConstraintType(Enum):
    """Classification of constraint types for soft constraint handling"""

    HARD = "hard"  # Must be satisfied (capacity, max_knock)
    SOFT = "soft"  # Can be violated with penalty (time window)


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
    """Vehicle type specifications with new cost model and penalty support"""

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
        self,
        used_capacity: int,
        time_violation_seconds: int = 0,
        time_violation_penalty_per_minute: float = 0.0,
        include_utilization_penalty: bool = True,
    ) -> float:
        """
        Calculate total cost for this vehicle with new cost model and time violation penalties

        Args:
            used_capacity: Number of parcels/capacity used
            time_violation_seconds: Seconds over time window limit
            time_violation_penalty_per_minute: Penalty cost per minute of violation
            include_utilization_penalty: Whether to apply utilization penalties

        Returns:
            Total cost (fixed cost + penalties)
        """
        # Base cost is fixed per vehicle (time_window * cost_per_hour)
        base_cost = self.fixed_cost_per_vehicle

        # Add time violation penalty
        time_penalty = 0.0
        if time_violation_seconds > 0:
            violation_minutes = time_violation_seconds / 60.0
            time_penalty = violation_minutes * time_violation_penalty_per_minute

        total_cost = base_cost + time_penalty

        if not include_utilization_penalty:
            return total_cost

        # Apply utilization modifier to encourage efficient packing
        utilization_score = self.calculate_utilization_score(used_capacity)
        utilization_penalty = max(0.5, utilization_score)  # Minimum 50% efficiency

        # Cost penalty for poor utilization (only applied to base cost, not time penalties)
        return (base_cost / utilization_penalty) + time_penalty

    def calculate_cost_per_parcel(
        self,
        used_capacity: int,
        time_violation_seconds: int = 0,
        time_violation_penalty_per_minute: float = 0.0,
    ) -> float:
        """Calculate cost per parcel for this vehicle configuration including penalties"""
        if used_capacity <= 0:
            return float("inf")

        total_cost = self.calculate_total_route_cost(
            used_capacity, time_violation_seconds, time_violation_penalty_per_minute
        )
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
    """Represents a vehicle route with enhanced cost tracking and soft constraint support"""

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

    # New fields for soft constraint support
    time_violation_seconds: int = 0  # How much over time window
    time_violation_penalty: float = 0.0  # Penalty cost for time violation
    is_time_feasible: bool = True  # Whether route meets time window
    constraint_violations: Dict[str, Dict] = None  # Track all constraint violations

    def __post_init__(self):
        self.total_size = sum(p.size for p in self.parcels)
        if self.total_size > self.vehicle_spec.capacity:
            raise ValueError(
                f"Route exceeds vehicle capacity: {self.total_size} > {self.vehicle_spec.capacity}"
            )

        # Initialize constraint violations tracking
        if self.constraint_violations is None:
            self.constraint_violations = {
                "time_window": {
                    "violated": False,
                    "violation_amount": 0,
                    "penalty": 0.0,
                },
                "capacity": {"violated": False, "violation_amount": 0, "penalty": 0.0},
            }

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

    def update_costs(self, time_violation_penalty_per_minute: float = 0.0):
        """Update cost calculations based on current route state including soft constraints"""
        if len(self.parcels) == 0:
            self.total_cost = 0.0
            self.cost_per_parcel = 0.0
            self.time_violation_penalty = 0.0
            return

        # Calculate time violation penalty
        self.time_violation_penalty = 0.0
        if self.time_violation_seconds > 0:
            violation_minutes = self.time_violation_seconds / 60.0
            self.time_violation_penalty = (
                violation_minutes * time_violation_penalty_per_minute
            )

        # Update constraint violations tracking
        self.constraint_violations["time_window"] = {
            "violated": self.time_violation_seconds > 0,
            "violation_amount": self.time_violation_seconds,
            "penalty": self.time_violation_penalty,
        }

        self.total_cost = self.vehicle_spec.calculate_total_route_cost(
            len(self.parcels),
            self.time_violation_seconds,
            time_violation_penalty_per_minute,
            include_utilization_penalty=True,
        )
        self.cost_per_parcel = self.total_cost / len(self.parcels)

    def update_time_feasibility(self, time_window_seconds: int):
        """Update time feasibility status and violation amounts"""
        if self.total_duration <= time_window_seconds:
            self.is_time_feasible = True
            self.time_violation_seconds = 0
        else:
            self.is_time_feasible = False
            self.time_violation_seconds = self.total_duration - time_window_seconds

    def is_time_feasible_check(self, time_window_seconds: int) -> bool:
        """Check if route meets time window constraint"""
        return self.total_duration <= time_window_seconds

    def get_constraint_violation_summary(self) -> Dict:
        """Get summary of all constraint violations for this route"""
        return {
            "has_violations": any(
                v["violated"] for v in self.constraint_violations.values()
            ),
            "violations": self.constraint_violations.copy(),
            "total_penalty": sum(
                v["penalty"] for v in self.constraint_violations.values()
            ),
            "time_feasible": self.is_time_feasible,
            "time_violation_minutes": (
                self.time_violation_seconds / 60.0
                if self.time_violation_seconds > 0
                else 0.0
            ),
        }

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
        self,
        new_parcels: List[Parcel],
        pickup_terminals: List[PickupTerminal] = None,
        time_violation_penalty_per_minute: float = 0.0,
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
        # Update costs with penalty consideration
        self.update_costs(time_violation_penalty_per_minute)

        # Mark as needing optimization
        self.is_optimized = False

    def remove_parcels(
        self,
        parcels_to_remove: List[Parcel],
        pickup_terminals: List[PickupTerminal] = None,
        time_violation_penalty_per_minute: float = 0.0,
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

        self.update_costs(time_violation_penalty_per_minute)
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
    """Complete solution representation with enhanced cost tracking and soft constraint support"""

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

        # Soft constraint support
        self.is_hard_feasible = False  # Satisfies capacity and max_knock
        self.is_soft_feasible = False  # Satisfies time windows
        self.is_fully_feasible = False  # Satisfies all constraints

        # Violation tracking
        self.total_time_violation_seconds = 0
        self.total_time_violation_penalty = 0.0
        self.routes_with_time_violations = 0
        self.constraint_violation_summary = {}
        self._max_knock = None

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

    def add_route(self, route: Route, time_violation_penalty_per_minute: float = 0.0):
        """Add a route to the solution with PROPER knock constraint validation"""
        
        # ADDED: Validate knock constraints BEFORE adding route
        temp_pickup_assignments = self.pickup_assignments.copy()
        
        # Build what the pickup assignments would be with this new route
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
        
        # Check knock constraints for each pickup terminal this route would serve
        max_knock = getattr(self, '_max_knock', 2)  # Default fallback
        for pickup_id in parcels_by_pickup.keys():
            if pickup_id not in temp_pickup_assignments:
                temp_pickup_assignments[pickup_id] = []
            
            # Check if adding this vehicle would violate knock constraint
            current_vehicles = set(temp_pickup_assignments[pickup_id])
            if route.vehicle_id not in current_vehicles:
                # This would be a new vehicle assignment
                if len(current_vehicles) >= max_knock:
                    raise ValueError(
                        f"Adding route {route.vehicle_id} would violate knock constraint "
                        f"for pickup {pickup_id}: {len(current_vehicles)} >= {max_knock}"
                    )
        
        # If we get here, knock constraints are satisfied - proceed with adding route
        self.routes.append(route)

        # Update route costs with penalty consideration
        route.update_costs(time_violation_penalty_per_minute)

        # Build pickup assignments based on actual parcels in the route
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
        """Update solution-level cost calculations including soft constraint penalties"""
        if not self.routes:
            self.total_cost = 0.0
            self.total_parcels = 0
            self.average_cost_per_parcel = 0.0
            self.total_time_violation_seconds = 0
            self.total_time_violation_penalty = 0.0
            self.routes_with_time_violations = 0
            return

        # Calculate total cost (sum of all vehicle costs including penalties)
        self.total_cost = sum(route.total_cost for route in self.routes)

        # Calculate total parcels
        self.total_parcels = sum(len(route.parcels) for route in self.routes)

        # Calculate average cost per parcel
        if self.total_parcels > 0:
            self.average_cost_per_parcel = self.total_cost / self.total_parcels
        else:
            self.average_cost_per_parcel = float("inf")

        # Calculate time violation statistics
        self.total_time_violation_seconds = sum(
            route.time_violation_seconds for route in self.routes
        )
        self.total_time_violation_penalty = sum(
            route.time_violation_penalty for route in self.routes
        )
        self.routes_with_time_violations = sum(
            1 for route in self.routes if route.time_violation_seconds > 0
        )

    def update_feasibility_status(self, max_knock: int, time_window_seconds: int):
        """Update all feasibility status indicators"""
        # Update time feasibility for all routes
        for route in self.routes:
            route.update_time_feasibility(time_window_seconds)

        # Check hard constraints (capacity is checked at route level, max_knock here)
        self.is_hard_feasible = len(
            self.unassigned_parcels
        ) == 0 and self.validate_knock_constraints(max_knock)

        # Check soft constraints (time windows)
        self.is_soft_feasible = all(route.is_time_feasible for route in self.routes)

        # Overall feasibility
        self.is_fully_feasible = self.is_hard_feasible and self.is_soft_feasible

        # Update constraint violation summary
        self.update_constraint_violation_summary()

    def update_constraint_violation_summary(self):
        """Update comprehensive constraint violation summary"""
        self.constraint_violation_summary = {
            "hard_constraints": {
                "capacity_violations": sum(
                    1
                    for route in self.routes
                    if route.total_size > route.vehicle_spec.capacity
                ),
                "unassigned_parcels": len(self.unassigned_parcels),
                "knock_violations": self._count_knock_violations(),
            },
            "soft_constraints": {
                "time_violations": {
                    "routes_count": self.routes_with_time_violations,
                    "total_violation_seconds": self.total_time_violation_seconds,
                    "total_violation_minutes": self.total_time_violation_seconds / 60.0,
                    "total_penalty": self.total_time_violation_penalty,
                    "average_violation_per_route": (
                        self.total_time_violation_seconds / len(self.routes)
                        if self.routes
                        else 0
                    ),
                }
            },
            "feasibility_status": {
                "hard_feasible": self.is_hard_feasible,
                "soft_feasible": self.is_soft_feasible,
                "fully_feasible": self.is_fully_feasible,
            },
        }

    def _count_knock_violations(self) -> int:
        """Count number of knock constraint violations"""
        violations = 0
        for pickup_id, vehicle_ids in self.pickup_assignments.items():
            if (
                len(set(vehicle_ids)) > 1
            ):  # Assuming max_knock is being checked elsewhere
                violations += 1
        return violations

    def calculate_cost_and_time(
        self,
        distance_matrix: Dict,
        eta_matrix: Dict,
        time_violation_penalty_per_minute: float = 0.0,
    ) -> Tuple[float, int]:
        """Calculate total solution cost and duration with penalty support"""
        total_duration = 0

        for route in self.routes:
            route_distance, route_duration = self._calculate_route_metrics(
                route, distance_matrix, eta_matrix
            )
            total_duration += route_duration

            # Update route metrics
            route.total_distance = route_distance
            route.total_duration = route_duration

            # Update costs with penalty consideration
            route.update_costs(time_violation_penalty_per_minute)

        self.total_duration = total_duration
        self.update_solution_costs()

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
        """Get detailed cost efficiency metrics including penalty analysis"""
        if not self.routes:
            return {
                "total_cost": 0.0,
                "total_parcels": 0,
                "average_cost_per_parcel": 0.0,
                "vehicle_efficiency": {},
                "utilization_stats": {},
                "penalty_analysis": {},
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
                    "total_time_penalties": 0.0,
                }

            vehicle_efficiency[vtype]["count"] += 1
            vehicle_efficiency[vtype]["total_cost"] += route.calculate_real_cost()
            vehicle_efficiency[vtype]["total_parcels"] += len(route.parcels)
            vehicle_efficiency[vtype]["average_utilization"] += utilization
            vehicle_efficiency[vtype][
                "total_time_penalties"
            ] += route.time_violation_penalty

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

        # Penalty analysis
        penalty_analysis = {
            "total_penalty": self.total_time_violation_penalty,
            "penalty_percentage_of_total_cost": (
                (self.total_time_violation_penalty / self.total_cost * 100)
                if self.total_cost > 0
                else 0
            ),
            "routes_with_penalties": self.routes_with_time_violations,
            "average_penalty_per_violation": (
                (self.total_time_violation_penalty / self.routes_with_time_violations)
                if self.routes_with_time_violations > 0
                else 0
            ),
        }

        return {
            "total_cost": self.total_cost,
            "total_parcels": self.total_parcels,
            "average_cost_per_parcel": self.average_cost_per_parcel,
            "vehicle_efficiency": vehicle_efficiency,
            "utilization_stats": utilization_stats,
            "penalty_analysis": penalty_analysis,
        }

    def validate_knock_constraints(self, max_knock: int) -> bool:
        """Validate knock constraints (hard constraint)"""
        for pickup_id, vehicle_ids in self.pickup_assignments.items():
            if len(set(vehicle_ids)) > max_knock:
                return False
        return True

    def validate_time_window(self, time_window_seconds: int) -> bool:
        """Validate that all routes meet time window constraints (soft constraint)"""
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
                    "time_violation_seconds": route.time_violation_seconds,
                    "time_violation_penalty": round(route.time_violation_penalty, 2),
                    "is_time_feasible": route.is_time_feasible,
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
        """Get detailed time analysis for each route including violations"""
        analysis = {
            "routes": [],
            "total_duration": self.total_duration,
            "max_route_duration": 0,
            "time_window_violations": [],
            "violation_summary": {
                "total_violations": self.routes_with_time_violations,
                "total_violation_seconds": self.total_time_violation_seconds,
                "total_violation_penalty": self.total_time_violation_penalty,
            },
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
                "time_violation_seconds": route.time_violation_seconds,
                "time_violation_penalty": route.time_violation_penalty,
                "is_time_feasible": route.is_time_feasible,
            }
            analysis["routes"].append(route_info)

            # Track violations
            if route.time_violation_seconds > 0:
                analysis["time_window_violations"].append(
                    {
                        "vehicle_id": route.vehicle_id,
                        "violation_seconds": route.time_violation_seconds,
                        "violation_minutes": round(
                            route.time_violation_seconds / 60, 2
                        ),
                        "penalty": route.time_violation_penalty,
                    }
                )

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

    def get_soft_constraint_solution_quality(self) -> Dict:
        """Get quality metrics specifically for soft constraint handling"""
        return {
            "constraint_satisfaction": {
                "hard_constraints_satisfied": self.is_hard_feasible,
                "soft_constraints_satisfied": self.is_soft_feasible,
                "fully_feasible": self.is_fully_feasible,
            },
            "time_violation_metrics": {
                "routes_with_violations": self.routes_with_time_violations,
                "total_routes": len(self.routes),
                "violation_rate": (
                    self.routes_with_time_violations / len(self.routes) * 100
                    if self.routes
                    else 0
                ),
                "total_violation_minutes": self.total_time_violation_seconds / 60.0,
                "average_violation_per_route": (
                    self.total_time_violation_seconds / len(self.routes) / 60.0
                    if self.routes
                    else 0
                ),
                "total_penalty_cost": self.total_time_violation_penalty,
                "penalty_as_percentage_of_total": (
                    self.total_time_violation_penalty / self.total_cost * 100
                    if self.total_cost > 0
                    else 0
                ),
            },
            "solution_degradation": {
                "cost_increase_due_to_penalties": self.total_time_violation_penalty,
                "base_cost_without_penalties": self.total_cost
                - self.total_time_violation_penalty,
                "penalty_overhead_percentage": (
                    self.total_time_violation_penalty
                    / (self.total_cost - self.total_time_violation_penalty)
                    * 100
                    if (self.total_cost - self.total_time_violation_penalty) > 0
                    else 0
                ),
            },
        }


class ProblemConfig:
    """Problem configuration with updated cost model and soft constraint support"""

    def __init__(
        self,
        max_knock: int,
        time_window_hours: float,
        vehicle_specs: Dict[VehicleType, VehicleSpec],
        time_violation_penalty_per_minute: float = 10.0,  # New parameter
        allow_time_violations: bool = True,  # New parameter
        prefer_time_feasible: bool = True,  # New parameter
    ):
        self.max_knock = max_knock
        self.time_window_hours = time_window_hours
        self.time_window_seconds = int(time_window_hours * 3600)
        self.vehicle_specs = vehicle_specs

        # Soft constraint parameters
        self.time_violation_penalty_per_minute = time_violation_penalty_per_minute
        self.allow_time_violations = allow_time_violations
        self.prefer_time_feasible = prefer_time_feasible

        # Set time window for all vehicle specs
        for spec in self.vehicle_specs.values():
            spec.set_time_window(time_window_hours)

        if max_knock < 1:
            raise ValueError("max_knock must be >= 1")
        if time_window_hours <= 0:
            raise ValueError("time_window_hours must be > 0")
        if time_violation_penalty_per_minute < 0:
            raise ValueError("time_violation_penalty_per_minute must be >= 0")

    def get_cost_summary(self) -> Dict:
        """Get summary of cost configuration including penalty settings"""
        return {
            "time_window_hours": self.time_window_hours,
            "time_violation_penalty_per_minute": self.time_violation_penalty_per_minute,
            "allow_time_violations": self.allow_time_violations,
            "prefer_time_feasible": self.prefer_time_feasible,
            "vehicle_costs": {
                vtype.value: {
                    "cost_per_hour": spec.cost_per_hour,
                    "fixed_cost_per_vehicle": spec.fixed_cost_per_vehicle,
                    "capacity": spec.capacity,
                }
                for vtype, spec in self.vehicle_specs.items()
            },
        }

    def get_constraint_configuration(self) -> Dict:
        """Get constraint configuration summary"""
        return {
            "hard_constraints": {
                "max_knock": self.max_knock,
                "vehicle_capacities": {
                    vtype.value: spec.capacity
                    for vtype, spec in self.vehicle_specs.items()
                },
            },
            "soft_constraints": {
                "time_window_seconds": self.time_window_seconds,
                "time_window_hours": self.time_window_hours,
                "violation_penalty_per_minute": self.time_violation_penalty_per_minute,
                "violations_allowed": self.allow_time_violations,
                "prefer_feasible_solutions": self.prefer_time_feasible,
            },
        }

    def calculate_time_violation_penalty(self, violation_seconds: int) -> float:
        """Calculate penalty for time window violation"""
        if violation_seconds <= 0:
            return 0.0
        violation_minutes = violation_seconds / 60.0
        return violation_minutes * self.time_violation_penalty_per_minute

    def is_solution_acceptable(
        self,
        solution: "Solution",
        max_acceptable_violation_minutes: float = float("inf"),
    ) -> bool:
        """
        Determine if a solution is acceptable based on constraint configuration

        Args:
            solution: Solution to evaluate
            max_acceptable_violation_minutes: Maximum acceptable time violation

        Returns:
            True if solution is acceptable
        """
        # Always reject if hard constraints are violated
        if not solution.is_hard_feasible:
            return False

        # If time violations not allowed, require full feasibility
        if not self.allow_time_violations:
            return solution.is_soft_feasible

        # If violations allowed, check against maximum acceptable violation
        total_violation_minutes = solution.total_time_violation_seconds / 60.0
        return total_violation_minutes <= max_acceptable_violation_minutes
