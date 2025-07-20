import time
import random

from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from ortools_optimizer import GlobalOptimizationController

from data_structures import (
    Route,
    Solution,
    VehicleType,
    VehicleSpec,
    Parcel,
    PickupTerminal,
    ProblemConfig,
)


class UpdatedALNSSolver:
    """
    Main ALNS solver with integrated global OR-Tools optimization
    """

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.constructor = UpdatedConstructionHeuristic(config)
        self.operators = UpdatedALNSOperators(config)
        self.global_optimizer = GlobalOptimizationController(config)
        self.random = random.Random(42)

    def solve_optimized(
        self,
        pickup_terminals: List[PickupTerminal],
        distance_matrix: Dict,
        eta_matrix: Dict,
        max_iterations: int = 1000,
    ) -> Solution:
        """
        Optimized solve method with global OR-Tools integration
        """
        start_time = time.time()

        # Adaptive parameters based on instance size
        instance_size = sum(len(terminal.parcels) for terminal in pickup_terminals)
        verbose_logging = True

        if verbose_logging:
            print(f"Starting optimization for {instance_size} parcels")

        optimization_intervals = max_iterations // 4
        stagnation_limit = optimization_intervals + 1

        # Generate initial solution
        if verbose_logging:
            print("Generating initial solution...")
        current_solution = self.constructor.greedy_construction(
            pickup_terminals, distance_matrix, eta_matrix
        )

        # Apply initial global optimization
        if verbose_logging:
            print("Initial global optimization...")
        try:
            initial_optimized = self.global_optimizer.optimize_if_needed(
                -1,
                current_solution,
                -1,
                distance_matrix,
                eta_matrix,
                optimization_intervals,
            )

            if (
                initial_optimized.average_cost_per_parcel
                < current_solution.average_cost_per_parcel * 0.99
            ):
                current_solution = initial_optimized
                if verbose_logging:
                    print("Initial optimization improved solution")
            else:
                if verbose_logging:
                    print("Initial optimization yielded minimal improvement")

        except Exception as e:
            if verbose_logging:
                print(f"Warning: Initial optimization failed: {e}")

        best_solution = deepcopy(current_solution)
        self._print_solution_stats(current_solution, "Initial", verbose_logging)

        if verbose_logging:
            print(
                f"Running {max_iterations} iterations (stagnation limit: {stagnation_limit})"
            )

        # ALNS main loop
        last_improvement = 0
        consecutive_no_improvement = 0

        for iteration in range(max_iterations):
            try:
                if verbose_logging:
                    print(f"Iteration number {iteration}:")
                    print(f"Last Improvement: {last_improvement}")
                # Intelligent operator selection
                repaired = self._apply_alns_operator(
                    current_solution,
                    iteration,
                    consecutive_no_improvement,
                    distance_matrix,
                    eta_matrix,
                )
                if verbose_logging:
                    print(f"ALNS Operator: {repaired}")
                if repaired is None:
                    consecutive_no_improvement += 1
                    continue

                # Apply global optimization at intervals
                iterations_since_improvement = iteration - last_improvement
                if verbose_logging:
                    print(
                        f"Iterations Since Improvement: {iterations_since_improvement}"
                    )
                globally_optimized = self.global_optimizer.optimize_if_needed(
                    iteration,
                    repaired,
                    iterations_since_improvement,
                    distance_matrix,
                    eta_matrix,
                    optimization_intervals,
                )

                # Evaluate solution
                globally_optimized.calculate_cost_and_time(distance_matrix, eta_matrix)
                globally_optimized.is_feasible = self._check_feasibility(
                    globally_optimized
                )

                # Acceptance criterion based on cost per parcel
                if self._should_accept_solution(
                    globally_optimized, best_solution, best_solution
                ):
                    current_solution = globally_optimized
                    consecutive_no_improvement = 0

                    # Update best solution if better
                    if globally_optimized.is_feasible and (
                        not best_solution.is_feasible
                        or globally_optimized.average_cost_per_parcel
                        < best_solution.average_cost_per_parcel
                    ):

                        improvement = (
                            (
                                best_solution.average_cost_per_parcel
                                - globally_optimized.average_cost_per_parcel
                            )
                            / best_solution.average_cost_per_parcel
                        ) * 100

                        best_solution = deepcopy(globally_optimized)
                        last_improvement = iteration

                        if improvement > 5.0:  # Significant improvement
                            if verbose_logging or iteration % 25 == 0:
                                print(
                                    f"Iteration {iteration}: Significant improvement ({improvement:.1f}%)"
                                )
                                self._print_solution_stats(
                                    best_solution,
                                    f"Iteration {iteration}",
                                    verbose_logging,
                                )
                        elif improvement > 1.0:  # Minor improvement
                            if verbose_logging:
                                print(
                                    f"Iteration {iteration}: Minor improvement ({improvement:.1f}%)"
                                )
                else:
                    consecutive_no_improvement += 1

                # Early termination checks
                if iteration - last_improvement > stagnation_limit:
                    if verbose_logging:
                        print(
                            f"Early termination: No improvement for {iteration - last_improvement} iterations"
                        )
                    break

                # Check if we have optimal single route (often can't be improved)
                if (
                    len(best_solution.routes) == 1
                    and best_solution.routes[0].get_utilization_percentage() > 80
                    and iteration - last_improvement > 10
                ):
                    if verbose_logging:
                        print("Early termination: Single high-utilization route found")
                    break

            except Exception as e:
                if verbose_logging:
                    print(f"Iteration {iteration}: Error: {e}")
                consecutive_no_improvement += 1
                continue

        processing_time = time.time() - start_time

        # Final global optimization
        if (
            best_solution.is_feasible
            and len(best_solution.routes) > 1
            and (
                iteration % optimization_intervals != 0
                or iteration % optimization_intervals != 1
            )
        ):
            if verbose_logging:
                print("Final global optimization...")
            try:
                final_optimized = self.global_optimizer.final_optimization(
                    best_solution, distance_matrix, eta_matrix
                )
                final_optimized.calculate_cost_and_time(distance_matrix, eta_matrix)

                if (
                    final_optimized.is_feasible
                    and final_optimized.average_cost_per_parcel
                    < best_solution.average_cost_per_parcel * 0.99
                ):
                    best_solution = final_optimized
                    if verbose_logging:
                        print("Final optimization improved solution")
                else:
                    if verbose_logging:
                        print("Final optimization yielded minimal improvement")
            except Exception as e:
                if verbose_logging:
                    print(f"Final optimization failed: {e}")

        if verbose_logging:
            print(f"Completed in {processing_time:.2f} seconds")

        self._print_final_stats(best_solution)
        return best_solution

    def _apply_alns_operator(
        self,
        solution: Solution,
        iteration: int,
        consecutive_no_improvement: int,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[Solution]:
        """Apply ALNS operator based on current state"""

        # If we have single route and it's well-utilized, focus on simple operations
        if (
            len(solution.routes) == 1
            and solution.routes[0].get_utilization_percentage() > 70
        ):
            return self._apply_simple_destroy_repair(
                solution, distance_matrix, eta_matrix
            )

        # If stagnating, try more disruptive operations
        if consecutive_no_improvement > 10:
            return self._apply_diversification(solution, distance_matrix, eta_matrix)

        # Normal operation selection
        operation_choice = self.random.random()

        if operation_choice < 0.6:  # Primary: destroy-repair
            return self._apply_destroy_repair(solution, distance_matrix, eta_matrix)
        elif operation_choice < 0.8:  # Secondary: route modification
            return self._apply_route_modification(solution, distance_matrix, eta_matrix)
        else:  # Tertiary: simple local search
            return self._apply_local_search(solution, distance_matrix, eta_matrix)

    def _apply_simple_destroy_repair(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Optional[Solution]:
        """Apply simple destroy-repair for single route solutions"""
        try:
            if self.random.random() < 0.5:
                destroyed = self.operators.random_removal(solution, removal_rate=0.2)
            else:
                destroyed = self.operators.terminal_removal(solution, num_terminals=1)

            return self.operators.greedy_insertion(
                destroyed, distance_matrix, eta_matrix
            )
        except Exception as e:
            return None

    def _apply_diversification(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Optional[Solution]:
        """Apply diversification operations when stagnating"""
        try:
            # More aggressive removal
            destroyed = self.operators.random_removal(solution, removal_rate=0.4)
            return self.operators.greedy_insertion(
                destroyed, distance_matrix, eta_matrix
            )
        except Exception as e:
            return None

    def _apply_destroy_repair(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Optional[Solution]:
        """Apply standard destroy-repair operations"""
        try:
            if self.random.random() < 0.5:
                destroyed = self.operators.random_removal(solution)
            else:
                destroyed = self.operators.terminal_removal(solution)

            return self.operators.greedy_insertion(
                destroyed, distance_matrix, eta_matrix
            )
        except Exception as e:
            return None

    def _apply_route_modification(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Optional[Solution]:
        """Apply route modification operations"""
        try:
            if self.random.random() < 0.7:
                return self.operators.route_splitting_operator(
                    solution, distance_matrix, eta_matrix
                )
            else:
                return self.operators.parcel_relocation_operator(
                    solution, distance_matrix, eta_matrix
                )
        except Exception as e:
            return None

    def _apply_local_search(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Optional[Solution]:
        """Apply local search operations"""
        try:
            return self.operators.local_parcel_swap(
                solution, distance_matrix, eta_matrix
            )
        except Exception as e:
            return None

    def _check_feasibility(self, solution: Solution) -> bool:
        """Check if solution is feasible"""
        return (
            len(solution.unassigned_parcels) == 0
            and solution.validate_knock_constraints(self.config.max_knock)
            and solution.validate_time_window(self.config.time_window_seconds)
        )

    def _should_accept_solution(
        self,
        new_solution: Solution,
        current_solution: Solution,
        best_solution: Solution,
    ) -> bool:
        """Determine whether to accept new solution based on cost per parcel"""
        if new_solution.is_feasible:
            # Always accept if better than current
            if (
                new_solution.average_cost_per_parcel
                < current_solution.average_cost_per_parcel
            ):
                return True
            # Accept slightly worse solutions occasionally for diversification
            elif self.random.random() < 0.05:  # 5% chance
                return True
        else:
            # Accept infeasible solutions very rarely
            if self.random.random() < 0.01:  # 1% chance
                return True

        return False

    def _print_solution_stats(self, solution: Solution, label: str, verbose: bool):
        """Print solution statistics"""
        if not verbose:
            return

        vehicle_counts = {}
        for route in solution.routes:
            vtype = route.vehicle_type.value
            vehicle_counts[vtype] = vehicle_counts.get(vtype, 0) + 1

        print(
            f"{label} solution: {len(solution.routes)} routes, "
            f"cost per parcel: {solution.average_cost_per_parcel:.2f}, "
            f"vehicles: {vehicle_counts}"
        )

    def _print_final_stats(self, solution: Solution):
        """Print comprehensive final solution statistics"""
        print("\n" + "=" * 60)
        print("FINAL SOLUTION SUMMARY")
        print("=" * 60)

        if solution.is_feasible:
            # Get cost efficiency metrics
            cost_metrics = solution.get_cost_efficiency_metrics()

            print(f"✅ FEASIBLE SOLUTION FOUND")
            print(f"📊 Total Routes: {len(solution.routes)}")
            print(f"🚗 Vehicle Distribution: {cost_metrics['vehicle_efficiency']}")
            print(f"💰 Total Cost: {solution.total_cost:.2f}")
            print(f"📦 Total Parcels: {solution.total_parcels}")
            print(f"💡 Average Cost per Parcel: {solution.average_cost_per_parcel:.2f}")
            print(f"⏱️  Total Duration: {solution.total_duration/3600:.2f} hours")
            print(
                f"⏱️  Max Route Duration: {max((getattr(r, 'total_duration', 0) for r in solution.routes), default=0)/3600:.2f} hours"
            )
            print(
                f"📈 Average Utilization: {cost_metrics['utilization_stats']['average_utilization']:.1f}%"
            )
            print(
                f"⚠️  Underutilized Routes: {cost_metrics['utilization_stats']['underutilized_routes']}"
            )

            # Time window and knock analysis
            time_violations = [
                r
                for r in solution.routes
                if not r.is_time_feasible(self.config.time_window_seconds)
            ]
            if time_violations:
                print(f"⚠️  Time Window Violations: {len(time_violations)}")
            else:
                print(f"✅ All routes meet time window constraints")

            knock_violations = []
            for pickup_id, vehicle_ids in solution.pickup_assignments.items():
                if len(set(vehicle_ids)) > self.config.max_knock:
                    knock_violations.append(pickup_id)

            if knock_violations:
                print(
                    f"⚠️  Knock Constraint Violations: {len(knock_violations)} terminals"
                )
            else:
                print(f"✅ All knock constraints satisfied")

        else:
            print(f"❌ NO FEASIBLE SOLUTION FOUND")
            print(f"📊 Best Attempt: {len(solution.routes)} routes")
            print(f"📦 Unassigned Parcels: {len(solution.unassigned_parcels)}")
            print(f"💰 Cost per Parcel: {solution.average_cost_per_parcel:.2f}")

        print("=" * 60)

    def format_output(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Dict:
        """Format solution to match expected output format"""
        # Get cost efficiency metrics
        cost_metrics = solution.get_cost_efficiency_metrics()

        # Build analysis
        capacity_utilization = solution.get_capacity_utilization()

        # Add empty vehicle types
        for vehicle_type in [VehicleType.BIKE, VehicleType.BIG_BOX, VehicleType.CARBOX]:
            if vehicle_type not in capacity_utilization:
                capacity_utilization[vehicle_type] = []

        # Knock analysis
        knock_analysis = []
        for pickup_id, vehicle_ids in solution.pickup_assignments.items():
            vehicles_info = []
            for vehicle_id in vehicle_ids:
                route = next(
                    (r for r in solution.routes if r.vehicle_id == vehicle_id), None
                )
                if route:
                    parcels_count = sum(
                        1
                        for p in route.parcels
                        if self._get_parcel_pickup_id(p, solution) == pickup_id
                    )
                    vehicles_info.append(
                        {"vehicle_id": vehicle_id, "parcels_from_pickup": parcels_count}
                    )

            knock_analysis.append(
                {
                    "pickup_id": pickup_id,
                    "total_knocks": len(set(vehicle_ids)),
                    "vehicles": vehicles_info,
                }
            )

        # Format routes
        formatted_routes = []
        total_duration_seconds = 0

        for route in solution.routes:
            route_distance = getattr(route, "total_distance", 0)
            route_duration = getattr(route, "total_duration", 0)
            total_duration_seconds += route_duration

            # Build physical route from optimized sequence
            physical_route = []
            route_indices = []

            pickup_locations_in_route = set()
            for action, location in route.route_sequence:
                if action == "pickup":
                    pickup_locations_in_route.add(location)

            for parcel in route.parcels:
                pickup_loc = parcel.pickup_location
                if pickup_loc not in pickup_locations_in_route:
                    # Insert pickup at the beginning if missing
                    route.route_sequence.insert(0, ("pickup", pickup_loc))

            for i, (action, location) in enumerate(route.route_sequence):
                physical_route.append((location[0], location[1]))
                route_indices.append(i)

            formatted_route = {
                "vehicle_id": route.vehicle_id,
                "vehicle_type": route.vehicle_type.value,
                "parcels": [float(p.id) for p in route.parcels],
                "route_indices": route_indices,
                "physical_route": physical_route,
                "route_sequence": route.route_sequence,
                "num_stops": len(route.route_sequence),
                "total_duration_seconds": int(route_duration),
                "total_distance_m": int(route_distance),
                "total_cost_kt": route.calculate_real_cost(),
                "cost_per_parcel": route.calculate_real_cost_per_parcel(),
                "capacity_used": route.total_size,
                "vehicle_capacity": route.vehicle_spec.capacity,
                "utilization_percent": route.get_utilization_percentage(),
                "time_window_feasible": route.is_time_feasible(
                    self.config.time_window_seconds
                ),
                "is_optimized": route.is_optimized,
                "parcels_per_pickup": {
                    pickup_id: len(
                        [
                            p
                            for p in route.parcels
                            if self._get_parcel_pickup_id(p, solution) == pickup_id
                        ]
                    )
                    for pickup_id in route.pickup_sequence
                },
            }
            formatted_routes.append(formatted_route)

        # Count vehicles by type
        vehicles_by_type = {}
        for route in solution.routes:
            vehicle_type = route.vehicle_type.value
            vehicles_by_type[vehicle_type] = vehicles_by_type.get(vehicle_type, 0) + 1

        # Total knocks per pickup
        total_knocks_per_pickup = []
        for pickup_id, vehicle_ids in solution.pickup_assignments.items():
            total_knocks_per_pickup.append({pickup_id: len(set(vehicle_ids))})

        # Time window validation
        time_window_violations = [
            r
            for r in solution.routes
            if not r.is_time_feasible(self.config.time_window_seconds)
        ]

        return {
            "input_summary": {
                "num_hubs": len(solution.pickup_terminals),
                "total_parcels": sum(len(t.parcels) for t in solution.pickup_terminals),
                "max_knock": self.config.max_knock,
                "time_window_seconds": self.config.time_window_seconds,
                "time_window_hours": self.config.time_window_hours,
            },
            "formatted_pickup_data": [
                {
                    "pickup_id": t.pickup_id,
                    "lat": t.lat,
                    "lon": t.lon,
                    "num_parcels": len(t.parcels),
                }
                for t in solution.pickup_terminals
            ],
            "results_summary": {
                "total_vehicles_used": len(solution.routes),
                "vehicles_by_type": vehicles_by_type,
                "total_knocks_per_pickup": total_knocks_per_pickup,
                "total_cost": solution.total_cost,
                "average_cost_per_parcel": solution.average_cost_per_parcel,
                "total_duration_seconds": total_duration_seconds,
                "max_route_duration_seconds": max(
                    (getattr(r, "total_duration", 0) for r in solution.routes),
                    default=0,
                ),
                "time_window_feasible": len(time_window_violations) == 0,
                "time_window_violations": len(time_window_violations),
                "global_optimization_applied": True,
                "cost_efficiency_metrics": cost_metrics,
                "map_filename": "vehicle_routes.html",
            },
            "vehicles": formatted_routes,
            "analysis": {
                "capacity_utilization": {
                    k.value: v for k, v in capacity_utilization.items()
                },
                "knock_analysis": knock_analysis,
                "time_analysis": {
                    "total_duration_hours": round(total_duration_seconds / 3600, 2),
                    "max_route_duration_hours": round(
                        max(
                            (getattr(r, "total_duration", 0) for r in solution.routes),
                            default=0,
                        )
                        / 3600,
                        2,
                    ),
                    "time_window_hours": self.config.time_window_hours,
                    "routes_violating_time_window": len(time_window_violations),
                    "feasible": len(time_window_violations) == 0,
                },
                "cost_analysis": cost_metrics,
            },
            "polygon_id": 0,
            "processing_time": 0.0,  # Will be set by caller
        }

    def _get_parcel_pickup_id(self, parcel: Parcel, solution: Solution) -> int:
        """Get pickup terminal ID for parcel"""
        for terminal in solution.pickup_terminals:
            if parcel in terminal.parcels:
                return terminal.pickup_id
        raise ValueError(f"Parcel {parcel.id} not found in any terminal")


class UpdatedConstructionHeuristic:
    """Construction heuristics updated for new cost model"""

    def __init__(self, config: ProblemConfig):
        self.config = config

    def greedy_construction(
        self,
        pickup_terminals: List[PickupTerminal],
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Solution:
        """Greedy construction heuristic optimized for cost per parcel"""
        solution = Solution(pickup_terminals, self.config.vehicle_specs)
        vehicle_counter = 1

        # Sort terminals by number of parcels (descending)
        sorted_terminals = sorted(
            pickup_terminals, key=lambda x: len(x.parcels), reverse=True
        )

        for terminal in sorted_terminals:
            remaining_parcels = terminal.parcels.copy()

            while remaining_parcels:
                # Select best vehicle type for cost per parcel optimization
                best_route = self._create_best_route(
                    remaining_parcels,
                    terminal,
                    vehicle_counter,
                    distance_matrix,
                    eta_matrix,
                )

                if best_route:
                    solution.add_route(best_route)
                    vehicle_counter += 1

                    # Remove assigned parcels
                    for parcel in best_route.parcels:
                        if parcel in remaining_parcels:
                            remaining_parcels.remove(parcel)
                else:
                    # Can't create feasible route, mark as unassigned
                    solution.unassigned_parcels.extend(remaining_parcels)
                    break

        # Calculate solution metrics
        solution.calculate_cost_and_time(distance_matrix, eta_matrix)
        solution.is_feasible = (
            len(solution.unassigned_parcels) == 0
            and solution.validate_knock_constraints(self.config.max_knock)
            and solution.validate_time_window(self.config.time_window_seconds)
        )

        return solution

    def _create_best_route(
        self,
        parcels: List[Parcel],
        terminal: PickupTerminal,
        vehicle_id: int,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[Route]:
        """Create best route for cost per parcel optimization"""
        best_route = None
        best_cost_per_parcel = float("inf")

        # Try each vehicle type
        for vehicle_type, vehicle_spec in self.config.vehicle_specs.items():
            # Pack parcels considering capacity and time constraints
            route_parcels = self._pack_parcels_for_cost_efficiency(
                parcels, vehicle_spec, terminal, distance_matrix, eta_matrix
            )

            if route_parcels:
                # Create route
                route = Route(
                    vehicle_id=vehicle_id,
                    vehicle_type=vehicle_type,
                    vehicle_spec=vehicle_spec,
                    parcels=route_parcels,
                    pickup_sequence=[terminal.pickup_id],
                    delivery_sequence=[p.delivery_location for p in route_parcels],
                )

                # Calculate route metrics
                route_distance, route_duration = self._calculate_route_metrics(
                    route, distance_matrix, eta_matrix
                )

                route.total_distance = route_distance
                route.total_duration = route_duration
                route.update_costs()

                # Check time feasibility
                if route.is_time_feasible(self.config.time_window_seconds):
                    if route.cost_per_parcel < best_cost_per_parcel:
                        best_cost_per_parcel = route.cost_per_parcel
                        best_route = route

        return best_route

    def _pack_parcels_for_cost_efficiency(
        self,
        parcels: List[Parcel],
        vehicle_spec: VehicleSpec,
        terminal: PickupTerminal,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> List[Parcel]:
        """Pack parcels to minimize cost per parcel"""
        # Sort parcels by delivery distance (closest first for time efficiency)
        terminal_location = (terminal.lat, terminal.lon)

        def delivery_distance(parcel):
            return distance_matrix.get(terminal_location, {}).get(
                parcel.delivery_location, float("inf")
            )

        sorted_parcels = sorted(parcels, key=delivery_distance)

        # Pack parcels greedily
        packed = []
        current_size = 0

        for parcel in sorted_parcels:
            if current_size + parcel.size <= vehicle_spec.capacity:
                # Check if adding this parcel improves cost per parcel
                temp_parcels = packed + [parcel]
                temp_cost_per_parcel = vehicle_spec.calculate_cost_per_parcel(
                    len(temp_parcels)
                )

                # Simple time check
                if self._estimated_time_feasible(
                    temp_parcels, terminal, distance_matrix, eta_matrix
                ):
                    packed.append(parcel)
                    current_size += parcel.size
                else:
                    # Stop packing if time window would be violated
                    break

        return packed

    def _estimated_time_feasible(
        self,
        parcels: List[Parcel],
        terminal: PickupTerminal,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> bool:
        """Quick estimation of time feasibility"""
        if not parcels:
            return True

        # Simple estimation: pickup + sum of delivery times
        total_time = 0
        current_location = (terminal.lat, terminal.lon)

        for parcel in parcels:
            delivery_location = parcel.delivery_location
            travel_time = eta_matrix.get(current_location, {}).get(
                delivery_location, 300
            )
            total_time += travel_time
            current_location = delivery_location

        return total_time <= self.config.time_window_seconds

    def _calculate_route_metrics(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Tuple[float, int]:
        """Calculate metrics for route"""
        total_distance = 0
        total_duration = 0

        if not route.pickup_sequence and not route.delivery_sequence:
            return 0, 0

        # Simple calculation: pickup -> deliveries
        route_locations = []

        if route.pickup_sequence:
            pickup_location = route.parcels[0].pickup_location
            route_locations.append(pickup_location)

        for delivery_location in route.delivery_sequence:
            route_locations.append(delivery_location)

        # Calculate cumulative metrics
        for i in range(len(route_locations) - 1):
            loc1 = route_locations[i]
            loc2 = route_locations[i + 1]

            distance = distance_matrix.get(loc1, {}).get(loc2, 0)
            travel_time = eta_matrix.get(loc1, {}).get(loc2, 0)

            total_distance += distance
            total_duration += travel_time

        return total_distance, total_duration


class UpdatedALNSOperators:
    """ALNS operators updated for new cost model and global optimization"""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.random = random.Random(42)

    def random_removal(self, solution: Solution, removal_rate: float = 0.3) -> Solution:
        """Randomly remove parcels from solution"""
        new_solution = deepcopy(solution)

        if not new_solution.routes:
            return new_solution

        # Calculate number of parcels to remove
        total_parcels = sum(len(route.parcels) for route in new_solution.routes)
        num_to_remove = max(1, int(total_parcels * removal_rate))

        # Collect all parcels with their route info
        all_parcels = []
        for route_idx, route in enumerate(new_solution.routes):
            for parcel in route.parcels:
                all_parcels.append((parcel, route_idx))

        # Randomly select parcels to remove
        to_remove = self.random.sample(
            all_parcels, min(num_to_remove, len(all_parcels))
        )

        # Remove parcels from routes
        for parcel, route_idx in to_remove:
            route = new_solution.routes[route_idx]
            route.remove_parcels([parcel], new_solution.pickup_terminals)
            new_solution.unassigned_parcels.append(parcel)

        # Remove empty routes
        new_solution.routes = [r for r in new_solution.routes if r.parcels]
        new_solution._rebuild_assignments()

        return new_solution

    def terminal_removal(self, solution: Solution, num_terminals: int = 1) -> Solution:
        """Remove all parcels from selected terminals"""
        new_solution = deepcopy(solution)

        if not new_solution.routes:
            return new_solution

        # Get all pickup terminals that have routes
        active_terminals = set()
        for route in new_solution.routes:
            active_terminals.update(route.pickup_sequence)

        if not active_terminals:
            return new_solution

        # Select terminals to remove
        terminals_to_remove = self.random.sample(
            list(active_terminals), min(num_terminals, len(active_terminals))
        )

        # Remove parcels from selected terminals
        for route in new_solution.routes:
            parcels_to_remove = []
            for parcel in route.parcels:
                # Find which terminal this parcel belongs to
                for terminal in new_solution.pickup_terminals:
                    if (
                        parcel in terminal.parcels
                        and terminal.pickup_id in terminals_to_remove
                    ):
                        parcels_to_remove.append(parcel)
                        break

            # Remove parcels and add to unassigned
            if parcels_to_remove:
                route.remove_parcels(parcels_to_remove)
                new_solution.unassigned_parcels.extend(parcels_to_remove)

        # Remove empty routes
        new_solution.routes = [r for r in new_solution.routes if r.parcels]
        new_solution._rebuild_assignments()

        return new_solution

    def greedy_insertion(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Greedily insert unassigned parcels with cost per parcel optimization"""
        new_solution = deepcopy(solution)

        # Sort unassigned parcels by size (largest first)
        unassigned = sorted(
            new_solution.unassigned_parcels, key=lambda x: x.size, reverse=True
        )

        for parcel in unassigned:
            best_insertion = self._find_best_insertion_for_cost(
                parcel, new_solution, distance_matrix, eta_matrix
            )

            if best_insertion:
                route_idx, _ = best_insertion
                route = new_solution.routes[route_idx]
                route.add_parcels([parcel], new_solution.pickup_terminals)
                new_solution.unassigned_parcels.remove(parcel)

                # Update pickup assignments
                parcel_pickup_id = self._get_parcel_pickup_id(parcel, new_solution)
                if parcel_pickup_id not in route.pickup_sequence:
                    route.pickup_sequence.append(parcel_pickup_id)

            else:
                # Create new route if no insertion found
                new_route = self._create_new_route_for_cost(
                    parcel, new_solution, distance_matrix, eta_matrix
                )
                if new_route:
                    new_solution.add_route(new_route)
                    new_solution.unassigned_parcels.remove(parcel)

        return new_solution

    def _find_best_insertion_for_cost(
        self,
        parcel: Parcel,
        solution: Solution,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[Tuple[int, float]]:
        """Find best insertion position for parcel based on cost per parcel"""
        best_route_idx = None
        best_cost_increase = float("inf")

        for route_idx, route in enumerate(solution.routes):
            # Check capacity constraint
            if route.total_size + parcel.size > route.vehicle_spec.capacity:
                continue

            # Check knock constraint
            parcel_pickup_id = self._get_parcel_pickup_id(parcel, solution)
            if parcel_pickup_id not in route.pickup_sequence:
                current_knocks = len(
                    solution.pickup_assignments.get(parcel_pickup_id, [])
                )
                if current_knocks >= self.config.max_knock:
                    continue

            # Calculate cost increase
            original_cost_per_parcel = route.cost_per_parcel
            new_parcel_count = len(route.parcels) + 1
            new_cost_per_parcel = route.vehicle_spec.calculate_cost_per_parcel(
                new_parcel_count
            )

            # Simple time feasibility check
            if self._quick_time_check(route, parcel, distance_matrix, eta_matrix):
                cost_increase = new_cost_per_parcel - original_cost_per_parcel

                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_route_idx = route_idx

        return (
            (best_route_idx, best_cost_increase) if best_route_idx is not None else None
        )

    def _quick_time_check(
        self, route: Route, parcel: Parcel, distance_matrix: Dict, eta_matrix: Dict
    ) -> bool:
        """Quick time feasibility check"""
        # Estimate additional time for adding this parcel
        if not route.parcels:
            return True

        # Simple estimation: add travel time to delivery location
        last_location = route.parcels[-1].delivery_location
        additional_time = eta_matrix.get(last_location, {}).get(
            parcel.delivery_location, 300
        )

        estimated_total_time = route.total_duration + additional_time
        return estimated_total_time <= self.config.time_window_seconds

    def _create_new_route_for_cost(
        self,
        parcel: Parcel,
        solution: Solution,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[Route]:
        """Create new route for parcel optimized for cost per parcel"""
        # Find parcel's pickup terminal
        pickup_terminal = None
        for terminal in solution.pickup_terminals:
            if parcel in terminal.parcels:
                pickup_terminal = terminal
                break

        if not pickup_terminal:
            return None

        # Select best vehicle type for single parcel
        best_vehicle_type = None
        best_cost_per_parcel = float("inf")

        for vehicle_type, spec in self.config.vehicle_specs.items():
            if spec.capacity >= parcel.size:
                cost_per_parcel = spec.calculate_cost_per_parcel(1)
                if cost_per_parcel < best_cost_per_parcel:
                    best_cost_per_parcel = cost_per_parcel
                    best_vehicle_type = vehicle_type

        if not best_vehicle_type:
            return None

        # Create route
        vehicle_id = max([r.vehicle_id for r in solution.routes], default=0) + 1
        route = Route(
            vehicle_id=vehicle_id,
            vehicle_type=best_vehicle_type,
            vehicle_spec=self.config.vehicle_specs[best_vehicle_type],
            parcels=[parcel],
            pickup_sequence=[pickup_terminal.pickup_id],
            delivery_sequence=[parcel.delivery_location],
        )

        # Calculate route metrics
        route_distance, route_duration = self._calculate_simple_route_metrics(
            route, distance_matrix, eta_matrix
        )

        route.total_distance = route_distance
        route.total_duration = route_duration
        route.update_costs()

        # Check time feasibility
        if route.is_time_feasible(self.config.time_window_seconds):
            return route

        return None

    def _calculate_simple_route_metrics(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Tuple[float, int]:
        """Calculate simple route metrics"""
        if not route.parcels:
            return 0, 0

        pickup_location = route.parcels[0].pickup_location
        delivery_location = route.parcels[0].delivery_location

        distance = distance_matrix.get(pickup_location, {}).get(delivery_location, 0)
        duration = eta_matrix.get(pickup_location, {}).get(delivery_location, 0)

        return distance, duration

    def _get_parcel_pickup_id(self, parcel: Parcel, solution: Solution) -> int:
        """Get pickup terminal ID for parcel"""
        for terminal in solution.pickup_terminals:
            if parcel in terminal.parcels:
                return terminal.pickup_id
        raise ValueError(f"Parcel {parcel.id} not found in any terminal")

    def route_splitting_operator(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Split over-utilized routes for better cost efficiency"""
        new_solution = deepcopy(solution)

        routes_to_split = []

        # Find routes that could benefit from splitting
        for route in new_solution.routes:
            utilization = route.get_utilization_percentage()

            # Split if over-utilized, time-constrained, or cost-inefficient
            if (
                utilization > 90
                or not route.is_time_feasible(self.config.time_window_seconds)
                or len(route.parcels) > 8
            ):  # Arbitrary threshold
                routes_to_split.append(route)

        # Try to split selected routes
        for route in routes_to_split:
            split_routes = self._try_split_route_for_cost(
                route, distance_matrix, eta_matrix
            )
            if split_routes and len(split_routes) > 1:
                # Check if splitting improves cost efficiency
                original_total_cost = route.total_cost
                split_total_cost = sum(r.total_cost for r in split_routes)

                if (
                    split_total_cost < original_total_cost * 1.1
                ):  # Allow 10% cost increase
                    # Replace original route with split routes
                    new_solution.remove_route(route)
                    for split_route in split_routes:
                        new_solution.add_route(split_route)

        return new_solution

    def _try_split_route_for_cost(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Optional[List[Route]]:
        """Try to split a route for better cost efficiency"""
        if len(route.parcels) < 2:
            return None

        # Simple 2-way split
        parcels = route.parcels
        split_point = len(parcels) // 2

        part1_parcels = parcels[:split_point]
        part2_parcels = parcels[split_point:]

        split_routes = []
        next_vehicle_id = route.vehicle_id

        for i, parcels_subset in enumerate([part1_parcels, part2_parcels]):
            if not parcels_subset:
                continue

            # Find best vehicle type for this subset
            best_route = self._create_best_route_for_parcels(
                parcels_subset,
                route.pickup_sequence,
                next_vehicle_id + i,
                distance_matrix,
                eta_matrix,
            )

            if best_route:
                split_routes.append(best_route)

        return split_routes if len(split_routes) >= 2 else None

    def _create_best_route_for_parcels(
        self,
        parcels: List[Parcel],
        pickup_sequence: List[int],
        vehicle_id: int,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[Route]:
        """Create best route for given parcels"""
        subset_size = sum(p.size for p in parcels)

        best_route = None
        best_cost_per_parcel = float("inf")

        for vehicle_type, vehicle_spec in self.config.vehicle_specs.items():
            if vehicle_spec.capacity >= subset_size:
                route = Route(
                    vehicle_id=vehicle_id,
                    vehicle_type=vehicle_type,
                    vehicle_spec=vehicle_spec,
                    parcels=parcels,
                    pickup_sequence=pickup_sequence.copy(),
                    delivery_sequence=[p.delivery_location for p in parcels],
                )

                # Calculate route metrics
                route_distance, route_duration = self._calculate_simple_route_metrics(
                    route, distance_matrix, eta_matrix
                )

                route.total_distance = route_distance
                route.total_duration = route_duration
                route.update_costs()

                # Check feasibility and cost
                if (
                    route.is_time_feasible(self.config.time_window_seconds)
                    and route.cost_per_parcel < best_cost_per_parcel
                ):
                    best_cost_per_parcel = route.cost_per_parcel
                    best_route = route

        return best_route

    def parcel_relocation_operator(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Relocate parcels between routes for better cost efficiency"""
        new_solution = deepcopy(solution)

        if len(new_solution.routes) < 2:
            return new_solution

        # Try relocating parcels between routes
        for _ in range(min(5, len(new_solution.routes))):  # Limit attempts
            # Select random source and target routes
            source_route = self.random.choice(new_solution.routes)
            target_route = self.random.choice(
                [r for r in new_solution.routes if r != source_route]
            )

            if not source_route.parcels:
                continue

            # Select random parcel to relocate
            parcel = self.random.choice(source_route.parcels)

            # Check if relocation is feasible
            if (
                target_route.total_size + parcel.size
                <= target_route.vehicle_spec.capacity
                and self._quick_time_check(
                    target_route, parcel, distance_matrix, eta_matrix
                )
            ):

                # Calculate cost impact
                original_cost = (
                    source_route.cost_per_parcel + target_route.cost_per_parcel
                )

                # Estimate new costs
                new_source_cost = (
                    source_route.vehicle_spec.calculate_cost_per_parcel(
                        len(source_route.parcels) - 1
                    )
                    if len(source_route.parcels) > 1
                    else 0
                )

                new_target_cost = target_route.vehicle_spec.calculate_cost_per_parcel(
                    len(target_route.parcels) + 1
                )

                new_total_cost = new_source_cost + new_target_cost

                # Apply relocation if beneficial
                if new_total_cost < original_cost:
                    source_route.remove_parcels([parcel], new_solution.pickup_terminals)
                    target_route.add_parcels([parcel], new_solution.pickup_terminals)

                    # Update pickup sequence if needed
                    parcel_pickup_id = self._get_parcel_pickup_id(parcel, new_solution)
                    if parcel_pickup_id not in target_route.pickup_sequence:
                        target_route.pickup_sequence.append(parcel_pickup_id)

                    break

        # Remove empty routes
        new_solution.routes = [r for r in new_solution.routes if r.parcels]
        new_solution._rebuild_assignments()

        return new_solution

    def local_parcel_swap(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Swap parcels between routes for local optimization"""
        new_solution = deepcopy(solution)

        if len(new_solution.routes) < 2:
            return new_solution

        # Try swapping parcels between random pairs of routes
        for _ in range(min(3, len(new_solution.routes))):  # Limit attempts
            routes = self.random.sample(new_solution.routes, 2)
            route1, route2 = routes[0], routes[1]

            if not route1.parcels or not route2.parcels:
                continue

            parcel1 = self.random.choice(route1.parcels)
            parcel2 = self.random.choice(route2.parcels)

            # Check if swap is feasible
            if (
                route1.total_size - parcel1.size + parcel2.size
                <= route1.vehicle_spec.capacity
                and route2.total_size - parcel2.size + parcel1.size
                <= route2.vehicle_spec.capacity
            ):

                # Calculate cost impact
                original_cost = route1.cost_per_parcel + route2.cost_per_parcel

                # Estimate new costs (same parcel counts)
                new_cost1 = route1.vehicle_spec.calculate_cost_per_parcel(
                    len(route1.parcels)
                )
                new_cost2 = route2.vehicle_spec.calculate_cost_per_parcel(
                    len(route2.parcels)
                )
                new_total_cost = new_cost1 + new_cost2

                # Apply swap if potentially beneficial (simple heuristic)
                if self.random.random() < 0.3:  # 30% chance to try swap
                    route1.remove_parcels([parcel1], new_solution.pickup_terminals)
                    route1.add_parcels([parcel2], new_solution.pickup_terminals)
                    route2.remove_parcels([parcel2], new_solution.pickup_terminals)
                    route2.add_parcels([parcel1], new_solution.pickup_terminals)
                    break

        return new_solution
