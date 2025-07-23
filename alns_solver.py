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
    ConstraintType,
)


class UpdatedALNSSolver:
    """
    Main ALNS solver with integrated global OR-Tools optimization and soft constraint support
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
        Optimized solve method with global OR-Tools integration and soft constraint support
        """
        start_time = time.time()

        # Adaptive parameters based on instance size
        instance_size = sum(len(terminal.parcels) for terminal in pickup_terminals)
        verbose_logging = True

        if verbose_logging:
            print(f"Starting optimization for {instance_size} parcels")
            print(f"Soft constraints enabled: {self.config.allow_time_violations}")
            print(
                f"Time violation penalty: {self.config.time_violation_penalty_per_minute:.2f} per minute"
            )

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
                    print(f"ALNS Operator result: {repaired is not None}")

                if repaired is None:
                    consecutive_no_improvement += 1
                    continue

                # Update feasibility status for the repaired solution
                repaired.update_feasibility_status(
                    self.config.max_knock, self.config.time_window_seconds
                )

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

                # Evaluate solution with soft constraint support
                globally_optimized.calculate_cost_and_time(
                    distance_matrix,
                    eta_matrix,
                    self.config.time_violation_penalty_per_minute,
                )
                globally_optimized.update_feasibility_status(
                    self.config.max_knock, self.config.time_window_seconds
                )

                # Enhanced acceptance criterion with soft constraint support
                if self._should_accept_solution(
                    globally_optimized, current_solution, best_solution
                ):
                    current_solution = globally_optimized
                    consecutive_no_improvement = 0

                    # Update best solution with soft constraint preference
                    if self._is_better_solution(globally_optimized, best_solution):
                        improvement = self._calculate_improvement_percentage(
                            best_solution, globally_optimized
                        )

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
        if self._is_solution_worth_final_optimization(
            best_solution, iteration, optimization_intervals
        ):
            if verbose_logging:
                print("Final global optimization...")
            try:
                final_optimized = self.global_optimizer.final_optimization(
                    best_solution, distance_matrix, eta_matrix
                )
                final_optimized.calculate_cost_and_time(
                    distance_matrix,
                    eta_matrix,
                    self.config.time_violation_penalty_per_minute,
                )
                final_optimized.update_feasibility_status(
                    self.config.max_knock, self.config.time_window_seconds
                )

                if self._is_better_solution(final_optimized, best_solution):
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
        """Apply ALNS operator based on current state with soft constraint awareness"""

        # If we have single route and it's well-utilized, focus on simple operations
        if (
            len(solution.routes) == 1
            and solution.routes[0].get_utilization_percentage() > 70
        ):
            return self._apply_simple_destroy_repair(
                solution, distance_matrix, eta_matrix
            )

        # If stagnating and we have time violations, try more aggressive operations
        if consecutive_no_improvement > 10:
            return self._apply_diversification(solution, distance_matrix, eta_matrix)

        # If we have time violations, prioritize operations that might fix them
        if not solution.is_soft_feasible and solution.routes_with_time_violations > 0:
            return self._apply_time_violation_repair(
                solution, distance_matrix, eta_matrix
            )

        # Normal operation selection
        operation_choice = self.random.random()

        if operation_choice < 0.6:  # Primary: destroy-repair
            return self._apply_destroy_repair(solution, distance_matrix, eta_matrix)
        elif operation_choice < 0.8:  # Secondary: route modification
            return self._apply_route_modification(solution, distance_matrix, eta_matrix)
        else:  # Tertiary: simple local search
            return self._apply_local_search(solution, distance_matrix, eta_matrix)

    def _apply_time_violation_repair(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Optional[Solution]:
        """Apply operations specifically designed to reduce time violations"""
        try:
            # Strategy 1: Split routes with severe time violations
            if self.random.random() < 0.5:
                return self.operators.split_time_violating_routes(
                    solution, distance_matrix, eta_matrix
                )
            # Strategy 2: Vehicle upsizing for over-capacity routes
            else:
                return self.operators.upsize_vehicles_for_time_feasibility(
                    solution, distance_matrix, eta_matrix
                )
        except Exception as e:
            return None

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

    def _is_better_solution(
        self, new_solution: Solution, current_best: Solution
    ) -> bool:
        """
        Determine if new solution is better than current best with soft constraint consideration
        Priority: Hard feasible > Soft feasible > Cost efficient
        """
        # Case 1: Both solutions are fully feasible - compare cost
        if new_solution.is_fully_feasible and current_best.is_fully_feasible:
            return (
                new_solution.average_cost_per_parcel
                < current_best.average_cost_per_parcel
            )

        # Case 2: New solution is fully feasible, current is not - new is better
        if new_solution.is_fully_feasible and not current_best.is_fully_feasible:
            return True

        # Case 3: Current is fully feasible, new is not - current is better
        if current_best.is_fully_feasible and not new_solution.is_fully_feasible:
            return False

        # Case 4: Neither is fully feasible - check hard constraints first
        if new_solution.is_hard_feasible and not current_best.is_hard_feasible:
            return True
        if current_best.is_hard_feasible and not new_solution.is_hard_feasible:
            return False

        # Case 5: Both have same hard feasibility - prefer less time violations if enabled
        if self.config.prefer_time_feasible:
            new_violations = new_solution.total_time_violation_seconds
            current_violations = current_best.total_time_violation_seconds

            if new_violations < current_violations:
                return True
            elif new_violations > current_violations:
                return False

        # Case 6: Same violation level - compare total cost including penalties
        return (
            new_solution.average_cost_per_parcel < current_best.average_cost_per_parcel
        )

    def _should_accept_solution(
        self,
        new_solution: Solution,
        current_solution: Solution,
        best_solution: Solution,
    ) -> bool:
        """
        Enhanced acceptance criterion with soft constraint support
        """
        # Always accept if better than current (using enhanced comparison)
        if self._is_better_solution(new_solution, current_solution):
            return True

        # If time violations are not allowed, reject infeasible solutions
        if not self.config.allow_time_violations and not new_solution.is_soft_feasible:
            return False

        # Accept hard-feasible solutions occasionally for diversification
        if new_solution.is_hard_feasible:
            # Higher acceptance rate for solutions with fewer time violations
            base_acceptance_rate = 0.05  # 5%
            if new_solution.is_soft_feasible:
                acceptance_rate = base_acceptance_rate * 2  # 10% for time-feasible
            else:
                # Reduce acceptance rate based on severity of time violations
                violation_factor = min(
                    1.0, new_solution.total_time_violation_seconds / 3600
                )  # Cap at 1 hour
                acceptance_rate = base_acceptance_rate * (1 - violation_factor * 0.5)

            return self.random.random() < acceptance_rate

        # Rarely accept solutions that violate hard constraints
        if self.random.random() < 0.01:  # 1% chance
            return True

        return False

    def _calculate_improvement_percentage(
        self, old_solution: Solution, new_solution: Solution
    ) -> float:
        """Calculate improvement percentage considering penalties"""
        if old_solution.average_cost_per_parcel == 0:
            return 0.0

        return (
            (
                old_solution.average_cost_per_parcel
                - new_solution.average_cost_per_parcel
            )
            / old_solution.average_cost_per_parcel
        ) * 100

    def _is_solution_worth_final_optimization(
        self, solution: Solution, iteration: int, optimization_intervals: int
    ) -> bool:
        """Determine if solution is worth final optimization"""
        return (
            solution.is_hard_feasible
            and len(solution.routes) > 1
            and (
                iteration % optimization_intervals != 0
                or iteration % optimization_intervals != 1
            )
        )

    def _print_solution_stats(self, solution: Solution, label: str, verbose: bool):
        """Print solution statistics with soft constraint information"""
        if not verbose:
            return

        vehicle_counts = {}
        for route in solution.routes:
            vtype = route.vehicle_type.value
            vehicle_counts[vtype] = vehicle_counts.get(vtype, 0) + 1

        violation_info = ""
        if solution.routes_with_time_violations > 0:
            violation_info = f", time violations: {solution.routes_with_time_violations} routes ({solution.total_time_violation_seconds/60:.1f} min)"

        print(
            f"{label} solution: {len(solution.routes)} routes, "
            f"cost per parcel: {solution.average_cost_per_parcel:.2f}, "
            f"vehicles: {vehicle_counts}, "
            f"hard feasible: {solution.is_hard_feasible}, "
            f"soft feasible: {solution.is_soft_feasible}"
            f"{violation_info}"
        )

    def _print_final_stats(self, solution: Solution):
        """Print comprehensive final solution statistics with soft constraint details"""
        print("\n" + "=" * 60)
        print("FINAL SOLUTION SUMMARY")
        print("=" * 60)

        if solution.is_hard_feasible:
            # Get cost efficiency metrics
            cost_metrics = solution.get_cost_efficiency_metrics()
            soft_metrics = solution.get_soft_constraint_solution_quality()

            feasibility_status = (
                "✅ FULLY FEASIBLE"
                if solution.is_fully_feasible
                else "⚠️  HARD FEASIBLE (Time Violations)"
            )
            print(f"{feasibility_status}")

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

            # Time window and constraint analysis
            if solution.routes_with_time_violations > 0:
                print(f"⚠️  Time Window Violations:")
                print(
                    f"   - Routes with violations: {solution.routes_with_time_violations}/{len(solution.routes)}"
                )
                print(
                    f"   - Total violation time: {solution.total_time_violation_seconds/60:.1f} minutes"
                )
                print(
                    f"   - Total penalty cost: {solution.total_time_violation_penalty:.2f}"
                )
                print(
                    f"   - Penalty as % of total cost: {soft_metrics['time_violation_metrics']['penalty_as_percentage_of_total']:.1f}%"
                )
            else:
                print(f"✅ All routes meet time window constraints")

            # Knock constraint analysis
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
            print(f"❌ NO HARD-FEASIBLE SOLUTION FOUND")
            print(f"📊 Best Attempt: {len(solution.routes)} routes")
            print(f"📦 Unassigned Parcels: {len(solution.unassigned_parcels)}")
            print(f"💰 Cost per Parcel: {solution.average_cost_per_parcel:.2f}")

            if solution.routes_with_time_violations > 0:
                print(
                    f"⚠️  Time violations: {solution.routes_with_time_violations} routes"
                )

        print("=" * 60)

    def format_output(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Dict:
        """Format solution to match expected output format with soft constraint information"""
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

        # Format routes with enhanced information
        formatted_routes = []
        total_duration_seconds = 0

        for route in solution.routes:
            route_distance = getattr(route, "total_distance", 0)
            route_duration = getattr(route, "total_duration", 0)
            total_duration_seconds += route_duration

            # Build physical route from optimized sequence
            physical_route = []
            route_indices = []

            route.rebuild_route_sequence()

            pickup_locations_in_route = set()
            for action, location in route.route_sequence:
                if action == "pickup":
                    pickup_locations_in_route.add(location)

            missing_pickups = set()
            for parcel in route.parcels:
                pickup_loc = parcel.pickup_location
                if pickup_loc not in pickup_locations_in_route:
                    missing_pickups.add(pickup_loc)

            if missing_pickups:
                # Create new sequence with missing pickups at start
                new_sequence = []

                # Add missing pickups first
                for pickup_loc in missing_pickups:
                    new_sequence.append(("pickup", pickup_loc))

                # Add existing sequence (skip existing pickups to avoid duplicates)
                for action, location in route.route_sequence:
                    if action == "pickup" and location not in missing_pickups:
                        new_sequence.append((action, location))
                    elif action == "delivery":
                        new_sequence.append((action, location))

                route.route_sequence = new_sequence

            for i, (action, location) in enumerate(route.route_sequence):
                physical_route.append((location[0], location[1]))
                route_indices.append(i)

            # Calculate SLA metrics for this route
            sla_metrics = self._calculate_route_sla_metrics(
                route, distance_matrix, eta_matrix
            )

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
                "time_window_feasible": route.is_time_feasible,
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
                # Cost breakdown for this route
                "cost_breakdown": {
                    "operational_cost": route.calculate_real_cost(),  # Real operational cost
                    "penalty_cost": route.time_violation_penalty,  # Penalty cost
                    "total_cost_with_penalties": route.total_cost,  # Total including penalties
                    "operational_cost_per_parcel": route.calculate_real_cost_per_parcel(),
                    "penalty_cost_per_parcel": (
                        round(route.time_violation_penalty / len(route.parcels), 2)
                        if len(route.parcels) > 0
                        else 0.0
                    ),
                    "total_cost_per_parcel_with_penalties": route.cost_per_parcel,
                },
                # Time violation details
                "time_violation_seconds": route.time_violation_seconds,
                "time_violation_minutes": round(route.time_violation_seconds / 60, 2),
                "time_violation_penalty": route.time_violation_penalty,
                "constraint_violations": route.get_constraint_violation_summary(),
                # SLA metrics
                "sla_metrics": sla_metrics,
            }
            formatted_routes.append(formatted_route)

        # Count vehicles by type
        vehicles_by_type = {}
        for route in solution.routes:
            vehicle_type = route.vehicle_type.value
            vehicles_by_type[vehicle_type] = vehicles_by_type.get(vehicle_type, 0) + 1

        # Calculate real costs (without penalties) for output
        total_real_cost = sum(route.calculate_real_cost() for route in solution.routes)
        total_parcels_count = sum(len(route.parcels) for route in solution.routes)
        average_real_cost_per_parcel = (
            total_real_cost / total_parcels_count if total_parcels_count > 0 else 0.0
        )

        # Total knocks per pickup
        total_knocks_per_pickup = []
        for pickup_id, vehicle_ids in solution.pickup_assignments.items():
            total_knocks_per_pickup.append({pickup_id: len(set(vehicle_ids))})

        # Enhanced constraint validation with SLA analysis
        time_window_violations = [r for r in solution.routes if not r.is_time_feasible]

        # Calculate aggregate SLA metrics across all routes
        aggregate_sla_metrics = self._calculate_aggregate_sla_metrics(
            solution, formatted_routes
        )

        return {
            "input_summary": {
                "num_hubs": len(solution.pickup_terminals),
                "total_parcels": sum(len(t.parcels) for t in solution.pickup_terminals),
                "max_knock": self.config.max_knock,
                "time_window_seconds": self.config.time_window_seconds,
                "time_window_hours": self.config.time_window_hours,
                "time_violation_penalty_per_minute": self.config.time_violation_penalty_per_minute,
                "allow_time_violations": self.config.allow_time_violations,
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
                "total_cost": total_real_cost,  # Real cost without penalties
                "average_cost_per_parcel": average_real_cost_per_parcel,  # Real cost per parcel
                "total_duration_seconds": total_duration_seconds,
                "max_route_duration_seconds": max(
                    (getattr(r, "total_duration", 0) for r in solution.routes),
                    default=0,
                ),
                "time_window_feasible": solution.is_soft_feasible,
                "hard_constraints_feasible": solution.is_hard_feasible,
                "fully_feasible": solution.is_fully_feasible,
                "time_window_violations": len(time_window_violations),
                "routes_with_time_violations": solution.routes_with_time_violations,
                "total_time_violation_seconds": solution.total_time_violation_seconds,
                "total_time_violation_penalty": solution.total_time_violation_penalty,
                "global_optimization_applied": True,
                "cost_efficiency_metrics": cost_metrics,
                "soft_constraint_metrics": solution.get_soft_constraint_solution_quality(),
                "map_filename": "vehicle_routes.html",
                # New aggregate SLA metrics
                "sla_metrics": aggregate_sla_metrics,
                # Cost breakdown for clarity
                "cost_breakdown": {
                    "total_operational_cost": total_real_cost,  # Real operational cost
                    "total_penalty_cost": solution.total_time_violation_penalty,  # Penalty cost
                    "total_cost_with_penalties": solution.total_cost,  # Total including penalties
                    "penalty_percentage_of_operational": round(
                        (
                            (
                                solution.total_time_violation_penalty
                                / total_real_cost
                                * 100
                            )
                            if total_real_cost > 0
                            else 0.0
                        ),
                        2,
                    ),
                },
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
                    "total_violation_minutes": solution.total_time_violation_seconds
                    / 60,
                    "violation_penalty": solution.total_time_violation_penalty,
                    "feasible": solution.is_soft_feasible,
                },
                "cost_analysis": cost_metrics,
                "constraint_analysis": solution.constraint_violation_summary,
            },
            "polygon_id": 0,
            "processing_time": 0.0,  # Will be set by caller
        }

    def _calculate_route_sla_metrics(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Dict:
        """
        Calculate SLA metrics for a route showing parcel-level time window compliance

        Args:
            route: Route to analyze
            distance_matrix: Distance matrix for locations
            eta_matrix: ETA matrix for locations

        Returns:
            Dictionary with SLA metrics including parcel counts and percentages
        """
        if not route.parcels or not route.route_sequence:
            return {
                "sla_compliance_percent": 100.0,
                "parcels_on_time": 0,
                "parcels_late": 0,
                "total_parcels": 0,
                "late_parcel_details": [],
                "average_lateness_minutes": 0.0,
                "max_lateness_minutes": 0.0,
                "sla_breakdown": {
                    "on_time": {"count": 0, "percentage": 100.0},
                    "late_0_to_30_min": {"count": 0, "percentage": 0.0},
                    "late_30_to_60_min": {"count": 0, "percentage": 0.0},
                    "late_60_plus_min": {"count": 0, "percentage": 0.0},
                },
            }

        # Calculate cumulative arrival times for each stop in the route
        arrival_times = self._calculate_cumulative_arrival_times(
            route, distance_matrix, eta_matrix
        )

        # Analyze each parcel's delivery time vs time window
        total_parcels = len(route.parcels)
        parcels_on_time = 0
        parcels_late = 0
        late_parcel_details = []
        lateness_values = []

        # SLA breakdown categories
        sla_breakdown = {
            "on_time": {"count": 0, "percentage": 0.0},
            "late_0_to_30_min": {"count": 0, "percentage": 0.0},
            "late_30_to_60_min": {"count": 0, "percentage": 0.0},
            "late_60_plus_min": {"count": 0, "percentage": 0.0},
        }

        for parcel in route.parcels:
            # Find when this parcel gets delivered
            delivery_time = self._find_parcel_delivery_time(
                parcel, route, arrival_times
            )

            # Check if delivery is within time window
            if delivery_time <= self.config.time_window_seconds:
                parcels_on_time += 1
                sla_breakdown["on_time"]["count"] += 1
            else:
                parcels_late += 1
                lateness_seconds = delivery_time - self.config.time_window_seconds
                lateness_minutes = lateness_seconds / 60.0
                lateness_values.append(lateness_minutes)

                # Categorize the lateness
                if lateness_minutes <= 30:
                    sla_breakdown["late_0_to_30_min"]["count"] += 1
                elif lateness_minutes <= 60:
                    sla_breakdown["late_30_to_60_min"]["count"] += 1
                else:
                    sla_breakdown["late_60_plus_min"]["count"] += 1

                late_parcel_details.append(
                    {
                        "parcel_id": parcel.id,
                        "delivery_time_seconds": delivery_time,
                        "lateness_seconds": lateness_seconds,
                        "lateness_minutes": round(lateness_minutes, 2),
                        "delivery_location": parcel.delivery_location,
                    }
                )

        # Calculate percentages for SLA breakdown
        for category in sla_breakdown.values():
            category["percentage"] = round((category["count"] / total_parcels) * 100, 1)

        # Calculate SLA compliance percentage
        sla_compliance_percent = (
            round((parcels_on_time / total_parcels) * 100, 1)
            if total_parcels > 0
            else 100.0
        )

        # Calculate average and max lateness
        average_lateness_minutes = (
            round(sum(lateness_values) / len(lateness_values), 2)
            if lateness_values
            else 0.0
        )
        max_lateness_minutes = (
            round(max(lateness_values), 2) if lateness_values else 0.0
        )

        return {
            "sla_compliance_percent": sla_compliance_percent,
            "parcels_on_time": parcels_on_time,
            "parcels_late": parcels_late,
            "total_parcels": total_parcels,
            "late_parcel_details": late_parcel_details,
            "average_lateness_minutes": average_lateness_minutes,
            "max_lateness_minutes": max_lateness_minutes,
            "sla_breakdown": sla_breakdown,
        }

    def _calculate_cumulative_arrival_times(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Dict[Tuple[float, float], int]:
        """
        Calculate cumulative arrival times for each location in the route sequence

        Returns:
            Dictionary mapping location -> arrival_time_seconds
        """
        arrival_times = {}
        current_time = 0

        if not route.route_sequence:
            return arrival_times

        # Start at first location (usually pickup)
        first_location = route.route_sequence[0][1]
        arrival_times[first_location] = 0

        # Calculate cumulative times
        for i in range(len(route.route_sequence) - 1):
            current_location = route.route_sequence[i][1]
            next_location = route.route_sequence[i + 1][1]

            # Add travel time to next location
            travel_time = eta_matrix.get(current_location, {}).get(next_location, 300)
            current_time += travel_time
            arrival_times[next_location] = current_time

        return arrival_times

    def _find_parcel_delivery_time(
        self,
        parcel: Parcel,
        route: Route,
        arrival_times: Dict[Tuple[float, float], int],
    ) -> int:
        """
        Find the delivery time for a specific parcel

        Returns:
            Delivery time in seconds from route start
        """
        delivery_location = parcel.delivery_location

        # Find the delivery time from arrival_times
        if delivery_location in arrival_times:
            return arrival_times[delivery_location]

        # Fallback: use route total duration (conservative estimate)
        return route.total_duration

    def _calculate_aggregate_sla_metrics(
        self, solution: Solution, formatted_routes: List[Dict]
    ) -> Dict:
        """
        Calculate aggregate SLA metrics across all routes in the solution

        Args:
            solution: Complete solution
            formatted_routes: List of formatted route dictionaries with SLA metrics

        Returns:
            Aggregate SLA metrics for the entire solution
        """
        if not formatted_routes:
            return {
                "overall_sla_compliance_percent": 100.0,
                "total_parcels_on_time": 0,
                "total_parcels_late": 0,
                "total_parcels": 0,
                "routes_meeting_sla": 0,
                "routes_violating_sla": 0,
                "average_route_sla_percent": 100.0,
                "worst_route_sla_percent": 100.0,
                "best_route_sla_percent": 100.0,
                "aggregate_sla_breakdown": {
                    "on_time": {"count": 0, "percentage": 100.0},
                    "late_0_to_30_min": {"count": 0, "percentage": 0.0},
                    "late_30_to_60_min": {"count": 0, "percentage": 0.0},
                    "late_60_plus_min": {"count": 0, "percentage": 0.0},
                },
                "sla_performance_by_vehicle_type": {},
            }

        # Aggregate counters
        total_parcels_on_time = 0
        total_parcels_late = 0
        total_parcels = 0
        route_sla_percentages = []
        routes_meeting_sla = 0  # Routes with 100% SLA compliance

        # Aggregate breakdown counters
        aggregate_breakdown = {
            "on_time": {"count": 0, "percentage": 0.0},
            "late_0_to_30_min": {"count": 0, "percentage": 0.0},
            "late_30_to_60_min": {"count": 0, "percentage": 0.0},
            "late_60_plus_min": {"count": 0, "percentage": 0.0},
        }

        # Performance by vehicle type
        vehicle_type_performance = {}

        # Process each route
        for route_data in formatted_routes:
            sla_metrics = route_data.get("sla_metrics", {})
            vehicle_type = route_data.get("vehicle_type", "unknown")

            # Aggregate parcel counts
            total_parcels_on_time += sla_metrics.get("parcels_on_time", 0)
            total_parcels_late += sla_metrics.get("parcels_late", 0)
            total_parcels += sla_metrics.get("total_parcels", 0)

            # Track route-level SLA
            route_sla = sla_metrics.get("sla_compliance_percent", 100.0)
            route_sla_percentages.append(route_sla)

            if route_sla >= 100.0:
                routes_meeting_sla += 1

            # Aggregate breakdown categories
            sla_breakdown = sla_metrics.get("sla_breakdown", {})
            for category, data in sla_breakdown.items():
                if category in aggregate_breakdown:
                    aggregate_breakdown[category]["count"] += data.get("count", 0)

            # Track performance by vehicle type
            if vehicle_type not in vehicle_type_performance:
                vehicle_type_performance[vehicle_type] = {
                    "routes": 0,
                    "total_parcels": 0,
                    "parcels_on_time": 0,
                    "parcels_late": 0,
                    "sla_compliance_percent": 0.0,
                    "average_route_sla": 0.0,
                    "route_sla_values": [],
                }

            vtype_data = vehicle_type_performance[vehicle_type]
            vtype_data["routes"] += 1
            vtype_data["total_parcels"] += sla_metrics.get("total_parcels", 0)
            vtype_data["parcels_on_time"] += sla_metrics.get("parcels_on_time", 0)
            vtype_data["parcels_late"] += sla_metrics.get("parcels_late", 0)
            vtype_data["route_sla_values"].append(route_sla)

        # Calculate aggregate percentages
        overall_sla_compliance = (
            round((total_parcels_on_time / total_parcels) * 100, 1)
            if total_parcels > 0
            else 100.0
        )

        for category in aggregate_breakdown.values():
            category["percentage"] = (
                round((category["count"] / total_parcels) * 100, 1)
                if total_parcels > 0
                else 0.0
            )

        # Calculate route-level statistics
        average_route_sla = (
            round(sum(route_sla_percentages) / len(route_sla_percentages), 1)
            if route_sla_percentages
            else 100.0
        )
        worst_route_sla = (
            round(min(route_sla_percentages), 1) if route_sla_percentages else 100.0
        )
        best_route_sla = (
            round(max(route_sla_percentages), 1) if route_sla_percentages else 100.0
        )
        routes_violating_sla = len(formatted_routes) - routes_meeting_sla

        # Calculate vehicle type performance
        for vtype, data in vehicle_type_performance.items():
            if data["total_parcels"] > 0:
                data["sla_compliance_percent"] = round(
                    (data["parcels_on_time"] / data["total_parcels"]) * 100, 1
                )
            if data["route_sla_values"]:
                data["average_route_sla"] = round(
                    sum(data["route_sla_values"]) / len(data["route_sla_values"]), 1
                )

            # Remove the temporary list
            del data["route_sla_values"]

        return {
            "overall_sla_compliance_percent": overall_sla_compliance,
            "total_parcels_on_time": total_parcels_on_time,
            "total_parcels_late": total_parcels_late,
            "total_parcels": total_parcels,
            "routes_meeting_sla": routes_meeting_sla,
            "routes_violating_sla": routes_violating_sla,
            "routes_with_perfect_sla_percent": (
                round((routes_meeting_sla / len(formatted_routes)) * 100, 1)
                if formatted_routes
                else 100.0
            ),
            "average_route_sla_percent": average_route_sla,
            "worst_route_sla_percent": worst_route_sla,
            "best_route_sla_percent": best_route_sla,
            "aggregate_sla_breakdown": aggregate_breakdown,
            "sla_performance_by_vehicle_type": vehicle_type_performance,
        }

    def _get_parcel_pickup_id(self, parcel: Parcel, solution: Solution) -> int:
        """Get pickup terminal ID for parcel"""
        for terminal in solution.pickup_terminals:
            if parcel in terminal.parcels:
                return terminal.pickup_id
        raise ValueError(f"Parcel {parcel.id} not found in any terminal")


class UpdatedConstructionHeuristic:
    """Construction heuristics updated for new cost model and soft constraint support"""

    def __init__(self, config: ProblemConfig):
        self.config = config

    def greedy_construction(
        self,
        pickup_terminals: List[PickupTerminal],
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Solution:
        """Enhanced greedy construction heuristic with STRICT knock constraint enforcement"""
        solution = Solution(pickup_terminals, self.config.vehicle_specs)
        solution._max_knock = self.config.max_knock  # Store for validation
        vehicle_counter = 1

        # Sort terminals by number of parcels (descending)
        sorted_terminals = sorted(
            pickup_terminals, key=lambda x: len(x.parcels), reverse=True
        )

        # Process each terminal with knock constraint awareness
        for terminal in sorted_terminals:
            remaining_parcels = terminal.parcels.copy()

            while remaining_parcels:
                # Check knock constraint before creating any route
                current_vehicles_for_pickup = len(solution.pickup_assignments.get(terminal.pickup_id, []))
                
                if current_vehicles_for_pickup >= self.config.max_knock:
                    # Cannot create more routes for this terminal - move parcels to unassigned
                    solution.unassigned_parcels.extend(remaining_parcels)
                    print(f"Warning: Terminal {terminal.pickup_id} already has {current_vehicles_for_pickup} vehicles (max_knock={self.config.max_knock})")
                    break

                # Try to create route (time-feasible first, then with violations)
                best_route = None
                
                # Phase 1: Try time-feasible route
                best_route = self._create_best_route(
                    remaining_parcels,
                    terminal,
                    vehicle_counter,
                    distance_matrix,
                    eta_matrix,
                    allow_time_violations=False
                )

                # Phase 2: If no time-feasible route and violations allowed, try with violations
                if not best_route and self.config.allow_time_violations:
                    best_route = self._create_best_route(
                        remaining_parcels,
                        terminal,
                        vehicle_counter,
                        distance_matrix,
                        eta_matrix,
                        allow_time_violations=True
                    )

                if best_route:
                    try:
                        solution.add_route(best_route, self.config.time_violation_penalty_per_minute)
                        vehicle_counter += 1

                        # Remove assigned parcels
                        for parcel in best_route.parcels:
                            if parcel in remaining_parcels:
                                remaining_parcels.remove(parcel)
                                
                    except ValueError as e:
                        # Route would violate knock constraints
                        print(f"Warning: Cannot add route due to knock constraint: {e}")
                        solution.unassigned_parcels.extend(remaining_parcels)
                        break
                else:
                    # Can't create feasible route
                    solution.unassigned_parcels.extend(remaining_parcels)
                    break

        # Calculate solution metrics with penalty support
        solution.calculate_cost_and_time(
            distance_matrix, eta_matrix, self.config.time_violation_penalty_per_minute
        )
        solution.update_feasibility_status(self.config.max_knock, self.config.time_window_seconds)

        return solution

    def _create_best_route(
        self,
        parcels: List[Parcel],
        terminal: PickupTerminal,
        vehicle_id: int,
        distance_matrix: Dict,
        eta_matrix: Dict,
        allow_time_violations: bool = False,
    ) -> Optional[Route]:
        """
        Create best route for cost per parcel optimization with time constraint handling
        """
        best_route = None
        best_cost_per_parcel = float("inf")

        # Try each vehicle type
        for vehicle_type, vehicle_spec in self.config.vehicle_specs.items():
            # Pack parcels considering capacity and time constraints
            route_parcels = self._pack_parcels_for_cost_efficiency(
                parcels,
                vehicle_spec,
                terminal,
                distance_matrix,
                eta_matrix,
                allow_time_violations,
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
                route.update_time_feasibility(self.config.time_window_seconds)
                route.update_costs(self.config.time_violation_penalty_per_minute)

                # Check constraints based on mode
                is_acceptable = True
                if not allow_time_violations:
                    # Strict mode: require time feasibility
                    is_acceptable = route.is_time_feasible
                else:
                    # Soft mode: always acceptable (penalties applied in cost)
                    is_acceptable = True

                if is_acceptable and route.cost_per_parcel < best_cost_per_parcel:
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
        allow_time_violations: bool = False,
    ) -> List[Parcel]:
        """Pack parcels to minimize cost per parcel with time constraint awareness"""
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
                # Check if adding this parcel violates time constraints
                temp_parcels = packed + [parcel]

                if allow_time_violations:
                    # In soft mode, always add if capacity allows
                    packed.append(parcel)
                    current_size += parcel.size
                else:
                    # In strict mode, check time feasibility
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
    """ALNS operators updated for new cost model, global optimization, and soft constraint support"""

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
            route.remove_parcels(
                [parcel],
                new_solution.pickup_terminals,
                self.config.time_violation_penalty_per_minute,
            )
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
                route.remove_parcels(
                    parcels_to_remove,
                    pickup_terminals=new_solution.pickup_terminals,
                    time_violation_penalty_per_minute=self.config.time_violation_penalty_per_minute,
                )
                new_solution.unassigned_parcels.extend(parcels_to_remove)

        # Remove empty routes
        new_solution.routes = [r for r in new_solution.routes if r.parcels]
        new_solution._rebuild_assignments()

        return new_solution

    def greedy_insertion(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """
        Enhanced greedy insertion with soft constraint support
        Prefers time-feasible insertions but allows violations if necessary
        """
        new_solution = deepcopy(solution)

        # Sort unassigned parcels by size (largest first)
        unassigned = sorted(
            new_solution.unassigned_parcels, key=lambda x: x.size, reverse=True
        )

        for parcel in unassigned:
            insertion_found = False

            # Phase 1: Try time-feasible insertion
            if self.config.prefer_time_feasible:
                best_insertion = self._find_best_insertion_for_cost(
                    parcel,
                    new_solution,
                    distance_matrix,
                    eta_matrix,
                    allow_time_violations=False,
                )

                if best_insertion:
                    route_idx, _ = best_insertion
                    route = new_solution.routes[route_idx]
                    route.add_parcels(
                        [parcel],
                        new_solution.pickup_terminals,
                        self.config.time_violation_penalty_per_minute,
                    )
                    new_solution.unassigned_parcels.remove(parcel)
                    insertion_found = True

                    # Update pickup assignments
                    parcel_pickup_id = self._get_parcel_pickup_id(parcel, new_solution)
                    if parcel_pickup_id not in route.pickup_sequence:
                        route.pickup_sequence.append(parcel_pickup_id)

            # Phase 2: Try insertion with time violations if allowed and phase 1 failed
            if not insertion_found and self.config.allow_time_violations:
                best_insertion = self._find_best_insertion_for_cost(
                    parcel,
                    new_solution,
                    distance_matrix,
                    eta_matrix,
                    allow_time_violations=True,
                )

                if best_insertion:
                    route_idx, _ = best_insertion
                    route = new_solution.routes[route_idx]
                    route.add_parcels(
                        [parcel],
                        new_solution.pickup_terminals,
                        self.config.time_violation_penalty_per_minute,
                    )
                    new_solution.unassigned_parcels.remove(parcel)
                    insertion_found = True

                    # Update pickup assignments
                    parcel_pickup_id = self._get_parcel_pickup_id(parcel, new_solution)
                    if parcel_pickup_id not in route.pickup_sequence:
                        route.pickup_sequence.append(parcel_pickup_id)

            # Phase 3: Create new route if no insertion found
            if not insertion_found:
                new_route = self._create_new_route_for_cost(
                    parcel, new_solution, distance_matrix, eta_matrix
                )
                if new_route:
                    new_solution.add_route(
                        new_route, self.config.time_violation_penalty_per_minute
                    )
                    new_solution.unassigned_parcels.remove(parcel)

        return new_solution

    def _find_best_insertion_for_cost(
        self,
        parcel: Parcel,
        solution: Solution,
        distance_matrix: Dict,
        eta_matrix: Dict,
        allow_time_violations: bool = False
    ) -> Optional[Tuple[int, float]]:
        """Find best insertion position for parcel with PROPER knock constraint checking"""
        best_route_idx = None
        best_cost_increase = float("inf")

        # Get pickup terminal for this parcel
        parcel_pickup_id = self._get_parcel_pickup_id(parcel, solution)

        for route_idx, route in enumerate(solution.routes):
            # Check capacity constraint
            if route.total_size + parcel.size > route.vehicle_spec.capacity:
                continue

            # FIXED: Proper knock constraint checking
            if parcel_pickup_id not in route.pickup_sequence:
                # This would be a NEW vehicle assignment to this pickup terminal
                current_vehicles_for_pickup = set(solution.pickup_assignments.get(parcel_pickup_id, []))
                
                # Check if adding this vehicle would violate knock constraint
                if len(current_vehicles_for_pickup) >= self.config.max_knock:
                    continue  # ✅ Reject: would exceed max_knock
            
            # Rest of the insertion logic...
            # Calculate cost impact with potential time violations
            original_cost_per_parcel = route.cost_per_parcel
            new_parcel_count = len(route.parcels) + 1

            # Estimate time impact of adding this parcel
            estimated_additional_time = self._estimate_additional_time(
                route, parcel, distance_matrix, eta_matrix
            )
            new_total_duration = route.total_duration + estimated_additional_time
            
            # Calculate new time violation
            new_time_violation = max(0, new_total_duration - self.config.time_window_seconds)
            
            # Check if insertion is acceptable based on time constraints
            if not allow_time_violations and new_time_violation > 0:
                continue

            # Calculate cost with penalties
            new_cost_per_parcel = route.vehicle_spec.calculate_cost_per_parcel(
                new_parcel_count, new_time_violation, self.config.time_violation_penalty_per_minute
            )

            cost_increase = new_cost_per_parcel - original_cost_per_parcel

            if cost_increase < best_cost_increase:
                best_cost_increase = cost_increase
                best_route_idx = route_idx

        return (best_route_idx, best_cost_increase) if best_route_idx is not None else None

    def _estimate_additional_time(
        self, route: Route, parcel: Parcel, distance_matrix: Dict, eta_matrix: Dict
    ) -> int:
        """Estimate additional time for adding this parcel to route"""
        if not route.parcels:
            # Empty route: pickup + delivery
            pickup_to_delivery = eta_matrix.get(parcel.pickup_location, {}).get(
                parcel.delivery_location, 300
            )
            return pickup_to_delivery

        # Non-empty route: additional delivery time
        last_location = route.parcels[-1].delivery_location
        additional_time = eta_matrix.get(last_location, {}).get(
            parcel.delivery_location, 300
        )
        return additional_time

    def _create_new_route_for_cost(
        self,
        parcel: Parcel,
        solution: Solution,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[Route]:
        """Create new route with PROPER knock constraint checking"""
        # Find parcel's pickup terminal
        pickup_terminal = None
        for terminal in solution.pickup_terminals:
            if parcel in terminal.parcels:
                pickup_terminal = terminal
                break

        if not pickup_terminal:
            return None

        # FIXED: Check knock constraint BEFORE creating new route
        parcel_pickup_id = pickup_terminal.pickup_id
        current_vehicles_for_pickup = set(solution.pickup_assignments.get(parcel_pickup_id, []))
        
        if len(current_vehicles_for_pickup) >= self.config.max_knock:
            return None  # ✅ Reject: would exceed max_knock

        # Rest of route creation logic...
        # Select best vehicle type for single parcel (considering penalties)
        best_vehicle_type = None
        best_cost_per_parcel = float("inf")

        for vehicle_type, spec in self.config.vehicle_specs.items():
            if spec.capacity >= parcel.size:
                # Estimate route time
                estimated_time = self._estimate_single_parcel_time(
                    parcel, distance_matrix, eta_matrix
                )
                time_violation = max(0, estimated_time - self.config.time_window_seconds)

                cost_per_parcel = spec.calculate_cost_per_parcel(
                    1, time_violation, self.config.time_violation_penalty_per_minute
                )
                
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
        route.update_time_feasibility(self.config.time_window_seconds)
        route.update_costs(self.config.time_violation_penalty_per_minute)

        return route

    def _estimate_single_parcel_time(
        self, parcel: Parcel, distance_matrix: Dict, eta_matrix: Dict
    ) -> int:
        """Estimate time for single parcel route"""
        return eta_matrix.get(parcel.pickup_location, {}).get(
            parcel.delivery_location, 300
        )

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

    def split_time_violating_routes(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Split routes that have severe time violations"""
        new_solution = deepcopy(solution)

        routes_to_split = [
            route
            for route in new_solution.routes
            if route.time_violation_seconds > 1800  # 30 minutes violation
        ]

        for route in routes_to_split:
            if len(route.parcels) > 2:  # Only split if it makes sense
                split_routes = self._split_route_for_time_feasibility(
                    route, distance_matrix, eta_matrix
                )

                if split_routes and len(split_routes) > 1:
                    # Check if splitting improves the situation
                    original_total_penalty = route.time_violation_penalty
                    split_total_penalty = sum(
                        r.time_violation_penalty for r in split_routes
                    )

                    if (
                        split_total_penalty < original_total_penalty * 0.8
                    ):  # 20% improvement
                        new_solution.remove_route(route)
                        for split_route in split_routes:
                            new_solution.add_route(
                                split_route,
                                self.config.time_violation_penalty_per_minute,
                            )

        return new_solution

    def _split_route_for_time_feasibility(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Optional[List[Route]]:
        """Split a route to reduce time violations"""
        if len(route.parcels) < 2:
            return None

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

    def upsize_vehicles_for_time_feasibility(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Try upsizing vehicles to larger types to potentially reduce time violations"""
        new_solution = deepcopy(solution)

        for route in new_solution.routes:
            if route.time_violation_seconds > 0:
                # Try larger vehicle types
                current_capacity = route.vehicle_spec.capacity
                better_vehicle_type = None
                best_cost_per_parcel = route.cost_per_parcel

                for vehicle_type, vehicle_spec in self.config.vehicle_specs.items():
                    if (
                        vehicle_spec.capacity > current_capacity
                        and vehicle_spec.capacity >= route.total_size
                    ):

                        # Calculate cost with larger vehicle
                        test_cost_per_parcel = vehicle_spec.calculate_cost_per_parcel(
                            len(route.parcels),
                            route.time_violation_seconds,
                            self.config.time_violation_penalty_per_minute,
                        )

                        if test_cost_per_parcel < best_cost_per_parcel:
                            best_cost_per_parcel = test_cost_per_parcel
                            better_vehicle_type = vehicle_type

                # Apply upsizing if beneficial
                if better_vehicle_type:
                    route.vehicle_type = better_vehicle_type
                    route.vehicle_spec = self.config.vehicle_specs[better_vehicle_type]
                    route.update_costs(self.config.time_violation_penalty_per_minute)

        return new_solution

    def route_splitting_operator(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Split over-utilized routes for better cost efficiency with time awareness"""
        new_solution = deepcopy(solution)

        routes_to_split = []

        # Find routes that could benefit from splitting
        for route in new_solution.routes:
            utilization = route.get_utilization_percentage()

            # Split if over-utilized, time-constrained, or cost-inefficient
            should_split = (
                utilization > 90
                or route.time_violation_seconds > 0
                or len(route.parcels) > 8
            )

            if should_split:
                routes_to_split.append(route)

        # Try to split selected routes
        for route in routes_to_split:
            split_routes = self._try_split_route_for_cost(
                route, distance_matrix, eta_matrix
            )
            if split_routes and len(split_routes) > 1:
                # Check if splitting improves cost efficiency or reduces violations
                original_total_cost = route.total_cost
                split_total_cost = sum(r.total_cost for r in split_routes)

                original_violations = route.time_violation_seconds
                split_violations = sum(r.time_violation_seconds for r in split_routes)

                # Accept if cost improvement or significant violation reduction
                if (
                    split_total_cost < original_total_cost * 1.1
                    or split_violations < original_violations * 0.7
                ):
                    # Replace original route with split routes
                    new_solution.remove_route(route)
                    for split_route in split_routes:
                        new_solution.add_route(
                            split_route, self.config.time_violation_penalty_per_minute
                        )

        return new_solution

    def _try_split_route_for_cost(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Optional[List[Route]]:
        """Try to split a route for better cost efficiency and time feasibility"""
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
        """Create best route for given parcels with soft constraint support"""
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
                route.update_time_feasibility(self.config.time_window_seconds)
                route.update_costs(self.config.time_violation_penalty_per_minute)

                # Consider route if it's acceptable
                is_acceptable = True
                if not self.config.allow_time_violations and not route.is_time_feasible:
                    is_acceptable = False

                if is_acceptable and route.cost_per_parcel < best_cost_per_parcel:
                    best_cost_per_parcel = route.cost_per_parcel
                    best_route = route

        return best_route

    def parcel_relocation_operator(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Relocate parcels between routes for better cost efficiency with time awareness"""
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
            ):
                # Estimate time impact
                source_time_before = source_route.total_duration
                target_time_before = target_route.total_duration

                # Estimate new times (simplified)
                estimated_source_time_after = max(
                    0, source_time_before - 600
                )  # Rough estimate
                estimated_target_time_after = (
                    target_time_before
                    + self._estimate_additional_time(
                        target_route, parcel, distance_matrix, eta_matrix
                    )
                )

                # Calculate cost impact with penalties
                source_violation_before = max(
                    0, source_time_before - self.config.time_window_seconds
                )
                target_violation_before = max(
                    0, target_time_before - self.config.time_window_seconds
                )
                source_violation_after = max(
                    0, estimated_source_time_after - self.config.time_window_seconds
                )
                target_violation_after = max(
                    0, estimated_target_time_after - self.config.time_window_seconds
                )

                original_cost = (
                    source_route.cost_per_parcel
                    + target_route.cost_per_parcel
                    + self.config.calculate_time_violation_penalty(
                        source_violation_before
                    )
                    + self.config.calculate_time_violation_penalty(
                        target_violation_before
                    )
                )

                # Estimate new costs
                new_source_cost = (
                    source_route.vehicle_spec.calculate_cost_per_parcel(
                        len(source_route.parcels) - 1,
                        source_violation_after,
                        self.config.time_violation_penalty_per_minute,
                    )
                    if len(source_route.parcels) > 1
                    else 0
                )

                new_target_cost = target_route.vehicle_spec.calculate_cost_per_parcel(
                    len(target_route.parcels) + 1,
                    target_violation_after,
                    self.config.time_violation_penalty_per_minute,
                )

                new_total_cost = new_source_cost + new_target_cost

                # Apply relocation if beneficial
                if new_total_cost < original_cost:
                    source_route.remove_parcels(
                        [parcel],
                        new_solution.pickup_terminals,
                        self.config.time_violation_penalty_per_minute,
                    )
                    target_route.add_parcels(
                        [parcel],
                        new_solution.pickup_terminals,
                        self.config.time_violation_penalty_per_minute,
                    )

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
        """Swap parcels between routes for local optimization with time awareness"""
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

            # Check if swap is feasible capacity-wise
            if (
                route1.total_size - parcel1.size + parcel2.size
                <= route1.vehicle_spec.capacity
                and route2.total_size - parcel2.size + parcel1.size
                <= route2.vehicle_spec.capacity
            ):

                # Estimate time impact (simplified)
                original_total_violations = max(
                    0, route1.total_duration - self.config.time_window_seconds
                ) + max(0, route2.total_duration - self.config.time_window_seconds)

                # Apply swap with some probability (simplified evaluation)
                if self.random.random() < 0.3:  # 30% chance to try swap
                    route1.remove_parcels(
                        [parcel1],
                        new_solution.pickup_terminals,
                        self.config.time_violation_penalty_per_minute,
                    )
                    route1.add_parcels(
                        [parcel2],
                        new_solution.pickup_terminals,
                        self.config.time_violation_penalty_per_minute,
                    )
                    route2.remove_parcels(
                        [parcel2],
                        new_solution.pickup_terminals,
                        self.config.time_violation_penalty_per_minute,
                    )
                    route2.add_parcels(
                        [parcel1],
                        new_solution.pickup_terminals,
                        self.config.time_violation_penalty_per_minute,
                    )
                    break

        return new_solution

