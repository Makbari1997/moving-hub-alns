import time
import constants

from copy import deepcopy
from typing import Dict, List, Tuple, Optional
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from vehicle_optimizer import UpdatedVehicleOptimizer
from data_structures import (
    Route,
    Solution,
    VehicleType,
    VehicleSpec,
    Parcel,
    ProblemConfig,
)


class GlobalRouteOptimizer:
    """
    Global OR-Tools optimizer that processes multiple routes simultaneously
    for better consolidation and sequence optimization with soft constraint support
    """

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.verbose = False
        self.optimization_timeout = 30  # seconds
        self.max_locations = 200  # Limit for performance

    def optimize_solution_globally(
        self,
        solution: Solution,
        distance_matrix: Dict,
        eta_matrix: Dict,
        enable_route_merging: bool = True,
        penalty_aware_optimization: bool = True,
        enable_ortools_time_constraints: bool = False,  # Disabled by default for speed
    ) -> Solution:
        """
        Globally optimize entire solution with route merging capabilities and penalty awareness

        Args:
            solution: Current solution to optimize
            distance_matrix: Distance matrix for locations
            eta_matrix: ETA matrix for locations
            enable_route_merging: Whether to attempt route merging
            penalty_aware_optimization: Whether to consider penalty costs in optimization
            enable_ortools_time_constraints: Whether to add time constraints to OR-Tools (slower but more precise)

        Returns:
            Optimized solution with potentially merged routes and reduced penalties
        """
        if not constants.ORTOOLS_AVAILABLE:
            if self.verbose:
                print("OR-Tools not available, using fallback optimization")
            return self._fallback_global_optimization(
                solution, distance_matrix, eta_matrix
            )

        if not solution.routes:
            return solution

        start_time = time.time()

        try:
            # Step 1: Optimize sequences within existing routes (penalty-aware)
            sequence_optimized = self._optimize_existing_routes(
                solution,
                distance_matrix,
                eta_matrix,
                penalty_aware_optimization,
                enable_ortools_time_constraints,
            )

            # Step 2: Attempt route merging if enabled
            if enable_route_merging and len(sequence_optimized.routes) > 1:
                merged_solution = self._attempt_route_merging(
                    sequence_optimized,
                    distance_matrix,
                    eta_matrix,
                    penalty_aware_optimization,
                )

                # Choose best solution based on cost per parcel (including penalties)
                if self._is_solution_better_with_penalties(
                    merged_solution, sequence_optimized
                ):
                    final_solution = merged_solution
                else:
                    final_solution = sequence_optimized
            else:
                final_solution = sequence_optimized

            # Step 3: Penalty-focused optimization for routes with violations
            if (
                penalty_aware_optimization
                and final_solution.routes_with_time_violations > 0
            ):
                penalty_optimized = self._optimize_for_penalty_reduction(
                    final_solution, distance_matrix, eta_matrix
                )

                if self._is_solution_better_with_penalties(
                    penalty_optimized, final_solution
                ):
                    final_solution = penalty_optimized

            # Update solution costs and feasibility
            final_solution.calculate_cost_and_time(
                distance_matrix,
                eta_matrix,
                self.config.time_violation_penalty_per_minute,
            )
            final_solution.update_feasibility_status(
                self.config.max_knock, self.config.time_window_seconds
            )

            if self.verbose:
                elapsed = time.time() - start_time
                print(f"Global optimization completed in {elapsed:.2f}s")
                print(f"Routes: {len(solution.routes)} -> {len(final_solution.routes)}")
                print(
                    f"Cost per parcel: {solution.average_cost_per_parcel:.2f} -> {final_solution.average_cost_per_parcel:.2f}"
                )
                if final_solution.total_time_violation_penalty > 0:
                    print(
                        f"Total penalties: {final_solution.total_time_violation_penalty:.2f}"
                    )

            return final_solution

        except Exception as e:
            if self.verbose:
                print(f"Global optimization failed: {e}")
            return self._fallback_global_optimization(
                solution, distance_matrix, eta_matrix
            )

    def _optimize_existing_routes(
        self,
        solution: Solution,
        distance_matrix: Dict,
        eta_matrix: Dict,
        penalty_aware: bool = True,
        enable_time_constraints: bool = False,
    ) -> Solution:
        """Optimize sequences within existing routes with optional time constraints"""
        optimized_solution = deepcopy(solution)

        for route in optimized_solution.routes:
            if len(route.parcels) > 2:  # Only optimize if worthwhile
                optimized_route = self._optimize_single_route_sequence(
                    route,
                    distance_matrix,
                    eta_matrix,
                    penalty_aware,
                    enable_time_constraints,
                )

                if optimized_route:
                    # Update route in solution
                    route.route_sequence = optimized_route.route_sequence
                    route.total_distance = optimized_route.total_distance
                    route.total_duration = optimized_route.total_duration
                    route.is_optimized = optimized_route.is_optimized

                    # Update time feasibility and costs
                    route.update_time_feasibility(self.config.time_window_seconds)
                    route.update_costs(self.config.time_violation_penalty_per_minute)

        return optimized_solution

    def _optimize_single_route_sequence(
        self,
        route: Route,
        distance_matrix: Dict,
        eta_matrix: Dict,
        penalty_aware: bool = True,
        enable_time_constraints: bool = False,
    ) -> Optional[Route]:
        """
        Optimize sequence for a single route using OR-Tools with optional time constraints

        Args:
            route: Route to optimize
            distance_matrix: Distance matrix
            eta_matrix: ETA matrix
            penalty_aware: Whether to use penalty-aware optimization
            enable_time_constraints: Whether to add time window constraints to OR-Tools
        """
        if len(route.parcels) <= 2:
            return route

        try:
            # Build locations and constraints
            locations = self._build_route_locations(route)

            if len(locations) < 3:  # Need at least start + 2 stops
                return route

            # Create OR-Tools model
            manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
            routing = pywrapcp.RoutingModel(manager)

            # Add distance/time callback based on optimization mode
            if penalty_aware and route.time_violation_seconds > 0:
                # Use time-based optimization for routes with violations
                def time_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    if from_node >= len(locations) or to_node >= len(locations):
                        return 0
                    from_loc = locations[from_node]
                    to_loc = locations[to_node]
                    travel_time = eta_matrix.get(from_loc, {}).get(to_loc, 300)
                    return max(1, int(travel_time / 10))  # Scale for OR-Tools

                transit_callback_index = routing.RegisterTransitCallback(time_callback)
            else:
                # Use distance-based optimization for time-feasible routes
                def distance_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    if from_node >= len(locations) or to_node >= len(locations):
                        return 0
                    from_loc = locations[from_node]
                    to_loc = locations[to_node]
                    distance = distance_matrix.get(from_loc, {}).get(to_loc, 0)
                    return max(1, int(distance / 100))  # Scale for OR-Tools

                transit_callback_index = routing.RegisterTransitCallback(
                    distance_callback
                )

            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            # Add time window constraints only if explicitly enabled (disabled by default for speed)
            if enable_time_constraints and penalty_aware:
                if self.verbose:
                    print(
                        "Adding time window constraints to OR-Tools (may slow down optimization)"
                    )
                self._add_time_window_constraints(
                    routing, manager, locations, eta_matrix, route
                )

            # Set search parameters based on route complexity and whether time constraints are used
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()

            if (
                enable_time_constraints and route.time_violation_seconds > 600
            ):  # 10+ minutes violation
                # More intensive search for severely violating routes with time constraints
                search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
                )
                search_parameters.local_search_metaheuristic = (
                    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
                )
                search_parameters.time_limit.seconds = 15
                search_parameters.solution_limit = 200
            else:
                # Fast optimization without time constraints (default)
                search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.SAVINGS
                )
                search_parameters.local_search_metaheuristic = (
                    routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
                )
                search_parameters.time_limit.seconds = (
                    5  # Reduced time for faster optimization
                )
                search_parameters.solution_limit = 50  # Reduced solutions for speed

            # Solve
            solution = routing.SolveWithParameters(search_parameters)

            if solution:
                optimized_route = self._extract_optimized_route(
                    route,
                    solution,
                    routing,
                    manager,
                    locations,
                    distance_matrix,
                    eta_matrix,
                )
                return optimized_route
            else:
                return route

        except Exception as e:
            if self.verbose:
                print(f"Route optimization failed: {e}")
            return route

    def _add_time_window_constraints(
        self,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        locations: List[Tuple[float, float]],
        eta_matrix: Dict,
        route: Route,
    ):
        """
        Add time window constraints to OR-Tools model (optional for performance)
        Note: This can significantly slow down optimization but may provide better results
        """
        try:
            # Add time dimension
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                if from_node >= len(locations) or to_node >= len(locations):
                    return 0
                from_loc = locations[from_node]
                to_loc = locations[to_node]
                return eta_matrix.get(from_loc, {}).get(to_loc, 300)

            time_callback_index = routing.RegisterTransitCallback(time_callback)
            routing.AddDimension(
                time_callback_index,
                300,  # Allow waiting time
                self.config.time_window_seconds,  # Maximum time per vehicle
                False,  # Don't force start cumul to zero
                "Time",
            )

            time_dimension = routing.GetDimensionOrDie("Time")

            # Set time windows for each location (soft constraints via penalty)
            for i in range(len(locations)):
                if i == 0:  # Depot/start location
                    continue
                index = manager.NodeToIndex(i)
                # Set a reasonable time window based on route duration
                time_dimension.CumulVar(index).SetRange(
                    0, self.config.time_window_seconds
                )

        except Exception as e:
            if self.verbose:
                print(f"Failed to add time constraints: {e}")

    def _build_route_locations(self, route: Route) -> List[Tuple[float, float]]:
        """Build ordered list of locations for a route"""
        locations = []

        # Add pickup locations (start points)
        pickup_locations = set()
        for parcel in route.parcels:
            pickup_locations.add(parcel.pickup_location)

        # Start with first pickup location
        if route.parcels:
            locations.append(route.parcels[0].pickup_location)

        # Add all unique delivery locations
        for parcel in route.parcels:
            if parcel.delivery_location not in locations:
                locations.append(parcel.delivery_location)

        return locations

    def _extract_optimized_route(
        self,
        original_route: Route,
        solution,
        routing,
        manager,
        locations: List[Tuple[float, float]],
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Route:
        """Extract optimized route from OR-Tools solution with metrics calculation"""
        optimized_route = deepcopy(original_route)

        # Get route sequence from solution
        route_sequence = []
        index = routing.Start(0)
        visited_locations = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node < len(locations):
                location = locations[node]
                visited_locations.append(location)
            index = solution.Value(routing.NextVar(index))

        # Build route sequence with pickup/delivery identification
        pickup_locations = set(p.pickup_location for p in original_route.parcels)

        for location in visited_locations:
            if location in pickup_locations:
                route_sequence.append(("pickup", location))
            else:
                route_sequence.append(("delivery", location))

        # Ensure all deliveries are included
        for parcel in original_route.parcels:
            if not any(
                loc == parcel.delivery_location and action == "delivery"
                for action, loc in route_sequence
            ):
                route_sequence.append(("delivery", parcel.delivery_location))

        optimized_route.route_sequence = route_sequence
        optimized_route.is_optimized = True

        # Calculate optimized route metrics
        optimized_route.total_distance, optimized_route.total_duration = (
            self._calculate_route_metrics_from_sequence(
                route_sequence, distance_matrix, eta_matrix
            )
        )

        return optimized_route

    def _calculate_route_metrics_from_sequence(
        self,
        route_sequence: List[Tuple[str, Tuple[float, float]]],
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Tuple[float, int]:
        """Calculate route metrics from optimized sequence"""
        total_distance = 0
        total_duration = 0

        if len(route_sequence) < 2:
            return 0, 0

        # Calculate cumulative metrics along the optimized sequence
        for i in range(len(route_sequence) - 1):
            _, loc1 = route_sequence[i]
            _, loc2 = route_sequence[i + 1]

            distance = distance_matrix.get(loc1, {}).get(loc2, 0)
            travel_time = eta_matrix.get(loc1, {}).get(loc2, 0)

            total_distance += distance
            total_duration += travel_time

        return total_distance, total_duration

    def _attempt_route_merging(
        self,
        solution: Solution,
        distance_matrix: Dict,
        eta_matrix: Dict,
        penalty_aware: bool = True,
    ) -> Solution:
        """
        Attempt to merge routes for better cost efficiency with penalty awareness
        """
        if len(solution.routes) <= 1:
            return solution

        best_solution = deepcopy(solution)
        best_cost_per_parcel = solution.average_cost_per_parcel

        # Group routes by pickup terminals for merging candidates
        terminal_groups = self._group_routes_by_terminals(solution.routes)

        # Try merging within each terminal group
        for terminal_id, routes in terminal_groups.items():
            if len(routes) > 1:
                merged_routes = self._merge_routes_for_terminal(
                    routes, distance_matrix, eta_matrix, penalty_aware
                )

                if merged_routes and len(merged_routes) < len(routes):
                    # Create new solution with merged routes
                    test_solution = self._create_solution_with_merged_routes(
                        solution, terminal_id, routes, merged_routes
                    )

                    test_solution.calculate_cost_and_time(
                        distance_matrix,
                        eta_matrix,
                        self.config.time_violation_penalty_per_minute,
                    )
                    test_solution.update_feasibility_status(
                        self.config.max_knock, self.config.time_window_seconds
                    )

                    if self._is_solution_better_with_penalties(
                        test_solution, best_solution
                    ):
                        best_solution = test_solution
                        best_cost_per_parcel = test_solution.average_cost_per_parcel

        # Try cross-terminal merging for routes with compatible pickup sequences
        if penalty_aware:
            cross_terminal_merged = self._attempt_cross_terminal_merging(
                best_solution, distance_matrix, eta_matrix
            )

            if self._is_solution_better_with_penalties(
                cross_terminal_merged, best_solution
            ):
                best_solution = cross_terminal_merged

        return best_solution

    def _is_solution_better_with_penalties(
        self, new_solution: Solution, current_solution: Solution
    ) -> bool:
        """Compare solutions considering both cost and penalty factors"""
        # Primary: Compare average cost per parcel (includes penalties)
        if (
            new_solution.average_cost_per_parcel
            < current_solution.average_cost_per_parcel
        ):
            return True

        # Secondary: If costs are similar, prefer solution with fewer violations
        cost_difference = abs(
            new_solution.average_cost_per_parcel
            - current_solution.average_cost_per_parcel
        )
        if cost_difference < 0.01:  # Very similar costs
            return (
                new_solution.total_time_violation_seconds
                < current_solution.total_time_violation_seconds
            )

        return False

    def _optimize_for_penalty_reduction(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Specifically optimize routes to reduce time violation penalties"""
        optimized_solution = deepcopy(solution)

        # Focus on routes with significant time violations
        violating_routes = [
            r
            for r in optimized_solution.routes
            if r.time_violation_seconds > 300  # 5+ minutes
        ]

        for route in violating_routes:
            # Try splitting the route if it has many parcels
            if len(route.parcels) > 4:
                split_result = self._try_penalty_reducing_split(
                    route, distance_matrix, eta_matrix
                )
                if split_result:
                    # Replace original route with split routes
                    optimized_solution.remove_route(route)
                    for split_route in split_result:
                        optimized_solution.add_route(
                            split_route, self.config.time_violation_penalty_per_minute
                        )

        return optimized_solution

    def _try_penalty_reducing_split(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Optional[List[Route]]:
        """Try splitting a route to reduce penalties"""
        if len(route.parcels) < 3:
            return None

        best_split = None
        best_penalty_reduction = 0

        # Try different split points
        for split_point in range(1, len(route.parcels)):
            part1_parcels = route.parcels[:split_point]
            part2_parcels = route.parcels[split_point:]

            # Create routes for each part
            split_routes = []
            total_penalty = 0

            for i, parcels_subset in enumerate([part1_parcels, part2_parcels]):
                if not parcels_subset:
                    continue

                # Find best vehicle type for this subset
                test_route = self._create_route_for_parcels(
                    parcels_subset,
                    route.pickup_sequence,
                    route.vehicle_id + i,
                    distance_matrix,
                    eta_matrix,
                )

                if test_route:
                    split_routes.append(test_route)
                    total_penalty += test_route.time_violation_penalty

            # Check if split reduces penalties
            if len(split_routes) == 2:
                penalty_reduction = route.time_violation_penalty - total_penalty
                if penalty_reduction > best_penalty_reduction:
                    best_penalty_reduction = penalty_reduction
                    best_split = split_routes

        return best_split if best_penalty_reduction > 0 else None

    def _create_route_for_parcels(
        self,
        parcels: List[Parcel],
        pickup_sequence: List[int],
        vehicle_id: int,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[Route]:
        """Create optimized route for given parcels"""
        subset_size = sum(p.size for p in parcels)

        # Find best vehicle type considering penalties
        best_route = None
        best_total_cost = float("inf")

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
                route_distance, route_duration = self._calculate_route_metrics(
                    route, distance_matrix, eta_matrix
                )

                route.total_distance = route_distance
                route.total_duration = route_duration
                route.update_time_feasibility(self.config.time_window_seconds)
                route.update_costs(self.config.time_violation_penalty_per_minute)

                if route.total_cost < best_total_cost:
                    best_total_cost = route.total_cost
                    best_route = route

        return best_route

    def _group_routes_by_terminals(self, routes: List[Route]) -> Dict[int, List[Route]]:
        """Group routes by their primary pickup terminal"""
        terminal_groups = {}

        for route in routes:
            # Use first pickup terminal as primary
            primary_terminal = route.pickup_sequence[0] if route.pickup_sequence else 0

            if primary_terminal not in terminal_groups:
                terminal_groups[primary_terminal] = []
            terminal_groups[primary_terminal].append(route)

        return terminal_groups

    def _merge_routes_for_terminal(
        self,
        routes: List[Route],
        distance_matrix: Dict,
        eta_matrix: Dict,
        penalty_aware: bool = True,
    ) -> Optional[List[Route]]:
        """Attempt to merge routes serving the same terminal with penalty awareness"""
        if len(routes) <= 1:
            return routes

        # Try different vehicle types for consolidation
        best_merged = None
        best_total_cost = float("inf")

        # Calculate total parcel requirements
        total_parcels = []
        for route in routes:
            total_parcels.extend(route.parcels)

        total_size = sum(p.size for p in total_parcels)
        current_total_cost = sum(r.total_cost for r in routes)

        # Try each vehicle type that can fit all parcels
        for vehicle_type, vehicle_spec in self.config.vehicle_specs.items():
            if vehicle_spec.capacity >= total_size:
                merged_route = self._create_merged_route(
                    routes, vehicle_type, vehicle_spec, total_parcels
                )

                if merged_route:
                    # Optimize the merged route
                    optimized_merged = self._optimize_single_route_sequence(
                        merged_route, distance_matrix, eta_matrix, penalty_aware
                    )

                    if optimized_merged:
                        # Calculate route metrics
                        route_distance, route_duration = self._calculate_route_metrics(
                            optimized_merged, distance_matrix, eta_matrix
                        )

                        optimized_merged.total_distance = route_distance
                        optimized_merged.total_duration = route_duration
                        optimized_merged.update_time_feasibility(
                            self.config.time_window_seconds
                        )
                        optimized_merged.update_costs(
                            self.config.time_violation_penalty_per_minute
                        )

                        # Check if merging is beneficial
                        if optimized_merged.total_cost < best_total_cost:
                            best_total_cost = optimized_merged.total_cost
                            best_merged = [optimized_merged]

        # Only return merged solution if it's significantly better
        if (
            best_merged and best_total_cost < current_total_cost * 0.95
        ):  # 5% improvement threshold
            return best_merged
        else:
            return routes

    def _create_merged_route(
        self,
        routes: List[Route],
        vehicle_type: VehicleType,
        vehicle_spec: VehicleSpec,
        total_parcels: List[Parcel],
    ) -> Optional[Route]:
        """Create a merged route from multiple routes"""
        if not routes:
            return None

        try:
            # Get unique pickup sequences
            pickup_sequences = set()
            for route in routes:
                pickup_sequences.update(route.pickup_sequence)

            # Create merged route
            merged_route = Route(
                vehicle_id=routes[0].vehicle_id,  # Use first vehicle ID
                vehicle_type=vehicle_type,
                vehicle_spec=vehicle_spec,
                parcels=total_parcels,
                pickup_sequence=list(pickup_sequences),
                delivery_sequence=[p.delivery_location for p in total_parcels],
            )

            return merged_route

        except Exception as e:
            if self.verbose:
                print(f"Failed to create merged route: {e}")
            return None

    def _attempt_cross_terminal_merging(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Attempt to merge routes from different terminals if beneficial with penalty awareness"""
        if len(solution.routes) <= 1:
            return solution

        best_solution = deepcopy(solution)

        # Try merging pairs of routes from different terminals
        for i in range(len(solution.routes)):
            for j in range(i + 1, len(solution.routes)):
                route1 = solution.routes[i]
                route2 = solution.routes[j]

                # Check if routes can be merged
                combined_parcels = route1.parcels + route2.parcels
                combined_size = sum(p.size for p in combined_parcels)

                # Try with different vehicle types
                for vehicle_type, vehicle_spec in self.config.vehicle_specs.items():
                    if vehicle_spec.capacity >= combined_size:
                        merged_route = self._create_merged_route(
                            [route1, route2],
                            vehicle_type,
                            vehicle_spec,
                            combined_parcels,
                        )

                        if merged_route:
                            # Optimize merged route
                            optimized_merged = self._optimize_single_route_sequence(
                                merged_route, distance_matrix, eta_matrix, True
                            )

                            if optimized_merged:
                                # Calculate metrics
                                route_distance, route_duration = (
                                    self._calculate_route_metrics(
                                        optimized_merged, distance_matrix, eta_matrix
                                    )
                                )

                                optimized_merged.total_distance = route_distance
                                optimized_merged.total_duration = route_duration
                                optimized_merged.update_time_feasibility(
                                    self.config.time_window_seconds
                                )
                                optimized_merged.update_costs(
                                    self.config.time_violation_penalty_per_minute
                                )

                                # Check if merging is beneficial (including penalty consideration)
                                original_total_cost = (
                                    route1.total_cost + route2.total_cost
                                )
                                if (
                                    optimized_merged.total_cost
                                    < original_total_cost * 0.9
                                ):  # 10% improvement for cross-terminal

                                    # Create new solution with merged route
                                    test_solution = deepcopy(solution)
                                    test_solution.remove_route(route1)
                                    test_solution.remove_route(route2)
                                    test_solution.add_route(
                                        optimized_merged,
                                        self.config.time_violation_penalty_per_minute,
                                    )

                                    test_solution.calculate_cost_and_time(
                                        distance_matrix,
                                        eta_matrix,
                                        self.config.time_violation_penalty_per_minute,
                                    )
                                    test_solution.update_feasibility_status(
                                        self.config.max_knock,
                                        self.config.time_window_seconds,
                                    )

                                    if self._is_solution_better_with_penalties(
                                        test_solution, best_solution
                                    ):
                                        best_solution = test_solution

        return best_solution

    def _create_solution_with_merged_routes(
        self,
        original_solution: Solution,
        terminal_id: int,
        old_routes: List[Route],
        new_routes: List[Route],
    ) -> Solution:
        """Create new solution with merged routes for a terminal"""
        new_solution = deepcopy(original_solution)

        # Remove old routes
        for route in old_routes:
            new_solution.remove_route(route)

        # Add new routes
        for route in new_routes:
            new_solution.add_route(route, self.config.time_violation_penalty_per_minute)

        return new_solution

    def _calculate_route_metrics(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Tuple[float, int]:
        """Calculate distance and duration for a route"""
        total_distance = 0
        total_duration = 0

        if not route.route_sequence:
            return 0, 0

        route_locations = route.get_locations_sequence()

        for i in range(len(route_locations) - 1):
            loc1 = route_locations[i]
            loc2 = route_locations[i + 1]

            distance = distance_matrix.get(loc1, {}).get(loc2, 0)
            travel_time = eta_matrix.get(loc1, {}).get(loc2, 0)

            total_distance += distance
            total_duration += travel_time

        return total_distance, total_duration

    def _fallback_global_optimization(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Fallback optimization when OR-Tools is not available with penalty awareness"""
        optimized_solution = deepcopy(solution)

        # Simple nearest neighbor optimization for each route
        for route in optimized_solution.routes:
            if len(route.parcels) > 2:
                optimized_sequence = self._simple_nearest_neighbor(
                    route, distance_matrix
                )
                route.route_sequence = optimized_sequence
                route.is_optimized = True

                # Recalculate metrics
                route.total_distance, route.total_duration = (
                    self._calculate_route_metrics(route, distance_matrix, eta_matrix)
                )
                route.update_time_feasibility(self.config.time_window_seconds)
                route.update_costs(self.config.time_violation_penalty_per_minute)

        # Simple route merging based on capacity and penalties
        optimized_solution = self._simple_route_merging_with_penalties(
            optimized_solution, distance_matrix, eta_matrix
        )

        return optimized_solution

    def _simple_nearest_neighbor(
        self, route: Route, distance_matrix: Dict
    ) -> List[Tuple[str, Tuple[float, float]]]:
        """Simple nearest neighbor heuristic for route optimization"""
        if not route.parcels:
            return []

        # Start with pickup location
        current_location = route.parcels[0].pickup_location
        route_sequence = [("pickup", current_location)]

        # Add deliveries in nearest neighbor order
        unvisited_deliveries = [p.delivery_location for p in route.parcels]

        while unvisited_deliveries:
            nearest_delivery = min(
                unvisited_deliveries,
                key=lambda loc: distance_matrix.get(current_location, {}).get(
                    loc, float("inf")
                ),
            )

            route_sequence.append(("delivery", nearest_delivery))
            current_location = nearest_delivery
            unvisited_deliveries.remove(nearest_delivery)

        return route_sequence

    def _simple_route_merging_with_penalties(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Simple route merging based on capacity constraints and penalty reduction"""
        if len(solution.routes) <= 1:
            return solution

        merged_solution = deepcopy(solution)

        # Try to merge routes with same pickup terminals
        terminal_groups = self._group_routes_by_terminals(merged_solution.routes)

        for terminal_id, routes in terminal_groups.items():
            if len(routes) > 1:
                # Calculate current total cost including penalties
                current_total_cost = sum(r.total_cost for r in routes)
                current_total_penalties = sum(r.time_violation_penalty for r in routes)

                # Try to merge into larger vehicle
                total_size = sum(sum(p.size for p in route.parcels) for route in routes)

                # Find suitable vehicle type
                best_merge = None
                best_total_cost = current_total_cost

                for vehicle_type, vehicle_spec in self.config.vehicle_specs.items():
                    if vehicle_spec.capacity >= total_size:
                        # Create merged route
                        all_parcels = []
                        for route in routes:
                            all_parcels.extend(route.parcels)

                        merged_route = Route(
                            vehicle_id=routes[0].vehicle_id,
                            vehicle_type=vehicle_type,
                            vehicle_spec=vehicle_spec,
                            parcels=all_parcels,
                            pickup_sequence=[terminal_id],
                            delivery_sequence=[
                                p.delivery_location for p in all_parcels
                            ],
                        )

                        # Estimate route duration (simplified)
                        estimated_duration = (
                            sum(r.total_duration for r in routes) * 0.8
                        )  # Assume 20% improvement
                        merged_route.total_duration = estimated_duration
                        merged_route.update_time_feasibility(
                            self.config.time_window_seconds
                        )
                        merged_route.update_costs(
                            self.config.time_violation_penalty_per_minute
                        )

                        # Check if merging is beneficial considering penalties
                        if merged_route.total_cost < best_total_cost:
                            best_total_cost = merged_route.total_cost
                            best_merge = merged_route

                # Apply merging if beneficial
                if (
                    best_merge and best_total_cost < current_total_cost * 0.95
                ):  # 5% improvement threshold
                    for route in routes:
                        merged_solution.remove_route(route)
                    merged_solution.add_route(
                        best_merge, self.config.time_violation_penalty_per_minute
                    )

        return merged_solution


class GlobalOptimizationController:
    """
    Enhanced controller for managing when and how to apply global optimization
    with comprehensive vehicle type optimization and penalty awareness
    """

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.optimizer = GlobalRouteOptimizer(config)
        self.vehicle_optimizer = UpdatedVehicleOptimizer(config)
        self.last_optimization_iteration = -1
        self.optimization_interval = 30  # Optimize every 30 iterations
        self.min_improvement_threshold = 0.01  # 1% improvement threshold

        # Performance settings for faster optimization
        self.enable_vehicle_optimization = True
        self.vehicle_optimization_frequency = 2  # Every 2nd global optimization
        self.penalty_focused_optimization = True  # Enable penalty-focused optimization
        self.enable_ortools_time_constraints = False  # Disabled by default for speed

    def should_optimize(
        self,
        iteration: int,
        solution: Solution,
        iterations_since_improvement: int,
        optimization_interval: int,
    ) -> bool:
        """
        Enhanced optimization trigger logic with penalty awareness

        Args:
            iteration: Current ALNS iteration
            solution: Current solution
            iterations_since_improvement: Iterations since last improvement
            optimization_interval: Optimization interval

        Returns:
            True if optimization should be applied
        """
        # Always optimize at start
        if iteration == -1:
            print(f"Should optimize as iteration is {iteration}")
            return True

        # Optimize at regular intervals
        if iteration - self.last_optimization_iteration >= optimization_interval:
            print(
                f"Should optimize as {iteration} - {self.last_optimization_iteration} >= {optimization_interval}"
            )
            return True

        # Prioritize optimization for solutions with high penalties
        if (
            self.penalty_focused_optimization
            and solution.total_time_violation_penalty > 0
        ):
            penalty_ratio = solution.total_time_violation_penalty / solution.total_cost
            if penalty_ratio > 0.15:  # Penalties are >15% of total cost
                print(f"Should optimize due to high penalty ratio: {penalty_ratio:.2f}")
                return True

        # Optimize when we have multiple routes with low utilization
        if len(solution.routes) > 1:
            avg_utilization = sum(
                r.get_utilization_percentage() for r in solution.routes
            ) / len(solution.routes)
            if avg_utilization < 60:  # Low utilization suggests merge opportunities
                print(
                    f"Should optimize as we have multiple routes with low utilization: {avg_utilization:.1f}%"
                )
                return True

        # Optimize when many routes have time violations
        if (
            solution.routes_with_time_violations > len(solution.routes) * 0.4
        ):  # >40% of routes
            print(
                f"Should optimize due to high violation rate: {solution.routes_with_time_violations}/{len(solution.routes)}"
            )
            return True

        return False

    def optimize_if_needed(
        self,
        iteration: int,
        solution: Solution,
        iterations_since_improvement: int,
        distance_matrix: Dict,
        eta_matrix: Dict,
        optimization_intervals: int,
    ) -> Solution:
        """
        Apply comprehensive optimization including route and vehicle optimization with penalty focus

        Args:
            iteration: Current ALNS iteration
            solution: Current solution
            iterations_since_improvement: Iterations since last improvement
            distance_matrix: Distance matrix
            eta_matrix: ETA matrix
            optimization_intervals: Optimization interval

        Returns:
            Optimized solution (or original if optimization not needed)
        """
        if not self.should_optimize(
            iteration, solution, iterations_since_improvement, optimization_intervals
        ):
            return solution

        print("Solution should be optimized...")
        original_cost = solution.average_cost_per_parcel
        original_penalties = solution.total_time_violation_penalty
        best_solution = solution

        # Step 1: Apply global route optimization (sequences + merging) with penalty awareness
        print("Global route optimization in progress...")
        route_optimized_solution = self.optimizer.optimize_solution_globally(
            solution,
            distance_matrix,
            eta_matrix,
            enable_route_merging=True,
            penalty_aware_optimization=self.penalty_focused_optimization,
            enable_ortools_time_constraints=self.enable_ortools_time_constraints,
        )
        print("Global route optimization done...")

        # Check if route optimization improved
        route_improvement = (
            original_cost - route_optimized_solution.average_cost_per_parcel
        ) / original_cost
        penalty_reduction = (
            original_penalties - route_optimized_solution.total_time_violation_penalty
        )

        if route_improvement >= self.min_improvement_threshold or penalty_reduction > 0:
            best_solution = route_optimized_solution
            print(
                f"Route optimization improved cost per parcel by {route_improvement*100:.1f}%"
            )
            if penalty_reduction > 0:
                print(
                    f"Route optimization reduced penalties by {penalty_reduction:.2f}"
                )

        # Step 2: Apply vehicle type optimization (if enabled and conditions are met)
        if (
            self.check_enable_vehicle_optimization()
            and self._should_apply_vehicle_optimization(iteration)
        ):
            print("Vehicle type optimization in progress...")

            # Choose optimization strategy based on solution state
            if best_solution.routes_with_time_violations > 0:
                # Focus on penalty reduction for solutions with violations
                vehicle_optimized_solution = (
                    self.vehicle_optimizer.optimize_for_time_feasibility(best_solution)
                )
                print("Applied time feasibility optimization...")
            else:
                # Standard cost optimization for feasible solutions
                vehicle_optimized_solution = (
                    self.vehicle_optimizer.optimize_vehicle_types_for_cost(
                        best_solution
                    )
                )
                print("Applied standard cost optimization...")

            # Check if vehicle optimization improved
            vehicle_improvement = (
                best_solution.average_cost_per_parcel
                - vehicle_optimized_solution.average_cost_per_parcel
            ) / best_solution.average_cost_per_parcel
            vehicle_penalty_reduction = (
                best_solution.total_time_violation_penalty
                - vehicle_optimized_solution.total_time_violation_penalty
            )

            if (
                vehicle_improvement >= self.min_improvement_threshold
                or vehicle_penalty_reduction > 0
            ):
                best_solution = vehicle_optimized_solution
                print(
                    f"Vehicle optimization improved cost per parcel by {vehicle_improvement*100:.1f}%"
                )
                if vehicle_penalty_reduction > 0:
                    print(
                        f"Vehicle optimization reduced penalties by {vehicle_penalty_reduction:.2f}"
                    )
            else:
                print("Vehicle optimization yielded no improvement")

            print("Vehicle type optimization done...")

        # Step 3: Penalty-focused optimization if significant violations remain
        if (
            self.penalty_focused_optimization
            and best_solution.total_time_violation_penalty > original_penalties * 0.7
        ):  # Still >70% of original penalties

            print("Penalty-focused optimization in progress...")
            penalty_optimized = self._apply_penalty_focused_optimization(
                best_solution, distance_matrix, eta_matrix
            )

            penalty_focused_improvement = (
                best_solution.total_time_violation_penalty
                - penalty_optimized.total_time_violation_penalty
            )
            if penalty_focused_improvement > 0:
                best_solution = penalty_optimized
                print(
                    f"Penalty-focused optimization reduced penalties by {penalty_focused_improvement:.2f}"
                )

        # Calculate total improvement
        total_improvement = (
            original_cost - best_solution.average_cost_per_parcel
        ) / original_cost
        total_penalty_reduction = (
            original_penalties - best_solution.total_time_violation_penalty
        )

        if (
            total_improvement >= self.min_improvement_threshold
            or total_penalty_reduction > 0
        ):
            self.last_optimization_iteration = iteration
            print(f"Total optimization improvement: {total_improvement*100:.1f}%")
            if total_penalty_reduction > 0:
                print(f"Total penalty reduction: {total_penalty_reduction:.2f}")
            return best_solution
        else:
            print("No significant improvement from optimization")
            return solution

    def _apply_penalty_focused_optimization(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Apply optimization strategies specifically focused on reducing penalties"""
        optimized_solution = deepcopy(solution)

        # Strategy 1: Vehicle upgrades for high-penalty routes
        high_penalty_routes = [
            r
            for r in optimized_solution.routes
            if r.time_violation_penalty > 0
            and r.time_violation_penalty / r.total_cost > 0.2
        ]

        for route in high_penalty_routes:
            # Try upgrading to a larger vehicle
            current_capacity = route.vehicle_spec.capacity
            for vehicle_type, vehicle_spec in self.config.vehicle_specs.items():
                if (
                    vehicle_spec.capacity > current_capacity
                    and vehicle_spec.capacity >= route.total_size
                ):

                    old_cost = route.total_cost

                    # Test the upgrade
                    route.vehicle_type = vehicle_type
                    route.vehicle_spec = vehicle_spec
                    # Assume some time improvement with larger vehicle
                    route.total_duration = int(
                        route.total_duration * 0.9
                    )  # 10% improvement
                    route.update_time_feasibility(self.config.time_window_seconds)
                    route.update_costs(self.config.time_violation_penalty_per_minute)

                    if route.total_cost < old_cost:
                        break  # Keep the upgrade
                    else:
                        # Revert if no improvement
                        route.vehicle_type = (
                            vehicle_type  # This needs to be reverted properly
                        )
                        route.vehicle_spec = vehicle_spec
                        route.update_costs(
                            self.config.time_violation_penalty_per_minute
                        )

        return optimized_solution

    def _should_apply_vehicle_optimization(self, iteration: int) -> bool:
        """
        Determine if vehicle optimization should be applied this iteration

        Args:
            iteration: Current iteration

        Returns:
            True if vehicle optimization should be applied
        """
        # Apply vehicle optimization less frequently than route optimization
        optimization_count = (iteration - (-1)) // self.optimization_interval

        # Apply every Nth global optimization
        if optimization_count % self.vehicle_optimization_frequency == 0:
            return True

        # Always apply at start
        if iteration == -1:
            return True

        return False

    def final_optimization(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """
        Apply final comprehensive optimization with penalty focus
        """
        print("Final comprehensive optimization...")

        original_cost = solution.average_cost_per_parcel
        original_penalties = solution.total_time_violation_penalty
        best_solution = solution

        # Step 1: Final route optimization with penalty awareness
        print("Final route optimization...")
        route_optimized = self.optimizer.optimize_solution_globally(
            solution,
            distance_matrix,
            eta_matrix,
            enable_route_merging=True,
            penalty_aware_optimization=True,
            enable_ortools_time_constraints=self.enable_ortools_time_constraints,
        )

        route_improvement = (
            original_cost - route_optimized.average_cost_per_parcel
        ) / original_cost
        route_penalty_reduction = (
            original_penalties - route_optimized.total_time_violation_penalty
        )

        if (
            route_improvement >= self.min_improvement_threshold
            or route_penalty_reduction > 0
        ):
            best_solution = route_optimized
            print(f"Final route optimization improved by {route_improvement*100:.1f}%")
            if route_penalty_reduction > 0:
                print(
                    f"Final route optimization reduced penalties by {route_penalty_reduction:.2f}"
                )

        # Step 2: Final vehicle optimization with penalty focus
        if self.enable_vehicle_optimization:
            print("Final vehicle optimization...")

            # Apply both cost and penalty optimization
            cost_optimized = self.vehicle_optimizer.optimize_vehicle_types_for_cost(
                best_solution
            )
            penalty_optimized = self.vehicle_optimizer.optimize_for_time_feasibility(
                best_solution
            )

            # Choose the better result
            if (
                cost_optimized.average_cost_per_parcel
                <= penalty_optimized.average_cost_per_parcel
                and cost_optimized.total_time_violation_penalty
                <= penalty_optimized.total_time_violation_penalty
            ):
                vehicle_optimized = cost_optimized
                print("Applied final cost optimization")
            else:
                vehicle_optimized = penalty_optimized
                print("Applied final penalty optimization")

            vehicle_improvement = (
                best_solution.average_cost_per_parcel
                - vehicle_optimized.average_cost_per_parcel
            ) / best_solution.average_cost_per_parcel
            vehicle_penalty_reduction = (
                best_solution.total_time_violation_penalty
                - vehicle_optimized.total_time_violation_penalty
            )

            if (
                vehicle_improvement >= self.min_improvement_threshold
                or vehicle_penalty_reduction > 0
            ):
                best_solution = vehicle_optimized
                print(
                    f"Final vehicle optimization improved by {vehicle_improvement*100:.1f}%"
                )
                if vehicle_penalty_reduction > 0:
                    print(
                        f"Final vehicle optimization reduced penalties by {vehicle_penalty_reduction:.2f}"
                    )

        total_improvement = (
            original_cost - best_solution.average_cost_per_parcel
        ) / original_cost
        total_penalty_reduction = (
            original_penalties - best_solution.total_time_violation_penalty
        )

        print(f"Final optimization total improvement: {total_improvement*100:.1f}%")
        if total_penalty_reduction > 0:
            print(
                f"Final optimization total penalty reduction: {total_penalty_reduction:.2f}"
            )

        return best_solution

    def get_optimization_statistics(self) -> Dict:
        """Get statistics about optimization performance including penalty metrics"""
        return {
            "last_optimization_iteration": self.last_optimization_iteration,
            "optimization_interval": self.optimization_interval,
            "min_improvement_threshold": self.min_improvement_threshold,
            "vehicle_optimization_enabled": self.enable_vehicle_optimization,
            "vehicle_optimization_frequency": self.vehicle_optimization_frequency,
            "penalty_focused_optimization": self.penalty_focused_optimization,
            "penalty_optimization_features": {
                "penalty_aware_route_optimization": True,
                "time_feasibility_optimization": True,
                "penalty_reduction_strategies": True,
                "vehicle_upgrade_optimization": True,
            },
        }

    def configure_performance_mode(self, fast_mode: bool = True):
        """
        Configure optimization for performance vs quality trade-off with penalty awareness

        Args:
            fast_mode: If True, optimize for speed. If False, optimize for quality.
        """
        if fast_mode:
            # Fast mode settings
            self.optimization_interval = 50  # Less frequent optimization
            self.min_improvement_threshold = 0.02  # Higher threshold
            self.vehicle_optimization_frequency = (
                3  # Less frequent vehicle optimization
            )
            self.enable_vehicle_optimization = True  # Keep it enabled but less frequent
            self.penalty_focused_optimization = (
                True  # Keep penalty focus even in fast mode
            )
            self.enable_ortools_time_constraints = False  # Disable for maximum speed
            print(
                "Performance mode: FAST - Optimized for speed without OR-Tools time constraints"
            )
        else:
            # Quality mode settings
            self.optimization_interval = 20  # More frequent optimization
            self.min_improvement_threshold = 0.005  # Lower threshold (0.5%)
            self.vehicle_optimization_frequency = 1  # Every global optimization
            self.enable_vehicle_optimization = True
            self.penalty_focused_optimization = True
            self.enable_ortools_time_constraints = True  # Enable for better quality
            print(
                "Performance mode: QUALITY - More thorough optimization with OR-Tools time constraints"
            )

    def configure_penalty_optimization(
        self,
        enable_penalty_focus: bool = True,
        penalty_optimization_threshold: float = 0.1,
    ):
        """
        Configure penalty-specific optimization settings

        Args:
            enable_penalty_focus: Whether to enable penalty-focused optimization
            penalty_optimization_threshold: Penalty ratio threshold for triggering optimization
        """
        self.penalty_focused_optimization = enable_penalty_focus
        self.penalty_optimization_threshold = penalty_optimization_threshold

        if enable_penalty_focus:
            print(
                f"Penalty-focused optimization enabled (threshold: {penalty_optimization_threshold:.1%})"
            )
        else:
            print("Penalty-focused optimization disabled")

    def disable_vehicle_optimization(self):
        """Disable vehicle optimization for maximum speed"""
        self.enable_vehicle_optimization = False
        print("Vehicle optimization disabled for maximum speed")

    def enable_ortools_time_constraints(self, enable: bool = True):
        """
        Enable or disable OR-Tools time window constraints

        Args:
            enable: Whether to enable time constraints in OR-Tools
        """
        self.enable_ortools_time_constraints = enable

        if enable:
            print(
                "OR-Tools time constraints enabled (slower but potentially better quality)"
            )
        else:
            print("OR-Tools time constraints disabled (faster optimization)")

    def check_enable_vehicle_optimization(self):
        """Enable vehicle optimization for better quality"""
        return self.enable_vehicle_optimization
