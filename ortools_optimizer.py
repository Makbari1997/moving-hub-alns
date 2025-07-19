from typing import Dict, List, Tuple, Optional, Set
from copy import deepcopy
import time
import math

# OR-Tools imports
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

from data_structures import (
    Route, Solution, VehicleType, VehicleSpec, Parcel, PickupTerminal, ProblemConfig
)


class GlobalRouteOptimizer:
    """
    Global OR-Tools optimizer that processes multiple routes simultaneously
    for better consolidation and sequence optimization
    """
    
    def __init__(self, config: ProblemConfig):
        self.config = config
        self.verbose = False
        self.optimization_timeout = 30  # seconds
        self.max_locations = 200  # Limit for performance
        
    def optimize_solution_globally(self, solution: Solution, distance_matrix: Dict, 
                                 eta_matrix: Dict, 
                                 enable_route_merging: bool = True) -> Solution:
        """
        Globally optimize entire solution with route merging capabilities
        
        Args:
            solution: Current solution to optimize
            distance_matrix: Distance matrix for locations
            eta_matrix: ETA matrix for locations
            enable_route_merging: Whether to attempt route merging
            
        Returns:
            Optimized solution with potentially merged routes
        """
        if not ORTOOLS_AVAILABLE:
            if self.verbose:
                print("OR-Tools not available, using fallback optimization")
            return self._fallback_global_optimization(solution, distance_matrix, eta_matrix)
        
        if not solution.routes:
            return solution
            
        start_time = time.time()
        
        try:
            # Step 1: Optimize sequences within existing routes
            sequence_optimized = self._optimize_existing_routes(solution, distance_matrix, eta_matrix)
            
            # Step 2: Attempt route merging if enabled
            if enable_route_merging and len(sequence_optimized.routes) > 1:
                merged_solution = self._attempt_route_merging(sequence_optimized, distance_matrix, eta_matrix)
                
                # Choose best solution based on cost per parcel
                if merged_solution.average_cost_per_parcel < sequence_optimized.average_cost_per_parcel:
                    final_solution = merged_solution
                else:
                    final_solution = sequence_optimized
            else:
                final_solution = sequence_optimized
            
            # Update solution costs
            final_solution.calculate_cost_and_time(distance_matrix, eta_matrix)
            
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"Global optimization completed in {elapsed:.2f}s")
                print(f"Routes: {len(solution.routes)} -> {len(final_solution.routes)}")
                print(f"Cost per parcel: {solution.average_cost_per_parcel:.2f} -> {final_solution.average_cost_per_parcel:.2f}")
            
            return final_solution
            
        except Exception as e:
            if self.verbose:
                print(f"Global optimization failed: {e}")
            return self._fallback_global_optimization(solution, distance_matrix, eta_matrix)
    
    def _optimize_existing_routes(self, solution: Solution, distance_matrix: Dict, 
                                eta_matrix: Dict) -> Solution:
        """Optimize sequences within existing routes"""
        optimized_solution = deepcopy(solution)
        
        for route in optimized_solution.routes:
            if len(route.parcels) > 2:  # Only optimize if worthwhile
                optimized_route = self._optimize_single_route_sequence(route, distance_matrix, eta_matrix)
                
                # Update route in solution
                route.route_sequence = optimized_route.route_sequence
                route.is_optimized = optimized_route.is_optimized
        
        return optimized_solution
    
    def _optimize_single_route_sequence(self, route: Route, distance_matrix: Dict, 
                                      eta_matrix: Dict) -> Route:
        """Optimize sequence for a single route using OR-Tools"""
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
            
            # Add distance callback
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                if from_node >= len(locations) or to_node >= len(locations):
                    return 0
                from_loc = locations[from_node]
                to_loc = locations[to_node]
                distance = distance_matrix.get(from_loc, {}).get(to_loc, 0)
                return max(1, int(distance / 100))  # Scale for OR-Tools
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            # Add time constraint
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                if from_node >= len(locations) or to_node >= len(locations):
                    return 0
                from_loc = locations[from_node]
                to_loc = locations[to_node]
                time_val = eta_matrix.get(from_loc, {}).get(to_loc, 300)
                return max(1, int(time_val / 60))  # Convert to minutes
            
            time_callback_index = routing.RegisterTransitCallback(time_callback)
            routing.AddDimension(
                time_callback_index,
                self.config.time_window_seconds // 60,
                self.config.time_window_seconds // 60,
                True,
                'Time'
            )
            
            # Set search parameters
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            search_parameters.time_limit.seconds = 10
            
            # Solve
            solution = routing.SolveWithParameters(search_parameters)
            
            if solution:
                return self._extract_optimized_route(route, solution, routing, manager, locations)
            else:
                return route
                
        except Exception as e:
            if self.verbose:
                print(f"Route optimization failed: {e}")
            return route
    
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
    
    def _extract_optimized_route(self, original_route: Route, solution, routing, 
                               manager, locations: List[Tuple[float, float]]) -> Route:
        """Extract optimized route from OR-Tools solution"""
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
                route_sequence.append(('pickup', location))
            else:
                route_sequence.append(('delivery', location))
        
        # Ensure all deliveries are included
        for parcel in original_route.parcels:
            if not any(loc == parcel.delivery_location and action == 'delivery' 
                      for action, loc in route_sequence):
                route_sequence.append(('delivery', parcel.delivery_location))
        
        optimized_route.route_sequence = route_sequence
        optimized_route.is_optimized = True
        
        return optimized_route
    
    def _attempt_route_merging(self, solution: Solution, distance_matrix: Dict, 
                             eta_matrix: Dict) -> Solution:
        """
        Attempt to merge routes for better cost efficiency
        Uses OR-Tools to find optimal merging combinations
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
                merged_routes = self._merge_routes_for_terminal(routes, distance_matrix, eta_matrix)
                
                if merged_routes and len(merged_routes) < len(routes):
                    # Create new solution with merged routes
                    test_solution = self._create_solution_with_merged_routes(
                        solution, terminal_id, routes, merged_routes
                    )
                    
                    test_solution.calculate_cost_and_time(distance_matrix, eta_matrix)
                    
                    if test_solution.average_cost_per_parcel < best_cost_per_parcel:
                        best_solution = test_solution
                        best_cost_per_parcel = test_solution.average_cost_per_parcel
        
        # Try cross-terminal merging for routes with compatible pickup sequences
        cross_terminal_merged = self._attempt_cross_terminal_merging(
            best_solution, distance_matrix, eta_matrix
        )
        
        if cross_terminal_merged.average_cost_per_parcel < best_cost_per_parcel:
            best_solution = cross_terminal_merged
        
        return best_solution
    
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
    
    def _merge_routes_for_terminal(self, routes: List[Route], distance_matrix: Dict, 
                                 eta_matrix: Dict) -> Optional[List[Route]]:
        """Attempt to merge routes serving the same terminal"""
        if len(routes) <= 1:
            return routes
            
        # Try different vehicle types for consolidation
        best_merged = None
        best_cost_per_parcel = float('inf')
        
        # Calculate total parcel requirements
        total_parcels = []
        for route in routes:
            total_parcels.extend(route.parcels)
        
        total_size = sum(p.size for p in total_parcels)
        
        # Try each vehicle type that can fit all parcels
        for vehicle_type, vehicle_spec in self.config.vehicle_specs.items():
            if vehicle_spec.capacity >= total_size:
                merged_route = self._create_merged_route(
                    routes, vehicle_type, vehicle_spec, total_parcels
                )
                
                if merged_route:
                    # Optimize the merged route
                    optimized_merged = self._optimize_single_route_sequence(
                        merged_route, distance_matrix, eta_matrix
                    )
                    
                    # Calculate route metrics
                    route_distance, route_duration = self._calculate_route_metrics(
                        optimized_merged, distance_matrix, eta_matrix
                    )
                    
                    optimized_merged.total_distance = route_distance
                    optimized_merged.total_duration = route_duration
                    optimized_merged.update_costs()
                    
                    # Check feasibility
                    if optimized_merged.is_time_feasible(self.config.time_window_seconds):
                        if optimized_merged.cost_per_parcel < best_cost_per_parcel:
                            best_merged = [optimized_merged]
                            best_cost_per_parcel = optimized_merged.cost_per_parcel
        
        return best_merged if best_merged else routes
    
    def _create_merged_route(self, routes: List[Route], vehicle_type: VehicleType, 
                           vehicle_spec: VehicleSpec, total_parcels: List[Parcel]) -> Optional[Route]:
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
    
    def _attempt_cross_terminal_merging(self, solution: Solution, distance_matrix: Dict, 
                                      eta_matrix: Dict) -> Solution:
        """Attempt to merge routes from different terminals if beneficial"""
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
                            [route1, route2], vehicle_type, vehicle_spec, combined_parcels
                        )
                        
                        if merged_route:
                            # Optimize merged route
                            optimized_merged = self._optimize_single_route_sequence(
                                merged_route, distance_matrix, eta_matrix
                            )
                            
                            # Calculate metrics
                            route_distance, route_duration = self._calculate_route_metrics(
                                optimized_merged, distance_matrix, eta_matrix
                            )
                            
                            optimized_merged.total_distance = route_distance
                            optimized_merged.total_duration = route_duration
                            optimized_merged.update_costs()
                            
                            # Check if merging is beneficial
                            if (optimized_merged.is_time_feasible(self.config.time_window_seconds) and
                                optimized_merged.cost_per_parcel < (route1.cost_per_parcel + route2.cost_per_parcel) / 2):
                                
                                # Create new solution with merged route
                                test_solution = deepcopy(solution)
                                test_solution.remove_route(route1)
                                test_solution.remove_route(route2)
                                test_solution.add_route(optimized_merged)
                                
                                test_solution.calculate_cost_and_time(distance_matrix, eta_matrix)
                                
                                if test_solution.average_cost_per_parcel < best_solution.average_cost_per_parcel:
                                    best_solution = test_solution
        
        return best_solution
    
    def _create_solution_with_merged_routes(self, original_solution: Solution, 
                                          terminal_id: int, old_routes: List[Route], 
                                          new_routes: List[Route]) -> Solution:
        """Create new solution with merged routes for a terminal"""
        new_solution = deepcopy(original_solution)
        
        # Remove old routes
        for route in old_routes:
            new_solution.remove_route(route)
        
        # Add new routes
        for route in new_routes:
            new_solution.add_route(route)
        
        return new_solution
    
    def _calculate_route_metrics(self, route: Route, distance_matrix: Dict, 
                               eta_matrix: Dict) -> Tuple[float, int]:
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
    
    def _fallback_global_optimization(self, solution: Solution, distance_matrix: Dict, 
                                    eta_matrix: Dict) -> Solution:
        """Fallback optimization when OR-Tools is not available"""
        optimized_solution = deepcopy(solution)
        
        # Simple nearest neighbor optimization for each route
        for route in optimized_solution.routes:
            if len(route.parcels) > 2:
                optimized_sequence = self._simple_nearest_neighbor(route, distance_matrix)
                route.route_sequence = optimized_sequence
                route.is_optimized = True
        
        # Simple route merging based on capacity
        optimized_solution = self._simple_route_merging(optimized_solution)
        
        return optimized_solution
    
    def _simple_nearest_neighbor(self, route: Route, distance_matrix: Dict) -> List[Tuple[str, Tuple[float, float]]]:
        """Simple nearest neighbor heuristic for route optimization"""
        if not route.parcels:
            return []
        
        # Start with pickup location
        current_location = route.parcels[0].pickup_location
        route_sequence = [('pickup', current_location)]
        
        # Add deliveries in nearest neighbor order
        unvisited_deliveries = [p.delivery_location for p in route.parcels]
        
        while unvisited_deliveries:
            nearest_delivery = min(unvisited_deliveries, 
                                 key=lambda loc: distance_matrix.get(current_location, {}).get(loc, float('inf')))
            
            route_sequence.append(('delivery', nearest_delivery))
            current_location = nearest_delivery
            unvisited_deliveries.remove(nearest_delivery)
        
        return route_sequence
    
    def _simple_route_merging(self, solution: Solution) -> Solution:
        """Simple route merging based on capacity constraints"""
        if len(solution.routes) <= 1:
            return solution
        
        merged_solution = deepcopy(solution)
        
        # Try to merge routes with same pickup terminals
        terminal_groups = self._group_routes_by_terminals(merged_solution.routes)
        
        for terminal_id, routes in terminal_groups.items():
            if len(routes) > 1:
                # Try to merge into larger vehicle
                total_size = sum(sum(p.size for p in route.parcels) for route in routes)
                
                # Find suitable vehicle type
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
                            delivery_sequence=[p.delivery_location for p in all_parcels],
                        )
                        
                        # Check if merging is cost-effective
                        if merged_route.cost_per_parcel < sum(r.cost_per_parcel for r in routes) / len(routes):
                            # Apply merging
                            for route in routes:
                                merged_solution.remove_route(route)
                            merged_solution.add_route(merged_route)
                            break
        
        return merged_solution


class GlobalOptimizationController:
    """
    Controller for managing when and how to apply global optimization
    """
    
    def __init__(self, config: ProblemConfig):
        self.config = config
        self.optimizer = GlobalRouteOptimizer(config)
        self.last_optimization_iteration = -1
        self.optimization_interval = 30  # Optimize every 30 iterations
        self.min_improvement_threshold = 0.01  # 1% improvement threshold
        
    def should_optimize(self, iteration: int, solution: Solution, 
                       iterations_since_improvement: int) -> bool:
        """
        Determine if global optimization should be applied
        
        Args:
            iteration: Current ALNS iteration
            solution: Current solution
            iterations_since_improvement: Iterations since last improvement
            
        Returns:
            True if optimization should be applied
        """
        # Always optimize at start
        if iteration == -1:
            print(f"Sould optimize as iteration is {iteration}")
            return True
        
        # Optimize at regular intervals
        if iteration - self.last_optimization_iteration >= self.optimization_interval:
            print(f"Should optimize as {iteration} - {self.last_optimization_iteration} >= {self.optimization_interval}")
            self.last_optimization_iteration = iteration
            return True
        
        # Optimize when stagnating
        # if iterations_since_improvement >= 20:
        #     return True
        
        # Optimize when we have multiple routes with low utilization
        if len(solution.routes) > 1:
            avg_utilization = sum(r.get_utilization_percentage() for r in solution.routes) / len(solution.routes)
            if avg_utilization < 60:  # Low utilization suggests merge opportunities
                print(f"Should optimize as we have multiple routes with low utilization")
                return True
        
        return False
    
    def optimize_if_needed(self, iteration: int, solution: Solution, 
                          iterations_since_improvement: int,
                          distance_matrix: Dict, eta_matrix: Dict) -> Solution:
        """
        Apply global optimization if conditions are met
        
        Args:
            iteration: Current ALNS iteration
            solution: Current solution
            iterations_since_improvement: Iterations since last improvement
            distance_matrix: Distance matrix
            eta_matrix: ETA matrix
            
        Returns:
            Optimized solution (or original if optimization not needed)
        """
        if not self.should_optimize(iteration, solution, iterations_since_improvement):
            return solution
        print("Solution should be optimized...")
        original_cost = solution.average_cost_per_parcel
        
        # Apply global optimization
        print("Global optimization in progress...")
        optimized_solution = self.optimizer.optimize_solution_globally(
            solution, distance_matrix, eta_matrix, enable_route_merging=True
        )
        print("Global Optimization Done...")
        # Check if optimization was beneficial
        improvement = (original_cost - optimized_solution.average_cost_per_parcel) / original_cost
        
        if improvement >= self.min_improvement_threshold:
            self.last_optimization_iteration = iteration
            return optimized_solution
        else:
            return solution
    
    def final_optimization(self, solution: Solution, distance_matrix: Dict, 
                         eta_matrix: Dict) -> Solution:
        """Apply final global optimization before returning solution"""
        return self.optimizer.optimize_solution_globally(
            solution, distance_matrix, eta_matrix, enable_route_merging=True
        )
