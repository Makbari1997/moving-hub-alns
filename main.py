import random

from typing import List, Dict, Tuple, Optional
import json
from copy import deepcopy
import time
import pandas as pd


# OR-Tools imports
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp

    ORTOOLS_AVAILABLE = True
except ImportError:
    print(
        "Warning: OR-Tools not available. Route optimization will use fallback methods."
    )
    ORTOOLS_AVAILABLE = False

from visualize import *
from preprocessing import *
from data_structures import *
from distance_calculator import *


class OptimizationCache:
    """Cache failed optimization attempts to avoid repeating them"""
    
    def __init__(self):
        self.failed_consolidations = set()
        self.successful_consolidations = {}
        self.time_window_violations = set()
        self.solution_hashes = {}  # Track solution states
        self.no_improvement_attempts = {}  # Track attempts that yield no improvement
        
    def add_failed_consolidation(self, terminal_id: int, route_sizes: tuple, vehicle_type: str):
        """Record a failed consolidation attempt"""
        key = (terminal_id, route_sizes, vehicle_type)
        self.failed_consolidations.add(key)
        
    def is_failed_consolidation(self, terminal_id: int, route_sizes: tuple, vehicle_type: str) -> bool:
        """Check if this consolidation has already failed"""
        key = (terminal_id, route_sizes, vehicle_type)
        return key in self.failed_consolidations
        
    def add_time_violation(self, route_signature: str):
        """Record a route that violates time windows"""
        self.time_window_violations.add(route_signature)
        
    def has_time_violation(self, route_signature: str) -> bool:
        """Check if this route signature has time violations"""
        return route_signature in self.time_window_violations
    
    def get_solution_hash(self, solution) -> str:
        """Generate a hash for solution state"""
        # Create a simplified hash based on routes and vehicle types
        route_data = []
        for route in solution.routes:
            route_data.append((
                route.vehicle_type.value,
                route.total_size,
                len(route.parcels),
                tuple(route.pickup_sequence)
            ))
        return str(hash(tuple(sorted(route_data))))
    
    def is_solution_seen(self, solution) -> bool:
        """Check if this solution state has been seen before"""
        solution_hash = self.get_solution_hash(solution)
        return solution_hash in self.solution_hashes
    
    def add_solution(self, solution):
        """Add solution to seen solutions"""
        solution_hash = self.get_solution_hash(solution)
        self.solution_hashes[solution_hash] = True
    
    def add_no_improvement_attempt(self, operation_key: str):
        """Record an operation that yielded no improvement"""
        self.no_improvement_attempts[operation_key] = self.no_improvement_attempts.get(operation_key, 0) + 1
    
    def should_skip_operation(self, operation_key: str, max_attempts: int = 3) -> bool:
        """Check if we should skip this operation due to repeated failures"""
        return self.no_improvement_attempts.get(operation_key, 0) >= max_attempts
        
    def clear_cache(self):
        """Clear all cached data"""
        self.failed_consolidations.clear()
        self.successful_consolidations.clear()
        self.time_window_violations.clear()
        self.solution_hashes.clear()
        self.no_improvement_attempts.clear()


class ALNSSolver:
    """Main ALNS solver with OR-Tools integration"""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.constructor = ConstructionHeuristic(config)
        self.operators = ALNSOperators(config)
        self.random = random.Random(42)

    def _should_try_vehicle_swap(self, solution: Solution) -> bool:
        """Check if vehicle swap is worth attempting"""
        if not hasattr(self.operators, 'vehicle_optimizer'):
            return True
        
        # Skip if all routes are already using optimal vehicle types
        optimal_count = 0
        for route in solution.routes:
            best_type = self.operators.vehicle_optimizer._get_best_vehicle_type_for_size(route.total_size)
            if route.vehicle_type == best_type:
                optimal_count += 1
        
        # Skip if >80% routes already optimal
        if optimal_count / len(solution.routes) > 0.8:
            return False
        
        return True

    def _select_operator_intelligently(self, iteration: int, current_solution: Solution, 
                                  consecutive_no_ops: int, vehicle_optimization_frequency: int,
                                  instance_size: int) -> str:
        """Intelligently select operator based on solution state"""
        
        # Skip expensive operations if they're unlikely to work
        if len(current_solution.routes) == 1:
            return "destroy_repair"  # Only useful operation for single route
        
        if consecutive_no_ops > 5:
            return "destroy_repair"  # Fall back to exploration
        
        # Check if vehicle swap is worth trying
        if not self._should_try_vehicle_swap(current_solution):
            if len(current_solution.routes) > 1:
                return "vehicle_consolidation"
            else:
                return "destroy_repair"
        
        # Instance-size based logic
        if instance_size > 100:
            # For large instances, reduce expensive vehicle operations
            choice = self.random.random()
            if choice < 0.5:
                return "destroy_repair"
            elif choice < 0.8:
                if iteration % vehicle_optimization_frequency == 0:
                    return "vehicle_consolidation"
                else:
                    return "skip"
            else:
                return "vehicle_swap"
        else:
            # Original logic for smaller instances
            choice = self.random.random()
            if choice < 0.3:
                return "destroy_repair"
            elif choice < 0.6:
                if iteration % vehicle_optimization_frequency == 0:
                    return "vehicle_consolidation"
                else:
                    return "vehicle_swap"
            else:
                return "destroy_repair"

    def solve_optimized(
        self,
        pickup_terminals: List[PickupTerminal],
        distance_matrix: Dict,
        eta_matrix: Dict,
        max_iterations: int = 1000,
    ) -> Solution:
        """Optimized solve method with enhanced stagnation detection"""
        start_time = time.time()
        
        # Clear cache at start
        if hasattr(self.operators, 'vehicle_optimizer'):
            self.operators.vehicle_optimizer.optimization_cache.clear_cache()
        
        # Reduce verbose logging for larger instances
        instance_size = sum(len(terminal.parcels) for terminal in pickup_terminals)
        verbose_logging = instance_size < 50
        
        if hasattr(self.operators, 'vehicle_optimizer'):
            self.operators.vehicle_optimizer.verbose_logging = verbose_logging
        
        print(f"Starting optimization for {instance_size} parcels")
        
        # Generate initial solution
        print("Generating initial solution...")
        current_solution = self.constructor.greedy_construction(
            pickup_terminals, distance_matrix, eta_matrix
        )
        
        # Apply initial vehicle optimization (once)
        print("Initial vehicle optimization...")
        try:
            optimized_initial = self.operators.vehicle_consolidation_operator(
                current_solution, distance_matrix, eta_matrix
            )
            
            # Check if initial optimization improved anything
            if optimized_initial.total_cost < current_solution.total_cost * 0.99:  # 1% improvement threshold
                current_solution = optimized_initial
                print("Initial optimization improved solution")
            else:
                print("Initial optimization yielded minimal improvement")
                
        except Exception as e:
            print(f"Warning: Initial vehicle optimization failed: {e}")
        
        best_solution = deepcopy(current_solution)
        self._print_solution_stats(current_solution, "Initial")
        
        # Adaptive parameters
        if instance_size > 100:
            max_iterations = min(max_iterations, 150)
            vehicle_optimization_frequency = 150  # Very infrequent
            stagnation_limit = 30
        elif instance_size > 50:
            max_iterations = min(max_iterations, 300)
            vehicle_optimization_frequency = 75
            stagnation_limit = 50
        else:
            vehicle_optimization_frequency = 25
            stagnation_limit = max_iterations // 6  # More generous for small instances
        
        print(f"Running {max_iterations} iterations (stagnation limit: {stagnation_limit})")
        
        # Enhanced stagnation tracking
        last_improvement = 0
        last_significant_improvement = 0  # Track major improvements
        consecutive_no_ops = 0  # Track consecutive operations with no effect
        solution_quality_history = [current_solution.total_cost]
        
        for iteration in range(max_iterations):
            try:
                # SINGLE INTELLIGENT OPERATOR SELECTION
                operator = self._select_operator_intelligently(
                    iteration, current_solution, consecutive_no_ops, 
                    vehicle_optimization_frequency, instance_size
                )
                
                repaired = None
                
                if operator == "destroy_repair":
                    if self.random.random() < 0.5:
                        destroyed = self.operators.random_removal(current_solution)
                    else:
                        destroyed = self.operators.terminal_removal(current_solution)
                    repaired = self.operators.greedy_insertion(
                        destroyed, distance_matrix, eta_matrix
                    )
                    
                elif operator == "vehicle_consolidation":
                    repaired = self.operators.vehicle_consolidation_operator(
                        current_solution, distance_matrix, eta_matrix
                    )
                    
                elif operator == "vehicle_swap":
                    repaired = self.operators.vehicle_type_swap_operator(
                        current_solution, distance_matrix, eta_matrix
                    )
                
                elif operator == "skip":
                    continue  # Skip this iteration
                
                if repaired is None:
                    consecutive_no_ops += 1
                    continue
                
                # Check if solution actually changed
                if self._solutions_are_equivalent(current_solution, repaired):
                    consecutive_no_ops += 1
                    if consecutive_no_ops > 10:
                        print(f"Iteration {iteration}: Too many no-ops, trying diversification")
                        # Force diversification
                        diversified = self.operators.random_removal(current_solution, removal_rate=0.5)
                        repaired = self.operators.greedy_insertion(diversified, distance_matrix, eta_matrix)
                    continue
                
                # Reset no-op counter if we got a different solution
                consecutive_no_ops = 0
                
                # Evaluate solution
                repaired.calculate_cost_and_time(distance_matrix, eta_matrix)
                repaired.is_feasible = (
                    len(repaired.unassigned_parcels) == 0
                    and repaired.validate_knock_constraints(self.config.max_knock)
                    and repaired.validate_time_window(self.config.time_window_seconds)
                )
                
                # Enhanced acceptance criterion
                improvement_threshold = 0.01  # 1% improvement threshold
                significant_improvement_threshold = 0.05  # 5% for significant improvement
                
                if repaired.is_feasible and (
                    not best_solution.is_feasible or repaired.total_cost < best_solution.total_cost
                ):
                    improvement = (best_solution.total_cost - repaired.total_cost) / best_solution.total_cost
                    
                    best_solution = deepcopy(repaired)
                    current_solution = repaired
                    last_improvement = iteration
                    
                    if improvement > significant_improvement_threshold:
                        last_significant_improvement = iteration
                        print(f"Iteration {iteration}: Significant improvement ({improvement*100:.1f}%)")
                        self._print_solution_stats(best_solution, f"Iteration {iteration}")
                    elif improvement > improvement_threshold:
                        if verbose_logging or iteration % 25 == 0:
                            print(f"Iteration {iteration}: Minor improvement ({improvement*100:.1f}%)")
                    
                    solution_quality_history.append(repaired.total_cost)
                    
                elif repaired.is_feasible and repaired.total_cost < current_solution.total_cost * 1.02:
                    current_solution = repaired
                elif self.random.random() < 0.01:  # Very rare diversification
                    current_solution = repaired
                
                # Enhanced early termination logic
                if iteration - last_improvement > stagnation_limit:
                    print(f"Early termination: No improvement for {iteration - last_improvement} iterations")
                    break
                
                # Check for solution quality plateau
                if len(solution_quality_history) > 20:
                    recent_variance = max(solution_quality_history[-10:]) - min(solution_quality_history[-10:])
                    if recent_variance < best_solution.total_cost * 0.001:  # Less than 0.1% variance
                        print(f"Early termination: Solution quality plateaued")
                        break
                        
            except Exception as e:
                if verbose_logging:
                    print(f"Iteration {iteration}: Error: {e}")
                continue
        
        processing_time = time.time() - start_time
        print(f"Completed in {processing_time:.2f} seconds")
        
        # Skip final optimization if we're already optimal
        if best_solution.is_feasible and len(best_solution.routes) > 1 and instance_size < 75:
            print("Final vehicle optimization...")
            try:
                final_optimized = self.operators.vehicle_consolidation_operator(
                    best_solution, distance_matrix, eta_matrix
                )
                final_optimized.calculate_cost_and_time(distance_matrix, eta_matrix)
                
                if final_optimized.is_feasible and final_optimized.total_cost < best_solution.total_cost * 0.99:
                    best_solution = final_optimized
                    print("Final optimization improved solution")
                else:
                    print("Final optimization yielded minimal improvement")
            except Exception as e:
                print(f"Final optimization failed: {e}")
        
        self._print_final_stats(best_solution)
        return best_solution

    def _solutions_are_equivalent(self, solution1: Solution, solution2: Solution) -> bool:
        """Check if two solutions are equivalent"""
        if len(solution1.routes) != len(solution2.routes):
            return False
        
        # Quick check: same total cost within small tolerance
        if abs(solution1.total_cost - solution2.total_cost) < 0.01:
            return True
        
        return False

    def solve_optimized_old(
        self,
        pickup_terminals: List[PickupTerminal],
        distance_matrix: Dict,
        eta_matrix: Dict,
        max_iterations: int = 1000,
    ) -> Solution:
        """Optimized solve method with enhanced stagnation detection"""
        start_time = time.time()
        
        # Clear cache at start
        if hasattr(self.operators, 'vehicle_optimizer'):
            self.operators.vehicle_optimizer.optimization_cache.clear_cache()
        
        # Reduce verbose logging for larger instances
        instance_size = sum(len(terminal.parcels) for terminal in pickup_terminals)
        verbose_logging = instance_size < 50
        
        if hasattr(self.operators, 'vehicle_optimizer'):
            self.operators.vehicle_optimizer.verbose_logging = verbose_logging
        
        print(f"Starting optimization for {instance_size} parcels")
        
        # Generate initial solution
        print("Generating initial solution...")
        current_solution = self.constructor.greedy_construction(
            pickup_terminals, distance_matrix, eta_matrix
        )
        
        # Apply initial vehicle optimization (once)
        print("Initial vehicle optimization...")
        try:
            optimized_initial = self.operators.vehicle_consolidation_operator(
                current_solution, distance_matrix, eta_matrix
            )
            
            # Check if initial optimization improved anything
            if optimized_initial.total_cost < current_solution.total_cost * 0.99:  # 1% improvement threshold
                current_solution = optimized_initial
                print("Initial optimization improved solution")
            else:
                print("Initial optimization yielded minimal improvement")
                
        except Exception as e:
            print(f"Warning: Initial vehicle optimization failed: {e}")
        
        best_solution = deepcopy(current_solution)
        self._print_solution_stats(current_solution, "Initial")
        
        # Adaptive parameters
        if instance_size > 100:
            max_iterations = min(max_iterations, 150)
            vehicle_optimization_frequency = 150  # Very infrequent
            stagnation_limit = 30
        elif instance_size > 50:
            max_iterations = min(max_iterations, 300)
            vehicle_optimization_frequency = 75
            stagnation_limit = 50
        else:
            vehicle_optimization_frequency = 25
            stagnation_limit = max_iterations // 6  # More generous for small instances
        
        print(f"Running {max_iterations} iterations (stagnation limit: {stagnation_limit})")
        
        # Enhanced stagnation tracking
        last_improvement = 0
        last_significant_improvement = 0  # Track major improvements
        consecutive_no_ops = 0  # Track consecutive operations with no effect
        solution_quality_history = [current_solution.total_cost]
        
        for iteration in range(max_iterations):
            try:
                # Skip expensive operations if we're stagnating
                if consecutive_no_ops > 5:
                    operation_choice = self.random.random()
                    if operation_choice < 0.8:  # Favor simple operations
                        if self.random.random() < 0.5:
                            destroyed = self.operators.random_removal(current_solution)
                        else:
                            destroyed = self.operators.terminal_removal(current_solution)
                        repaired = self.operators.greedy_insertion(
                            destroyed, distance_matrix, eta_matrix
                        )
                    else:
                        continue  # Skip iteration
                else:
                    # Normal operation selection
                    operator_choice = self.random.random()
                    
                    if instance_size > 100:
                        # For large instances, reduce expensive vehicle operations
                        if operator_choice < 0.5:
                            if self.random.random() < 0.5:
                                destroyed = self.operators.random_removal(current_solution)
                            else:
                                destroyed = self.operators.terminal_removal(current_solution)
                            repaired = self.operators.greedy_insertion(
                                destroyed, distance_matrix, eta_matrix
                            )
                        elif operator_choice < 0.8:
                            if iteration % vehicle_optimization_frequency == 0:
                                repaired = self.operators.vehicle_consolidation_operator(
                                    current_solution, distance_matrix, eta_matrix
                                )
                            else:
                                continue
                        else:
                            repaired = self.operators.vehicle_type_swap_operator(
                                current_solution, distance_matrix, eta_matrix
                            )
                    else:
                        # Original logic for smaller instances
                        if operator_choice < 0.3:
                            if self.random.random() < 0.5:
                                destroyed = self.operators.random_removal(current_solution)
                            else:
                                destroyed = self.operators.terminal_removal(current_solution)
                            repaired = self.operators.greedy_insertion(
                                destroyed, distance_matrix, eta_matrix
                            )
                        elif operator_choice < 0.6:
                            if iteration % vehicle_optimization_frequency == 0:
                                repaired = self.operators.vehicle_consolidation_operator(
                                    current_solution, distance_matrix, eta_matrix
                                )
                            else:
                                repaired = self.operators.vehicle_type_swap_operator(
                                    current_solution, distance_matrix, eta_matrix
                                )
                        else:
                            if self.random.random() < 0.5:
                                destroyed = self.operators.random_removal(current_solution)
                            else:
                                destroyed = self.operators.terminal_removal(current_solution)
                            repaired = self.operators.greedy_insertion(
                                destroyed, distance_matrix, eta_matrix
                            )
                
                if repaired is None:
                    consecutive_no_ops += 1
                    continue
                
                # Check if solution actually changed
                if self._solutions_are_equivalent(current_solution, repaired):
                    consecutive_no_ops += 1
                    if consecutive_no_ops > 10:
                        print(f"Iteration {iteration}: Too many no-ops, trying diversification")
                        # Force diversification
                        diversified = self.operators.random_removal(current_solution, removal_rate=0.5)
                        repaired = self.operators.greedy_insertion(diversified, distance_matrix, eta_matrix)
                    continue
                
                # Reset no-op counter if we got a different solution
                consecutive_no_ops = 0
                
                # Evaluate solution
                repaired.calculate_cost_and_time(distance_matrix, eta_matrix)
                repaired.is_feasible = (
                    len(repaired.unassigned_parcels) == 0
                    and repaired.validate_knock_constraints(self.config.max_knock)
                    and repaired.validate_time_window(self.config.time_window_seconds)
                )
                
                # Enhanced acceptance criterion
                improvement_threshold = 0.01  # 1% improvement threshold
                significant_improvement_threshold = 0.05  # 5% for significant improvement
                
                if repaired.is_feasible and (
                    not best_solution.is_feasible or repaired.total_cost < best_solution.total_cost
                ):
                    improvement = (best_solution.total_cost - repaired.total_cost) / best_solution.total_cost
                    
                    best_solution = deepcopy(repaired)
                    current_solution = repaired
                    last_improvement = iteration
                    
                    if improvement > significant_improvement_threshold:
                        last_significant_improvement = iteration
                        print(f"Iteration {iteration}: Significant improvement ({improvement*100:.1f}%)")
                        self._print_solution_stats(best_solution, f"Iteration {iteration}")
                    elif improvement > improvement_threshold:
                        if verbose_logging or iteration % 25 == 0:
                            print(f"Iteration {iteration}: Minor improvement ({improvement*100:.1f}%)")
                    
                    solution_quality_history.append(repaired.total_cost)
                    
                elif repaired.is_feasible and repaired.total_cost < current_solution.total_cost * 1.02:
                    current_solution = repaired
                elif self.random.random() < 0.01:  # Very rare diversification
                    current_solution = repaired
                
                # Enhanced early termination logic
                if iteration - last_improvement > stagnation_limit:
                    print(f"Early termination: No improvement for {iteration - last_improvement} iterations")
                    break
                
                # Check for solution quality plateau
                if len(solution_quality_history) > 20:
                    recent_variance = max(solution_quality_history[-10:]) - min(solution_quality_history[-10:])
                    if recent_variance < best_solution.total_cost * 0.001:  # Less than 0.1% variance
                        print(f"Early termination: Solution quality plateaued")
                        break
                        
            except Exception as e:
                if verbose_logging:
                    print(f"Iteration {iteration}: Error: {e}")
                continue
        
        processing_time = time.time() - start_time
        print(f"Completed in {processing_time:.2f} seconds")
        
        # Skip final optimization if we're already optimal
        if best_solution.is_feasible and len(best_solution.routes) > 1 and instance_size < 75:
            print("Final vehicle optimization...")
            try:
                final_optimized = self.operators.vehicle_consolidation_operator(
                    best_solution, distance_matrix, eta_matrix
                )
                final_optimized.calculate_cost_and_time(distance_matrix, eta_matrix)
                
                if final_optimized.is_feasible and final_optimized.total_cost < best_solution.total_cost * 0.99:
                    best_solution = final_optimized
                    print("Final optimization improved solution")
                else:
                    print("Final optimization yielded minimal improvement")
            except Exception as e:
                print(f"Final optimization failed: {e}")
        
        self._print_final_stats(best_solution)
        return best_solution

    def _print_final_stats(self, solution: Solution):
        """Print comprehensive final solution statistics"""
        print("\n" + "="*60)
        print("FINAL SOLUTION SUMMARY")
        print("="*60)
        
        if solution.is_feasible:
            # Vehicle distribution
            vehicle_counts = {}
            utilization_summary = {}
            optimized_routes = 0
            total_utilization = 0
            
            for route in solution.routes:
                vtype = route.vehicle_type.value
                vehicle_counts[vtype] = vehicle_counts.get(vtype, 0) + 1
                if route.is_optimized:
                    optimized_routes += 1
                    
                utilization = (route.total_size / route.vehicle_spec.capacity) * 100
                total_utilization += utilization
                
                if vtype not in utilization_summary:
                    utilization_summary[vtype] = []
                utilization_summary[vtype].append(utilization)
            
            # Calculate average utilizations
            avg_utilizations = {
                vtype: round(sum(utils) / len(utils), 1)
                for vtype, utils in utilization_summary.items()
            }
            
            print(f"✅ FEASIBLE SOLUTION FOUND")
            print(f"📊 Total Routes: {len(solution.routes)}")
            print(f"🚗 Vehicle Distribution: {vehicle_counts}")
            print(f"📈 Average Utilizations: {avg_utilizations}")
            print(f"💰 Total Cost: {solution.total_cost:.2f}")
            print(f"⏱️  Total Duration: {solution.total_duration/3600:.2f} hours")
            print(f"⏱️  Max Route Duration: {max((getattr(r, 'total_duration', 0) for r in solution.routes), default=0)/3600:.2f} hours")
            print(f"🎯 OR-Tools Optimized Routes: {optimized_routes}/{len(solution.routes)} ({optimized_routes/len(solution.routes)*100:.1f}%)")
            print(f"📏 Average Utilization: {total_utilization/len(solution.routes):.1f}%")
            
            # Time window analysis
            time_violations = [
                r for r in solution.routes 
                if not r.is_time_feasible(self.config.time_window_seconds)
            ]
            if time_violations:
                print(f"⚠️  Time Window Violations: {len(time_violations)}")
            else:
                print(f"✅ All routes meet time window constraints")
                
            # Knock analysis
            knock_violations = []
            for pickup_id, vehicle_ids in solution.pickup_assignments.items():
                if len(set(vehicle_ids)) > self.config.max_knock:
                    knock_violations.append(pickup_id)
            
            if knock_violations:
                print(f"⚠️  Knock Constraint Violations: {len(knock_violations)} terminals")
            else:
                print(f"✅ All knock constraints satisfied")
                
        else:
            print(f"❌ NO FEASIBLE SOLUTION FOUND")
            print(f"📊 Best Attempt: {len(solution.routes)} routes")
            print(f"📦 Unassigned Parcels: {len(solution.unassigned_parcels)}")
            
            if solution.unassigned_parcels:
                unassigned_sizes = [p.size for p in solution.unassigned_parcels]
                print(f"📦 Unassigned Sizes: min={min(unassigned_sizes)}, max={max(unassigned_sizes)}, total={sum(unassigned_sizes)}")
            
            # Analyze why infeasible
            time_violations = [
                r for r in solution.routes 
                if hasattr(r, 'total_duration') and not r.is_time_feasible(self.config.time_window_seconds)
            ]
            print(f"⏱️  Routes with Time Violations: {len(time_violations)}")
            
            knock_violations = 0
            for pickup_id, vehicle_ids in solution.pickup_assignments.items():
                if len(set(vehicle_ids)) > self.config.max_knock:
                    knock_violations += 1
            print(f"🚪 Terminals with Knock Violations: {knock_violations}")
        
        print("="*60)

    def _print_detailed_route_analysis(self, solution: Solution):
        """Print detailed analysis of each route"""
        print("\n" + "-"*50)
        print("DETAILED ROUTE ANALYSIS")
        print("-"*50)
        
        for i, route in enumerate(solution.routes, 1):
            utilization = (route.total_size / route.vehicle_spec.capacity) * 100
            duration_hours = getattr(route, 'total_duration', 0) / 3600
            distance_km = getattr(route, 'total_distance', 0) / 1000
            
            print(f"\nRoute {i} (Vehicle {route.vehicle_id}):")
            print(f"  Type: {route.vehicle_type.value}")
            print(f"  Parcels: {len(route.parcels)}")
            print(f"  Capacity: {route.total_size}/{route.vehicle_spec.capacity} ({utilization:.1f}%)")
            print(f"  Distance: {distance_km:.2f} km")
            print(f"  Duration: {duration_hours:.2f} hours")
            print(f"  Optimized: {'✅' if route.is_optimized else '❌'}")
            print(f"  Time Feasible: {'✅' if route.is_time_feasible(self.config.time_window_seconds) else '❌'}")
            print(f"  Pickup Terminals: {route.pickup_sequence}")
            print(f"  Deliveries: {len(route.delivery_sequence)}")

    def _print_solution_comparison(self, initial_solution: Solution, final_solution: Solution):
        """Compare initial vs final solution"""
        print("\n" + "-"*50)
        print("SOLUTION IMPROVEMENT ANALYSIS")
        print("-"*50)
        
        # Count vehicles by type
        def count_vehicles(sol):
            counts = {}
            for route in sol.routes:
                vtype = route.vehicle_type.value
                counts[vtype] = counts.get(vtype, 0) + 1
            return counts
        
        initial_vehicles = count_vehicles(initial_solution)
        final_vehicles = count_vehicles(final_solution)
        
        print(f"Routes: {len(initial_solution.routes)} → {len(final_solution.routes)} "
            f"({len(final_solution.routes) - len(initial_solution.routes):+d})")
        
        if hasattr(initial_solution, 'total_cost') and hasattr(final_solution, 'total_cost'):
            cost_improvement = ((initial_solution.total_cost - final_solution.total_cost) / initial_solution.total_cost) * 100
            print(f"Cost: {initial_solution.total_cost:.2f} → {final_solution.total_cost:.2f} "
                f"({cost_improvement:+.1f}%)")
        
        print(f"Initial vehicles: {initial_vehicles}")
        print(f"Final vehicles: {final_vehicles}")
        
        # Calculate optimization rate
        initial_optimized = sum(1 for r in initial_solution.routes if r.is_optimized)
        final_optimized = sum(1 for r in final_solution.routes if r.is_optimized)
        
        initial_opt_rate = initial_optimized / len(initial_solution.routes) * 100 if initial_solution.routes else 0
        final_opt_rate = final_optimized / len(final_solution.routes) * 100 if final_solution.routes else 0
        
        print(f"OR-Tools optimization rate: {initial_opt_rate:.1f}% → {final_opt_rate:.1f}%")


    def _print_solution_stats(self, solution: Solution, label: str):
        """Print solution statistics"""
        vehicle_counts = {}
        optimized_routes = 0
        for route in solution.routes:
            vtype = route.vehicle_type.value
            vehicle_counts[vtype] = vehicle_counts.get(vtype, 0) + 1
            if route.is_optimized:
                optimized_routes += 1

        print(f"{label} solution: {len(solution.routes)} routes, "
            f"cost: {solution.total_cost:.2f}, "
            f"vehicles: {vehicle_counts}, "
            f"optimized: {optimized_routes}/{len(solution.routes)}")

    def _accept_solution(self, new_solution: Solution, current_solution: Solution, 
                        best_solution: Solution) -> bool:
        """Determine whether to accept new solution"""
        if new_solution.is_feasible:
            if (not best_solution.is_feasible or 
                new_solution.total_cost < current_solution.total_cost * 1.05):
                return True
        else:
            # Accept infeasible solutions occasionally
            if self.random.random() < 0.05:
                return True
        return False

    def _create_emergency_solution(self, pickup_terminals: List[PickupTerminal]) -> Solution:
        """Create basic emergency solution when everything fails"""
        print("Creating emergency solution...")
        solution = Solution(pickup_terminals, self.config.vehicle_specs)
        
        # Use largest vehicle type for all parcels
        largest_vehicle = max(self.config.vehicle_specs.items(), 
                            key=lambda x: x[1].capacity)
        vehicle_type, vehicle_spec = largest_vehicle
        
        vehicle_counter = 1
        for terminal in pickup_terminals:
            for parcel in terminal.parcels:
                route = Route(
                    vehicle_id=vehicle_counter,
                    vehicle_type=vehicle_type,
                    vehicle_spec=vehicle_spec,
                    parcels=[parcel],
                    pickup_sequence=[terminal.pickup_id],
                    delivery_sequence=[parcel.delivery_location],
                )
                solution.add_route(route)
                vehicle_counter += 1
        
        solution.is_feasible = True
        return solution

    def solve_with_error_handling(
        self,
        pickup_terminals: List[PickupTerminal],
        distance_matrix: Dict,
        eta_matrix: Dict,
        max_iterations: int = 1000,
    ) -> Solution:
        """Solve with comprehensive error handling"""
        start_time = time.time()

        try:
            # Generate initial solution
            print("Generating initial solution with OR-Tools optimization...")
            current_solution = self.constructor.greedy_construction(
                pickup_terminals, distance_matrix, eta_matrix
            )

            # Apply initial vehicle optimization with error handling
            print("Optimizing initial vehicle assignments...")
            try:
                current_solution = self.operators.vehicle_consolidation_operator(
                    current_solution, distance_matrix, eta_matrix
                )
            except Exception as e:
                print(f"Warning: Initial vehicle optimization failed: {e}")
                # Continue with non-optimized solution

            best_solution = deepcopy(current_solution)

            # Print initial solution stats
            self._print_solution_stats(current_solution, "Initial")

            if not current_solution.is_feasible:
                print(f"Warning: Initial solution is not feasible!")
                print(f"  - Unassigned parcels: {len(current_solution.unassigned_parcels)}")

            # ALNS main loop with error handling
            successful_iterations = 0
            
            for iteration in range(max_iterations):
                try:
                    # Choose operator type with error handling
                    operator_choice = self.random.random()
                    repaired = None

                    if operator_choice < 0.25:
                        # Standard destroy-repair
                        try:
                            if self.random.random() < 0.5:
                                destroyed = self.operators.random_removal(current_solution)
                            else:
                                destroyed = self.operators.terminal_removal(current_solution)
                            repaired = self.operators.greedy_insertion(
                                destroyed, distance_matrix, eta_matrix
                            )
                        except Exception as e:
                            print(f"Iteration {iteration}: Destroy-repair failed: {e}")
                            continue

                    elif operator_choice < 0.5:
                        # Vehicle type optimization
                        try:
                            if self.random.random() < 0.4:
                                repaired = self.operators.vehicle_consolidation_operator(
                                    current_solution, distance_matrix, eta_matrix
                                )
                            elif self.random.random() < 0.7:
                                repaired = self.operators.vehicle_type_swap_operator(
                                    current_solution, distance_matrix, eta_matrix
                                )
                            else:
                                repaired = self.operators.route_splitting_operator(
                                    current_solution, distance_matrix, eta_matrix
                                )
                        except Exception as e:
                            print(f"Iteration {iteration}: Vehicle optimization failed: {e}")
                            continue
                    else:
                        # Combined operations
                        try:
                            if self.random.random() < 0.5:
                                destroyed = self.operators.random_removal(current_solution)
                            else:
                                destroyed = self.operators.terminal_removal(current_solution)

                            repaired = self.operators.greedy_insertion(
                                destroyed, distance_matrix, eta_matrix
                            )
                            repaired = self.operators.vehicle_consolidation_operator(
                                repaired, distance_matrix, eta_matrix
                            )
                        except Exception as e:
                            print(f"Iteration {iteration}: Combined operation failed: {e}")
                            continue

                    if repaired is None:
                        continue

                    # Evaluate solution with error handling
                    try:
                        repaired.calculate_cost_and_time(distance_matrix, eta_matrix)
                        repaired.is_feasible = (
                            len(repaired.unassigned_parcels) == 0
                            and repaired.validate_knock_constraints(self.config.max_knock)
                            and repaired.validate_time_window(self.config.time_window_seconds)
                        )
                    except Exception as e:
                        print(f"Iteration {iteration}: Solution evaluation failed: {e}")
                        continue

                    # Acceptance criterion
                    if self._accept_solution(repaired, current_solution, best_solution):
                        current_solution = repaired
                        if repaired.is_feasible and (not best_solution.is_feasible or 
                                                repaired.total_cost < best_solution.total_cost):
                            best_solution = deepcopy(repaired)
                            self._print_solution_stats(best_solution, f"Iteration {iteration}")

                    successful_iterations += 1

                except Exception as e:
                    print(f"Iteration {iteration}: Unexpected error: {e}")
                    continue

            print(f"Completed {successful_iterations}/{max_iterations} successful iterations")
            
            # Final optimization attempt
            if best_solution.is_feasible:
                try:
                    print("Applying final vehicle optimization...")
                    final_optimized = self.operators.vehicle_consolidation_operator(
                        best_solution, distance_matrix, eta_matrix
                    )
                    final_optimized.calculate_cost_and_time(distance_matrix, eta_matrix)

                    if (final_optimized.is_feasible and 
                        final_optimized.total_cost < best_solution.total_cost):
                        best_solution = final_optimized
                        print("Final optimization improved solution")
                except Exception as e:
                    print(f"Final optimization failed: {e}")

            processing_time = time.time() - start_time
            print(f"Solved in {processing_time:.2f} seconds")

            self._print_final_stats(best_solution)
            return best_solution

        except Exception as e:
            print(f"Critical error in solve method: {e}")
            # Return a basic feasible solution
            return self._create_emergency_solution(pickup_terminals)

    def solve(
        self,
        pickup_terminals: List[PickupTerminal],
        distance_matrix: Dict,
        eta_matrix: Dict,
        max_iterations: int = 1000,
    ) -> Solution:
        """Enhanced solve method with better logging and error handling"""
        start_time = time.time()

        # Generate initial solution
        print("Generating initial solution with OR-Tools optimization...")
        try:
            current_solution = self.constructor.greedy_construction(
                pickup_terminals, distance_matrix, eta_matrix
            )
        except Exception as e:
            print(f"Error in initial construction: {e}")
            return self._create_emergency_solution(pickup_terminals)

        # Store initial solution for comparison
        initial_solution = deepcopy(current_solution)

        # Apply initial vehicle optimization
        print("Optimizing initial vehicle assignments...")
        try:
            current_solution = self.operators.vehicle_consolidation_operator(
                current_solution, distance_matrix, eta_matrix
            )
        except Exception as e:
            print(f"Warning: Initial vehicle optimization failed: {e}")
            # Continue with unoptimized solution

        best_solution = deepcopy(current_solution)

        # Print initial stats
        self._print_solution_stats(current_solution, "Initial")

        if not current_solution.is_feasible:
            print(f"Warning: Initial solution is not feasible!")
            print(f"  - Unassigned parcels: {len(current_solution.unassigned_parcels)}")
            time_violations = [
                r for r in current_solution.routes
                if not r.is_time_feasible(self.config.time_window_seconds)
            ]
            print(f"  - Time window violations: {len(time_violations)}")

        # ALNS main loop with enhanced error handling
        successful_iterations = 0
        last_improvement = 0
        
        for iteration in range(max_iterations):
            try:
                # Choose operator type
                operator_choice = self.random.random()
                repaired = None

                if operator_choice < 0.25:
                    # Standard destroy-repair
                    try:
                        if self.random.random() < 0.5:
                            destroyed = self.operators.random_removal(current_solution)
                        else:
                            destroyed = self.operators.terminal_removal(current_solution)
                        repaired = self.operators.greedy_insertion(
                            destroyed, distance_matrix, eta_matrix
                        )
                    except Exception as e:
                        print(f"Iteration {iteration}: Destroy-repair failed: {e}")
                        continue

                elif operator_choice < 0.5:
                    # Vehicle type optimization
                    try:
                        if self.random.random() < 0.4:
                            repaired = self.operators.vehicle_consolidation_operator(
                                current_solution, distance_matrix, eta_matrix
                            )
                        elif self.random.random() < 0.7:
                            repaired = self.operators.vehicle_type_swap_operator(
                                current_solution, distance_matrix, eta_matrix
                            )
                        else:
                            repaired = self.operators.route_splitting_operator(
                                current_solution, distance_matrix, eta_matrix
                            )
                    except Exception as e:
                        print(f"Iteration {iteration}: Vehicle optimization failed: {e}")
                        continue
                else:
                    # Combined destroy-repair + vehicle optimization
                    try:
                        if self.random.random() < 0.5:
                            destroyed = self.operators.random_removal(current_solution)
                        else:
                            destroyed = self.operators.terminal_removal(current_solution)

                        repaired = self.operators.greedy_insertion(
                            destroyed, distance_matrix, eta_matrix
                        )
                        repaired = self.operators.vehicle_consolidation_operator(
                            repaired, distance_matrix, eta_matrix
                        )
                    except Exception as e:
                        print(f"Iteration {iteration}: Combined operation failed: {e}")
                        continue

                if repaired is None:
                    continue

                # Evaluate solution
                try:
                    repaired.calculate_cost_and_time(distance_matrix, eta_matrix)
                    repaired.is_feasible = (
                        len(repaired.unassigned_parcels) == 0
                        and repaired.validate_knock_constraints(self.config.max_knock)
                        and repaired.validate_time_window(self.config.time_window_seconds)
                    )
                except Exception as e:
                    print(f"Iteration {iteration}: Solution evaluation failed: {e}")
                    continue

                # Acceptance criterion
                accept_solution = False

                if repaired.is_feasible:
                    if (
                        not best_solution.is_feasible
                        or repaired.total_cost < best_solution.total_cost
                    ):
                        best_solution = deepcopy(repaired)
                        current_solution = repaired
                        accept_solution = True
                        last_improvement = iteration

                        # Print improvement details
                        self._print_solution_stats(best_solution, f"Iteration {iteration}")

                    elif repaired.total_cost < current_solution.total_cost * 1.05:
                        current_solution = repaired
                        accept_solution = True
                else:
                    # Accept infeasible solutions occasionally
                    if self.random.random() < 0.05:
                        current_solution = repaired
                        accept_solution = True

                successful_iterations += 1

                # Early termination if no improvement for a while
                if iteration - last_improvement > max_iterations // 4:
                    print(f"Early termination: No improvement for {iteration - last_improvement} iterations")
                    break

            except Exception as e:
                print(f"Iteration {iteration}: Unexpected error: {e}")
                continue

        processing_time = time.time() - start_time
        print(f"Completed {successful_iterations}/{iteration + 1} successful iterations")

        # Final vehicle optimization
        if best_solution.is_feasible:
            print("Applying final vehicle optimization...")
            try:
                final_optimized = self.operators.vehicle_consolidation_operator(
                    best_solution, distance_matrix, eta_matrix
                )
                final_optimized.calculate_cost_and_time(distance_matrix, eta_matrix)

                if (
                    final_optimized.is_feasible
                    and final_optimized.total_cost < best_solution.total_cost
                ):
                    best_solution = final_optimized
                    print("Final optimization improved solution")
            except Exception as e:
                print(f"Final optimization failed: {e}")

        print(f"Total processing time: {processing_time:.2f} seconds")

        # Print comprehensive final statistics
        self._print_final_stats(best_solution)
        
        # Print solution comparison
        self._print_solution_comparison(initial_solution, best_solution)

        return best_solution


    def format_output(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Dict:
        """Format solution to match expected output format with OR-Tools analysis"""
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
                # Find route for this vehicle
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
        optimized_routes_count = 0

        for route in solution.routes:
            route_distance = getattr(route, "total_distance", 0)
            route_duration = getattr(route, "total_duration", 0)
            total_duration_seconds += route_duration

            if route.is_optimized:
                optimized_routes_count += 1

            # Build physical route from optimized sequence
            physical_route = []
            route_indices = []

            for i, (action, location) in enumerate(route.route_sequence):
                physical_route.append((location[0], location[1]))
                route_indices.append(i)

            formatted_route = {
                "vehicle_id": route.vehicle_id,
                "vehicle_type": route.vehicle_type.value,
                "parcels": [float(p.id) for p in route.parcels],
                "route_indices": route_indices,
                "physical_route": physical_route,
                "route_sequence": route.route_sequence,  # Include optimized sequence
                "num_stops": len(route.route_sequence),
                "total_duration_seconds": int(route_duration),
                "total_distance_m": int(route_distance),
                "total_cost_kt": (route_distance * route.vehicle_spec.cost_per_km)
                / 1000,
                "capacity_used": route.total_size,
                "vehicle_capacity": route.vehicle_spec.capacity,
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
                "total_duration_seconds": total_duration_seconds,
                "max_route_duration_seconds": max(
                    (getattr(r, "total_duration", 0) for r in solution.routes),
                    default=0,
                ),
                "time_window_feasible": len(time_window_violations) == 0,
                "time_window_violations": len(time_window_violations),
                "ortools_optimized_routes": optimized_routes_count,
                "ortools_optimization_rate": (
                    round(optimized_routes_count / len(solution.routes) * 100, 1)
                    if solution.routes
                    else 0
                ),
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
                "ortools_analysis": {
                    "optimized_routes": optimized_routes_count,
                    "total_routes": len(solution.routes),
                    "optimization_rate": (
                        round(optimized_routes_count / len(solution.routes) * 100, 1)
                        if solution.routes
                        else 0
                    ),
                    "available": ORTOOLS_AVAILABLE,
                },
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


class ORToolsRouteOptimizer:
    """OR-Tools integration for route sequence optimization"""
    
    def __init__(self, config: ProblemConfig):
        self.config = config
        
    def optimize_route_sequence(self, route: Route, distance_matrix: Dict, 
                              eta_matrix: Dict) -> Route:
        """
        Optimize pickup and delivery sequence for a single route using OR-Tools
        
        Args:
            route: Route to optimize
            distance_matrix: Distance matrix
            eta_matrix: ETA matrix
            
        Returns:
            Optimized route
        """
        if not ORTOOLS_AVAILABLE:
            return self._fallback_route_optimization(route, distance_matrix, eta_matrix)
        
        # For small routes, use fallback (OR-Tools overhead not worth it)
        if len(route.parcels) <= 2:
            return self._fallback_route_optimization(route, distance_matrix, eta_matrix)
        
        try:
            # Build locations list and pickup-delivery pairs
            locations = []
            location_to_index = {}
            pickup_delivery_pairs = []
            
            # Add depot (first pickup location as starting point)
            depot_location = route.parcels[0].pickup_location if route.parcels else (0, 0)
            locations.append(depot_location)
            location_to_index[depot_location] = 0
            
            # Add unique pickup locations
            pickup_locations_set = set()
            for parcel in route.parcels:
                pickup_locations_set.add(parcel.pickup_location)
            
            for pickup_loc in pickup_locations_set:
                if pickup_loc not in location_to_index:
                    location_to_index[pickup_loc] = len(locations)
                    locations.append(pickup_loc)
            
            # Add delivery locations and create pickup-delivery pairs
            for parcel in route.parcels:
                pickup_loc = parcel.pickup_location
                delivery_loc = parcel.delivery_location
                
                # Add delivery location if not already added
                if delivery_loc not in location_to_index:
                    location_to_index[delivery_loc] = len(locations)
                    locations.append(delivery_loc)
                
                # Create pickup-delivery pair
                pickup_idx = location_to_index[pickup_loc]
                delivery_idx = location_to_index[delivery_loc]
                
                # Only add unique pairs (avoid duplicates for same pickup-delivery)
                if (pickup_idx, delivery_idx) not in pickup_delivery_pairs:
                    pickup_delivery_pairs.append((pickup_idx, delivery_idx))
            
            # Ensure we have enough locations
            if len(locations) < 2:
                return self._fallback_route_optimization(route, distance_matrix, eta_matrix)
            
            # Create OR-Tools model
            manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
            routing = pywrapcp.RoutingModel(manager)
            
            # Create distance callback with scaling for OR-Tools
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                if from_node >= len(locations) or to_node >= len(locations):
                    return 0
                from_loc = locations[from_node]
                to_loc = locations[to_node]
                distance = distance_matrix.get(from_loc, {}).get(to_loc, 0)
                # Scale down distance for OR-Tools (it works better with smaller numbers)
                return max(1, int(distance / 100))  # Convert to hectometers
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            # Simplified approach: just optimize the TSP without complex constraints
            # that might make the problem infeasible
            
            # Only add time constraint (much more relaxed)
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                if from_node >= len(locations) or to_node >= len(locations):
                    return 0
                from_loc = locations[from_node]
                to_loc = locations[to_node]
                time_val = eta_matrix.get(from_loc, {}).get(to_loc, 300)  # Default 5 min
                return max(1, int(time_val / 60))  # Convert to minutes
            
            time_callback_index = routing.RegisterTransitCallback(time_callback)
            routing.AddDimension(
                time_callback_index,
                self.config.time_window_seconds // 60,  # Allow slack in minutes
                self.config.time_window_seconds // 60,  # Max time in minutes
                True,  # Start cumul to zero
                'Time'
            )
            
            # Simpler search parameters for better success rate
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
            )
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
            )
            search_parameters.time_limit.seconds = 10  # Shorter time limit
            search_parameters.solution_limit = 1  # Just find one solution
            
            # Solve
            solution = routing.SolveWithParameters(search_parameters)
            
            if solution:
                # Extract optimized sequence
                optimized_route = self._extract_optimized_route_simple(
                    route, solution, routing, manager, locations
                )
                return optimized_route
            else:
                # OR-Tools couldn't solve - use fallback
                return self._fallback_route_optimization(route, distance_matrix, eta_matrix)
                
        except Exception as e:
            # Any error - use fallback
            return self._fallback_route_optimization(route, distance_matrix, eta_matrix)
    
    def _extract_optimized_route_simple(self, original_route: Route, solution, routing, 
                                       manager, locations: List[Tuple[float, float]]) -> Route:
        """Extract optimized route from OR-Tools solution (simplified approach)"""
        optimized_route = deepcopy(original_route)
        
        # Get the route sequence from OR-Tools solution
        route_sequence = []
        index = routing.Start(0)
        visited_locations = []
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node < len(locations):
                location = locations[node]
                visited_locations.append(location)
            index = solution.Value(routing.NextVar(index))
        
        # Now build the route sequence based on the optimized order
        # We need to determine pickup vs delivery for each location
        pickup_locations = set()
        delivery_locations = set()
        
        for parcel in original_route.parcels:
            pickup_locations.add(parcel.pickup_location)
            delivery_locations.add(parcel.delivery_location)
        
        # Build sequence respecting the OR-Tools order
        for location in visited_locations:
            if location in pickup_locations:
                route_sequence.append(('pickup', location))
            elif location in delivery_locations:
                route_sequence.append(('delivery', location))
        
        # Ensure we have all deliveries (add any missing ones at the end)
        for parcel in original_route.parcels:
            delivery_location = parcel.delivery_location
            if not any(loc == delivery_location and action == 'delivery' 
                      for action, loc in route_sequence):
                route_sequence.append(('delivery', delivery_location))
        
        # Update route with optimized sequence
        optimized_route.route_sequence = route_sequence
        optimized_route.is_optimized = True
        
        return optimized_route
    
    def _extract_optimized_route(self, original_route: Route, solution, routing, 
                               manager, locations: List[Tuple[float, float]]) -> Route:
        """Extract optimized route from OR-Tools solution"""
        optimized_route = deepcopy(original_route)
        
        # Get the route sequence
        route_sequence = []
        index = routing.Start(0)
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            location = locations[node]
            
            # Determine if this is a pickup or delivery
            is_pickup = False
            is_delivery = False
            
            # Check if this location is a pickup
            for pickup_id in original_route.pickup_sequence:
                pickup_location = original_route.parcels[0].pickup_location if original_route.parcels else (0, 0)
                if pickup_location == location:
                    is_pickup = True
                    break
            
            # Check if this location is a delivery
            for parcel in original_route.parcels:
                if parcel.delivery_location == location:
                    is_delivery = True
                    break
            
            # Add to sequence
            if is_pickup:
                route_sequence.append(('pickup', location))
            elif is_delivery:
                route_sequence.append(('delivery', location))
            
            index = solution.Value(routing.NextVar(index))
        
        # Update route with optimized sequence
        optimized_route.route_sequence = route_sequence
        optimized_route.is_optimized = True
        
        return optimized_route
    
    def _fallback_route_optimization(self, route: Route, distance_matrix: Dict, 
                                   eta_matrix: Dict) -> Route:
        """Fallback optimization when OR-Tools is not available"""
        optimized_route = deepcopy(route)
        
        # Simple TSP optimization for pickup sequence
        if len(route.pickup_sequence) > 1:
            optimized_pickups = self._simple_tsp_optimization(
                route.pickup_sequence, distance_matrix, route.parcels[0].pickup_location
            )
            optimized_route.pickup_sequence = optimized_pickups
        
        # Simple TSP optimization for delivery sequence
        if len(route.delivery_sequence) > 1:
            optimized_deliveries = self._simple_tsp_optimization(
                route.delivery_sequence, distance_matrix
            )
            optimized_route.delivery_sequence = optimized_deliveries
        
        # Rebuild route sequence
        route_sequence = []
        for pickup_id in optimized_route.pickup_sequence:
            pickup_location = route.parcels[0].pickup_location if route.parcels else (0, 0)
            route_sequence.append(('pickup', pickup_location))
        
        for delivery_loc in optimized_route.delivery_sequence:
            route_sequence.append(('delivery', delivery_loc))
        
        optimized_route.route_sequence = route_sequence
        optimized_route.is_optimized = True
        
        return optimized_route
    
    def _simple_tsp_optimization(self, locations, distance_matrix, start_location=None):
        """Simple TSP optimization using nearest neighbor heuristic"""
        if len(locations) <= 1:
            return locations
        
        if start_location is None:
            start_location = locations[0]
        
        unvisited = locations.copy()
        if start_location in unvisited:
            unvisited.remove(start_location)
        
        route = [start_location] if start_location in locations else []
        current = start_location
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: distance_matrix.get(current, {}).get(x, float('inf')))
            route.append(nearest)
            current = nearest
            unvisited.remove(nearest)
        
        return route


class VehicleOptimizer:
    """Handles global vehicle type optimization and fleet management"""

    # def __init__(self, config: ProblemConfig):
    #     self.config = config
    #     self.vehicle_specs = config.vehicle_specs
    #     self.route_optimizer = ORToolsRouteOptimizer(config)
    def __init__(self, config: ProblemConfig):
        self.config = config
        self.vehicle_specs = config.vehicle_specs
        self.route_optimizer = ORToolsRouteOptimizer(config)
        self.optimization_cache = OptimizationCache()  # Use enhanced cache
        self.verbose_logging = False

    def _solutions_are_equivalent(self, solution1: Solution, solution2: Solution) -> bool:
        """Check if two solutions are equivalent"""
        if len(solution1.routes) != len(solution2.routes):
            return False
        
        # Quick check: same total cost within small tolerance
        if abs(solution1.total_cost - solution2.total_cost) < 0.01:
            return True
        
        return False

    def optimize_vehicle_assignments_with_stagnation_detection(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Globally optimize vehicle type assignments with stagnation detection"""
        
        # Quick exit for single route scenarios
        if len(solution.routes) <= 1:
            if self.verbose_logging:
                print(f"Skipping vehicle optimization: only {len(solution.routes)} route(s)")
            return solution
        
        # Check if we've seen this solution before
        if self.optimization_cache.is_solution_seen(solution):
            if self.verbose_logging:
                print("Skipping vehicle optimization: solution already processed")
            return solution
        
        # Record this solution
        self.optimization_cache.add_solution(solution)
        
        new_solution = deepcopy(solution)
        original_cost = solution.total_cost
        
        if self.verbose_logging:
            print(f"Starting vehicle optimization with {len(new_solution.routes)} routes")

        # Group routes by pickup terminals for consolidation opportunities
        terminal_routes = self._group_routes_by_terminals(new_solution.routes)
        
        improved_routes = []
        any_improvement = False

        for terminal_id, routes in terminal_routes.items():
            operation_key = f"terminal_{terminal_id}_{len(routes)}_routes"
            
            # Skip if we've tried this operation too many times without improvement
            if self.optimization_cache.should_skip_operation(operation_key):
                if self.verbose_logging:
                    print(f"Skipping terminal {terminal_id}: too many failed attempts")
                improved_routes.extend(routes)
                continue
            
            if self.verbose_logging:
                print(f"Optimizing terminal {terminal_id} with {len(routes)} routes")
            
            if len(routes) == 1:
                # Single route - check if optimization is worth it
                route = routes[0]
                
                # Skip if route is already well-optimized
                if self._is_route_well_optimized(route):
                    if self.verbose_logging:
                        print(f"  Skipping single route: already well-optimized")
                    improved_routes.extend(routes)
                    continue
                
                if self.verbose_logging:
                    print(f"  Single route optimization")
                optimized = self._optimize_single_route_vehicle(
                    route, distance_matrix, eta_matrix
                )
                
                # Check if optimization actually improved anything
                if self._routes_are_equivalent(routes, optimized):
                    self.optimization_cache.add_no_improvement_attempt(operation_key)
                    if self.verbose_logging:
                        print(f"  No improvement from single route optimization")
                else:
                    any_improvement = True
                    
                improved_routes.extend(optimized)
            else:
                # Multiple routes - try consolidation
                if self.verbose_logging:
                    print(f"  Multi-route consolidation")
                optimized = self._optimize_multi_route_consolidation_optimized(
                    routes, terminal_id, distance_matrix, eta_matrix
                )
                
                # Check if consolidation actually improved anything
                if self._routes_are_equivalent(routes, optimized):
                    self.optimization_cache.add_no_improvement_attempt(operation_key)
                    if self.verbose_logging:
                        print(f"  No improvement from consolidation")
                else:
                    any_improvement = True
                    
                improved_routes.extend(optimized)

        new_solution.routes = improved_routes
        new_solution._rebuild_assignments()
        
        # Calculate new cost to verify improvement
        new_solution.calculate_cost_and_time(distance_matrix, eta_matrix)
        
        improvement_pct = ((original_cost - new_solution.total_cost) / original_cost) * 100 if original_cost > 0 else 0
        
        if self.verbose_logging or improvement_pct > 1:  # Only log significant improvements
            print(f"Vehicle optimization complete: {len(solution.routes)} -> {len(new_solution.routes)} routes")
            if improvement_pct > 1:
                print(f"Cost improvement: {improvement_pct:.1f}%")
        
        return new_solution

    def _is_route_well_optimized(self, route: Route) -> bool:
        """Check if a route is already well-optimized"""
        # Check utilization
        utilization = route.total_size / route.vehicle_spec.capacity
        
        # Well-optimized criteria
        return (
            route.is_optimized and  # Already OR-Tools optimized
            0.6 <= utilization <= 1.0 and  # Good utilization
            route.is_time_feasible(self.config.time_window_seconds) and  # Time feasible
            route.vehicle_type == self._get_best_vehicle_type_for_size(route.total_size)  # Optimal vehicle type
        )

    def _get_best_vehicle_type_for_size(self, total_size: int) -> VehicleType:
        """Get the most cost-effective vehicle type for given size"""
        best_type = None
        best_cost_efficiency = float("inf")
        
        for vehicle_type, spec in self.vehicle_specs.items():
            if spec.capacity >= total_size:
                cost_efficiency = spec.cost_per_km / spec.capacity
                if cost_efficiency < best_cost_efficiency:
                    best_cost_efficiency = cost_efficiency
                    best_type = vehicle_type
        
        return best_type or VehicleType.CARBOX 

    def _routes_are_equivalent(self, routes1: List[Route], routes2: List[Route]) -> bool:
        """Check if two route lists are equivalent"""
        if len(routes1) != len(routes2):
            return False
        
        # Sort routes by vehicle_id for comparison
        sorted_routes1 = sorted(routes1, key=lambda r: r.vehicle_id)
        sorted_routes2 = sorted(routes2, key=lambda r: r.vehicle_id)
        
        for r1, r2 in zip(sorted_routes1, sorted_routes2):
            if (r1.vehicle_type != r2.vehicle_type or 
                r1.total_size != r2.total_size or
                len(r1.parcels) != len(r2.parcels)):
                return False
        
        return True

    def _try_consolidation_with_vehicle_optimized(
        self,
        routes: List[Route],
        vehicle_type: VehicleType,
        vehicle_spec: VehicleSpec,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[List[Route]]:
        """Optimized consolidation with caching and early exits"""
        total_size = sum(route.total_size for route in routes)
        
        # Quick capacity check
        if total_size > vehicle_spec.capacity:
            return None
        
        # Create cache key
        route_sizes = tuple(sorted([r.total_size for r in routes]))
        terminal_id = routes[0].pickup_sequence[0] if routes[0].pickup_sequence else 0
        
        # Check cache for failed attempts
        if self.optimization_cache.is_failed_consolidation(terminal_id, route_sizes, vehicle_type.value):
            if self.verbose_logging:
                print(f"  Skipping {vehicle_type.value}: cached as failed")
            return None
        
        # Generate route signature for time window check
        route_signature = f"{vehicle_type.value}_{total_size}_{len(routes)}"
        if self.optimization_cache.has_time_violation(route_signature):
            if self.verbose_logging:
                print(f"  Skipping {vehicle_type.value}: cached time violation")
            return None
        
        # Combine all parcels
        all_parcels = []
        pickup_sequences = set()
        
        for route in routes:
            all_parcels.extend(route.parcels)
            pickup_sequences.update(route.pickup_sequence)
        
        # Double-check total size calculation
        calculated_total_size = sum(p.size for p in all_parcels)
        if calculated_total_size > vehicle_spec.capacity:
            if self.verbose_logging:
                print(f"Warning: Calculated size {calculated_total_size} exceeds {vehicle_type.value} capacity {vehicle_spec.capacity}")
            # Cache this failure
            self.optimization_cache.add_failed_consolidation(terminal_id, route_sizes, vehicle_type.value)
            return None
        
        try:
            # Create consolidated route
            consolidated_route = Route(
                vehicle_id=routes[0].vehicle_id,
                vehicle_type=vehicle_type,
                vehicle_spec=vehicle_spec,
                parcels=all_parcels,
                pickup_sequence=list(pickup_sequences),
                delivery_sequence=[p.delivery_location for p in all_parcels],
            )
        except ValueError as e:
            if self.verbose_logging:
                print(f"Error creating consolidated route for {vehicle_type.value}: {e}")
            # Cache this failure
            self.optimization_cache.add_failed_consolidation(terminal_id, route_sizes, vehicle_type.value)
            return None
        
        # Quick time estimate before OR-Tools optimization
        estimated_duration = self._estimate_route_duration(consolidated_route, distance_matrix, eta_matrix)
        if estimated_duration > self.config.time_window_seconds * 1.1:  # 10% buffer
            if self.verbose_logging:
                print(f"Consolidated route for {vehicle_type.value} estimated to violate time window")
            # Cache time violation
            self.optimization_cache.add_time_violation(route_signature)
            self.optimization_cache.add_failed_consolidation(terminal_id, route_sizes, vehicle_type.value)
            return None
        
        # Optimize route sequence with OR-Tools
        optimized_consolidated = self.route_optimizer.optimize_route_sequence(
            consolidated_route, distance_matrix, eta_matrix
        )
        
        # Calculate metrics
        optimized_consolidated.total_distance, optimized_consolidated.total_duration = (
            self._calculate_route_metrics(
                optimized_consolidated, distance_matrix, eta_matrix
            )
        )
        
        # Check time feasibility
        if not optimized_consolidated.is_time_feasible(self.config.time_window_seconds):
            if self.verbose_logging:
                print(f"Consolidated route for {vehicle_type.value} violates time window")
            # Cache both failures
            self.optimization_cache.add_time_violation(route_signature)
            self.optimization_cache.add_failed_consolidation(terminal_id, route_sizes, vehicle_type.value)
            return None
        
        if self.verbose_logging:
            print(f"Successfully consolidated {len(routes)} routes into 1 {vehicle_type.value} vehicle")
        return [optimized_consolidated]

    def _estimate_route_duration(self, route: Route, distance_matrix: Dict, eta_matrix: Dict) -> int:
        """Quick estimation of route duration without full optimization"""
        if not route.parcels:
            return 0
    
        # Simple estimation: pickup + all deliveries in sequence
        total_duration = 0
        current_location = route.parcels[0].pickup_location
        
        # Add deliveries
        for parcel in route.parcels:
            delivery_location = parcel.delivery_location
            travel_time = eta_matrix.get(current_location, {}).get(delivery_location, 300)  # 5 min default
            total_duration += travel_time
            current_location = delivery_location
        
        return total_duration

    def _optimize_multi_route_consolidation_optimized(
        self,
        routes: List[Route],
        terminal_id: int,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> List[Route]:
        """Optimized multi-route consolidation with early exits"""
        if len(routes) <= 1:
            return routes
        
        # Calculate current cost once
        current_cost = sum(
            self._calculate_route_total_cost(r, distance_matrix, eta_matrix)
            for r in routes
        )
        
        best_routes = routes
        best_cost = current_cost
        
        if self.verbose_logging:
            print(f"Trying to consolidate {len(routes)} routes for terminal {terminal_id}")
        
        total_size_needed = sum(route.total_size for route in routes)
        if self.verbose_logging:
            print(f"Total size needed: {total_size_needed}")
        
        # Quick check: if all routes together violate the largest vehicle capacity, skip
        max_capacity = max(spec.capacity for spec in self.vehicle_specs.values())
        if total_size_needed > max_capacity:
            if self.verbose_logging:
                print(f"  Total size {total_size_needed} exceeds maximum vehicle capacity {max_capacity}")
            return routes
        
        # Try consolidation with each vehicle type (sorted by capacity)
        sorted_vehicle_types = sorted(
            self.vehicle_specs.items(),
            key=lambda x: x[1].capacity,
            reverse=True
        )
        
        consolidation_attempted = False
        for vehicle_type, vehicle_spec in sorted_vehicle_types:
            if total_size_needed > vehicle_spec.capacity:
                continue
                
            consolidation_attempted = True
            if self.verbose_logging:
                print(f"Trying consolidation with {vehicle_type.value} (capacity: {vehicle_spec.capacity})")
            
            consolidated_routes = self._try_consolidation_with_vehicle_optimized(
                routes, vehicle_type, vehicle_spec, distance_matrix, eta_matrix
            )
            
            if consolidated_routes:
                consolidated_cost = sum(
                    self._calculate_route_total_cost(r, distance_matrix, eta_matrix)
                    for r in consolidated_routes
                )
                
                if self.verbose_logging:
                    print(f"  Consolidation successful: cost {consolidated_cost:.2f} vs current {best_cost:.2f}")
                
                if consolidated_cost < best_cost:
                    best_routes = consolidated_routes
                    best_cost = consolidated_cost
                    if self.verbose_logging:
                        print(f"  New best consolidation found!")
                    break  # Found improvement, stop trying other vehicle types
        
        # Only try partial consolidations if no full consolidation worked and we have many routes
        if best_routes == routes and len(routes) > 2 and consolidation_attempted:
            if self.verbose_logging:
                print("Trying partial consolidations...")
            
            # Try only 2-route combinations (most likely to succeed)
            partial_consolidations = self._try_partial_consolidations_optimized(
                routes, 2, distance_matrix, eta_matrix
            )
            
            if partial_consolidations:
                partial_cost = sum(
                    self._calculate_route_total_cost(r, distance_matrix, eta_matrix)
                    for r in partial_consolidations
                )
                
                if partial_cost < best_cost:
                    best_routes = partial_consolidations
                    best_cost = partial_cost
                    if self.verbose_logging:
                        print(f"  Partial consolidation improved cost to {partial_cost:.2f}")
        
        return best_routes
    
    def _try_partial_consolidations_optimized(
        self,
        routes: List[Route],
        combo_size: int,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[List[Route]]:
        """Optimized partial consolidation with limited attempts"""
        from itertools import combinations
        
        if combo_size >= len(routes):
            return None
        
        best_combination = None
        best_cost_saving = 0
        max_attempts = 5  # Limit number of combinations to try
        attempts = 0
        
        # Try combinations sorted by total size (smaller first, more likely to succeed)
        combinations_list = list(combinations(routes, combo_size))
        combinations_list.sort(key=lambda combo: sum(r.total_size for r in combo))
        
        for route_combo in combinations_list[:max_attempts]:
            attempts += 1
            remaining_routes = [r for r in routes if r not in route_combo]
            combo_total_size = sum(route.total_size for route in route_combo)
            
            # Try only the most suitable vehicle type
            suitable_vehicle = None
            for vehicle_type, vehicle_spec in sorted(self.vehicle_specs.items(), 
                                                key=lambda x: x[1].capacity):
                if combo_total_size <= vehicle_spec.capacity:
                    suitable_vehicle = (vehicle_type, vehicle_spec)
                    break
            
            if not suitable_vehicle:
                continue
                
            vehicle_type, vehicle_spec = suitable_vehicle
            consolidated = self._try_consolidation_with_vehicle_optimized(
                list(route_combo),
                vehicle_type,
                vehicle_spec,
                distance_matrix,
                eta_matrix,
            )
            
            if consolidated:
                # Calculate cost saving
                original_cost = sum(
                    self._calculate_route_total_cost(r, distance_matrix, eta_matrix)
                    for r in route_combo
                )
                new_cost = self._calculate_route_total_cost(
                    consolidated[0], distance_matrix, eta_matrix
                )
                cost_saving = original_cost - new_cost
                
                if cost_saving > best_cost_saving:
                    best_cost_saving = cost_saving
                    best_combination = consolidated + remaining_routes
                    break  # Found improvement, stop trying
        
        return best_combination

    def optimize_vehicle_assignments(self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict) -> Solution:
        """Globally optimize vehicle type assignments with stagnation detection"""
        
        # Quick exit for single route scenarios
        if len(solution.routes) <= 1:
            if self.verbose_logging:
                print(f"Skipping vehicle optimization: only {len(solution.routes)} route(s)")
            return solution
        
        # Check if we've seen this solution before
        if self.optimization_cache.is_solution_seen(solution):
            if self.verbose_logging:
                print("Skipping vehicle optimization: solution already processed")
            return solution
        
        # Record this solution
        self.optimization_cache.add_solution(solution)
        
        new_solution = deepcopy(solution)
        original_cost = solution.total_cost
        
        if self.verbose_logging:
            print(f"Starting vehicle optimization with {len(new_solution.routes)} routes")

        # Group routes by pickup terminals for consolidation opportunities
        terminal_routes = self._group_routes_by_terminals(new_solution.routes)
        
        improved_routes = []
        any_improvement = False

        for terminal_id, routes in terminal_routes.items():
            operation_key = f"terminal_{terminal_id}_{len(routes)}_routes"
            
            # Skip if we've tried this operation too many times without improvement
            if self.optimization_cache.should_skip_operation(operation_key):
                if self.verbose_logging:
                    print(f"Skipping terminal {terminal_id}: too many failed attempts")
                improved_routes.extend(routes)
                continue
            
            if self.verbose_logging:
                print(f"Optimizing terminal {terminal_id} with {len(routes)} routes")
            
            if len(routes) == 1:
                # Single route - check if optimization is worth it
                route = routes[0]
                
                # Skip if route is already well-optimized
                if self._is_route_well_optimized(route):
                    if self.verbose_logging:
                        print(f"  Skipping single route: already well-optimized")
                    improved_routes.extend(routes)
                    continue
                
                if self.verbose_logging:
                    print(f"  Single route optimization")
                optimized = self._optimize_single_route_vehicle(
                    route, distance_matrix, eta_matrix
                )
                
                # Check if optimization actually improved anything
                if self._routes_are_equivalent(routes, optimized):
                    self.optimization_cache.add_no_improvement_attempt(operation_key)
                    if self.verbose_logging:
                        print(f"  No improvement from single route optimization")
                else:
                    any_improvement = True
                    
                improved_routes.extend(optimized)
            else:
                # Multiple routes - try consolidation
                if self.verbose_logging:
                    print(f"  Multi-route consolidation")
                optimized = self._optimize_multi_route_consolidation_optimized(
                    routes, terminal_id, distance_matrix, eta_matrix
                )
                
                # Check if consolidation actually improved anything
                if self._routes_are_equivalent(routes, optimized):
                    self.optimization_cache.add_no_improvement_attempt(operation_key)
                    if self.verbose_logging:
                        print(f"  No improvement from consolidation")
                else:
                    any_improvement = True
                    
                improved_routes.extend(optimized)

        new_solution.routes = improved_routes
        new_solution._rebuild_assignments()
        
        # Calculate new cost to verify improvement
        new_solution.calculate_cost_and_time(distance_matrix, eta_matrix)
        
        improvement_pct = ((original_cost - new_solution.total_cost) / original_cost) * 100 if original_cost > 0 else 0
        
        if self.verbose_logging or improvement_pct > 1:  # Only log significant improvements
            print(f"Vehicle optimization complete: {len(solution.routes)} -> {len(new_solution.routes)} routes")
            if improvement_pct > 1:
                print(f"Cost improvement: {improvement_pct:.1f}%")
        
        return new_solution

    def _group_routes_by_terminals(self, routes: List[Route]) -> Dict[int, List[Route]]:
        """Group routes by their pickup terminals"""
        terminal_routes = {}

        for route in routes:
            # Use first pickup as primary terminal (simplified)
            primary_terminal = route.pickup_sequence[0] if route.pickup_sequence else 0

            if primary_terminal not in terminal_routes:
                terminal_routes[primary_terminal] = []
            terminal_routes[primary_terminal].append(route)

        return terminal_routes

    def _optimize_single_route_vehicle(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> List[Route]:
        """Optimize vehicle type for a single route with OR-Tools sequence optimization"""
        current_cost = self._calculate_route_total_cost(
            route, distance_matrix, eta_matrix
        )
        best_route = route
        best_cost = current_cost

        # Try each vehicle type
        for vehicle_type, vehicle_spec in self.vehicle_specs.items():
            if vehicle_spec.capacity < route.total_size:
                continue  # Can't fit

            # Create test route with different vehicle
            test_route = deepcopy(route)
            test_route.vehicle_type = vehicle_type
            test_route.vehicle_spec = vehicle_spec

            # Optimize route sequence with OR-Tools
            optimized_test_route = self.route_optimizer.optimize_route_sequence(
                test_route, distance_matrix, eta_matrix
            )

            # Recalculate metrics
            optimized_test_route.total_distance, optimized_test_route.total_duration = (
                self._calculate_route_metrics(
                    optimized_test_route, distance_matrix, eta_matrix
                )
            )

            # Check time feasibility
            if not optimized_test_route.is_time_feasible(
                self.config.time_window_seconds
            ):
                continue

            # Calculate cost
            test_cost = self._calculate_route_total_cost(
                optimized_test_route, distance_matrix, eta_matrix
            )

            if test_cost < best_cost:
                best_route = optimized_test_route
                best_cost = test_cost

        return [best_route]

    def _optimize_multi_route_consolidation(
        self,
        routes: List[Route],
        terminal_id: int,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> List[Route]:
        """Try to consolidate multiple routes into fewer, larger vehicles with OR-Tools optimization"""
        if len(routes) <= 1:
            return routes

        # Current total cost
        current_cost = sum(
            self._calculate_route_total_cost(r, distance_matrix, eta_matrix)
            for r in routes
        )

        best_routes = routes
        best_cost = current_cost

        print(f"Trying to consolidate {len(routes)} routes for terminal {terminal_id}")
        
        # Calculate total size needed for all routes
        total_size_needed = sum(route.total_size for route in routes)
        print(f"Total size needed: {total_size_needed}")

        # Try consolidation with each vehicle type (sorted by capacity)
        sorted_vehicle_types = sorted(
            self.vehicle_specs.items(), 
            key=lambda x: x[1].capacity, 
            reverse=True  # Try largest capacity first
        )
        
        for vehicle_type, vehicle_spec in sorted_vehicle_types:
            print(f"Trying consolidation with {vehicle_type.value} (capacity: {vehicle_spec.capacity})")
            
            # Skip if total size exceeds vehicle capacity
            if total_size_needed > vehicle_spec.capacity:
                print(f"  Skipping {vehicle_type.value}: total size {total_size_needed} > capacity {vehicle_spec.capacity}")
                continue
                
            consolidated_routes = self._try_consolidation_with_vehicle(
                routes, vehicle_type, vehicle_spec, distance_matrix, eta_matrix
            )

            if consolidated_routes:
                consolidated_cost = sum(
                    self._calculate_route_total_cost(r, distance_matrix, eta_matrix)
                    for r in consolidated_routes
                )

                print(f"  Consolidation successful: cost {consolidated_cost:.2f} vs current {best_cost:.2f}")
                
                if consolidated_cost < best_cost:
                    best_routes = consolidated_routes
                    best_cost = consolidated_cost
                    print(f"  New best consolidation found!")

        # Try partial consolidations (2 routes at a time, 3 routes, etc.)
        if len(routes) > 2:
            for combo_size in range(2, len(routes)):
                print(f"Trying partial consolidations of size {combo_size}")
                partial_consolidations = self._try_partial_consolidations(
                    routes, combo_size, distance_matrix, eta_matrix
                )

                if partial_consolidations:
                    partial_cost = sum(
                        self._calculate_route_total_cost(r, distance_matrix, eta_matrix)
                        for r in partial_consolidations
                    )

                    if partial_cost < best_cost:
                        best_routes = partial_consolidations
                        best_cost = partial_cost
                        print(f"  Partial consolidation improved cost to {partial_cost:.2f}")

        return best_routes

    def _try_consolidation_with_vehicle(
        self,
        routes: List[Route],
        vehicle_type: VehicleType,
        vehicle_spec: VehicleSpec,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[List[Route]]:
        """Try to consolidate all routes into single vehicle type with OR-Tools optimization"""
        total_size = sum(route.total_size for route in routes)

        # CRITICAL FIX: Check capacity constraint BEFORE creating route
        if total_size > vehicle_spec.capacity:
            return None  # Won't fit - return early

        # Combine all parcels
        all_parcels = []
        pickup_sequences = set()

        for route in routes:
            all_parcels.extend(route.parcels)
            pickup_sequences.update(route.pickup_sequence)

        # Double-check total size calculation
        calculated_total_size = sum(p.size for p in all_parcels)
        if calculated_total_size > vehicle_spec.capacity:
            print(f"Warning: Calculated size {calculated_total_size} exceeds {vehicle_type.value} capacity {vehicle_spec.capacity}")
            return None

        # Create consolidated route
        try:
            consolidated_route = Route(
                vehicle_id=routes[0].vehicle_id,  # Use first vehicle ID
                vehicle_type=vehicle_type,
                vehicle_spec=vehicle_spec,
                parcels=all_parcels,
                pickup_sequence=list(pickup_sequences),
                delivery_sequence=[p.delivery_location for p in all_parcels],
            )
        except ValueError as e:
            print(f"Error creating consolidated route for {vehicle_type.value}: {e}")
            return None

        # Optimize route sequence with OR-Tools
        optimized_consolidated = self.route_optimizer.optimize_route_sequence(
            consolidated_route, distance_matrix, eta_matrix
        )

        # Calculate metrics
        optimized_consolidated.total_distance, optimized_consolidated.total_duration = (
            self._calculate_route_metrics(
                optimized_consolidated, distance_matrix, eta_matrix
            )
        )

        # Check time feasibility
        if not optimized_consolidated.is_time_feasible(self.config.time_window_seconds):
            print(f"Consolidated route for {vehicle_type.value} violates time window")
            return None

        print(f"Successfully consolidated {len(routes)} routes into 1 {vehicle_type.value} vehicle")
        return [optimized_consolidated]

    def _try_partial_consolidations(
        self,
        routes: List[Route],
        combo_size: int,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[List[Route]]:
        """Try consolidating subsets of routes with OR-Tools optimization"""
        from itertools import combinations

        if combo_size >= len(routes):
            return None

        best_combination = None
        best_cost_saving = 0

        # Try all combinations of combo_size routes
        for route_combo in combinations(routes, combo_size):
            remaining_routes = [r for r in routes if r not in route_combo]
            
            # Calculate total size for this combination
            combo_total_size = sum(route.total_size for route in route_combo)

            # Try consolidating this combination with each vehicle type
            for vehicle_type, vehicle_spec in self.vehicle_specs.items():
                # Check capacity constraint early
                if combo_total_size > vehicle_spec.capacity:
                    continue
                    
                consolidated = self._try_consolidation_with_vehicle(
                    list(route_combo),
                    vehicle_type,
                    vehicle_spec,
                    distance_matrix,
                    eta_matrix,
                )

                if consolidated:
                    # Calculate cost saving
                    original_cost = sum(
                        self._calculate_route_total_cost(r, distance_matrix, eta_matrix)
                        for r in route_combo
                    )
                    new_cost = self._calculate_route_total_cost(
                        consolidated[0], distance_matrix, eta_matrix
                    )
                    cost_saving = original_cost - new_cost

                    if cost_saving > best_cost_saving:
                        best_cost_saving = cost_saving
                        best_combination = consolidated + remaining_routes

        return best_combination

    def _calculate_route_metrics(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Tuple[float, int]:
        """Calculate distance and duration for route using optimized sequence"""
        total_distance = 0
        total_duration = 0

        if not route.route_sequence:
            return 0, 0

        # Get locations from optimized sequence
        route_locations = route.get_locations_sequence()

        # Calculate cumulative metrics
        for i in range(len(route_locations) - 1):
            loc1 = route_locations[i]
            loc2 = route_locations[i + 1]

            distance = distance_matrix.get(loc1, {}).get(loc2, 0)
            travel_time = eta_matrix.get(loc1, {}).get(loc2, 0)

            total_distance += distance
            total_duration += travel_time

        return total_distance, total_duration

    def _calculate_route_total_cost(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> float:
        """Calculate total cost including utilization penalties"""
        if not hasattr(route, "total_distance") or route.total_distance == 0:
            route.total_distance, route.total_duration = self._calculate_route_metrics(
                route, distance_matrix, eta_matrix
            )

        return route.vehicle_spec.calculate_total_route_cost(
            route.total_distance, route.total_size, include_fixed=True
        )

    def calculate_fleet_cost(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Dict:
        """Calculate comprehensive fleet cost analysis"""
        vehicle_counts = {}
        total_variable_cost = 0
        total_fixed_cost = 0
        utilization_analysis = {}

        for route in solution.routes:
            vehicle_type = route.vehicle_type

            # Count vehicles
            vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1

            # Calculate costs
            route_cost = self._calculate_route_total_cost(
                route, distance_matrix, eta_matrix
            )
            variable_cost = route.total_distance * route.vehicle_spec.cost_per_km
            fixed_cost = route.vehicle_spec.fixed_cost_per_vehicle

            total_variable_cost += variable_cost
            total_fixed_cost += fixed_cost

            # Utilization analysis
            if vehicle_type not in utilization_analysis:
                utilization_analysis[vehicle_type] = {
                    "routes": [],
                    "avg_utilization": 0,
                    "total_capacity": 0,
                    "total_used": 0,
                }

            utilization = route.total_size / route.vehicle_spec.capacity
            utilization_analysis[vehicle_type]["routes"].append(
                {
                    "vehicle_id": route.vehicle_id,
                    "utilization": round(utilization * 100, 1),
                    "capacity_used": route.total_size,
                    "capacity_total": route.vehicle_spec.capacity,
                    "is_optimized": route.is_optimized,
                }
            )
            utilization_analysis[vehicle_type][
                "total_capacity"
            ] += route.vehicle_spec.capacity
            utilization_analysis[vehicle_type]["total_used"] += route.total_size

        # Calculate average utilizations
        for vehicle_type, analysis in utilization_analysis.items():
            if analysis["total_capacity"] > 0:
                analysis["avg_utilization"] = round(
                    (analysis["total_used"] / analysis["total_capacity"]) * 100, 1
                )

        return {
            "vehicle_counts": {vt.value: count for vt, count in vehicle_counts.items()},
            "total_cost": total_variable_cost + total_fixed_cost,
            "variable_cost": total_variable_cost,
            "fixed_cost": total_fixed_cost,
            "utilization_analysis": {
                vt.value: analysis for vt, analysis in utilization_analysis.items()
            },
            "cost_per_vehicle_type": {
                vt.value: {
                    "count": vehicle_counts.get(vt, 0),
                    "total_cost": vehicle_counts.get(vt, 0)
                    * spec.fixed_cost_per_vehicle,
                }
                for vt, spec in self.vehicle_specs.items()
            },
        }


class ConstructionHeuristic:
    """Construction heuristics for initial solution"""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.route_optimizer = ORToolsRouteOptimizer(config)

    def greedy_construction(
        self,
        pickup_terminals: List[PickupTerminal],
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Solution:
        """Greedy construction heuristic with time window validation and OR-Tools optimization"""
        solution = Solution(pickup_terminals, self.config.vehicle_specs)
        vehicle_counter = 1

        # Sort terminals by number of parcels (descending)
        sorted_terminals = sorted(
            pickup_terminals, key=lambda x: len(x.parcels), reverse=True
        )

        for terminal in sorted_terminals:
            remaining_parcels = terminal.parcels.copy()

            while remaining_parcels:
                # Select best vehicle type for remaining parcels
                best_vehicle_type = self._select_best_vehicle_type(remaining_parcels)
                vehicle_spec = self.config.vehicle_specs[best_vehicle_type]

                # Create route with maximum parcels that fit (considering time)
                route_parcels = self._pack_parcels_with_time_constraint(
                    remaining_parcels,
                    vehicle_spec,
                    terminal,
                    distance_matrix,
                    eta_matrix,
                )

                if not route_parcels:
                    # If no parcels fit, mark as unassigned
                    solution.unassigned_parcels.extend(remaining_parcels)
                    break

                # Create route
                route = Route(
                    vehicle_id=vehicle_counter,
                    vehicle_type=best_vehicle_type,
                    vehicle_spec=vehicle_spec,
                    parcels=route_parcels,
                    pickup_sequence=[terminal.pickup_id],
                    delivery_sequence=[p.delivery_location for p in route_parcels],
                )

                # Optimize route sequence with OR-Tools
                optimized_route = self.route_optimizer.optimize_route_sequence(
                    route, distance_matrix, eta_matrix
                )

                # Calculate route metrics
                optimized_route.total_distance, optimized_route.total_duration = (
                    solution._calculate_route_metrics(
                        optimized_route, distance_matrix, eta_matrix
                    )
                )

                # Check time window constraint
                if optimized_route.is_time_feasible(self.config.time_window_seconds):
                    solution.add_route(optimized_route)
                    vehicle_counter += 1

                    # Remove assigned parcels
                    for parcel in route_parcels:
                        remaining_parcels.remove(parcel)
                else:
                    # If time constraint violated, try with fewer parcels
                    if len(route_parcels) > 1:
                        # Retry with half the parcels
                        reduced_parcels = route_parcels[: len(route_parcels) // 2]
                        reduced_route = Route(
                            vehicle_id=vehicle_counter,
                            vehicle_type=best_vehicle_type,
                            vehicle_spec=vehicle_spec,
                            parcels=reduced_parcels,
                            pickup_sequence=[terminal.pickup_id],
                            delivery_sequence=[
                                p.delivery_location for p in reduced_parcels
                            ],
                        )

                        # Optimize reduced route
                        optimized_reduced_route = (
                            self.route_optimizer.optimize_route_sequence(
                                reduced_route, distance_matrix, eta_matrix
                            )
                        )

                        (
                            optimized_reduced_route.total_distance,
                            optimized_reduced_route.total_duration,
                        ) = solution._calculate_route_metrics(
                            optimized_reduced_route, distance_matrix, eta_matrix
                        )

                        if optimized_reduced_route.is_time_feasible(
                            self.config.time_window_seconds
                        ):
                            solution.add_route(optimized_reduced_route)
                            vehicle_counter += 1

                            # Remove assigned parcels
                            for parcel in reduced_parcels:
                                remaining_parcels.remove(parcel)
                        else:
                            # Can't fit even reduced parcels, mark as unassigned
                            solution.unassigned_parcels.extend(remaining_parcels)
                            break
                    else:
                        # Single parcel doesn't fit time window
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

    def _select_best_vehicle_type(self, parcels: List[Parcel]) -> VehicleType:
        """Select best vehicle type for given parcels"""
        total_size = sum(p.size for p in parcels)

        # Find cheapest vehicle that can fit all parcels
        best_type = None
        best_cost_efficiency = float("inf")

        for vehicle_type, spec in self.config.vehicle_specs.items():
            if spec.capacity >= total_size:
                cost_efficiency = spec.cost_per_km / spec.capacity
                if cost_efficiency < best_cost_efficiency:
                    best_cost_efficiency = cost_efficiency
                    best_type = vehicle_type

        # If no single vehicle can fit all, choose highest capacity
        if best_type is None:
            best_type = max(
                self.config.vehicle_specs.keys(),
                key=lambda x: self.config.vehicle_specs[x].capacity,
            )

        return best_type

    def _pack_parcels_with_time_constraint(
        self,
        parcels: List[Parcel],
        vehicle_spec: VehicleSpec,
        terminal: PickupTerminal,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> List[Parcel]:
        """Pack parcels considering both capacity and time constraints"""
        # Sort parcels by delivery distance (closest first for time efficiency)
        terminal_location = (terminal.lat, terminal.lon)

        def delivery_distance(parcel):
            return distance_matrix.get(terminal_location, {}).get(
                parcel.delivery_location, float("inf")
            )

        sorted_parcels = sorted(parcels, key=delivery_distance)

        packed = []
        current_size = 0

        for parcel in sorted_parcels:
            if current_size + parcel.size <= vehicle_spec.capacity:
                # Create temporary route to check time constraint
                temp_parcels = packed + [parcel]
                temp_route = Route(
                    vehicle_id=0,  # Temporary
                    vehicle_type=vehicle_spec.vehicle_type,
                    vehicle_spec=vehicle_spec,
                    parcels=temp_parcels,
                    pickup_sequence=[terminal.pickup_id],
                    delivery_sequence=[p.delivery_location for p in temp_parcels],
                )

                # Calculate route duration with simple estimation
                route_distance, route_duration = self._calculate_temp_route_metrics(
                    temp_route, distance_matrix, eta_matrix
                )

                # Check if adding this parcel violates time window
                if route_duration <= self.config.time_window_seconds:
                    packed.append(parcel)
                    current_size += parcel.size
                else:
                    # Stop packing if time window would be violated
                    break

        return packed

    def _calculate_temp_route_metrics(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Tuple[float, int]:
        """Calculate metrics for temporary route during construction"""
        total_distance = 0
        total_duration = 0

        if not route.pickup_sequence and not route.delivery_sequence:
            return 0, 0

        # Build route locations (simple: pickups then deliveries)
        route_locations = []

        # Add pickup location
        if route.pickup_sequence:
            pickup_location = route.parcels[0].pickup_location
            route_locations.append(pickup_location)

        # Add delivery locations
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


class ALNSOperators:
    """ALNS destroy and repair operators with vehicle optimization and OR-Tools"""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.random = random.Random(42)  # For reproducibility
        self.vehicle_optimizer = VehicleOptimizer(config)
        self.route_optimizer = ORToolsRouteOptimizer(config)

    def _should_try_vehicle_swap(self, solution: Solution) -> bool:
        """Check if vehicle swap is worth attempting"""
        # Skip if all routes are already using optimal vehicle types
        optimal_count = 0
        for route in solution.routes:
            best_type = self.vehicle_optimizer._get_best_vehicle_type_for_size(route.total_size)
            if route.vehicle_type == best_type:
                optimal_count += 1
        
        # Skip if >80% routes already optimal
        if optimal_count / len(solution.routes) > 0.8:
            if self.vehicle_optimizer.verbose_logging:
                print("Skipping vehicle swap: most routes already optimal")
            return False
        
        return True

    # Vehicle Type Optimization Operators
    def vehicle_consolidation_operator(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Try to consolidate routes using larger vehicles with OR-Tools optimization"""
        return self.vehicle_optimizer.optimize_vehicle_assignments_with_stagnation_detection(
            solution, distance_matrix, eta_matrix
        )

    def vehicle_type_swap_operator(
        self,
        solution: Solution,
        distance_matrix: Dict,
        eta_matrix: Dict,
        num_swaps: int = 2,
    ) -> Solution:
        """Optimized vehicle type swap with early capacity filtering"""
        new_solution = deepcopy(solution)

        if not new_solution.routes:
            return new_solution

        # Select random routes to modify
        routes_to_modify = self.random.sample(
            new_solution.routes, min(num_swaps, len(new_solution.routes))
        )

        for route in routes_to_modify:
            original_type = route.vehicle_type
            original_spec = route.vehicle_spec
            
            # PRE-FILTER: Only test vehicle types that can fit the route
            feasible_types = []
            for vt, spec in self.config.vehicle_specs.items():
                if vt != original_type and spec.capacity >= route.total_size:
                    feasible_types.append((vt, spec))
            
            if not feasible_types:
                if self.vehicle_optimizer.verbose_logging:
                    print(f"  Skipping route {route.vehicle_id}: no feasible vehicle types")
                continue

            # Sort by cost efficiency (try cheapest first)
            feasible_types.sort(key=lambda x: x[1].cost_per_km / x[1].capacity)
            
            # Try only the most promising vehicle type (not all types)
            new_type, new_spec = feasible_types[0]
            
            if self.vehicle_optimizer.verbose_logging:
                print(f"  Testing {original_type.value} → {new_type.value} for route {route.vehicle_id}")

            try:
                # Update vehicle type
                route.vehicle_type = new_type
                route.vehicle_spec = new_spec

                # Quick time estimate before expensive OR-Tools call
                estimated_duration = self.vehicle_optimizer._estimate_route_duration(
                    route, distance_matrix, eta_matrix
                )
                
                if estimated_duration > self.config.time_window_seconds * 1.1:
                    if self.vehicle_optimizer.verbose_logging:
                        print(f"  Skipping {new_type.value}: estimated time violation")
                    # Revert and skip
                    route.vehicle_type = original_type
                    route.vehicle_spec = original_spec
                    continue

                # Only now call expensive OR-Tools optimization
                optimized_route = self.route_optimizer.optimize_route_sequence(
                    route, distance_matrix, eta_matrix
                )

                # Update route with optimized version
                route.route_sequence = optimized_route.route_sequence
                route.is_optimized = optimized_route.is_optimized

                # Recalculate metrics
                route.total_distance, route.total_duration = (
                    self.vehicle_optimizer._calculate_route_metrics(
                        route, distance_matrix, eta_matrix
                    )
                )

                # Final time constraint check
                if not route.is_time_feasible(self.config.time_window_seconds):
                    if self.vehicle_optimizer.verbose_logging:
                        print(f"  Reverting {new_type.value}: time window violated")
                    route.vehicle_type = original_type
                    route.vehicle_spec = original_spec
                    route.is_optimized = False
                else:
                    if self.vehicle_optimizer.verbose_logging:
                        print(f"  Successfully swapped to {new_type.value}")
                        
            except Exception as e:
                if self.vehicle_optimizer.verbose_logging:
                    print(f"  Error swapping to {new_type.value}: {e}")
                # Revert to original
                route.vehicle_type = original_type
                route.vehicle_spec = original_spec
                route.is_optimized = False

        return new_solution

    def route_splitting_operator(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Split over-utilized or large routes into smaller vehicles with OR-Tools optimization"""
        new_solution = deepcopy(solution)

        routes_to_split = []

        # Find routes that could benefit from splitting
        for route in new_solution.routes:
            utilization = route.total_size / route.vehicle_spec.capacity

            # Split if over-utilized or time-constrained
            if (
                utilization > 0.9
                or not route.is_time_feasible(self.config.time_window_seconds)
                or len(route.parcels) > 10
            ):  # Arbitrary large route threshold
                routes_to_split.append(route)

        # Try to split selected routes
        for route in routes_to_split:
            split_routes = self._try_split_route(route, distance_matrix, eta_matrix)
            if split_routes and len(split_routes) > 1:
                # Calculate costs
                original_cost = self.vehicle_optimizer._calculate_route_total_cost(
                    route, distance_matrix, eta_matrix
                )
                split_cost = sum(
                    self.vehicle_optimizer._calculate_route_total_cost(
                        r, distance_matrix, eta_matrix
                    )
                    for r in split_routes
                )

                # Accept if cost improvement or better feasibility
                if (
                    split_cost < original_cost * 1.1  # Allow 10% cost increase
                    or any(
                        not r.is_time_feasible(self.config.time_window_seconds)
                        for r in [route]
                    )
                    and all(
                        r.is_time_feasible(self.config.time_window_seconds)
                        for r in split_routes
                    )
                ):

                    # Replace original route with split routes
                    new_solution.routes.remove(route)
                    new_solution.routes.extend(split_routes)

        return new_solution

    def _validate_route_before_creation(self, parcels: List[Parcel], vehicle_spec: VehicleSpec) -> bool:
        """Validate that parcels can fit in vehicle before creating route"""
        total_size = sum(p.size for p in parcels)
        if total_size > vehicle_spec.capacity:
            print(f"Warning: Cannot fit {total_size} units in {vehicle_spec.vehicle_type.value} (capacity: {vehicle_spec.capacity})")
            return False
        return True

    def _try_split_route(
        self, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> Optional[List[Route]]:
        """Attempt to split a route into smaller routes with OR-Tools optimization"""
        if len(route.parcels) < 2:
            return None

        # Sort parcels by delivery distance for better splitting
        if route.pickup_sequence:
            pickup_location = route.parcels[0].pickup_location
            sorted_parcels = sorted(
                route.parcels,
                key=lambda p: distance_matrix.get(pickup_location, {}).get(
                    p.delivery_location, 0
                ),
            )
        else:
            sorted_parcels = route.parcels.copy()

        # Try splitting into 2 parts
        split_point = len(sorted_parcels) // 2
        part1_parcels = sorted_parcels[:split_point]
        part2_parcels = sorted_parcels[split_point:]

        # Find suitable vehicle types for each part
        split_routes = []
        next_vehicle_id = max([r.vehicle_id for r in [route]], default=0) + 1

        for i, parcels_subset in enumerate([part1_parcels, part2_parcels]):
            if not parcels_subset:
                continue

            subset_size = sum(p.size for p in parcels_subset)

            # Find best vehicle type for this subset
            best_vehicle_type = None
            best_cost = float("inf")

            for vehicle_type, vehicle_spec in self.config.vehicle_specs.items():
                if vehicle_spec.capacity >= subset_size:
                    # Create test route
                    test_route = Route(
                        vehicle_id=next_vehicle_id + i,
                        vehicle_type=vehicle_type,
                        vehicle_spec=vehicle_spec,
                        parcels=parcels_subset,
                        pickup_sequence=route.pickup_sequence.copy(),
                        delivery_sequence=[p.delivery_location for p in parcels_subset],
                    )

                    # Optimize with OR-Tools
                    optimized_test_route = self.route_optimizer.optimize_route_sequence(
                        test_route, distance_matrix, eta_matrix
                    )

                    # Calculate metrics
                    (
                        optimized_test_route.total_distance,
                        optimized_test_route.total_duration,
                    ) = self.vehicle_optimizer._calculate_route_metrics(
                        optimized_test_route, distance_matrix, eta_matrix
                    )

                    # Check time feasibility
                    if optimized_test_route.is_time_feasible(
                        self.config.time_window_seconds
                    ):
                        test_cost = self.vehicle_optimizer._calculate_route_total_cost(
                            optimized_test_route, distance_matrix, eta_matrix
                        )

                        if test_cost < best_cost:
                            best_cost = test_cost
                            best_vehicle_type = vehicle_type

            # Create final route with best vehicle type
            if best_vehicle_type:
                final_route = Route(
                    vehicle_id=next_vehicle_id + i,
                    vehicle_type=best_vehicle_type,
                    vehicle_spec=self.config.vehicle_specs[best_vehicle_type],
                    parcels=parcels_subset,
                    pickup_sequence=route.pickup_sequence.copy(),
                    delivery_sequence=[p.delivery_location for p in parcels_subset],
                )

                # Optimize with OR-Tools
                optimized_final_route = self.route_optimizer.optimize_route_sequence(
                    final_route, distance_matrix, eta_matrix
                )

                # Set metrics
                (
                    optimized_final_route.total_distance,
                    optimized_final_route.total_duration,
                ) = self.vehicle_optimizer._calculate_route_metrics(
                    optimized_final_route, distance_matrix, eta_matrix
                )

                split_routes.append(optimized_final_route)

        return split_routes if len(split_routes) >= 2 else None

    # Destroy operators
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
            new_solution.routes[route_idx].parcels.remove(parcel)
            new_solution.unassigned_parcels.append(parcel)

            # Update delivery sequence
            if (
                parcel.delivery_location
                in new_solution.routes[route_idx].delivery_sequence
            ):
                new_solution.routes[route_idx].delivery_sequence.remove(
                    parcel.delivery_location
                )

        # Remove empty routes and re-optimize remaining routes
        remaining_routes = []
        for route in new_solution.routes:
            if route.parcels:
                # Re-optimize route after parcel removal
                optimized_route = self.route_optimizer.optimize_route_sequence(
                    route, {}, {}  # Will be recalculated later
                )
                remaining_routes.append(optimized_route)

        new_solution.routes = remaining_routes

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
            for parcel in parcels_to_remove:
                route.parcels.remove(parcel)
                new_solution.unassigned_parcels.append(parcel)

                # Update delivery sequence
                if parcel.delivery_location in route.delivery_sequence:
                    route.delivery_sequence.remove(parcel.delivery_location)

        # Remove empty routes and re-optimize remaining routes
        # remaining_routes = []
        # for route in new_solution.routes:
        #     if
        # Remove empty routes and re-optimize remaining routes
        remaining_routes = []
        for route in new_solution.routes:
            if route.parcels:
                # Re-optimize route after parcel removal
                optimized_route = self.route_optimizer.optimize_route_sequence(
                    route, {}, {}  # Will be recalculated later
                )
                remaining_routes.append(optimized_route)

        new_solution.routes = remaining_routes

        return new_solution

    # Repair operators
    def greedy_insertion(
        self, solution: Solution, distance_matrix: Dict, eta_matrix: Dict
    ) -> Solution:
        """Greedily insert unassigned parcels with time constraint validation and OR-Tools optimization"""
        new_solution = deepcopy(solution)

        # Sort unassigned parcels by size (largest first)
        unassigned = sorted(
            new_solution.unassigned_parcels, key=lambda x: x.size, reverse=True
        )

        for parcel in unassigned:
            best_insertion = self._find_best_insertion_with_time(
                parcel, new_solution, distance_matrix, eta_matrix
            )

            if best_insertion:
                route_idx, insertion_cost = best_insertion
                new_solution.routes[route_idx].parcels.append(parcel)
                new_solution.unassigned_parcels.remove(parcel)

                # Update route delivery sequence
                new_solution.routes[route_idx].delivery_sequence.append(
                    parcel.delivery_location
                )

                # Re-optimize route with OR-Tools
                optimized_route = self.route_optimizer.optimize_route_sequence(
                    new_solution.routes[route_idx], distance_matrix, eta_matrix
                )

                # Update route with optimized version
                new_solution.routes[route_idx] = optimized_route

                # Recalculate route metrics
                route = new_solution.routes[route_idx]
                route.total_distance, route.total_duration = (
                    new_solution._calculate_route_metrics(
                        route, distance_matrix, eta_matrix
                    )
                )

            else:
                # Create new route if no insertion found
                new_route = self._create_new_route_with_time(
                    parcel, new_solution, distance_matrix, eta_matrix
                )
                if new_route:
                    new_solution.add_route(new_route)
                    new_solution.unassigned_parcels.remove(parcel)

        return new_solution

    def _find_best_insertion_with_time(
        self,
        parcel: Parcel,
        solution: Solution,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[Tuple[int, float]]:
        """Find best insertion position for parcel considering time constraints"""
        best_route_idx = None
        best_cost = float("inf")

        for route_idx, route in enumerate(solution.routes):
            # Check capacity constraint
            if route.total_size + parcel.size > route.vehicle_spec.capacity:
                continue

            # Check knock constraint
            parcel_pickup_id = self._get_parcel_pickup_id(parcel, solution)
            if parcel_pickup_id not in route.pickup_sequence:
                # Check if adding this pickup violates knock constraint
                current_knocks = len(
                    solution.pickup_assignments.get(parcel_pickup_id, [])
                )
                if current_knocks >= self.config.max_knock:
                    continue

            # Check time constraint by testing insertion
            temp_route = deepcopy(route)
            temp_route.parcels.append(parcel)
            temp_route.delivery_sequence.append(parcel.delivery_location)

            # Optimize temporary route
            optimized_temp_route = self.route_optimizer.optimize_route_sequence(
                temp_route, distance_matrix, eta_matrix
            )

            # Calculate new route duration
            route_distance, route_duration = solution._calculate_route_metrics(
                optimized_temp_route, distance_matrix, eta_matrix
            )

            # Skip if time window violated
            if route_duration > self.config.time_window_seconds:
                continue

            # Calculate insertion cost
            insertion_cost = self._calculate_insertion_cost_with_time(
                parcel, route, distance_matrix, eta_matrix
            )

            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_route_idx = route_idx

        return (best_route_idx, best_cost) if best_route_idx is not None else None

    def _calculate_insertion_cost_with_time(
        self, parcel: Parcel, route: Route, distance_matrix: Dict, eta_matrix: Dict
    ) -> float:
        """Calculate cost of inserting parcel into route including time penalty"""
        # Base cost: distance increase
        base_cost = parcel.size * route.vehicle_spec.cost_per_km

        # Time penalty: penalize routes that are getting close to time limit
        current_duration = getattr(route, "total_duration", 0)
        time_window = self.config.time_window_seconds

        if current_duration > 0 and time_window > 0:
            time_utilization = current_duration / time_window
            time_penalty = (
                base_cost * time_utilization * 0.5
            )  # 50% penalty for high time utilization
        else:
            time_penalty = 0

        return base_cost + time_penalty

    def _create_new_route_with_time(
        self,
        parcel: Parcel,
        solution: Solution,
        distance_matrix: Dict,
        eta_matrix: Dict,
    ) -> Optional[Route]:
        """Create new route for parcel with time validation and OR-Tools optimization"""
        # Find parcel's pickup terminal
        pickup_terminal = None
        for terminal in solution.pickup_terminals:
            if parcel in terminal.parcels:
                pickup_terminal = terminal
                break

        if not pickup_terminal:
            return None

        # Select best vehicle type
        best_vehicle_type = None
        best_cost_efficiency = float("inf")
        best_optimized_route = None

        for vehicle_type, spec in self.config.vehicle_specs.items():
            if spec.capacity >= parcel.size:
                # Create temporary route
                temp_route = Route(
                    vehicle_id=0,  # Temporary
                    vehicle_type=vehicle_type,
                    vehicle_spec=spec,
                    parcels=[parcel],
                    pickup_sequence=[pickup_terminal.pickup_id],
                    delivery_sequence=[parcel.delivery_location],
                )

                # Optimize with OR-Tools
                optimized_temp_route = self.route_optimizer.optimize_route_sequence(
                    temp_route, distance_matrix, eta_matrix
                )

                # Calculate route duration
                route_distance, route_duration = solution._calculate_route_metrics(
                    optimized_temp_route, distance_matrix, eta_matrix
                )

                # Check time feasibility
                if route_duration <= self.config.time_window_seconds:
                    cost_efficiency = spec.cost_per_km / spec.capacity
                    if cost_efficiency < best_cost_efficiency:
                        best_cost_efficiency = cost_efficiency
                        best_vehicle_type = vehicle_type
                        best_optimized_route = optimized_temp_route

        if not best_vehicle_type or not best_optimized_route:
            return None

        # Create final route
        vehicle_id = max([r.vehicle_id for r in solution.routes], default=0) + 1
        final_route = deepcopy(best_optimized_route)
        final_route.vehicle_id = vehicle_id

        # Calculate and set route metrics
        final_route.total_distance, final_route.total_duration = (
            solution._calculate_route_metrics(final_route, distance_matrix, eta_matrix)
        )

        return final_route

    def _get_parcel_pickup_id(self, parcel: Parcel, solution: Solution) -> int:
        """Get pickup terminal ID for parcel"""
        for terminal in solution.pickup_terminals:
            if parcel in terminal.parcels:
                return terminal.pickup_id
        raise ValueError(f"Parcel {parcel.id} not found in any terminal")


def create_problem_from_csv(
    df: pd.DataFrame,
    max_knock: int = 4,
    time_window_hours: float = 6.0,
    default_parcel_size: int = 1,
    proximity_threshold_meters: float = 100.0,
) -> Tuple[ProblemConfig, List[PickupTerminal], Dict]:
    """
    Main function to create problem from CSV data

    Args:
        csv_file_path: Path to CSV file
        max_knock: Maximum knocks per pickup terminal
        time_window_hours: Time window in hours
        default_parcel_size: Default size for parcels
        proximity_threshold_meters: Distance threshold to group pickup locations

    Returns:
        Tuple of (ProblemConfig, List[PickupTerminal], data_summary)
    """
    # Define vehicle types (customize as needed)
    vehicle_specs = {
        VehicleType.BIKE: VehicleSpec(
            VehicleType.BIKE,
            capacity=28,
            cost_per_km=7.27,
            fixed_cost_per_vehicle=0,
            min_utilization_threshold=0.5,
            optimal_utilization_range=(0.6, 1.0),
        ),
        VehicleType.CARBOX: VehicleSpec(
            VehicleType.CARBOX,
            capacity=212,
            cost_per_km=4.2,
            fixed_cost_per_vehicle=0,
            min_utilization_threshold=0.5,
            optimal_utilization_range=(0.6, 1.0),
        ),
        VehicleType.BIG_BOX: VehicleSpec(
            VehicleType.BIG_BOX,
            capacity=44,
            cost_per_km=7.95,
            fixed_cost_per_vehicle=0,
            min_utilization_threshold=0.5,
            optimal_utilization_range=(0.6, 1.0),
        ),
    }

    # Process data
    processor = DataProcessor(default_parcel_size, proximity_threshold_meters)
    config, pickup_terminals = processor.process_csv_data(
        df, vehicle_specs, max_knock, time_window_hours
    )

    # Get summary
    data_summary = processor.get_data_summary(pickup_terminals)

    return config, pickup_terminals, data_summary


def solve_from_csv_per_polygon(
    df: pd.DataFrame,
    max_knock: int = 4,
    time_window_hours: float = 6.0,
    default_parcel_size: int = 1,
    proximity_threshold_meters: float = 100.0,
    max_iterations: int = 1000,
) -> Dict:
    """
    Complete workflow: Load CSV data, solve VRP, return formatted results

    Args:
        csv_file_path: Path to CSV file with columns:
                      - order_request_id, latitude, longitude, pickup_latitude, pickup_longitude
        max_knock: Maximum knocks per pickup terminal
        time_window_hours: Time window in hours
        default_parcel_size: Default size for parcels
        proximity_threshold_meters: Distance threshold to group pickup locations
        max_iterations: ALNS iterations

    Returns:
        Formatted solution dictionary
    """
    all_etas = {}
    all_distances = {}
    aggregated_results = {"polygon_results": {}}
    for polygon_id, group in df.groupby(by="polygon_id"):
        start_time = time.time()

        # Load and process data
        print("Loading and processing CSV data...")
        config, pickup_terminals, data_summary = create_problem_from_csv(
            group,
            max_knock,
            time_window_hours,
            default_parcel_size,
            proximity_threshold_meters,
        )

        print("Data Summary:")
        print(f"  - Pickup terminals: {data_summary['num_pickup_terminals']}")
        print(f"  - Total parcels: {data_summary['total_parcels']}")
        print(
            f"  - Average parcels per terminal: {data_summary['avg_parcels_per_terminal']:.1f}"
        )

        # Build distance matrix
        print("Building distance matrix...")
        all_locations = []
        for terminal in pickup_terminals:
            all_locations.append((terminal.lat, terminal.lon))
            for parcel in terminal.parcels:
                all_locations.append(parcel.delivery_location)

        # Remove duplicates
        all_locations = list(set(all_locations))
        distance_matrix, eta_matrix = DistanceCalculator.build_matrices(all_locations)
        all_etas.update(eta_matrix)
        all_distances.update(distance_matrix)
        # Solve
        print("Solving VRP with ALNS...")
        solver = ALNSSolver(config)
        solution = solver.solve_optimized(
            pickup_terminals, distance_matrix, eta_matrix, max_iterations
        )

        # Format output
        result = solver.format_output(solution, distance_matrix, eta_matrix)
        result["processing_time"] = time.time() - start_time
        aggregated_results["polygon_results"][polygon_id] = result

    visualize_vrp_solutions(
        aggregated_results["polygon_results"], all_distances, all_etas
    )

    return aggregated_results


if __name__ == "__main__":
    df = pd.read_csv("./Sample_SDD 2.csv")

    result = solve_from_csv_per_polygon(df, 2, 3, 4, 0, 200)
    print(result)
