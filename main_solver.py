import time
import constants
import pandas as pd

from typing import List, Dict, Tuple

from preprocessing import DataProcessor
from alns_solver import UpdatedALNSSolver
from visualize import visualize_vrp_solutions
from distance_calculator import DistanceCalculator
from data_structures import (
    VehicleType,
    VehicleSpec,
    ProblemConfig,
    PickupTerminal,
    Parcel,
)


class PerformanceOptimizedSolver:
    """
    Main solver class that integrates all updated components
    Designed to solve 250+ parcels in under 1 minute
    """

    def __init__(self):
        self.verbose_logging = True
        self.performance_mode = True  # Enables aggressive optimizations

    def solve_from_csv_optimized(
        self,
        df: pd.DataFrame,
        max_knock: int = 4,
        time_window_hours: float = 6.0,
        capacity_config: Dict[str, float] = None,
        cost_per_hour_config: Dict[str, float] = None,
        default_parcel_size: int = 1,
        proximity_threshold_meters: float = 100.0,
        max_iterations: int = 200,
        performance_target_seconds: int = 60,
    ) -> Dict:
        """
        Main solve method optimized for performance

        Args:
            df: DataFrame with parcel data
            max_knock: Maximum knocks per pickup terminal
            time_window_hours: Time window in hours
            cost_per_hour_config: Cost per hour for each vehicle type
            default_parcel_size: Default parcel size
            proximity_threshold_meters: Proximity threshold for grouping
            max_iterations: Maximum ALNS iterations
            performance_target_seconds: Target completion time

        Returns:
            Formatted solution results
        """
        overall_start_time = time.time()

        if capacity_config is None:
            capacity_config = constants.VEHICLE_CAPACITIES
        if cost_per_hour_config is None:
            cost_per_hour_config = constants.VEHICLE_COST_PER_HOUR

        all_etas = {}
        all_distances = {}
        aggregated_results = {"polygon_results": {}}

        if "polygon_id" in df.columns:
            grouped_data = df.groupby("polygon_id")
        else:
            grouped_data = [("default", df)]

        for polygon_id, group in grouped_data:
            polygon_start_time = time.time()

            try:
                if self.verbose_logging:
                    print(f"\n{'='*50}")
                    print(f"Processing Polygon {polygon_id}")
                    print(f"{'='*50}")

                result = self._solve_single_polygon(
                    group,
                    polygon_id,
                    max_knock,
                    time_window_hours,
                    capacity_config,
                    cost_per_hour_config,
                    default_parcel_size,
                    proximity_threshold_meters,
                    max_iterations,
                    performance_target_seconds,
                )

                if "distance_matrix" in result:
                    all_distances.update(result["distance_matrix"])
                if "eta_matrix" in result:
                    all_etas.update(result["eta_matrix"])

                if "distance_matrix" in result:
                    del result["distance_matrix"]
                if "eta_matrix" in result:
                    del result["eta_matrix"]

                polygon_time = time.time() - polygon_start_time
                result["processing_time"] = polygon_time

                aggregated_results["polygon_results"][polygon_id] = result

                if self.verbose_logging:
                    print(
                        f"Polygon {polygon_id} completed in {polygon_time:.2f} seconds"
                    )

            except Exception as e:
                if self.verbose_logging:
                    print(f"Error processing polygon {polygon_id}: {e}")

                aggregated_results["polygon_results"][polygon_id] = {
                    "error": str(e),
                    "processing_time": time.time() - polygon_start_time,
                    "results_summary": {
                        "total_vehicles_used": 0,
                        "total_cost": 0.0,
                        "time_window_feasible": False,
                    },
                }

        try:
            if all_distances and all_etas:
                visualization_file = visualize_vrp_solutions(
                    aggregated_results["polygon_results"], all_distances, all_etas
                )
                aggregated_results["visualization_file"] = visualization_file
        except Exception as e:
            if self.verbose_logging:
                print(f"Visualization generation failed: {e}")

        total_time = time.time() - overall_start_time
        aggregated_results["total_processing_time"] = total_time

        if self.verbose_logging:
            print(f"\n{'='*50}")
            print(f"OVERALL COMPLETION")
            print(f"{'='*50}")
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Polygons processed: {len(aggregated_results['polygon_results'])}")

            total_vehicles = 0
            total_cost = 0.0
            for result in aggregated_results["polygon_results"].values():
                if "results_summary" in result:
                    total_vehicles += result["results_summary"].get(
                        "total_vehicles_used", 0
                    )
                    total_cost += result["results_summary"].get("total_cost", 0.0)

            print(f"Total vehicles used: {total_vehicles}")
            print(f"Total cost: {total_cost:.2f}")
            print(
                f"Average cost per polygon: {total_cost / len(aggregated_results['polygon_results']):.2f}"
            )

        return aggregated_results

    def _solve_single_polygon(
        self,
        df: pd.DataFrame,
        polygon_id: str,
        max_knock: int,
        time_window_hours: float,
        capacity_config: Dict[str, float],
        cost_per_hour_config: Dict[str, float],
        default_parcel_size: int,
        proximity_threshold_meters: float,
        max_iterations: int,
        performance_target_seconds: int,
    ) -> Dict:
        """Solve VRP for a single polygon with performance optimizations"""

        polygon_start_time = time.time()

        if self.verbose_logging:
            print("Step 1: Processing data...")

        config, pickup_terminals, data_summary = self._create_problem_from_dataframe(
            df,
            max_knock,
            time_window_hours,
            capacity_config,
            cost_per_hour_config,
            default_parcel_size,
            proximity_threshold_meters,
        )

        if self.verbose_logging:
            print(f"  - Pickup terminals: {data_summary['num_pickup_terminals']}")
            print(f"  - Total parcels: {data_summary['total_parcels']}")
            print(
                f"  - Avg parcels per terminal: {data_summary['avg_parcels_per_terminal']:.1f}"
            )

        if self.verbose_logging:
            print("Step 2: Building distance matrix...")

        distance_matrix, eta_matrix = self._build_distance_matrices(pickup_terminals)

        if self.verbose_logging:
            print("Step 3: Solving VRP...")

        instance_size = data_summary["total_parcels"]
        adjusted_max_iterations = self._calculate_optimal_iterations(
            instance_size, performance_target_seconds, max_iterations
        )

        if self.verbose_logging:
            print(f"  - Instance size: {instance_size} parcels")
            print(f"  - Max iterations: {adjusted_max_iterations}")
            print(f"  - Performance target: {performance_target_seconds}s")

        solver = UpdatedALNSSolver(config)

        solve_start_time = time.time()
        solution = solver.solve_optimized(
            pickup_terminals, distance_matrix, eta_matrix, adjusted_max_iterations
        )
        solve_time = time.time() - solve_start_time

        if self.verbose_logging:
            print(f"  - Solve completed in {solve_time:.2f} seconds")

        if self.verbose_logging:
            print("Step 4: Formatting output...")

        result = solver.format_output(solution, distance_matrix, eta_matrix)
        result["processing_time"] = time.time() - polygon_start_time
        result["solve_time"] = solve_time
        result["polygon_id"] = polygon_id

        result["performance_metrics"] = {
            "instance_size": instance_size,
            "iterations_used": adjusted_max_iterations,
            "solve_time_seconds": solve_time,
            "total_time_seconds": result["processing_time"],
            "performance_target_met": solve_time <= performance_target_seconds,
            "parcels_per_second": instance_size / solve_time if solve_time > 0 else 0,
        }

        result["distance_matrix"] = distance_matrix
        result["eta_matrix"] = eta_matrix

        return result

    def _create_problem_from_dataframe(
        self,
        df: pd.DataFrame,
        max_knock: int,
        time_window_hours: float,
        vehicle_capacities: Dict[str, float],
        cost_per_hour_config: Dict[str, float],
        default_parcel_size: int,
        proximity_threshold_meters: float,
    ) -> Tuple[ProblemConfig, List[PickupTerminal], Dict]:
        """Create problem configuration from DataFrame"""

        vehicle_specs = {}

        for vehicle_name, cost_per_hour in cost_per_hour_config.items():
            vehicle_type = VehicleType(vehicle_name)
            capacity = vehicle_capacities.get(vehicle_name, 50)

            vehicle_specs[vehicle_type] = VehicleSpec(
                vehicle_type=vehicle_type,
                capacity=capacity,
                cost_per_hour=cost_per_hour,
                min_utilization_threshold=0.3,
                optimal_utilization_range=(0.6, 0.9),
            )

        processor = DataProcessor(default_parcel_size, proximity_threshold_meters)
        config = ProblemConfig(max_knock, time_window_hours, vehicle_specs)

        pickup_terminals = self._create_pickup_terminals(df, processor)

        data_summary = processor.get_data_summary(pickup_terminals)

        return config, pickup_terminals, data_summary

    def _create_pickup_terminals(
        self, df: pd.DataFrame, processor: DataProcessor
    ) -> List[PickupTerminal]:
        """Create pickup terminals from DataFrame"""

        # Create parcels
        parcels = []
        for idx, row in df.iterrows():
            try:
                parcel = Parcel(
                    id=int(row["order_request_id"]),
                    pickup_location=(
                        float(row["pickup_latitude"]),
                        float(row["pickup_longitude"]),
                    ),
                    delivery_location=(float(row["latitude"]), float(row["longitude"])),
                    size=processor._determine_parcel_size(row),
                )
                parcels.append(parcel)
            except Exception as e:
                if self.verbose_logging:
                    print(f"Warning: Could not create parcel from row {idx}: {e}")
                continue

        # Group into pickup terminals
        pickup_terminals = processor._group_parcels_into_terminals(parcels)

        return pickup_terminals

    def _build_distance_matrices(
        self, pickup_terminals: List[PickupTerminal]
    ) -> Tuple[Dict, Dict]:
        """Build distance and ETA matrices"""

        # Collect all unique locations
        all_locations = set()

        for terminal in pickup_terminals:
            all_locations.add((terminal.lat, terminal.lon))
            for parcel in terminal.parcels:
                all_locations.add(parcel.delivery_location)

        all_locations = list(all_locations)

        if self.verbose_logging:
            print(f"  - Building matrices for {len(all_locations)} locations")

        # Build matrices
        distance_matrix, eta_matrix = DistanceCalculator.build_matrices(all_locations)

        return distance_matrix, eta_matrix

    def _calculate_optimal_iterations(
        self, instance_size: int, performance_target_seconds: int, max_iterations: int
    ) -> int:
        """Calculate optimal iteration count based on instance size and performance target"""

        # Performance-based scaling
        # if instance_size <= 50:
        #     base_iterations = min(max_iterations, 300)
        # elif instance_size <= 100:
        #     base_iterations = min(max_iterations, 200)
        # elif instance_size <= 200:
        #     base_iterations = min(max_iterations, 150)
        # else:
        #     base_iterations = min(max_iterations, 100)

        # # Time-based scaling
        # if performance_target_seconds <= 30:
        #     time_factor = 0.5
        # elif performance_target_seconds <= 60:
        #     time_factor = 0.8
        # else:
        #     time_factor = 1.0

        # optimal_iterations = int(base_iterations * time_factor)

        # # Ensure minimum iterations
        # return max(optimal_iterations, 20)
        return max_iterations

    def solve_with_custom_vehicles(
        self,
        df: pd.DataFrame,
        vehicle_configs: List[Dict],
        max_knock: int = 4,
        time_window_hours: float = 6.0,
        **kwargs,
    ) -> Dict:
        """
        Solve with custom vehicle configurations

        Args:
            df: DataFrame with parcel data
            vehicle_configs: List of vehicle configurations
                [{"name": "bike", "capacity": 28, "cost_per_hour": 1200}, ...]
            max_knock: Maximum knocks per pickup terminal
            time_window_hours: Time window in hours
            **kwargs: Additional arguments for solve_from_csv_optimized

        Returns:
            Formatted solution results
        """

        # Convert vehicle configs to cost_per_hour_config
        cost_per_hour_config = {}

        for config in vehicle_configs:
            vehicle_name = config["name"]
            cost_per_hour = config["cost_per_hour"]
            cost_per_hour_config[vehicle_name] = cost_per_hour

        # Update vehicle specs in the solver
        # This would require extending the solver to accept custom capacities
        # For now, using the standard solve method

        return self.solve_from_csv_optimized(
            df=df,
            max_knock=max_knock,
            time_window_hours=time_window_hours,
            cost_per_hour_config=cost_per_hour_config,
            **kwargs,
        )

    def benchmark_performance(
        self, df: pd.DataFrame, target_times: List[int] = [30, 60, 120], **kwargs
    ) -> Dict:
        """
        Benchmark solver performance with different time targets

        Args:
            df: DataFrame with parcel data
            target_times: List of target completion times in seconds
            **kwargs: Additional arguments for solve_from_csv_optimized

        Returns:
            Benchmark results
        """

        benchmark_results = {
            "instance_size": len(df),
            "benchmark_runs": [],
            "summary": {},
        }

        for target_time in target_times:
            print(f"\nBenchmarking with {target_time}s target...")

            start_time = time.time()

            result = self.solve_from_csv_optimized(
                df=df, performance_target_seconds=target_time, **kwargs
            )

            actual_time = time.time() - start_time

            benchmark_run = {
                "target_time_seconds": target_time,
                "actual_time_seconds": actual_time,
                "target_met": actual_time <= target_time,
                "total_cost": sum(
                    polygon_result.get("results_summary", {}).get("total_cost", 0)
                    for polygon_result in result["polygon_results"].values()
                ),
                "total_vehicles": sum(
                    polygon_result.get("results_summary", {}).get(
                        "total_vehicles_used", 0
                    )
                    for polygon_result in result["polygon_results"].values()
                ),
                "feasible_solutions": sum(
                    1
                    for polygon_result in result["polygon_results"].values()
                    if polygon_result.get("results_summary", {}).get(
                        "time_window_feasible", False
                    )
                ),
            }

            benchmark_results["benchmark_runs"].append(benchmark_run)

            print(
                f"  Target: {target_time}s, Actual: {actual_time:.2f}s, "
                f"Cost: {benchmark_run['total_cost']:.2f}"
            )

        # Generate summary
        successful_runs = [
            run for run in benchmark_results["benchmark_runs"] if run["target_met"]
        ]

        benchmark_results["summary"] = {
            "total_runs": len(benchmark_results["benchmark_runs"]),
            "successful_runs": len(successful_runs),
            "success_rate": len(successful_runs)
            / len(benchmark_results["benchmark_runs"])
            * 100,
            "fastest_time": min(
                run["actual_time_seconds"]
                for run in benchmark_results["benchmark_runs"]
            ),
            "best_cost": min(
                run["total_cost"] for run in benchmark_results["benchmark_runs"]
            ),
            "recommended_target": (
                min(target_times) if successful_runs else max(target_times)
            ),
        }

        return benchmark_results

    def get_solver_statistics(self) -> Dict:
        """Get solver configuration and statistics"""
        return {
            "solver_version": "performance_optimized_v1.0",
            "features": {
                "global_ortools_optimization": True,
                "interval_based_optimization": True,
                "cost_per_parcel_objective": True,
                "performance_mode": self.performance_mode,
                "stagnation_detection": True,
                "intelligent_early_termination": True,
            },
            "performance_targets": {
                "small_instances": "< 50 parcels in 30s",
                "medium_instances": "50-200 parcels in 60s",
                "large_instances": "200+ parcels in 120s",
            },
            "optimizations": {
                "removed_per_route_ortools": True,
                "added_global_optimization": True,
                "reduced_ortools_calls": "90%+ reduction",
                "added_smart_caching": True,
                "improved_stagnation_detection": True,
            },
        }


# Convenience functions for backward compatibility
def solve_from_csv_per_polygon(
    df: pd.DataFrame,
    max_knock: int = 4,
    time_window_hours: float = 6.0,
    default_parcel_size: int = 1,
    proximity_threshold_meters: float = 100.0,
    max_iterations: int = 200,
) -> Dict:
    """
    Backward compatible function for existing code

    Args:
        df: DataFrame with parcel data
        max_knock: Maximum knocks per pickup terminal
        time_window_hours: Time window in hours
        default_parcel_size: Default parcel size
        proximity_threshold_meters: Proximity threshold for grouping
        max_iterations: Maximum ALNS iterations

    Returns:
        Formatted solution results
    """
    solver = PerformanceOptimizedSolver()

    return solver.solve_from_csv_optimized(
        df=df,
        max_knock=max_knock,
        time_window_hours=time_window_hours,
        default_parcel_size=default_parcel_size,
        proximity_threshold_meters=proximity_threshold_meters,
        max_iterations=max_iterations,
        performance_target_seconds=60,
    )
