import constants
import pandas as pd

from typing import List, Dict, Tuple

from main_solver import PerformanceOptimizedSolver
from data_structures import ProblemConfig, PickupTerminal


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
        performance_target_seconds=60,  # Default 60s target
    )


def solve_with_time_budget(
    df: pd.DataFrame,
    time_budget_seconds: int,
    max_knock: int = 4,
    time_window_hours: float = 6.0,
    **kwargs,
) -> Dict:
    """
    Solve with a specific time budget

    Args:
        df: DataFrame with parcel data
        time_budget_seconds: Maximum time allowed for solving
        max_knock: Maximum knocks per pickup terminal
        time_window_hours: Time window in hours
        **kwargs: Additional arguments

    Returns:
        Formatted solution results
    """
    solver = PerformanceOptimizedSolver()

    return solver.solve_from_csv_optimized(
        df=df,
        max_knock=max_knock,
        time_window_hours=time_window_hours,
        performance_target_seconds=time_budget_seconds,
        **kwargs,
    )


def create_problem_from_csv(
    df: pd.DataFrame,
    max_knock: int = 4,
    time_window_hours: float = 6.0,
    default_parcel_size: int = 1,
    proximity_threshold_meters: float = 100.0,
) -> Tuple[ProblemConfig, List[PickupTerminal], Dict]:
    """
    Backward compatible function for problem creation

    Args:
        df: DataFrame with parcel data
        max_knock: Maximum knocks per pickup terminal
        time_window_hours: Time window in hours
        default_parcel_size: Default parcel size
        proximity_threshold_meters: Proximity threshold for grouping

    Returns:
        Tuple of (ProblemConfig, List[PickupTerminal], data_summary)
    """
    solver = PerformanceOptimizedSolver()

    # Default cost configuration
    cost_per_hour_config = {"bike": 1200.0, "carbox": 2000.0, "big-box": 1500.0}

    config, pickup_terminals, data_summary = solver._create_problem_from_dataframe(
        df,
        max_knock,
        time_window_hours,
        cost_per_hour_config,
        default_parcel_size,
        proximity_threshold_meters,
    )

    return config, pickup_terminals, data_summary


# Example usage
if __name__ == "__main__":
    # Example of how to use the updated solver

    # Load your data
    df = pd.read_csv("sampled_unique.csv")
    # df = df[df["polygon_id"] == "polyon_2"].reset_index(drop=True)
    # Create solver
    solver = PerformanceOptimizedSolver()

    # Solve with performance optimization
    result = solver.solve_from_csv_optimized(
        df=df,
        max_knock=2,
        time_window_hours=3.0,
        capacity_config=constants.VEHICLE_CAPACITIES,
        cost_per_hour_config=constants.VEHICLE_COST_PER_HOUR,
        performance_target_seconds=900,  # Complete in 60 seconds
        max_iterations=200,
        default_parcel_size=4,
        proximity_threshold_meters=0,
    )
    print(result)
    # Print results
    print(f"Solved in {result['total_processing_time']:.2f} seconds")

    for polygon_id, polygon_result in result["polygon_results"].items():
        summary = polygon_result["results_summary"]
        print(f"Polygon {polygon_id}:")
        print(f"  - Vehicles: {summary['total_vehicles_used']}")
        print(f"  - Cost: {summary['total_cost']:.2f}")
        print(f"  - Total duration (seconds): {summary['total_duration_seconds']:.2f}")
        print(f"  - Feasible: {summary['time_window_feasible']}")
