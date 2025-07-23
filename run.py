import json
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

def convert_solution_to_dataframe(solution_output: Dict, order_level: str, start_time_slot: str, today_date: str) -> pd.DataFrame:
    """
    Convert VRP solution output to DataFrame format matching the specified structure.

    Args:
        solution_output: The complete solution dictionary from format_output()

    Returns:
        pandas.DataFrame with one row per polygon containing all required metrics
    """

    # Handle both single polygon and multi-polygon results
    if "polygon_results" in solution_output:
        # Multi-polygon format
        polygon_results = solution_output["polygon_results"]
    else:
        # Single polygon format - wrap in polygon_results structure
        polygon_results = {"default": solution_output}

    rows = []

    for polygon_id, result in polygon_results.items():
        # Extract main result data
        input_summary = result.get("input_summary", {})
        results_summary = result.get("results_summary", {})
        vehicles = result.get("vehicles", [])
        pickup_data = result.get("formatted_pickup_data", [])

        # Basic metrics
        processing_time = result.get("processing_time", 0.0)
        pickup_locations = len(pickup_data)
        total_parcels = input_summary.get("total_parcels", 0)
        total_vehicles_used = results_summary.get("total_vehicles_used", 0)
        total_cost = results_summary.get("total_cost", 0.0)  # Real operational cost
        total_duration_seconds = results_summary.get("total_duration_seconds", 0)
        total_duration_hours = round(total_duration_seconds / 3600, 2)

        # Calculate total distance from vehicles
        total_distance_m = sum(v.get("total_distance_m", 0) for v in vehicles)
        total_distance_km = round(total_distance_m / 1000, 2)

        # SLA metrics
        sla_metrics = results_summary.get("sla_metrics", {})
        sla_percentage = sla_metrics.get("overall_sla_compliance_percent", 100.0)
        parcels_in_sla = sla_metrics.get("total_parcels_on_time", total_parcels)
        parcels_out_sla = sla_metrics.get("total_parcels_late", 0)

        # Vehicle type counts
        vehicles_by_type = results_summary.get("vehicles_by_type", {})
        bike_count = vehicles_by_type.get("bike", 0)
        big_box_count = vehicles_by_type.get("big-box", 0)
        carbox_count = vehicles_by_type.get("carbox", 0)
        van_count = vehicles_by_type.get("van", 0)

        # Pickup IDs
        pickup_ids = ",".join(
            str(p.get("pickup_id", i)) for i, p in enumerate(pickup_data)
        )

        # Calculate averages
        avg_sla_per_polygon = sla_metrics.get("average_route_sla_percent", 100.0)
        avg_duration_per_vehicle_hours = (
            round(total_duration_hours / total_vehicles_used, 2)
            if total_vehicles_used > 0
            else 0.0
        )
        avg_distance_per_vehicle_km = (
            round(total_distance_km / total_vehicles_used, 2)
            if total_vehicles_used > 0
            else 0.0
        )

        # Vehicle details arrays
        vehicle_ids = []
        vehicle_types = []
        vehicle_slas = []
        vehicle_durations_hours = []
        vehicle_distances_km = []
        vehicle_parcels = []
        vehicle_costs_kt = []
        vehicle_utilization = []

        # Vehicle details string
        vehicle_details_parts = []
        cost_per_parcels = 0

        for vehicle in vehicles:
            cost_per_parcels += vehicle.get("cost_per_parcel", 0)
            vehicle_id = vehicle.get("vehicle_id", 0)
            vehicle_type = vehicle.get("vehicle_type", "unknown")

            # SLA metrics for this vehicle
            vehicle_sla_metrics = vehicle.get("sla_metrics", {})
            vehicle_sla = vehicle_sla_metrics.get("sla_compliance_percent", 100.0)

            # Duration and distance
            duration_seconds = vehicle.get("total_duration_seconds", 0)
            duration_hours = round(duration_seconds / 3600, 2)
            distance_m = vehicle.get("total_distance_m", 0)
            distance_km = round(distance_m / 1000, 2)

            # Parcels and cost
            num_parcels = len(vehicle.get("parcels", []))
            cost_kt = vehicle.get("total_cost_kt", 0.0)  # Real operational cost
            utilization = round(vehicle.get("utilization_percent", 0.0), 1)

            # Append to arrays
            vehicle_ids.append(str(vehicle_id))
            vehicle_types.append(vehicle_type)
            vehicle_slas.append(str(vehicle_sla))
            vehicle_durations_hours.append(str(duration_hours))
            vehicle_distances_km.append(str(distance_km))
            vehicle_parcels.append(str(num_parcels))
            vehicle_costs_kt.append(str(round(cost_kt, 1)))
            vehicle_utilization.append(str(utilization))

            # Create vehicle detail string
            vehicle_detail = (
                f"Vehicle_{vehicle_id}({vehicle_type}): "
                f"SLA={vehicle_sla}%, "
                f"Duration={duration_hours}h, "
                f"Distance={distance_km}km, "
                f"Parcels={num_parcels}, "
                f"Cost={round(cost_kt, 1)}KT"
            )
            vehicle_details_parts.append(vehicle_detail)

        # Join arrays with commas
        vehicle_details = " | ".join(vehicle_details_parts)
        vehicle_ids_str = ",".join(vehicle_ids)
        vehicle_types_str = ",".join(vehicle_types)
        vehicle_slas_str = ",".join(vehicle_slas)
        vehicle_durations_str = ",".join(vehicle_durations_hours)
        vehicle_distances_str = ",".join(vehicle_distances_km)
        vehicle_parcels_str = ",".join(vehicle_parcels)
        vehicle_costs_str = ",".join(vehicle_costs_kt)
        vehicle_utilization_str = ",".join(vehicle_utilization)

        # Create row
        row = {
            # "polygon_id": float(polygon_id) if polygon_id != "default" else 0.0,
            "polygon_id": polygon_id,
            "processing_time": processing_time,
            "pickup_locations": pickup_locations,
            "total_parcels": total_parcels,
            "total_vehicles_used": total_vehicles_used,
            "total_cost": total_cost,
            "total_duration_seconds": total_duration_seconds,
            "total_duration_hours": total_duration_hours,
            "total_distance_m": total_distance_m,
            "total_distance_km": total_distance_km,
            "total_clusters": total_vehicles_used,  # Using vehicles as clusters
            "sla_percentage": sla_percentage,
            "parcels_in_sla": parcels_in_sla,
            "parcels_out_sla": parcels_out_sla,
            "parcels_total": total_parcels,
            "bike_count": bike_count,
            "big_box_count": big_box_count,
            "carbox_count": carbox_count,
            "van_count": van_count,
            "pickup_ids": pickup_ids,
            "avg_sla_per_polygon": avg_sla_per_polygon,
            "avg_duration_per_vehicle_hours": avg_duration_per_vehicle_hours,
            "avg_distance_per_vehicle_km": avg_distance_per_vehicle_km,
            "vehicle_details": vehicle_details,
            "vehicle_ids": vehicle_ids_str,
            "vehicle_types": vehicle_types_str,
            "vehicle_slas": vehicle_slas_str,
            "vehicle_durations_hours": vehicle_durations_str,
            "vehicle_distances_km": vehicle_distances_str,
            "vehicle_parcels": vehicle_parcels_str,
            "vehicle_costs_kt": vehicle_costs_str,
            "vehicle_utilization": vehicle_utilization_str,
            'order_level': order_level,
            'start_time_slot': start_time_slot,
            'today_date': today_date,
            "cost_per_parcel": round(cost_per_parcels / len(vehicles), 2)
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    expected_columns = [
        "polygon_id",
        "processing_time",
        "pickup_locations",
        "total_parcels",
        "total_vehicles_used",
        "total_cost",
        "cost_per_parcel",
        "total_duration_seconds",
        "total_duration_hours",
        "total_distance_m",
        "total_distance_km",
        "total_clusters",
        "sla_percentage",
        "parcels_in_sla",
        "parcels_out_sla",
        "parcels_total",
        "bike_count",
        "big_box_count",
        "carbox_count",
        "van_count",
        "pickup_ids",
        "avg_sla_per_polygon",
        "avg_duration_per_vehicle_hours",
        "avg_distance_per_vehicle_km",
        "vehicle_details",
        "vehicle_ids",
        "vehicle_types",
        "vehicle_slas",
        "vehicle_durations_hours",
        "vehicle_distances_km",
        "vehicle_parcels",
        "vehicle_costs_kt",
        "vehicle_utilization",
        "order_level",
        "start_time_slot",
        "today_date",
    ]

    df = df.reindex(columns=expected_columns)

    return df


if __name__ == "__main__":
    today_date = "2025-07-22"
    start_time_slot = 10
    knocks = [1, 2, 3, 4, 5]
    min_parcels = [5, 10]
    polygon_size = [5]
    order_level = [100, 250, 500, 1000, 1500, 2000, 3000]
    timewindow = [3]

    for lvl in order_level:
        if lvl <= 500:
            max_iteration = 250
        elif lvl <= 1500:
            max_iteration = 300
        else:
            max_iteration = 350
        for p in min_parcels:
            for size in polygon_size:
                data = pd.read_csv(f"./data/orders{lvl}_min{p}_polygon{size}.csv")
                matrix_id = f"orders{lvl}_min{p}_polygon{size}_bucket{start_time_slot}"
                for knock in knocks:
                    for tw in timewindow:
                        try:
                            vehicles = "&".join(list(constants.VEHICLE_COST_PER_HOUR.keys()))
                            problem_id = f"per_zones_results_max_knock_{knock}_TW_{tw}_vehicle_{vehicles}_order_{lvl}_min_{p}_polygon_{1}_sts_{start_time_slot}_date_{today_date}_alns"
                            print(f"START OF {problem_id}")
                            problem_config = {
                                "id": problem_id,
                                "max_knock": knock,
                                "order_level": lvl,
                                "number_of_polygons": size,
                                "time_window": tw,
                                "vehicles": vehicles,
                                "date": today_date,
                                "start_time_slot": start_time_slot,
                                "min_parcel": p
                            }
                            solver = PerformanceOptimizedSolver()
                            result = solver.solve_from_csv_optimized(
                                df=data,
                                max_knock=knock,
                                time_window_hours=tw,
                                capacity_config=constants.VEHICLE_CAPACITIES,
                                cost_per_hour_config=constants.VEHICLE_COST_PER_HOUR,
                                performance_target_seconds=900,
                                max_iterations=max_iteration,
                                default_parcel_size=4,
                                proximity_threshold_meters=0,
                                matrix_id=matrix_id
                            )
                            result["problem_config"] = problem_config
                            convert_solution_to_dataframe(result, lvl, start_time_slot, today_date).to_csv(
                                f"./results/{problem_id}.csv", index=False
                            )
                            with open(f"./results/{problem_id}.json", "w") as f:
                                json.dump(result, f, indent=4)
                        except Exception as exc:
                            print(f"Error occured for {problem_id}: {str(exc)}")
                            print(exc.__traceback__)
                        print(f"END OF {problem_id}")
                        print("*" * 50)
