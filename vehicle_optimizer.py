from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import time

from data_structures import Route, Solution, VehicleType, VehicleSpec, ProblemConfig


class UpdatedVehicleOptimizer:
    """
    Updated vehicle optimizer focused on cost per parcel optimization
    Works with global OR-Tools optimizer instead of individual route optimization
    """

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.vehicle_specs = config.vehicle_specs
        self.verbose_logging = False

    def optimize_vehicle_types_for_cost(self, solution: Solution) -> Solution:
        """
        Optimize vehicle types based on cost per parcel

        Args:
            solution: Current solution to optimize

        Returns:
            Solution with optimized vehicle types
        """
        if not solution.routes:
            return solution

        optimized_solution = deepcopy(solution)
        improvements_made = False

        if self.verbose_logging:
            print(f"Optimizing vehicle types for {len(solution.routes)} routes")

        for route in optimized_solution.routes:
            # Find best vehicle type for this route's parcel count
            best_vehicle_type = self._find_best_vehicle_type_for_route(route)

            if best_vehicle_type != route.vehicle_type:
                old_cost_per_parcel = route.cost_per_parcel

                # Update vehicle type
                route.vehicle_type = best_vehicle_type
                route.vehicle_spec = self.vehicle_specs[best_vehicle_type]
                route.update_costs()

                if self.verbose_logging:
                    print(
                        f"  Route {route.vehicle_id}: {route.vehicle_type.value} -> {best_vehicle_type.value}"
                    )
                    print(
                        f"    Cost per parcel: {old_cost_per_parcel:.2f} -> {route.cost_per_parcel:.2f}"
                    )

                improvements_made = True

        if improvements_made:
            optimized_solution.update_solution_costs()

        return optimized_solution

    def _find_best_vehicle_type_for_route(self, route: Route) -> VehicleType:
        """Find the best vehicle type for a route based on cost per parcel"""
        parcel_count = len(route.parcels)
        route_size = route.total_size

        best_vehicle_type = route.vehicle_type
        best_cost_per_parcel = route.cost_per_parcel

        # Try each vehicle type that can fit the route
        for vehicle_type, vehicle_spec in self.vehicle_specs.items():
            if vehicle_spec.capacity >= route_size:
                cost_per_parcel = vehicle_spec.calculate_cost_per_parcel(parcel_count)

                if cost_per_parcel < best_cost_per_parcel:
                    best_cost_per_parcel = cost_per_parcel
                    best_vehicle_type = vehicle_type

        return best_vehicle_type

    def suggest_route_consolidation_opportunities(
        self, solution: Solution
    ) -> List[Dict]:
        """
        Suggest route consolidation opportunities based on cost efficiency

        Args:
            solution: Current solution

        Returns:
            List of consolidation suggestions
        """
        suggestions = []

        if len(solution.routes) < 2:
            return suggestions

        # Group routes by pickup terminals
        terminal_groups = self._group_routes_by_terminals(solution.routes)

        for terminal_id, routes in terminal_groups.items():
            if len(routes) > 1:
                consolidation_suggestion = self._analyze_consolidation_for_terminal(
                    terminal_id, routes
                )
                if consolidation_suggestion:
                    suggestions.append(consolidation_suggestion)

        # Check cross-terminal consolidation opportunities
        cross_terminal_suggestions = self._analyze_cross_terminal_consolidation(
            solution
        )
        suggestions.extend(cross_terminal_suggestions)

        return suggestions

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

    def _analyze_consolidation_for_terminal(
        self, terminal_id: int, routes: List[Route]
    ) -> Optional[Dict]:
        """Analyze consolidation opportunity for routes from same terminal"""
        if len(routes) <= 1:
            return None

        # Calculate current costs
        current_total_cost = sum(route.total_cost for route in routes)
        current_total_parcels = sum(len(route.parcels) for route in routes)
        current_cost_per_parcel = current_total_cost / current_total_parcels

        # Calculate total size needed
        total_size = sum(route.total_size for route in routes)

        # Find best vehicle type for consolidation
        best_consolidation = None
        best_cost_per_parcel = current_cost_per_parcel

        for vehicle_type, vehicle_spec in self.vehicle_specs.items():
            if vehicle_spec.capacity >= total_size:
                consolidated_cost_per_parcel = vehicle_spec.calculate_cost_per_parcel(
                    current_total_parcels
                )

                if consolidated_cost_per_parcel < best_cost_per_parcel:
                    best_cost_per_parcel = consolidated_cost_per_parcel
                    best_consolidation = {
                        "terminal_id": terminal_id,
                        "current_routes": len(routes),
                        "consolidated_routes": 1,
                        "vehicle_type": vehicle_type.value,
                        "current_cost_per_parcel": current_cost_per_parcel,
                        "consolidated_cost_per_parcel": consolidated_cost_per_parcel,
                        "improvement_percent": (
                            (current_cost_per_parcel - consolidated_cost_per_parcel)
                            / current_cost_per_parcel
                        )
                        * 100,
                        "total_parcels": current_total_parcels,
                        "total_size": total_size,
                        "vehicle_capacity": vehicle_spec.capacity,
                        "utilization_percent": (total_size / vehicle_spec.capacity)
                        * 100,
                    }

        return best_consolidation

    def _analyze_cross_terminal_consolidation(self, solution: Solution) -> List[Dict]:
        """Analyze cross-terminal consolidation opportunities"""
        suggestions = []

        routes = solution.routes
        if len(routes) < 2:
            return suggestions

        # Check pairs of routes from different terminals
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                route1, route2 = routes[i], routes[j]

                # Skip if same terminal
                if (
                    route1.pickup_sequence
                    and route2.pickup_sequence
                    and route1.pickup_sequence[0] == route2.pickup_sequence[0]
                ):
                    continue

                # Calculate combined requirements
                combined_parcels = len(route1.parcels) + len(route2.parcels)
                combined_size = route1.total_size + route2.total_size

                # Current cost
                current_cost = route1.total_cost + route2.total_cost
                current_cost_per_parcel = current_cost / combined_parcels

                # Check if consolidation is possible
                for vehicle_type, vehicle_spec in self.vehicle_specs.items():
                    if vehicle_spec.capacity >= combined_size:
                        consolidated_cost_per_parcel = (
                            vehicle_spec.calculate_cost_per_parcel(combined_parcels)
                        )

                        # Check if improvement is significant
                        improvement_percent = (
                            (current_cost_per_parcel - consolidated_cost_per_parcel)
                            / current_cost_per_parcel
                        ) * 100

                        if improvement_percent > 5:  # At least 5% improvement
                            suggestions.append(
                                {
                                    "type": "cross_terminal",
                                    "route1_id": route1.vehicle_id,
                                    "route2_id": route2.vehicle_id,
                                    "route1_terminal": (
                                        route1.pickup_sequence[0]
                                        if route1.pickup_sequence
                                        else 0
                                    ),
                                    "route2_terminal": (
                                        route2.pickup_sequence[0]
                                        if route2.pickup_sequence
                                        else 0
                                    ),
                                    "vehicle_type": vehicle_type.value,
                                    "current_cost_per_parcel": current_cost_per_parcel,
                                    "consolidated_cost_per_parcel": consolidated_cost_per_parcel,
                                    "improvement_percent": improvement_percent,
                                    "total_parcels": combined_parcels,
                                    "total_size": combined_size,
                                    "vehicle_capacity": vehicle_spec.capacity,
                                    "utilization_percent": (
                                        combined_size / vehicle_spec.capacity
                                    )
                                    * 100,
                                }
                            )
                            break  # Found best vehicle type for this pair

        return suggestions

    def calculate_fleet_efficiency_metrics(self, solution: Solution) -> Dict:
        """Calculate comprehensive fleet efficiency metrics"""
        if not solution.routes:
            return {
                "total_vehicles": 0,
                "total_cost": 0.0,
                "total_parcels": 0,
                "average_cost_per_parcel": 0.0,
                "vehicle_type_analysis": {},
                "efficiency_recommendations": [],
            }

        # Basic metrics
        total_vehicles = len(solution.routes)
        total_cost = solution.total_cost
        total_parcels = solution.total_parcels
        average_cost_per_parcel = solution.average_cost_per_parcel

        # Vehicle type analysis
        vehicle_type_analysis = {}
        for route in solution.routes:
            vtype = route.vehicle_type.value

            if vtype not in vehicle_type_analysis:
                vehicle_type_analysis[vtype] = {
                    "count": 0,
                    "total_cost": 0.0,
                    "total_parcels": 0,
                    "total_capacity": 0,
                    "total_used_capacity": 0,
                    "routes": [],
                }

            vtype_data = vehicle_type_analysis[vtype]
            vtype_data["count"] += 1
            vtype_data["total_cost"] += route.total_cost
            vtype_data["total_parcels"] += len(route.parcels)
            vtype_data["total_capacity"] += route.vehicle_spec.capacity
            vtype_data["total_used_capacity"] += route.total_size

            vtype_data["routes"].append(
                {
                    "vehicle_id": route.vehicle_id,
                    "parcels": len(route.parcels),
                    "capacity_used": route.total_size,
                    "capacity_total": route.vehicle_spec.capacity,
                    "utilization_percent": route.get_utilization_percentage(),
                    "cost_per_parcel": route.cost_per_parcel,
                }
            )

        # Calculate averages for each vehicle type
        for vtype, data in vehicle_type_analysis.items():
            if data["count"] > 0:
                data["average_cost_per_parcel"] = (
                    data["total_cost"] / data["total_parcels"]
                )
                data["average_utilization"] = (
                    data["total_used_capacity"] / data["total_capacity"]
                ) * 100
                data["cost_efficiency_rank"] = data[
                    "average_cost_per_parcel"
                ]  # Lower is better

        # Generate efficiency recommendations
        efficiency_recommendations = self._generate_efficiency_recommendations(solution)

        return {
            "total_vehicles": total_vehicles,
            "total_cost": total_cost,
            "total_parcels": total_parcels,
            "average_cost_per_parcel": average_cost_per_parcel,
            "vehicle_type_analysis": vehicle_type_analysis,
            "efficiency_recommendations": efficiency_recommendations,
            "consolidation_opportunities": self.suggest_route_consolidation_opportunities(
                solution
            ),
        }

    def _generate_efficiency_recommendations(self, solution: Solution) -> List[Dict]:
        """Generate recommendations for improving fleet efficiency"""
        recommendations = []

        if not solution.routes:
            return recommendations

        # Check for underutilized vehicles
        underutilized_routes = [
            r for r in solution.routes if r.get_utilization_percentage() < 50
        ]
        if underutilized_routes:
            recommendations.append(
                {
                    "type": "underutilization",
                    "priority": "high",
                    "description": f"{len(underutilized_routes)} routes have <50% capacity utilization",
                    "suggestion": "Consider consolidating these routes or switching to smaller vehicles",
                    "affected_routes": [r.vehicle_id for r in underutilized_routes],
                    "potential_savings": self._calculate_underutilization_savings(
                        underutilized_routes
                    ),
                }
            )

        # Check for oversized vehicles
        oversized_routes = []
        for route in solution.routes:
            # Find smallest vehicle that can fit this route
            min_capacity_needed = route.total_size
            current_capacity = route.vehicle_spec.capacity

            smaller_vehicles = [
                (vtype, spec)
                for vtype, spec in self.vehicle_specs.items()
                if spec.capacity >= min_capacity_needed
                and spec.capacity < current_capacity
            ]

            if smaller_vehicles:
                # Find the best smaller vehicle
                best_smaller = min(
                    smaller_vehicles,
                    key=lambda x: x[1].calculate_cost_per_parcel(len(route.parcels)),
                )
                oversized_routes.append((route, best_smaller))

        if oversized_routes:
            total_savings = sum(
                route.cost_per_parcel
                - smaller_spec.calculate_cost_per_parcel(len(route.parcels))
                for route, (smaller_type, smaller_spec) in oversized_routes
            )

            recommendations.append(
                {
                    "type": "oversized_vehicles",
                    "priority": "medium",
                    "description": f"{len(oversized_routes)} routes could use smaller, more cost-effective vehicles",
                    "suggestion": "Switch to smaller vehicle types for these routes",
                    "affected_routes": [
                        route.vehicle_id for route, _ in oversized_routes
                    ],
                    "potential_savings_per_parcel": total_savings
                    / len(oversized_routes),
                }
            )

        # Check for consolidation opportunities
        consolidation_opportunities = self.suggest_route_consolidation_opportunities(
            solution
        )
        high_value_opportunities = [
            opp
            for opp in consolidation_opportunities
            if opp.get("improvement_percent", 0) > 15
        ]

        if high_value_opportunities:
            recommendations.append(
                {
                    "type": "consolidation",
                    "priority": "high",
                    "description": f"{len(high_value_opportunities)} high-value consolidation opportunities found",
                    "suggestion": "Merge routes to reduce total vehicle count and cost",
                    "opportunities": high_value_opportunities,
                    "potential_improvement": sum(
                        opp.get("improvement_percent", 0)
                        for opp in high_value_opportunities
                    ),
                }
            )

        # Check for balanced utilization
        utilizations = [r.get_utilization_percentage() for r in solution.routes]
        if utilizations:
            utilization_std = (
                sum(
                    (u - sum(utilizations) / len(utilizations)) ** 2
                    for u in utilizations
                )
                / len(utilizations)
            ) ** 0.5

            if utilization_std > 25:  # High variation in utilization
                recommendations.append(
                    {
                        "type": "unbalanced_utilization",
                        "priority": "medium",
                        "description": f"High variation in capacity utilization (std dev: {utilization_std:.1f}%)",
                        "suggestion": "Redistribute parcels between routes for more balanced utilization",
                        "utilization_range": f"{min(utilizations):.1f}% - {max(utilizations):.1f}%",
                    }
                )

        return recommendations

    def _calculate_underutilization_savings(
        self, underutilized_routes: List[Route]
    ) -> float:
        """Calculate potential savings from addressing underutilization"""
        if not underutilized_routes:
            return 0.0

        # Simple estimation: if we could consolidate half of these routes
        routes_to_consolidate = len(underutilized_routes) // 2

        # Average cost per route
        avg_cost_per_route = sum(r.total_cost for r in underutilized_routes) / len(
            underutilized_routes
        )

        # Potential savings from eliminating routes
        potential_savings = routes_to_consolidate * avg_cost_per_route

        return potential_savings

    def optimize_vehicle_assignment_simple(self, solution: Solution) -> Solution:
        """
        Simple vehicle assignment optimization without complex consolidation
        Useful when global optimizer is not available or for quick improvements
        """
        optimized_solution = deepcopy(solution)
        improvements_made = False

        # First pass: optimize individual route vehicle types
        for route in optimized_solution.routes:
            original_cost = route.cost_per_parcel
            best_vehicle_type = self._find_best_vehicle_type_for_route(route)

            if best_vehicle_type != route.vehicle_type:
                route.vehicle_type = best_vehicle_type
                route.vehicle_spec = self.vehicle_specs[best_vehicle_type]
                route.update_costs()
                improvements_made = True

                if self.verbose_logging:
                    print(
                        f"Route {route.vehicle_id}: vehicle type optimized, "
                        f"cost per parcel: {original_cost:.2f} -> {route.cost_per_parcel:.2f}"
                    )

        # Second pass: simple consolidation within same terminals
        terminal_groups = self._group_routes_by_terminals(optimized_solution.routes)

        for terminal_id, routes in terminal_groups.items():
            if len(routes) > 1:
                consolidated_route = self._try_simple_consolidation(routes)
                if consolidated_route:
                    # Calculate cost improvement
                    original_total_cost = sum(r.total_cost for r in routes)
                    if consolidated_route.total_cost < original_total_cost:
                        # Apply consolidation
                        for route in routes:
                            optimized_solution.remove_route(route)
                        optimized_solution.add_route(consolidated_route)
                        improvements_made = True

                        if self.verbose_logging:
                            print(
                                f"Terminal {terminal_id}: consolidated {len(routes)} routes into 1, "
                                f"cost: {original_total_cost:.2f} -> {consolidated_route.total_cost:.2f}"
                            )

        if improvements_made:
            optimized_solution.update_solution_costs()

        return optimized_solution

    def _try_simple_consolidation(self, routes: List[Route]) -> Optional[Route]:
        """Try simple consolidation of routes from same terminal"""
        if len(routes) <= 1:
            return None

        # Calculate total requirements
        total_parcels = []
        total_size = 0
        pickup_sequences = set()

        for route in routes:
            total_parcels.extend(route.parcels)
            total_size += route.total_size
            pickup_sequences.update(route.pickup_sequence)

        # Find best vehicle type for consolidation
        best_vehicle_type = None
        best_cost_per_parcel = float("inf")

        for vehicle_type, vehicle_spec in self.vehicle_specs.items():
            if vehicle_spec.capacity >= total_size:
                cost_per_parcel = vehicle_spec.calculate_cost_per_parcel(
                    len(total_parcels)
                )
                if cost_per_parcel < best_cost_per_parcel:
                    best_cost_per_parcel = cost_per_parcel
                    best_vehicle_type = vehicle_type

        if not best_vehicle_type:
            return None

        # Create consolidated route
        try:
            consolidated_route = Route(
                vehicle_id=routes[0].vehicle_id,  # Use first vehicle ID
                vehicle_type=best_vehicle_type,
                vehicle_spec=self.vehicle_specs[best_vehicle_type],
                parcels=total_parcels,
                pickup_sequence=list(pickup_sequences),
                delivery_sequence=[p.delivery_location for p in total_parcels],
            )

            return consolidated_route

        except Exception as e:
            if self.verbose_logging:
                print(f"Failed to create consolidated route: {e}")
            return None

    def get_vehicle_type_recommendations(
        self, solution: Solution
    ) -> Dict[VehicleType, Dict]:
        """Get recommendations for each vehicle type usage"""
        recommendations = {}

        for vehicle_type, vehicle_spec in self.vehicle_specs.items():
            # Find routes using this vehicle type
            routes_with_type = [
                r for r in solution.routes if r.vehicle_type == vehicle_type
            ]

            if not routes_with_type:
                recommendations[vehicle_type] = {
                    "current_usage": 0,
                    "recommendation": "unused",
                    "cost_per_parcel": vehicle_spec.calculate_cost_per_parcel(1),
                    "optimal_parcel_range": self._find_optimal_parcel_range(
                        vehicle_spec
                    ),
                    "notes": "Consider using for single-parcel deliveries or small batches",
                }
                continue

            # Calculate usage statistics
            utilizations = [r.get_utilization_percentage() for r in routes_with_type]
            costs_per_parcel = [r.cost_per_parcel for r in routes_with_type]

            avg_utilization = sum(utilizations) / len(utilizations)
            avg_cost_per_parcel = sum(costs_per_parcel) / len(costs_per_parcel)

            # Generate recommendation
            if avg_utilization < 40:
                recommendation = "underutilized"
                notes = "Consider consolidating routes or switching to smaller vehicles"
            elif avg_utilization > 90:
                recommendation = "overutilized"
                notes = "Consider splitting routes or switching to larger vehicles"
            else:
                recommendation = "well_utilized"
                notes = "Good utilization level"

            recommendations[vehicle_type] = {
                "current_usage": len(routes_with_type),
                "recommendation": recommendation,
                "average_utilization": avg_utilization,
                "average_cost_per_parcel": avg_cost_per_parcel,
                "utilization_range": f"{min(utilizations):.1f}% - {max(utilizations):.1f}%",
                "optimal_parcel_range": self._find_optimal_parcel_range(vehicle_spec),
                "notes": notes,
            }

        return recommendations

    def _find_optimal_parcel_range(self, vehicle_spec: VehicleSpec) -> Tuple[int, int]:
        """Find optimal parcel count range for a vehicle type"""
        # Calculate cost per parcel for different parcel counts
        costs = []
        for parcel_count in range(1, vehicle_spec.capacity + 1):
            cost = vehicle_spec.calculate_cost_per_parcel(parcel_count)
            costs.append((parcel_count, cost))

        # Find the range where cost per parcel is minimized
        min_cost = min(costs, key=lambda x: x[1])[1]

        # Find range within 10% of minimum cost
        optimal_range = [
            parcel_count for parcel_count, cost in costs if cost <= min_cost * 1.1
        ]

        if optimal_range:
            return (min(optimal_range), max(optimal_range))
        else:
            return (1, vehicle_spec.capacity)
