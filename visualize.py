import folium
import json
import random
from typing import Dict, List, Tuple, Optional
import branca.colormap as cm
from folium import plugins
import pandas as pd

class VRPRouteVisualizer:
    """
    Advanced Folium visualization for VRP solutions with multiple polygons
    """
    
    def __init__(self):
        # Define distinct colors for individual routes
        self.route_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3',
            '#FF9F43', '#10AC84', '#EE5A24', '#0097E6', '#8C7AE6',
            '#FF3838', '#17A2B8', '#28A745', '#FFC107', '#6F42C1',
            '#FD7E14', '#20C997', '#E83E8C', '#6C757D', '#343A40',
            '#FF1744', '#00BCD4', '#4CAF50', '#FF9800', '#9C27B0',
            '#FF5722', '#009688', '#E91E63', '#795548', '#607D8B'
        ]
        
        # Define icons for different terminal types
        self.terminal_icons = {
            'pickup': 'arrow-up',
            'delivery': 'arrow-down',
            'both': 'arrows'
        }
        
        # Polygon colors for different solutions (lighter colors for backgrounds)
        self.polygon_colors = [
            '#FFE0E0', '#E0F7F7', '#E0F0FF', '#E0F7E0', '#FFF7E0',
            '#F0E0F0', '#FFE0FF', '#E0E8FF', '#E8E0FF', '#E0FFFF'
        ]
    
    def create_multi_polygon_map(self, solutions_dict: Dict[str, Dict], 
                                distance_matrix: Optional[Dict] = None,
                                eta_matrix: Optional[Dict] = None,
                                center_lat: float = None, 
                                center_lon: float = None,
                                zoom_start: int = 11) -> folium.Map:
        """
        Create a comprehensive map with multiple VRP solutions
        
        Args:
            solutions_dict: Dictionary where keys are polygon_ids and values are solution dictionaries
            distance_matrix: Optional distance matrix for edge information
            eta_matrix: Optional ETA matrix for time information
            center_lat: Map center latitude (auto-calculated if None)
            center_lon: Map center longitude (auto-calculated if None)
            zoom_start: Initial zoom level
            
        Returns:
            Folium map object
        """
        # Auto-calculate center if not provided
        if center_lat is None or center_lon is None:
            center_lat, center_lon = self._calculate_map_center(solutions_dict)
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
        
        # Create feature groups for each polygon
        polygon_groups = {}
        
        for polygon_id, solution in solutions_dict.items():
            # Create feature group for this polygon
            polygon_color = self.polygon_colors[hash(str(polygon_id)) % len(self.polygon_colors)]
            group_name = f"Polygon {polygon_id}"
            
            polygon_group = folium.FeatureGroup(name=group_name)
            polygon_groups[polygon_id] = polygon_group
            
            # Add routes and terminals for this polygon
            self._add_polygon_routes(
                polygon_group, solution, polygon_id, polygon_color,
                distance_matrix, eta_matrix
            )
            
            # Add summary info for this polygon
            self._add_polygon_summary(polygon_group, solution, polygon_id, polygon_color)
            
            # Add to map
            polygon_group.add_to(m)
        
        # Add layer control at the end
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add legend
        self._add_legend(m, solutions_dict)
        
        # Add distance/time measurement tool
        plugins.MeasureControl().add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        return m
    
    def _calculate_map_center(self, solutions_dict: Dict[str, Dict]) -> Tuple[float, float]:
        """Calculate optimal map center from all solutions"""
        all_lats = []
        all_lons = []
        
        for solution in solutions_dict.values():
            # Get pickup terminal coordinates
            for pickup_data in solution.get('formatted_pickup_data', []):
                all_lats.append(pickup_data['lat'])
                all_lons.append(pickup_data['lon'])
            
            # Get delivery coordinates from routes
            for vehicle in solution.get('vehicles', []):
                for coord in vehicle.get('physical_route', []):
                    if len(coord) == 2:
                        all_lats.append(coord[0])
                        all_lons.append(coord[1])
        
        if all_lats and all_lons:
            center_lat = sum(all_lats) / len(all_lats)
            center_lon = sum(all_lons) / len(all_lons)
        else:
            center_lat, center_lon = 35.6892, 51.3890  # Default to Tehran
        
        return center_lat, center_lon
    
    def _add_polygon_routes(self, feature_group: folium.FeatureGroup, 
                           solution: Dict, polygon_id: str, polygon_color: str,
                           distance_matrix: Optional[Dict], eta_matrix: Optional[Dict]):
        """Add all routes for a single polygon"""
        
        # Add pickup terminals
        self._add_pickup_terminals(feature_group, solution, polygon_color)
        
        # Add vehicle routes with unique colors for each route
        for idx, vehicle in enumerate(solution.get('vehicles', [])):
            # Assign unique color to each route
            route_color = self.route_colors[idx % len(self.route_colors)]
            
            self._add_vehicle_route(
                feature_group, vehicle, solution, polygon_id, route_color,
                distance_matrix, eta_matrix
            )
    
    def _add_pickup_terminals(self, feature_group: folium.FeatureGroup, 
                             solution: Dict, polygon_color: str):
        """Add pickup terminal markers"""
        
        for pickup_data in solution.get('formatted_pickup_data', []):
            pickup_id = pickup_data['pickup_id']
            lat = pickup_data['lat']
            lon = pickup_data['lon']
            num_parcels = pickup_data['num_parcels']
            
            # Find knock information
            knock_info = None
            for knock_data in solution.get('analysis', {}).get('knock_analysis', []):
                if knock_data['pickup_id'] == pickup_id:
                    knock_info = knock_data
                    break
            
            # Create pickup marker
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; width: 250px;">
                <h4 style="color: {polygon_color}; margin: 0;">📦 Pickup Terminal {pickup_id}</h4>
                <hr style="margin: 5px 0;">
                <b>Total Parcels:</b> {num_parcels}<br>
                <b>Total Knocks:</b> {knock_info['total_knocks'] if knock_info else 'N/A'}<br>
                <b>Vehicles Serving:</b><br>
            """
            
            if knock_info:
                for vehicle_info in knock_info.get('vehicles', []):
                    popup_html += f"  • Vehicle {vehicle_info['vehicle_id']}: {vehicle_info['parcels_from_pickup']} parcels<br>"
            
            popup_html += "</div>"
            
            # Add marker with larger size
            folium.CircleMarker(
                location=[lat, lon],
                radius=12,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Pickup Terminal {pickup_id} ({num_parcels} parcels)",
                color='red',
                fillColor='red',
                fillOpacity=0.8,
                weight=3
            ).add_to(feature_group)
            
            # Add terminal ID label
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html=f"""<div style="font-size: 10pt; color: white; font-weight: bold; 
                             text-align: center; margin-top: -5px;">{pickup_id}</div>""",
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                )
            ).add_to(feature_group)
    
    def _add_vehicle_route(self, feature_group: folium.FeatureGroup, 
                          vehicle: Dict, solution: Dict, polygon_id: str, 
                          route_color: str, distance_matrix: Optional[Dict], 
                          eta_matrix: Optional[Dict]):
        """Add a single vehicle route with detailed information"""
        
        vehicle_id = vehicle['vehicle_id']
        vehicle_type = vehicle['vehicle_type']
        route_coords = vehicle.get('physical_route', [])
        
        if len(route_coords) < 2:
            return
        
        # Add route polyline with segments
        self._add_route_segments(
            feature_group, route_coords, vehicle, route_color, 
            distance_matrix, eta_matrix
        )
        
        # Add delivery markers
        self._add_delivery_markers(
            feature_group, vehicle, route_coords, route_color, polygon_id
        )
    
    def _add_route_segments(self, feature_group: folium.FeatureGroup, 
                           route_coords: List[Tuple[float, float]], 
                           vehicle: Dict, route_color: str,
                           distance_matrix: Optional[Dict], eta_matrix: Optional[Dict]):
        """Add route segments with detailed edge information"""
        
        for i in range(len(route_coords) - 1):
            start_coord = route_coords[i]
            end_coord = route_coords[i + 1]
            
            # Calculate edge information
            edge_info = self._calculate_edge_info(
                start_coord, end_coord, distance_matrix, eta_matrix
            )
            
            # Determine segment type
            segment_type = "pickup" if i == 0 else "delivery"
            segment_style = {
                "pickup": {"weight": 5, "opacity": 0.9, "dashArray": None},
                "delivery": {"weight": 4, "opacity": 0.8, "dashArray": None}
            }
            
            # Create popup for edge
            popup_html = f"""
            <div style="font-family: Arial; font-size: 11px; width: 220px;">
                <h5 style="color: {route_color}; margin: 0;">🚗 Route Segment</h5>
                <hr style="margin: 3px 0;">
                <b>Vehicle:</b> {vehicle['vehicle_id']} ({vehicle['vehicle_type']})<br>
                <b>Segment:</b> {segment_type.title()}<br>
                <b>Distance:</b> {edge_info['distance']:.0f}m<br>
                <b>ETA:</b> {edge_info['eta']:.1f} min<br>
                <b>Sequence:</b> {i+1}/{len(route_coords)-1}
            </div>
            """
            
            # Add polyline
            folium.PolyLine(
                locations=[start_coord, end_coord],
                color=route_color,
                weight=segment_style[segment_type]["weight"],
                opacity=segment_style[segment_type]["opacity"],
                dashArray=segment_style[segment_type]["dashArray"],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Vehicle {vehicle['vehicle_id']}: {edge_info['distance']:.0f}m, {edge_info['eta']:.1f}min"
            ).add_to(feature_group)
            
            # Add direction arrow at midpoint
            self._add_direction_arrow(feature_group, start_coord, end_coord, route_color)
    
    def _add_delivery_markers(self, feature_group: folium.FeatureGroup, 
                             vehicle: Dict, route_coords: List[Tuple[float, float]], 
                             route_color: str, polygon_id: str):
        """Add delivery point markers"""
        
        # Skip first coordinate (pickup terminal)
        delivery_coords = route_coords[1:]
        parcels = vehicle.get('parcels', [])
        
        for idx, coord in enumerate(delivery_coords):
            parcel_id = parcels[idx] if idx < len(parcels) else f"Unknown_{idx}"
            
            popup_html = f"""
            <div style="font-family: Arial; font-size: 11px; width: 200px;">
                <h5 style="color: {route_color}; margin: 0;">📦 Delivery Point</h5>
                <hr style="margin: 3px 0;">
                <b>Parcel ID:</b> {parcel_id}<br>
                <b>Vehicle:</b> {vehicle['vehicle_id']} ({vehicle['vehicle_type']})<br>
                <b>Polygon:</b> {polygon_id}<br>
                <b>Delivery Order:</b> {idx + 1} of {len(delivery_coords)}<br>
                <b>Capacity Used:</b> {vehicle.get('capacity_used', 'N/A')}/{vehicle.get('vehicle_capacity', 'N/A')}
            </div>
            """
            
            folium.CircleMarker(
                location=coord,
                radius=6,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Delivery #{idx+1} - Parcel {parcel_id}",
                color=route_color,
                fillColor=route_color,
                fillOpacity=0.8,
                weight=2
            ).add_to(feature_group)
    
    def _add_vehicle_info_marker(self, feature_group: folium.FeatureGroup, 
                                vehicle: Dict, start_coord: Tuple[float, float], 
                                vehicle_color: str, polygon_id: str):
        """Add comprehensive vehicle information marker"""
        
        # Calculate utilization
        capacity_used = vehicle.get('capacity_used', 0)
        vehicle_capacity = vehicle.get('vehicle_capacity', 1)
        utilization = (capacity_used / vehicle_capacity) * 100
        
        # Time information
        duration_hours = vehicle.get('total_duration_seconds', 0) / 3600
        distance_km = vehicle.get('total_distance_m', 0) / 1000
        
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 280px;">
            <h4 style="color: {vehicle_color}; margin: 0;">🚗 Vehicle {vehicle['vehicle_id']}</h4>
            <hr style="margin: 5px 0;">
            <b>Type:</b> {vehicle['vehicle_type']}<br>
            <b>Polygon:</b> {polygon_id}<br>
            <b>Parcels:</b> {len(vehicle.get('parcels', []))}<br>
            <b>Capacity:</b> {capacity_used}/{vehicle_capacity} ({utilization:.1f}%)<br>
            <b>Distance:</b> {distance_km:.2f} km<br>
            <b>Duration:</b> {duration_hours:.2f} hours<br>
            <b>Cost:</b> {vehicle.get('total_cost_kt', 0):.2f} KT<br>
            <b>Stops:</b> {vehicle.get('num_stops', 0)}<br>
            <b>Time Feasible:</b> {vehicle.get('time_window_feasible', 'N/A')}<br>
            <hr style="margin: 5px 0;">
            <b>Route Summary:</b><br>
            • Start: Pickup Terminal<br>
            • Deliveries: {len(vehicle.get('parcels', []))} stops<br>
            • End: Last delivery
        </div>
        """
        
        # Vehicle icon based on type
        icon_map = {
            'bike': 'bicycle',
            'carbox': 'car',
            'big-box': 'truck',
            'van': 'shuttle-van',
            'truck': 'truck'
        }
        
        folium.Marker(
            location=[start_coord[0] + 0.001, start_coord[1] + 0.001],  # Slight offset
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"Vehicle {vehicle['vehicle_id']} ({vehicle['vehicle_type']})",
            icon=folium.Icon(
                color='green',
                icon=icon_map.get(vehicle['vehicle_type'], 'car'),
                prefix='fa'
            )
        ).add_to(feature_group)
    
    def _add_direction_arrow(self, feature_group: folium.FeatureGroup, 
                            start_coord: Tuple[float, float], 
                            end_coord: Tuple[float, float], color: str):
        """Add direction arrow to route segment using a small circle marker"""
        
        # Calculate midpoint
        mid_lat = (start_coord[0] + end_coord[0]) / 2
        mid_lon = (start_coord[1] + end_coord[1]) / 2
        
        # Add small direction indicator
        folium.CircleMarker(
            location=[mid_lat, mid_lon],
            radius=3,
            color=color,
            fillColor='white',
            fillOpacity=1,
            weight=2,
            tooltip="Route direction"
        ).add_to(feature_group)
    
    def _calculate_edge_info(self, start_coord: Tuple[float, float], 
                           end_coord: Tuple[float, float],
                           distance_matrix: Optional[Dict], 
                           eta_matrix: Optional[Dict]) -> Dict:
        """Calculate distance and ETA for route segment"""
        
        if distance_matrix and start_coord in distance_matrix:
            distance = distance_matrix[start_coord].get(end_coord, 0)
        else:
            # Calculate haversine distance
            distance = self._haversine_distance(start_coord, end_coord)
        
        if eta_matrix and start_coord in eta_matrix:
            eta_seconds = eta_matrix[start_coord].get(end_coord, 0)
        else:
            # Estimate ETA (assuming 25 km/h average speed)
            eta_seconds = (distance / 1000) * 3600 / 25
        
        return {
            'distance': distance,
            'eta': eta_seconds / 60  # Convert to minutes
        }
    
    def _haversine_distance(self, coord1: Tuple[float, float], 
                           coord2: Tuple[float, float]) -> float:
        """Calculate haversine distance between two coordinates"""
        import math
        
        R = 6371000  # Earth's radius in meters
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _add_polygon_summary(self, feature_group: folium.FeatureGroup, 
                            solution: Dict, polygon_id: str, polygon_color: str):
        """Add polygon summary information"""
        
        results = solution.get('results_summary', {})
        input_summary = solution.get('input_summary', {})
        
        # Find a good location for summary (top-left of bounding box)
        summary_lat, summary_lon = self._find_summary_location(solution)
        
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 300px;">
            <h3 style="color: {polygon_color}; margin: 0;">📊 Polygon {polygon_id} Summary</h3>
            <hr style="margin: 5px 0;">
            <b>Input:</b><br>
            • Pickup Terminals: {input_summary.get('num_hubs', 0)}<br>
            • Total Parcels: {input_summary.get('total_parcels', 0)}<br>
            • Max Knock: {input_summary.get('max_knock', 0)}<br>
            • Time Window: {input_summary.get('time_window_hours', 0)} hours<br>
            <br>
            <b>Solution:</b><br>
            • Total Vehicles: {results.get('total_vehicles_used', 0)}<br>
            • Vehicle Types: {results.get('vehicles_by_type', {})}<br>
            • Total Cost: {results.get('total_cost', 0):.2f}<br>
            • Total Duration: {results.get('total_duration_seconds', 0)/3600:.2f} hours<br>
            • Time Feasible: {results.get('time_window_feasible', 'N/A')}<br>
            • Processing Time: {solution.get('processing_time', 0):.2f} seconds<br>
        </div>
        """
        
        folium.Marker(
            location=[summary_lat, summary_lon],
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"Polygon {polygon_id} Summary",
            icon=folium.Icon(
                color='purple',
                icon='info-circle',
                prefix='fa'
            )
        ).add_to(feature_group)
    
    def _find_summary_location(self, solution: Dict) -> Tuple[float, float]:
        """Find good location for polygon summary"""
        
        # Get all coordinates
        all_coords = []
        
        for pickup_data in solution.get('formatted_pickup_data', []):
            all_coords.append((pickup_data['lat'], pickup_data['lon']))
        
        for vehicle in solution.get('vehicles', []):
            all_coords.extend(vehicle.get('physical_route', []))
        
        if all_coords:
            # Find top-left corner
            min_lat = min(coord[0] for coord in all_coords)
            max_lat = max(coord[0] for coord in all_coords)
            min_lon = min(coord[1] for coord in all_coords)
            max_lon = max(coord[1] for coord in all_coords)
            
            # Place summary at top-left with small offset
            return max_lat + 0.01, min_lon - 0.01
        
        return 35.6892, 51.3890  # Default location
    
    def _add_legend(self, m: folium.Map, solutions_dict: Dict[str, Dict]):
        """Add legend for routes and symbols"""
        
        legend_html = """
        <div style='position: fixed; 
                    top: 10px; left: 10px; width: 180px; height: auto; 
                    border:2px solid grey; z-index:9999; 
                    font-size:12px; background-color:white;
                    padding: 10px;
                    '>
        <h4>VRP Routes Legend</h4>
        <h5>Symbols:</h5>
        <i style="background:red; width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:5px;"></i>Pickup Terminal<br>
        <i style="background:blue; width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px;"></i>Delivery Point<br>
        <i style="background:gray; width:6px; height:6px; border-radius:50%; display:inline-block; margin-right:5px;"></i>Route Direction<br>
        <br>
        <h5>Routes:</h5>
        Each route has a unique color<br>
        Thicker lines = pickup segments<br>
        Thinner lines = delivery segments
        </div>
        """
        
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def save_map(self, m: folium.Map, filename: str = "vrp_routes.html"):
        """Save the map to HTML file"""
        m.save(filename)
        print(f"Map saved to {filename}")
        return filename


# Example usage function
def visualize_vrp_solutions(solutions_dict: Dict[str, Dict], 
                           distance_matrix: Optional[Dict] = None,
                           eta_matrix: Optional[Dict] = None,
                           output_file: str = "vrp_routes_multi.html") -> str:
    """
    Main function to visualize VRP solutions
    
    Args:
        solutions_dict: Dictionary of {polygon_id: solution_dict}
        distance_matrix: Optional distance matrix
        eta_matrix: Optional ETA matrix
        output_file: Output HTML filename
        
    Returns:
        Path to saved HTML file
    """
    
    visualizer = VRPRouteVisualizer()
    
    # Create map
    m = visualizer.create_multi_polygon_map(
        solutions_dict, 
        distance_matrix, 
        eta_matrix
    )
    
    # Save map
    return visualizer.save_map(m, output_file)
