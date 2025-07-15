import math
import json
import time
import requests

from typing import Dict, Tuple, List


URL_MATRIX = ""
APIKEY_MATRIX = ""

USE_HAVERSINE = False


class DistanceCalculator:
    """Handles distance and time calculations"""

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in meters"""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c


    @staticmethod
    def build_matrices_smapp(
        locations: List[Tuple[float, float]]
    ) -> Dict[Tuple[float, float], Dict[Tuple[float, float], float]]:
        """Build distance matrix for all locations using batched requests"""
        eta_matrix = {}
        distance_matrix = {}
        
        # Initialize matrices
        for loc in locations:
            eta_matrix[loc] = {}
            distance_matrix[loc] = {}
        
        # Process each location as a source
        for source_loc in locations:
            print(f"Processing source: {source_loc}")
            
            # Get all other locations as targets (excluding self)
            target_locations = [loc for loc in locations if loc != source_loc]
            
            # Set self-distance to 0
            distance_matrix[source_loc][source_loc] = 0
            eta_matrix[source_loc][source_loc] = 0
            
            # Process targets in batches of 50
            batch_size = 50
            for i in range(0, len(target_locations), batch_size):
                batch_targets = target_locations[i:i + batch_size]
                
                # Prepare batch payload
                payload = {
                    "sources": [
                        {
                            "lat": source_loc[0], 
                            "lon": source_loc[1]
                        }
                    ],
                    "targets": [
                        {
                            "lat": target[0], 
                            "lon": target[1]
                        } for target in batch_targets
                    ]
                }
                
                params = {
                    "json": json.dumps(payload),
                    "engine": "ocelot",
                    "no_traffic": "false"
                }
                
                try:
                    response = requests.get(
                        URL_MATRIX, 
                        headers={"X-Smapp-Key": APIKEY_MATRIX}, 
                        params=params
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        sources_to_targets = result["sources_to_targets"][0]  # First (and only) source
                        
                        # Process each target in the batch
                        for j, target_result in enumerate(sources_to_targets):
                            target_loc = batch_targets[j]
                            
                            if target_result["status"] == "Success":
                                eta_matrix[source_loc][target_loc] = target_result["time"]
                                distance_matrix[source_loc][target_loc] = target_result["distance"]
                            else:
                                # Handle failed calculations - you might want to set to None or a default value
                                print(f"Failed to calculate distance from {source_loc} to {target_loc}")
                                eta_matrix[source_loc][target_loc] = None
                                distance_matrix[source_loc][target_loc] = None
                                
                    else:
                        print(f"API request failed with status {response.status_code}")
                        response.raise_for_status()
                        
                except requests.RequestException as e:
                    print(f"Request failed: {e}")
                    # Set all targets in this batch to None
                    for target_loc in batch_targets:
                        eta_matrix[source_loc][target_loc] = None
                        distance_matrix[source_loc][target_loc] = None
                
                # Add delay between batches to avoid rate limiting
                time.sleep(2)
        
        return distance_matrix, eta_matrix
    

    @staticmethod
    def build_matrices(locations: List[Tuple[float, float]]) -> Dict[Tuple[float, float], Dict[Tuple[float, float], float]]:
        if USE_HAVERSINE:
            distance_matrix = DistanceCalculator.build_distance_matrix_haversine(locations)
            eta_matrix = DistanceCalculator.build_eta_matrix(locations)
        else:
            distance_matrix, eta_matrix = DistanceCalculator.build_matrices_smapp(locations)
        return distance_matrix, eta_matrix

    @staticmethod
    def build_distance_matrix_haversine(locations: List[Tuple[float, float]]) -> Dict[Tuple[float, float], Dict[Tuple[float, float], float]]:
        """Build distance matrix for all locations"""
        distance_matrix = {}
        
        for loc1 in locations:
            distance_matrix[loc1] = {}
            for loc2 in locations:
                if loc1 == loc2:
                    distance_matrix[loc1][loc2] = 0
                else:
                    distance = DistanceCalculator.haversine_distance(
                        loc1[0], loc1[1], loc2[0], loc2[1]
                    )
                    distance_matrix[loc1][loc2] = distance
        
        return distance_matrix
    
    @staticmethod
    def build_eta_matrix(locations: List[Tuple[float, float]], 
                        average_speed_kmh: float = 25.0,
                        service_time_seconds: int = 0) -> Dict[Tuple[float, float], Dict[Tuple[float, float], int]]:
        """
        Build ETA (Estimated Time of Arrival) matrix for all locations
        
        Args:
            locations: List of (lat, lon) tuples
            average_speed_kmh: Average vehicle speed in km/h
            service_time_seconds: Service time at each location in seconds
            
        Returns:
            ETA matrix with travel times in seconds
        """
        eta_matrix = {}
        
        for loc1 in locations:
            eta_matrix[loc1] = {}
            for loc2 in locations:
                if loc1 == loc2:
                    eta_matrix[loc1][loc2] = service_time_seconds
                else:
                    distance_meters = DistanceCalculator.haversine_distance(
                        loc1[0], loc1[1], loc2[0], loc2[1]
                    )
                    # Convert to travel time: distance(m) / speed(m/s) + service_time
                    speed_ms = (average_speed_kmh * 1000) / 3600  # Convert km/h to m/s
                    travel_time = int(distance_meters / speed_ms) + service_time_seconds
                    eta_matrix[loc1][loc2] = travel_time
        
        return eta_matrix
