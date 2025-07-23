import os
import math
import pickle
import json
import time
import requests
import constants

from typing import Dict, Tuple, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_result


class DistanceCalculator:
    """Handles distance and time calculations with optimized API usage"""

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in meters"""
        R = 6371000 # Earth's radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_not_result(
            lambda result: result is not None and result.status_code == 200
        ),
    )
    def _make_api_request(payload: dict) -> requests.Response:
        """Make API request with retry mechanism"""
        params = {
            "json": json.dumps(payload),
            "engine": "ocelot",
            "no_traffic": "false",
        }

        try:
            response = requests.get(
                constants.URL_MATRIX,
                headers={"X-Smapp-Key": constants.APIKEY_MATRIX},
                params=params,
                timeout=30, # Add timeout to prevent hanging requests
            )

            # Check if response is successful
            if response.status_code == 200:
                return response
            else:
                print(
                    f"API request failed with status {response.status_code}, retrying..."
                )
                return None # This will trigger a retry

        except requests.RequestException as e:
            print(f"Request failed with exception: {e}, retrying...")
            return None # This will trigger a retry

    @staticmethod
    def build_matrices_smapp_optimized(
        locations: List[Tuple[float, float]], 
        matrix_id: str,
        waiting_time: int = 120,
        max_locations_per_call: int = 50, # Reduced for better reliability
        delay_between_calls: float = 0.3 # Reduced delay
    ) -> Tuple[Dict[Tuple[float, float], Dict[Tuple[float, float], float]], 
               Dict[Tuple[float, float], Dict[Tuple[float, float], float]]]:
        """
        Build distance matrix for all locations using optimized batched requests
        
        Args:
            locations: List of (lat, lon) tuples
            waiting_time: Service time to add to travel time
            max_locations_per_call: Maximum locations per API call
            delay_between_calls: Delay between API calls to avoid rate limiting
            
        Returns:
            Tuple of (distance_matrix, eta_matrix)
        """
        eta_matrix = {}
        distance_matrix = {}
        
        # Initialize matrices
        for loc in locations:
            eta_matrix[loc] = {}
            distance_matrix[loc] = {}
            # Set self-distance to 0
            distance_matrix[loc][loc] = 0
            eta_matrix[loc][loc] = 0

        num_locations = len(locations)
        print(f"Building matrices for {num_locations} locations using optimized batching")
        
        # Strategy 1: If small number of locations, use full matrix approach
        if num_locations <= max_locations_per_call:
            return DistanceCalculator._build_full_matrix(locations, matrix_id, waiting_time)
        
        # Strategy 2: For larger datasets, use chunked approach
        return DistanceCalculator._build_chunked_matrix(
            locations, matrix_id, waiting_time, max_locations_per_call, delay_between_calls
        )

    @staticmethod
    def _build_full_matrix(
        locations: List[Tuple[float, float]],
        matrix_id: str,
        waiting_time: int = 120
    ) -> Tuple[Dict, Dict]:
        """Build full matrix with single API call for small datasets"""
        print("Using full matrix approach (single API call)")
        
        eta_matrix = {}
        distance_matrix = {}
        
        # Initialize matrices
        for loc in locations:
            eta_matrix[loc] = {}
            distance_matrix[loc] = {}
            distance_matrix[loc][loc] = 0
            eta_matrix[loc][loc] = 0

        # Prepare payload for full matrix
        payload = {
            "sources": [{"lat": loc[0], "lon": loc[1]} for loc in locations],
            "targets": [{"lat": loc[0], "lon": loc[1]} for loc in locations],
        }

        try:
            response = DistanceCalculator._make_api_request(payload)
            
            if response is not None:
                result = response.json()
                sources_to_targets = result["sources_to_targets"]
                
                # Process all source-target pairs
                for source_idx, source_results in enumerate(sources_to_targets):
                    source_loc = locations[source_idx]
                    
                    for target_idx, target_result in enumerate(source_results):
                        target_loc = locations[target_idx]
                        
                        if target_result["status"] == "Success":
                            eta_matrix[source_loc][target_loc] = (
                                target_result["time"] + waiting_time
                            )
                            distance_matrix[source_loc][target_loc] = target_result["distance"]
                        else:
                            print(f"Failed calculation: {source_loc} -> {target_loc}")
                            # Use haversine as fallback
                            distance = DistanceCalculator.haversine_distance(
                                source_loc[0], source_loc[1], target_loc[0], target_loc[1]
                            )
                            distance_matrix[source_loc][target_loc] = distance
                            # Estimate time based on distance (25 km/h average)
                            eta_matrix[source_loc][target_loc] = int(distance / 1000 * 3600 / 25) + waiting_time
                with open(f"./matrices/{matrix_id}_eta.pkl", "wb") as f:
                    pickle.dump(eta_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f"./matrices/{matrix_id}_distance.pkl", "wb") as f:
                    pickle.dump(distance_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print("Full matrix API call failed, using haversine fallback")
                return DistanceCalculator._build_haversine_fallback(locations, waiting_time)
                
        except Exception as e:
            print(f"Full matrix optimization failed: {e}, using haversine fallback")
            return DistanceCalculator._build_haversine_fallback(locations, waiting_time)

        print("Full matrix built successfully")
        return distance_matrix, eta_matrix

    @staticmethod
    def _build_chunked_matrix(
        locations: List[Tuple[float, float]],
        matrix_id: str,
        waiting_time: int,
        max_locations_per_call: int,
        delay_between_calls: float
    ) -> Tuple[Dict, Dict]:
        """Build matrix using chunked approach for large datasets"""
        num_locations = len(locations)
        print(f"Using chunked matrix approach ({max_locations_per_call} locations per call)")
        
        eta_matrix = {}
        distance_matrix = {}
        
        # Initialize matrices
        for loc in locations:
            eta_matrix[loc] = {}
            distance_matrix[loc] = {}
            distance_matrix[loc][loc] = 0
            eta_matrix[loc][loc] = 0

        # Calculate number of chunks needed
        chunk_size = max_locations_per_call
        num_chunks = (num_locations + chunk_size - 1) // chunk_size
        
        print(f"Processing {num_chunks} chunks of up to {chunk_size} locations each")
        
        # Process chunks
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_locations)
            chunk_locations = locations[start_idx:end_idx]
            
            print(f"Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_locations)} locations)")
            
            try:
                # Build matrix for this chunk against all locations
                chunk_distance, chunk_eta = DistanceCalculator._build_chunk_to_all_matrix(
                    chunk_locations, locations, waiting_time
                )
                
                # Merge results
                for source_loc in chunk_locations:
                    distance_matrix[source_loc].update(chunk_distance[source_loc])
                    eta_matrix[source_loc].update(chunk_eta[source_loc])
                
                # Add delay between chunks to avoid rate limiting
                if chunk_idx < num_chunks - 1: # Don't delay after last chunk
                    time.sleep(delay_between_calls)
                    
            except Exception as e:
                print(f"Chunk {chunk_idx + 1} failed: {e}, using haversine for this chunk")
                # Fallback to haversine for failed chunk
                for source_loc in chunk_locations:
                    for target_loc in locations:
                        if target_loc not in distance_matrix[source_loc]:
                            distance = DistanceCalculator.haversine_distance(
                                source_loc[0], source_loc[1], target_loc[0], target_loc[1]
                            )
                            distance_matrix[source_loc][target_loc] = distance
                            eta_matrix[source_loc][target_loc] = int(distance / 1000 * 3600 / 25) + waiting_time

        with open(f"./matrices/{matrix_id}_eta.pkl", "wb") as f:
            pickle.dump(eta_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"./matrices/{matrix_id}_distance.pkl", "wb") as f:
            pickle.dump(distance_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Chunked matrix built successfully")
        return distance_matrix, eta_matrix

    @staticmethod
    def _build_chunk_to_all_matrix(
        chunk_locations: List[Tuple[float, float]], 
        all_locations: List[Tuple[float, float]], 
        waiting_time: int
    ) -> Tuple[Dict, Dict]:
        """Build matrix from chunk locations to all locations"""
        chunk_distance = {}
        chunk_eta = {}
        
        # Initialize for chunk locations
        for loc in chunk_locations:
            chunk_distance[loc] = {}
            chunk_eta[loc] = {}

        # Prepare payload
        payload = {
            "sources": [{"lat": loc[0], "lon": loc[1]} for loc in chunk_locations],
            "targets": [{"lat": loc[0], "lon": loc[1]} for loc in all_locations],
        }

        response = DistanceCalculator._make_api_request(payload)
        
        if response is not None:
            result = response.json()
            sources_to_targets = result["sources_to_targets"]
            
            # Process results
            for source_idx, source_results in enumerate(sources_to_targets):
                source_loc = chunk_locations[source_idx]
                
                for target_idx, target_result in enumerate(source_results):
                    target_loc = all_locations[target_idx]
                    
                    if target_result["status"] == "Success":
                        chunk_eta[source_loc][target_loc] = (
                            target_result["time"] + waiting_time
                        )
                        chunk_distance[source_loc][target_loc] = target_result["distance"]
                    else:
                        # Fallback to haversine for failed individual calculations
                        distance = DistanceCalculator.haversine_distance(
                            source_loc[0], source_loc[1], target_loc[0], target_loc[1]
                        )
                        chunk_distance[source_loc][target_loc] = distance
                        chunk_eta[source_loc][target_loc] = int(distance / 1000 * 3600 / 25) + waiting_time
        else:
            raise Exception("API call failed for chunk")

        return chunk_distance, chunk_eta

    @staticmethod
    def _build_haversine_fallback(
        locations: List[Tuple[float, float]], 
        waiting_time: int
    ) -> Tuple[Dict, Dict]:
        """Fallback to haversine calculations when API fails"""
        print("Using haversine fallback for all calculations")
        
        distance_matrix = DistanceCalculator.build_distance_matrix_haversine(locations)
        eta_matrix = DistanceCalculator.build_eta_matrix(locations, service_time_seconds=waiting_time)
        
        return distance_matrix, eta_matrix

    @staticmethod
    def build_matrices_smapp_ultra_fast(
        locations: List[Tuple[float, float]], 
        waiting_time: int = 40,
        sample_ratio: float = 0.25 # Use 30% of locations for API, rest haversine
    ) -> Tuple[Dict[Tuple[float, float], Dict[Tuple[float, float], float]], 
               Dict[Tuple[float, float], Dict[Tuple[float, float], float]]]:
        """
        Ultra-fast hybrid approach: API for subset, haversine for rest with correction factors
        
        Args:
            locations: List of (lat, lon) tuples
            waiting_time: Service time to add to travel time
            sample_ratio: Ratio of locations to use for API calibration
            
        Returns:
            Tuple of (distance_matrix, eta_matrix)
        """
        num_locations = len(locations)
        sample_size = max(10, int(num_locations * sample_ratio)) # At least 10 locations
        sample_size = min(sample_size, 100) # At most 100 locations for API
        
        print(f"Ultra-fast mode: Using API for {sample_size}/{num_locations} locations, haversine for rest")
        
        # Step 1: Get sample locations for API calibration
        if num_locations <= sample_size:
            # Small dataset, use full API
            return DistanceCalculator._build_full_matrix(locations, waiting_time)
        
        # Select representative sample (spread across the dataset)
        import random
        random.seed(42) # For reproducibility
        sample_locations = random.sample(locations, sample_size)
        
        # Step 2: Build API matrix for sample
        try:
            print("Building calibration matrix with API...")
            api_distance, api_eta = DistanceCalculator._build_full_matrix(sample_locations, waiting_time)
            
            # Step 3: Calculate correction factors
            distance_correction, time_correction = DistanceCalculator._calculate_correction_factors(
                sample_locations, api_distance, api_eta, waiting_time
            )
            
            print(f"Correction factors - Distance: {distance_correction:.3f}, Time: {time_correction:.3f}")
            
        except Exception as e:
            print(f"API calibration failed: {e}, using pure haversine")
            return DistanceCalculator._build_haversine_fallback(locations, waiting_time)
        
        # Step 4: Build full matrix with haversine + corrections
        print("Building full matrix with corrected haversine...")
        distance_matrix, eta_matrix = DistanceCalculator._build_corrected_haversine_matrix(
            locations, waiting_time, distance_correction, time_correction
        )
        
        # Step 5: Insert exact API values for sample locations
        print("Inserting exact API values for sample...")
        for source_loc in sample_locations:
            for target_loc in sample_locations:
                if source_loc in api_distance and target_loc in api_distance[source_loc]:
                    distance_matrix[source_loc][target_loc] = api_distance[source_loc][target_loc]
                    eta_matrix[source_loc][target_loc] = api_eta[source_loc][target_loc]
        
        print("Ultra-fast matrix built successfully")
        return distance_matrix, eta_matrix

    @staticmethod
    def _calculate_correction_factors(
        sample_locations: List[Tuple[float, float]], 
        api_distance: Dict, 
        api_eta: Dict, 
        waiting_time: int
    ) -> Tuple[float, float]:
        """Calculate correction factors by comparing API vs haversine for sample"""
        distance_ratios = []
        time_ratios = []
        
        for source_loc in sample_locations:
            for target_loc in sample_locations:
                if source_loc != target_loc:
                    # Get API values
                    api_dist = api_distance[source_loc][target_loc]
                    api_time = api_eta[source_loc][target_loc] - waiting_time # Remove waiting time
                    
                    # Calculate haversine
                    haversine_dist = DistanceCalculator.haversine_distance(
                        source_loc[0], source_loc[1], target_loc[0], target_loc[1]
                    )
                    haversine_time = haversine_dist / 1000 * 3600 / 25 # 25 km/h average
                    
                    # Calculate ratios (avoid division by zero)
                    if haversine_dist > 0 and haversine_time > 0:
                        distance_ratios.append(api_dist / haversine_dist)
                        time_ratios.append(api_time / haversine_time)
        
        # Calculate median correction factors (more robust than mean)
        distance_correction = sorted(distance_ratios)[len(distance_ratios) // 2] if distance_ratios else 1.3
        time_correction = sorted(time_ratios)[len(time_ratios) // 2] if time_ratios else 1.5
        
        # Clamp correction factors to reasonable ranges
        distance_correction = max(1.1, min(2.0, distance_correction))
        time_correction = max(1.2, min(3.0, time_correction))
        
        return distance_correction, time_correction

    @staticmethod
    def _build_corrected_haversine_matrix(
        locations: List[Tuple[float, float]], 
        waiting_time: int,
        distance_correction: float,
        time_correction: float
    ) -> Tuple[Dict, Dict]:
        """Build matrix using corrected haversine calculations"""
        distance_matrix = {}
        eta_matrix = {}
        
        for source_loc in locations:
            distance_matrix[source_loc] = {}
            eta_matrix[source_loc] = {}
            
            for target_loc in locations:
                if source_loc == target_loc:
                    distance_matrix[source_loc][target_loc] = 0
                    eta_matrix[source_loc][target_loc] = waiting_time
                else:
                    # Calculate haversine distance
                    haversine_dist = DistanceCalculator.haversine_distance(
                        source_loc[0], source_loc[1], target_loc[0], target_loc[1]
                    )
                    
                    # Apply corrections
                    corrected_distance = haversine_dist * distance_correction
                    corrected_time = (haversine_dist / 1000 * 3600 / 25) * time_correction + waiting_time
                    
                    distance_matrix[source_loc][target_loc] = corrected_distance
                    eta_matrix[source_loc][target_loc] = int(corrected_time)
        
        return distance_matrix, eta_matrix

    @staticmethod
    def build_matrices_smapp(
        locations: List[Tuple[float, float]], waiting_time: int = 120
    ) -> Tuple[Dict[Tuple[float, float], Dict[Tuple[float, float], float]], 
               Dict[Tuple[float, float], Dict[Tuple[float, float], float]]]:
        """
        Legacy method - now redirects to optimized version
        Kept for backward compatibility
        """
        return DistanceCalculator.build_matrices_smapp_optimized(locations, waiting_time)

    @staticmethod
    def build_matrices(
        locations: List[Tuple[float, float]],
        matrix_id: str,
        optimization_mode: str = "accurate"
    ) -> Tuple[Dict[Tuple[float, float], Dict[Tuple[float, float], float]], 
               Dict[Tuple[float, float], Dict[Tuple[float, float], float]]]:
        """
        Enhanced build_matrices with optimization modes
        
        Args:
            locations: List of (lat, lon) tuples
            optimization_mode: 
                - "fast": Ultra-fast hybrid approach (30% API, 70% corrected haversine)
                - "balanced": Optimized batching with full API coverage
                - "accurate": Full API matrix (slowest but most accurate)
        """
        if constants.USE_HAVERSINE:
            distance_matrix = DistanceCalculator.build_distance_matrix_haversine(locations)
            eta_matrix = DistanceCalculator.build_eta_matrix(locations)
            return distance_matrix, eta_matrix
        
        num_locations = len(locations)
        print(f"Building matrices for {num_locations} locations in '{optimization_mode}' mode")
        
        if optimization_mode == "fast":
            return DistanceCalculator.build_matrices_smapp_ultra_fast(locations)
        elif optimization_mode == "balanced":
            return DistanceCalculator.build_matrices_smapp_optimized(locations)
        elif optimization_mode == "accurate":
            if os.path.exists(f"./matrices/{matrix_id}_eta.pkl") and os.path.isfile(f"./matrices/{matrix_id}_eta.pkl"):
                try:
                    with open(f"./matrices/{matrix_id}_eta.pkl", "rb") as f:
                        eta_matrix = pickle.load(f)
                    with open(f"./matrices/{matrix_id}_distance.pkl", "rb") as f:
                        distance_matrix = pickle.load(f)
                    return distance_matrix, eta_matrix
                except Exception as exc:
                    print(f"Error occured for {matrix_id}: {exc}")
            if num_locations <= 50:
                return DistanceCalculator._build_full_matrix(locations, matrix_id, 120)
            else:
                print("Warning: 'accurate' mode with >100 locations will be very slow")
                return DistanceCalculator.build_matrices_smapp_optimized(locations, matrix_id)
        else:
            raise ValueError(f"Unknown optimization_mode: {optimization_mode}")

    @staticmethod
    def build_distance_matrix_haversine(
        locations: List[Tuple[float, float]],
    ) -> Dict[Tuple[float, float], Dict[Tuple[float, float], float]]:
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
    def build_eta_matrix(
        locations: List[Tuple[float, float]],
        average_speed_kmh: float = 25.0,
        service_time_seconds: int = 120,
    ) -> Dict[Tuple[float, float], Dict[Tuple[float, float], int]]:
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
                    speed_ms = (average_speed_kmh * 1000) / 3600 # Convert km/h to m/s
                    travel_time = int(distance_meters / speed_ms) + service_time_seconds
                    eta_matrix[loc1][loc2] = travel_time

        return eta_matrix
