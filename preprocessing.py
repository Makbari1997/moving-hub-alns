import pandas as pd
from collections import defaultdict

from data_structures import *
from distance_calculator import *


class DataProcessor:
    """Processes raw CSV data into problem format"""

    def __init__(
        self, default_parcel_size: int = 1, proximity_threshold_meters: float = 100.0
    ):
        """
        Initialize data processor

        Args:
            default_parcel_size: Default size for parcels when not specified
            proximity_threshold_meters: Distance threshold to group parcels into same pickup terminal
        """
        self.default_parcel_size = default_parcel_size
        self.proximity_threshold_meters = proximity_threshold_meters

    def process_csv_data(
        self,
        df: pd.DataFrame,
        vehicle_specs: Dict[VehicleType, VehicleSpec],
        max_knock: int = 4,
        time_window_hours: float = 6.0,
    ) -> Tuple[ProblemConfig, List[PickupTerminal]]:
        """
        Process CSV data into problem format

        Args:
            csv_file_path: Path to CSV file
            vehicle_specs: Vehicle specifications
            max_knock: Maximum knocks per pickup terminal
            time_window_hours: Time window in hours

        Returns:
            Tuple of (ProblemConfig, List[PickupTerminal])
        """
        # Validate required columns
        required_columns = [
            "order_request_id",
            "latitude",
            "longitude",
            "pickup_latitude",
            "pickup_longitude",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Clean and validate data
        df = self._clean_data(df)

        # Create parcels from CSV rows
        parcels = self._create_parcels_from_df(df)

        # Group parcels into pickup terminals
        pickup_terminals = self._group_parcels_into_terminals(parcels)

        # Create problem configuration
        config = ProblemConfig(max_knock, time_window_hours, vehicle_specs)

        print(
            f"Processed {len(parcels)} parcels into {len(pickup_terminals)} pickup terminals"
        )

        return config, pickup_terminals

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate CSV data"""
        # Remove rows with missing coordinates
        initial_count = len(df)
        df = df.dropna(
            subset=["latitude", "longitude", "pickup_latitude", "pickup_longitude"]
        )

        if len(df) < initial_count:
            print(f"Removed {initial_count - len(df)} rows with missing coordinates")

        # Validate coordinate ranges
        def validate_coordinates(lat, lon, coord_type="delivery"):
            if not (-90 <= lat <= 90):
                raise ValueError(f"Invalid {coord_type} latitude: {lat}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Invalid {coord_type} longitude: {lon}")

        # Validate all coordinates
        for idx, row in df.iterrows():
            try:
                validate_coordinates(row["latitude"], row["longitude"], "delivery")
                validate_coordinates(
                    row["pickup_latitude"], row["pickup_longitude"], "pickup"
                )
            except ValueError as e:
                print(f"Warning: Row {idx} has invalid coordinates: {e}")
                df = df.drop(idx)

        # Reset index after dropping rows
        df = df.reset_index(drop=True)

        return df

    def _create_parcels_from_df(self, df: pd.DataFrame) -> List[Parcel]:
        """Create Parcel objects from DataFrame"""
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
                    size=self._determine_parcel_size(row),
                )
                parcels.append(parcel)

            except Exception as e:
                print(f"Warning: Could not create parcel from row {idx}: {e}")
                continue

        return parcels

    def _determine_parcel_size(self, row: pd.Series) -> int:
        """Determine parcel size from row data"""
        # Default implementation - can be enhanced based on your business logic
        # You can add columns like 'weight', 'volume', 'package_type' to determine size

        # Example business logic (customize as needed):
        # - Check if there's a size/weight/volume column
        # - Use time slots to determine urgency/size
        # - Use contact info or other metadata

        if "package_size" in row and pd.notna(row["package_size"]):
            return int(row["package_size"])
        elif "weight" in row and pd.notna(row["weight"]):
            # Convert weight to size units (customize conversion)
            return max(1, int(row["weight"] / 0.5))  # Assuming 0.5kg per size unit
        elif "volume" in row and pd.notna(row["volume"]):
            # Convert volume to size units
            return max(1, int(row["volume"] * 10))  # Customize conversion factor
        else:
            # Default size
            return self.default_parcel_size

    def _group_parcels_into_terminals(
        self, parcels: List[Parcel]
    ) -> List[PickupTerminal]:
        """Group parcels into pickup terminals based on proximity"""
        # Group parcels by pickup location with proximity threshold
        terminal_groups = defaultdict(list)
        pickup_locations = {}  # Maps terminal_id to (lat, lon)
        terminal_id_counter = 1

        for parcel in parcels:
            pickup_lat, pickup_lon = parcel.pickup_location

            # Find existing terminal within proximity threshold
            assigned_terminal = None
            for terminal_id, (existing_lat, existing_lon) in pickup_locations.items():
                distance = DistanceCalculator.haversine_distance(
                    pickup_lat, pickup_lon, existing_lat, existing_lon
                )

                if distance <= self.proximity_threshold_meters:
                    assigned_terminal = terminal_id
                    break

            # Assign to existing terminal or create new one
            if assigned_terminal:
                terminal_groups[assigned_terminal].append(parcel)
            else:
                terminal_groups[terminal_id_counter].append(parcel)
                pickup_locations[terminal_id_counter] = (pickup_lat, pickup_lon)
                terminal_id_counter += 1

        # Create PickupTerminal objects
        pickup_terminals = []
        for terminal_id, parcels_list in terminal_groups.items():
            lat, lon = pickup_locations[terminal_id]

            # Update parcel pickup locations to use terminal location (for consistency)
            for parcel in parcels_list:
                parcel.pickup_location = (lat, lon)

            terminal = PickupTerminal(
                pickup_id=terminal_id, lat=lat, lon=lon, parcels=parcels_list
            )
            pickup_terminals.append(terminal)

        return pickup_terminals

    def get_data_summary(self, pickup_terminals: List[PickupTerminal]) -> Dict:
        """Get summary statistics of processed data"""
        total_parcels = sum(len(terminal.parcels) for terminal in pickup_terminals)
        total_size = sum(
            sum(p.size for p in terminal.parcels) for terminal in pickup_terminals
        )

        terminal_sizes = [len(terminal.parcels) for terminal in pickup_terminals]

        return {
            "num_pickup_terminals": len(pickup_terminals),
            "total_parcels": total_parcels,
            "total_size": total_size,
            "avg_parcels_per_terminal": (
                total_parcels / len(pickup_terminals) if pickup_terminals else 0
            ),
            "min_parcels_per_terminal": min(terminal_sizes) if terminal_sizes else 0,
            "max_parcels_per_terminal": max(terminal_sizes) if terminal_sizes else 0,
            "terminals_info": [
                {
                    "terminal_id": terminal.pickup_id,
                    "lat": terminal.lat,
                    "lon": terminal.lon,
                    "num_parcels": len(terminal.parcels),
                    "total_size": sum(p.size for p in terminal.parcels),
                }
                for terminal in pickup_terminals
            ],
        }
