"""
AU Storage System
Store and manage AU measurements for each tracked face ID
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque, defaultdict
from dataclasses import dataclass, field
import time


@dataclass
class AUMeasurement:
    """Single AU measurement with timestamp"""
    timestamp: float
    au1: float  # Inner Brow Raiser
    au4: float  # Brow Lowerer
    au6: float  # Cheek Raiser
    au12: float  # Lip Corner Puller
    au15: float  # Lip Corner Depressor

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'au1': self.au1,
            'au4': self.au4,
            'au6': self.au6,
            'au12': self.au12,
            'au15': self.au15
        }


@dataclass
class FaceAUHistory:
    """AU measurement history for a single face ID"""
    face_id: int
    measurements: deque = field(default_factory=lambda: deque(maxlen=1800))  # 60 sec @ 30fps

    def add_measurement(self, aus: Dict[str, float]):
        """
        Add new AU measurement

        Args:
            aus: Dictionary with AU values {'au1': float, 'au4': float, ...}
        """
        measurement = AUMeasurement(
            timestamp=time.time(),
            au1=aus.get('au1', 0.0),
            au4=aus.get('au4', 0.0),
            au6=aus.get('au6', 0.0),
            au12=aus.get('au12', 0.0),
            au15=aus.get('au15', 0.0)
        )
        self.measurements.append(measurement)

    def get_latest(self) -> Optional[AUMeasurement]:
        """Get most recent measurement"""
        return self.measurements[-1] if self.measurements else None

    def get_recent(self, n: int = 30) -> List[AUMeasurement]:
        """
        Get last N measurements

        Args:
            n: Number of recent measurements (default 30 = 1 sec @ 30fps)
        """
        return list(self.measurements)[-n:]

    def get_statistics(self, window_seconds: float = 5.0) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for recent time window

        Args:
            window_seconds: Time window in seconds (default 5.0)

        Returns:
            Dictionary with statistics for each AU:
            {
                'au1': {'mean': float, 'std': float, 'max': float, 'min': float},
                ...
            }
        """
        if not self.measurements:
            return {}

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Filter measurements within time window
        recent = [m for m in self.measurements if m.timestamp >= cutoff_time]

        if not recent:
            return {}

        # Compute statistics for each AU
        stats = {}
        for au_name in ['au1', 'au4', 'au6', 'au12', 'au15']:
            values = [getattr(m, au_name) for m in recent]
            stats[au_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values),
                'median': np.median(values)
            }

        return stats

    def clear(self):
        """Clear all measurements"""
        self.measurements.clear()

    def get_count(self) -> int:
        """Get total number of measurements"""
        return len(self.measurements)


class AUStorageManager:
    """
    Manage AU measurements for multiple tracked faces
    """

    def __init__(self, max_history_seconds: float = 60.0, fps: int = 30):
        """
        Initialize AU storage manager

        Args:
            max_history_seconds: Maximum history to keep per face (default 60 seconds)
            fps: Expected frame rate for calculating max history length
        """
        self.max_history_length = int(max_history_seconds * fps)
        self.face_histories: Dict[int, FaceAUHistory] = {}

    def update(self, face_id: int, aus: Dict[str, float]):
        """
        Update AU measurements for a face ID

        Args:
            face_id: Face ID
            aus: AU measurements {'au1': float, 'au4': float, ...}
        """
        # Create history if doesn't exist
        if face_id not in self.face_histories:
            self.face_histories[face_id] = FaceAUHistory(
                face_id=face_id,
                measurements=deque(maxlen=self.max_history_length)
            )

        # Add measurement
        self.face_histories[face_id].add_measurement(aus)

    def get_latest(self, face_id: int) -> Optional[AUMeasurement]:
        """Get latest AU measurement for a face ID"""
        if face_id in self.face_histories:
            return self.face_histories[face_id].get_latest()
        return None

    def get_statistics(self, face_id: int, window_seconds: float = 5.0) -> Dict[str, Dict[str, float]]:
        """Get AU statistics for a face ID"""
        if face_id in self.face_histories:
            return self.face_histories[face_id].get_statistics(window_seconds)
        return {}

    def get_all_latest(self) -> Dict[int, AUMeasurement]:
        """Get latest measurements for all tracked faces"""
        return {
            face_id: history.get_latest()
            for face_id, history in self.face_histories.items()
            if history.get_latest() is not None
        }

    def remove_face(self, face_id: int):
        """Remove AU history for a face ID"""
        if face_id in self.face_histories:
            del self.face_histories[face_id]

    def clear_all(self):
        """Clear all AU histories"""
        self.face_histories.clear()

    def get_active_faces(self) -> List[int]:
        """Get list of face IDs with AU data"""
        return list(self.face_histories.keys())

    def get_face_count(self) -> int:
        """Get number of tracked faces"""
        return len(self.face_histories)

    def export_to_dict(self) -> Dict[int, List[Dict]]:
        """
        Export all AU data to dictionary format

        Returns:
            {
                face_id: [
                    {'timestamp': float, 'au1': float, ...},
                    ...
                ]
            }
        """
        export_data = {}
        for face_id, history in self.face_histories.items():
            export_data[face_id] = [
                m.to_dict() for m in history.measurements
            ]
        return export_data


# Test
if __name__ == "__main__":
    print("=" * 70)
    print(" " * 25 + "AU Storage Test")
    print("=" * 70)

    storage = AUStorageManager(max_history_seconds=5.0, fps=30)

    # Simulate measurements for face ID 0
    print("\nSimulating AU measurements for Face ID 0...")
    for i in range(150):  # 5 seconds @ 30fps
        aus = {
            'au1': np.random.random() * 0.5,
            'au4': np.random.random() * 0.3,
            'au6': np.random.random() * 0.8,
            'au12': np.random.random() * 0.9,
            'au15': np.random.random() * 0.2
        }
        storage.update(face_id=0, aus=aus)
        time.sleep(0.01)  # Simulate frame delay

    # Get latest
    latest = storage.get_latest(0)
    if latest:
        print(f"\nLatest measurement:")
        print(f"  AU1: {latest.au1:.3f}")
        print(f"  AU4: {latest.au4:.3f}")
        print(f"  AU6: {latest.au6:.3f}")
        print(f"  AU12: {latest.au12:.3f}")
        print(f"  AU15: {latest.au15:.3f}")

    # Get statistics
    stats = storage.get_statistics(0, window_seconds=5.0)
    print(f"\nStatistics (5-second window):")
    for au_name, au_stats in stats.items():
        print(f"  {au_name.upper()}:")
        print(f"    Mean: {au_stats['mean']:.3f}")
        print(f"    Std:  {au_stats['std']:.3f}")
        print(f"    Max:  {au_stats['max']:.3f}")
        print(f"    Min:  {au_stats['min']:.3f}")

    # Test multiple faces
    print(f"\nAdding measurements for Face ID 1...")
    for i in range(30):
        aus = {
            'au1': 0.8,  # High AU1
            'au4': 0.2,
            'au6': 0.1,
            'au12': 0.1,
            'au15': 0.9  # High AU15 (sad)
        }
        storage.update(face_id=1, aus=aus)

    print(f"\nActive faces: {storage.get_active_faces()}")
    print(f"Total tracked faces: {storage.get_face_count()}")

    # Export
    export_data = storage.export_to_dict()
    print(f"\nExport data summary:")
    for face_id, measurements in export_data.items():
        print(f"  Face ID {face_id}: {len(measurements)} measurements")

    print("\n[OK] AU storage test completed")
