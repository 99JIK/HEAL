"""
Landmark Storage System for Privacy-Preserving Data Collection
Stores facial landmarks as normalized coordinates instead of actual images
Can be exported to time-series databases
"""

import json
import csv
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path


class LandmarkFrame:
    """Single frame of landmark data"""

    def __init__(
        self,
        timestamp: float,
        face_id: int,
        landmarks: np.ndarray,
        bbox: np.ndarray,
        aus: Dict[str, float],
        emotion: str,
        emotion_intensity: float
    ):
        """
        Initialize landmark frame

        Args:
            timestamp: Unix timestamp (seconds since epoch)
            face_id: Face tracker ID
            landmarks: Raw landmarks array (478, 3)
            bbox: Face bounding box [x1, y1, x2, y2]
            aus: AU measurements {'au1': float, ...}
            emotion: Classified emotion
            emotion_intensity: Emotion intensity (0-1)
        """
        self.timestamp = timestamp
        self.face_id = face_id
        self.emotion = emotion
        self.emotion_intensity = emotion_intensity
        self.aus = aus

        # Normalize landmarks to relative coordinates (0-1)
        self.normalized_landmarks = self._normalize_landmarks(landmarks, bbox)

        # Store bbox size for reference (not absolute position)
        x1, y1, x2, y2 = bbox
        self.bbox_width = x2 - x1
        self.bbox_height = y2 - y1

    def _normalize_landmarks(self, landmarks: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to relative coordinates within bounding box
        This removes absolute position information for privacy

        Args:
            landmarks: Absolute landmark coordinates (478, 3)
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Normalized landmarks (478, 3) with x,y in range [0, 1]
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        normalized = landmarks.copy()

        # Normalize x and y to [0, 1] relative to bbox
        normalized[:, 0] = (landmarks[:, 0] - x1) / width
        normalized[:, 1] = (landmarks[:, 1] - y1) / height
        # Keep z as is (depth is already relative)

        return normalized

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'face_id': self.face_id,
            'emotion': self.emotion,
            'emotion_intensity': float(self.emotion_intensity),
            'aus': {k: float(v) for k, v in self.aus.items()},
            'bbox_size': {
                'width': int(self.bbox_width),
                'height': int(self.bbox_height)
            },
            'landmarks': self.normalized_landmarks.tolist()  # (478, 3) list
        }

    def to_flat_dict(self) -> Dict:
        """
        Convert to flat dictionary for time-series DB or CSV
        Only essential features (no full landmark array)
        """
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'face_id': self.face_id,
            'emotion': self.emotion,
            'emotion_intensity': float(self.emotion_intensity),
            'au1': float(self.aus.get('au1', 0)),
            'au4': float(self.aus.get('au4', 0)),
            'au6': float(self.aus.get('au6', 0)),
            'au12': float(self.aus.get('au12', 0)),
            'au15': float(self.aus.get('au15', 0)),
            'bbox_width': int(self.bbox_width),
            'bbox_height': int(self.bbox_height)
        }


class LandmarkStorage:
    """Storage manager for landmark data"""

    def __init__(self, output_dir: str = "data/recordings", sample_rate: int = 1):
        """
        Initialize landmark storage

        Args:
            output_dir: Directory to save data files
            sample_rate: Save every Nth frame (1 = all frames, 5 = every 5th frame, etc.)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffer
        self.frames: List[LandmarkFrame] = []

        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = None

        # Sampling control
        self.sample_rate = max(1, sample_rate)  # At least 1
        self.frame_counter = 0

    def start_session(self):
        """Start a new recording session"""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.frames.clear()
        self.frame_counter = 0
        print(f"[Recording] Session started: {self.session_id} (sample rate: 1/{self.sample_rate})")

    def add_frame(
        self,
        timestamp: float,
        face_id: int,
        landmarks: np.ndarray,
        bbox: np.ndarray,
        aus: Dict[str, float],
        emotion: str,
        emotion_intensity: float
    ):
        """Add a frame to the recording (respects sample_rate)"""
        # Increment counter
        self.frame_counter += 1

        # Only save if this frame matches the sample rate
        if self.frame_counter % self.sample_rate != 0:
            return

        frame = LandmarkFrame(
            timestamp=timestamp,
            face_id=face_id,
            landmarks=landmarks,
            bbox=bbox,
            aus=aus,
            emotion=emotion,
            emotion_intensity=emotion_intensity
        )
        self.frames.append(frame)

    def export_json(self, filename: Optional[str] = None) -> str:
        """
        Export full landmark data to JSON
        Includes all 478 landmarks per frame

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"landmarks_{self.session_id}.json"

        filepath = self.output_dir / filename

        data = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'total_frames': len(self.frames),
            'frames': [frame.to_dict() for frame in self.frames]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Saved] Full landmark data: {filepath}")
        return str(filepath)

    def export_csv(self, filename: Optional[str] = None) -> str:
        """
        Export summary data to CSV (time-series format)
        Only AUs and emotion, no full landmarks

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"timeseries_{self.session_id}.csv"

        filepath = self.output_dir / filename

        if not self.frames:
            print("[Warning] No frames to export")
            return str(filepath)

        # Get fieldnames from first frame
        fieldnames = list(self.frames[0].to_flat_dict().keys())

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for frame in self.frames:
                writer.writerow(frame.to_flat_dict())

        print(f"[Saved] Time-series CSV: {filepath}")
        return str(filepath)

    def export_influxdb_format(self, filename: Optional[str] = None) -> str:
        """
        Export in InfluxDB line protocol format
        For direct import to time-series databases

        Format: measurement,tag=value field=value timestamp

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"influxdb_{self.session_id}.txt"

        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            for frame in self.frames:
                # Convert to nanoseconds for InfluxDB
                timestamp_ns = int(frame.timestamp * 1e9)

                # Tags (indexed)
                tags = f"face_id={frame.face_id},emotion={frame.emotion}"

                # Fields (not indexed)
                fields = (
                    f"emotion_intensity={frame.emotion_intensity},"
                    f"au1={frame.aus.get('au1', 0)},"
                    f"au4={frame.aus.get('au4', 0)},"
                    f"au6={frame.aus.get('au6', 0)},"
                    f"au12={frame.aus.get('au12', 0)},"
                    f"au15={frame.aus.get('au15', 0)},"
                    f"bbox_width={frame.bbox_width},"
                    f"bbox_height={frame.bbox_height}"
                )

                # Write line
                line = f"emotion_data,{tags} {fields} {timestamp_ns}\n"
                f.write(line)

        print(f"[Saved] InfluxDB format: {filepath}")
        return str(filepath)

    def export_all(self):
        """Export data in all formats"""
        self.export_json()
        self.export_csv()
        self.export_influxdb_format()

        print(f"\n[Summary] Exported {len(self.frames)} frames")
        if self.frames:
            duration = self.frames[-1].timestamp - self.frames[0].timestamp
            print(f"  Duration: {duration:.1f} seconds")
            print(f"  Avg FPS: {len(self.frames) / duration:.1f}")

    def get_statistics(self) -> Dict:
        """Get recording statistics"""
        if not self.frames:
            return {}

        emotions = [f.emotion for f in self.frames]
        from collections import Counter
        emotion_counts = Counter(emotions)

        return {
            'total_frames': len(self.frames),
            'duration_seconds': self.frames[-1].timestamp - self.frames[0].timestamp,
            'unique_faces': len(set(f.face_id for f in self.frames)),
            'emotion_distribution': dict(emotion_counts),
            'avg_emotion_intensity': np.mean([f.emotion_intensity for f in self.frames])
        }


class LandmarkVisualizer:
    """Visualize landmarks from saved data (for verification)"""

    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load landmark data from JSON"""
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def render_frame(frame_data: Dict, canvas_size: Tuple[int, int] = (400, 400)) -> np.ndarray:
        """
        Render a frame from normalized landmarks

        Args:
            frame_data: Frame dictionary from JSON
            canvas_size: Output image size (width, height)

        Returns:
            Rendered image (BGR)
        """
        import cv2

        width, height = canvas_size
        image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        # Get landmarks
        landmarks = np.array(frame_data['landmarks'])  # (478, 3)

        # Scale to canvas
        scaled_landmarks = landmarks.copy()
        scaled_landmarks[:, 0] *= width
        scaled_landmarks[:, 1] *= height

        # Draw landmarks
        for i, (x, y, z) in enumerate(scaled_landmarks):
            x, y = int(x), int(y)
            cv2.circle(image, (x, y), 2, (100, 100, 100), -1)

        # Draw eye contours (same as simple mode)
        left_eye_contour = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_contour = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

        eye_color = (0, 200, 200)
        for contour in [left_eye_contour, right_eye_contour]:
            for i in range(len(contour)):
                start_idx = contour[i]
                end_idx = contour[(i + 1) % len(contour)]
                start_pt = tuple(scaled_landmarks[start_idx, :2].astype(int))
                end_pt = tuple(scaled_landmarks[end_idx, :2].astype(int))
                cv2.line(image, start_pt, end_pt, eye_color, 2)

        # Add text info
        emotion = frame_data.get('emotion', 'Unknown')
        intensity = frame_data.get('emotion_intensity', 0)
        text = f"{emotion}: {intensity:.0%}"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return image


# Test
if __name__ == "__main__":
    print("=" * 70)
    print(" " * 20 + "Landmark Storage Test")
    print("=" * 70)

    # Create test storage
    storage = LandmarkStorage(output_dir="data/test_recordings")
    storage.start_session()

    # Add some test frames
    print("\nAdding test frames...")
    for i in range(10):
        # Create dummy data
        timestamp = datetime.now().timestamp() + i * 0.033  # ~30 FPS
        landmarks = np.random.rand(478, 3) * 100  # Random landmarks
        bbox = np.array([100, 100, 200, 200])
        aus = {
            'au1': np.random.rand(),
            'au4': np.random.rand(),
            'au6': np.random.rand(),
            'au12': np.random.rand(),
            'au15': np.random.rand()
        }
        emotion = np.random.choice(['Neutral', 'Happy', 'Sad'])
        intensity = np.random.rand()

        storage.add_frame(timestamp, 0, landmarks, bbox, aus, emotion, intensity)

    print(f"Added {len(storage.frames)} frames")

    # Export
    print("\nExporting data...")
    storage.export_all()

    # Statistics
    print("\nStatistics:")
    stats = storage.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n[OK] Landmark storage test completed")
