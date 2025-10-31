"""
Lightweight Face Tracker using Centroid Tracking
Tracks multiple faces across frames without face recognition
With stable detection threshold (1 second minimum)
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import OrderedDict


class FaceTracker:
    """Lightweight face tracker using centroid + IoU with stability threshold"""

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3, min_stable_frames: int = 30, max_ids: int = 1000000):
        """
        Initialize face tracker

        Args:
            max_disappeared: Maximum frames a face can disappear before removal
            iou_threshold: Minimum IoU to consider same face
            min_stable_frames: Minimum frames to be detected before assigning ID (default 30 = 1 sec @ 30fps)
            max_ids: Maximum number of IDs (circular queue, default 1000000 = 0-999999)
        """
        # Circular ID queue
        self.max_ids = max_ids
        self.available_ids = set(range(max_ids))  # Pool of available IDs

        self.objects = OrderedDict()  # {id: bbox} - confirmed objects with ID
        self.disappeared = OrderedDict()  # {id: disappeared_count}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

        # Stability tracking for new detections
        self.min_stable_frames = min_stable_frames
        self.pending_objects = OrderedDict()  # {pending_id: {'bbox': ..., 'count': ..., 'confidence': ...}}
        self.next_pending_id = 0

    def register(self, bbox: np.ndarray) -> int:
        """
        Register new face with unique ID from circular queue (after stability check)

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            Assigned ID from circular queue
        """
        # Get ID from available pool (circular queue)
        if self.available_ids:
            face_id = min(self.available_ids)  # Get lowest available ID
            self.available_ids.remove(face_id)
        else:
            # All IDs in use - reuse oldest ID (shouldn't happen with proper max_disappeared)
            face_id = min(self.objects.keys()) if self.objects else 0
            self.deregister(face_id)

        self.objects[face_id] = bbox
        self.disappeared[face_id] = 0
        return face_id

    def deregister(self, face_id: int):
        """Remove face from tracking and return ID to circular queue"""
        if face_id in self.objects:
            del self.objects[face_id]
            del self.disappeared[face_id]
            # Return ID to available pool for reuse
            self.available_ids.add(face_id)

    def register_pending(self, bbox: np.ndarray, confidence: float) -> int:
        """Register a pending (not yet stable) detection"""
        pending_id = self.next_pending_id
        self.pending_objects[pending_id] = {
            'bbox': bbox,
            'count': 1,
            'confidence': confidence
        }
        self.next_pending_id += 1
        return pending_id

    def deregister_pending(self, pending_id: int):
        """Remove pending detection"""
        if pending_id in self.pending_objects:
            del self.pending_objects[pending_id]

    @staticmethod
    def compute_centroid(bbox: np.ndarray) -> Tuple[float, float]:
        """Compute centroid of bounding box"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx, cy)

    @staticmethod
    def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute Intersection over Union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracked faces with new detections

        Args:
            detections: List of {'bbox': [x1,y1,x2,y2], 'confidence': float}

        Returns:
            Dictionary {id: {'bbox': [...], 'confidence': float}} - only stable/confirmed objects
        """
        # No detections - mark all as disappeared and clear pending
        if len(detections) == 0:
            for face_id in list(self.disappeared.keys()):
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)

            # Clear all pending objects
            self.pending_objects.clear()

            return {}

        # Extract bboxes
        input_bboxes = np.array([d['bbox'] for d in detections])
        input_confidences = [d['confidence'] for d in detections]

        # Match with confirmed objects first
        result = {}
        used_detection_indices = set()

        if len(self.objects) > 0:
            object_ids = list(self.objects.keys())
            object_bboxes = np.array(list(self.objects.values()))

            object_centroids = np.array([self.compute_centroid(bbox) for bbox in object_bboxes])
            input_centroids = np.array([self.compute_centroid(bbox) for bbox in input_bboxes])

            # Compute distance matrix
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    D[i, j] = np.linalg.norm(obj_centroid - input_centroid)

            # Compute IoU matrix
            IoU = np.zeros((len(object_bboxes), len(input_bboxes)))
            for i, obj_bbox in enumerate(object_bboxes):
                for j, input_bbox in enumerate(input_bboxes):
                    IoU[i, j] = self.compute_iou(obj_bbox, input_bbox)

            # Find best matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_detection_indices:
                    continue

                # Validate with IoU
                if IoU[row, col] < self.iou_threshold:
                    continue

                face_id = object_ids[row]
                self.objects[face_id] = input_bboxes[col]
                self.disappeared[face_id] = 0

                result[face_id] = {
                    'bbox': input_bboxes[col],
                    'confidence': input_confidences[col]
                }

                used_rows.add(row)
                used_detection_indices.add(col)

            # Mark unmatched confirmed objects as disappeared
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                face_id = object_ids[row]
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)

        # Now handle pending objects and new detections
        remaining_detections = [i for i in range(len(detections)) if i not in used_detection_indices]

        if len(remaining_detections) > 0:
            remaining_bboxes = input_bboxes[remaining_detections]
            remaining_confidences = [input_confidences[i] for i in remaining_detections]

            # Match with pending objects
            used_pending_ids = set()
            used_remaining_indices = set()

            if len(self.pending_objects) > 0:
                pending_ids = list(self.pending_objects.keys())
                pending_bboxes = np.array([p['bbox'] for p in self.pending_objects.values()])

                pending_centroids = np.array([self.compute_centroid(bbox) for bbox in pending_bboxes])
                remaining_centroids = np.array([self.compute_centroid(bbox) for bbox in remaining_bboxes])

                # Distance matrix
                D_pending = np.zeros((len(pending_centroids), len(remaining_centroids)))
                for i, pend_centroid in enumerate(pending_centroids):
                    for j, rem_centroid in enumerate(remaining_centroids):
                        D_pending[i, j] = np.linalg.norm(pend_centroid - rem_centroid)

                # IoU matrix
                IoU_pending = np.zeros((len(pending_bboxes), len(remaining_bboxes)))
                for i, pend_bbox in enumerate(pending_bboxes):
                    for j, rem_bbox in enumerate(remaining_bboxes):
                        IoU_pending[i, j] = self.compute_iou(pend_bbox, rem_bbox)

                # Match
                rows = D_pending.min(axis=1).argsort()
                cols = D_pending.argmin(axis=1)[rows]

                for (row, col) in zip(rows, cols):
                    if row in used_pending_ids or col in used_remaining_indices:
                        continue

                    if IoU_pending[row, col] < self.iou_threshold:
                        continue

                    pending_id = pending_ids[row]
                    detection_idx = remaining_detections[col]

                    # Increment count
                    self.pending_objects[pending_id]['bbox'] = input_bboxes[detection_idx]
                    self.pending_objects[pending_id]['count'] += 1
                    self.pending_objects[pending_id]['confidence'] = input_confidences[detection_idx]

                    # Check if stable enough to promote to confirmed
                    if self.pending_objects[pending_id]['count'] >= self.min_stable_frames:
                        face_id = self.register(input_bboxes[detection_idx])
                        result[face_id] = {
                            'bbox': input_bboxes[detection_idx],
                            'confidence': input_confidences[detection_idx]
                        }
                        self.deregister_pending(pending_id)

                    used_pending_ids.add(row)
                    used_remaining_indices.add(col)

                # Remove pending objects that weren't matched
                unused_pending_rows = set(range(len(pending_ids))) - used_pending_ids
                for row in unused_pending_rows:
                    pending_id = pending_ids[row]
                    self.deregister_pending(pending_id)

            # Register completely new detections as pending
            unused_remaining = set(range(len(remaining_detections))) - used_remaining_indices
            for idx in unused_remaining:
                detection_idx = remaining_detections[idx]
                self.register_pending(input_bboxes[detection_idx], input_confidences[detection_idx])

        return result

    def get_active_count(self) -> int:
        """Get number of actively tracked faces (confirmed only)"""
        return len(self.objects)

    def get_pending_count(self) -> int:
        """Get number of pending (not yet stable) faces"""
        return len(self.pending_objects)


# Test
if __name__ == "__main__":
    print("=" * 70)
    print(" " * 20 + "Face Tracker Test (with stability)")
    print("=" * 70)

    tracker = FaceTracker(max_disappeared=30, min_stable_frames=3)  # 3 frames for quick test

    # Simulate detections
    print("\nFrame 1: New face appears")
    detections = [
        {'bbox': np.array([100, 100, 200, 200]), 'confidence': 0.95}
    ]
    result = tracker.update(detections)
    print(f"Confirmed IDs: {list(result.keys())} (should be empty - not stable yet)")
    print(f"Pending count: {tracker.get_pending_count()}")

    print("\nFrame 2: Same face")
    detections = [
        {'bbox': np.array([102, 102, 202, 202]), 'confidence': 0.94}
    ]
    result = tracker.update(detections)
    print(f"Confirmed IDs: {list(result.keys())} (should still be empty)")
    print(f"Pending count: {tracker.get_pending_count()}")

    print("\nFrame 3: Same face - NOW CONFIRMED")
    detections = [
        {'bbox': np.array([104, 104, 204, 204]), 'confidence': 0.96}
    ]
    result = tracker.update(detections)
    print(f"Confirmed IDs: {list(result.keys())} (should show ID 0 now)")
    print(f"Pending count: {tracker.get_pending_count()}")

    print("\nFrame 4: Second face appears")
    detections = [
        {'bbox': np.array([106, 106, 206, 206]), 'confidence': 0.95},
        {'bbox': np.array([300, 150, 400, 250]), 'confidence': 0.92}
    ]
    result = tracker.update(detections)
    print(f"Confirmed IDs: {list(result.keys())} (should show ID 0 only)")
    print(f"Pending count: {tracker.get_pending_count()}")

    print("\n[OK] Face tracker test completed")
