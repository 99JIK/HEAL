"""
Face Detector using MediaPipe Blaze Face
Accurate face detection with mask support
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import mediapipe as mp
from .face_tracker import FaceTracker


class FaceDetector:
    """MediaPipe Blaze Face detector with mask support"""

    def __init__(self, enable_tracking: bool = True, model_selection: int = 1, min_detection_confidence: float = 0.5):
        """
        Initialize MediaPipe Blaze Face detector

        Args:
            enable_tracking: Enable face tracking for persistent IDs
            model_selection: 0 for short-range (2m), 1 for full-range (5m)
            min_detection_confidence: Minimum confidence threshold (0.0 to 1.0)
        """
        print("Loading MediaPipe Blaze Face detector...")

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )

        print("[OK] Face detector loaded (Blaze Face)")

        self.conf_threshold = min_detection_confidence

        # Face tracker
        self.enable_tracking = enable_tracking
        self.tracker = FaceTracker(max_disappeared=30) if enable_tracking else None
        if enable_tracking:
            print("[OK] Face tracker enabled")

    def detect(self, image: np.ndarray, detect_multiple: bool = False):
        """
        Detect face bounding box(es)

        Args:
            image: Input image (BGR)
            detect_multiple: If True, return all faces; if False, return best face only

        Returns:
            If tracking disabled:
                If detect_multiple=False:
                    Dictionary with:
                        - bbox: [x1, y1, x2, y2] face bounding box
                        - confidence: detection confidence
                    Returns None if no face detected

                If detect_multiple=True:
                    List of dictionaries (one per face)
                    Returns empty list if no faces detected

            If tracking enabled:
                Dictionary {id: {'bbox': [...], 'confidence': float}}
                Empty dict if no faces detected
        """
        h, w = image.shape[:2]

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run detection
        results = self.face_detection.process(image_rgb)

        # Collect all valid detections
        faces = []

        if results.detections:
            for detection in results.detections:
                # Get confidence score
                confidence = detection.score[0]

                # Get bounding box (normalized coordinates)
                bbox_rel = detection.location_data.relative_bounding_box

                # Convert to absolute pixel coordinates
                x1 = int(bbox_rel.xmin * w)
                y1 = int(bbox_rel.ymin * h)
                x2 = int((bbox_rel.xmin + bbox_rel.width) * w)
                y2 = int((bbox_rel.ymin + bbox_rel.height) * h)

                # Clamp to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                bbox = np.array([x1, y1, x2, y2])

                faces.append({
                    'bbox': bbox,
                    'confidence': float(confidence)
                })

        # If tracking enabled, use tracker
        if self.enable_tracking and self.tracker:
            tracked_faces = self.tracker.update(faces)
            return tracked_faces

        # No tracking - return raw detections
        if not faces:
            return [] if detect_multiple else None

        if detect_multiple:
            # Return all faces sorted by confidence
            return sorted(faces, key=lambda x: x['confidence'], reverse=True)
        else:
            # Return best face only
            return max(faces, key=lambda x: x['confidence'])

    def visualize(
        self,
        image: np.ndarray,
        detections
    ) -> np.ndarray:
        """
        Draw bounding box(es) on image

        Args:
            image: Input image
            detections: Output from detect() - dict with IDs or list

        Returns:
            Image with bounding box(es)
        """
        output = image.copy()

        if detections is None or (isinstance(detections, (list, dict)) and len(detections) == 0):
            return output

        # Handle tracking mode (dict with IDs)
        if self.enable_tracking and isinstance(detections, dict) and not any(k in detections for k in ['bbox', 'confidence']):
            for face_id, detection in detections.items():
                bbox = detection['bbox']
                confidence = detection['confidence']

                # Draw face bounding box
                x1, y1, x2, y2 = bbox

                # Color based on ID for visual distinction
                color = self._get_color_for_id(face_id)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

                # Draw ID and confidence
                label = f"ID {face_id}: {confidence:.2f}"
                cv2.putText(output, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            return output

        # Handle non-tracking mode (list or single dict)
        if isinstance(detections, dict):
            detections = [detections]

        # Draw all faces
        for idx, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']

            # Draw face bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw confidence and face number
            label = f"Face {idx+1}: {confidence:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return output

    @staticmethod
    def _get_color_for_id(face_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for each face ID"""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Orange
        ]
        return colors[face_id % len(colors)]

    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


# Test
if __name__ == "__main__":
    print("=" * 70)
    print(" " * 12 + "MediaPipe Blaze Face Detector + Tracker Test")
    print("=" * 70)

    detector = FaceDetector(enable_tracking=True)

    print("\nMediaPipe Blaze Face + Centroid Tracking")
    print("Mask-friendly detection with unique IDs")
    print("Each person gets a distinct color")
    print("Controls: q - quit, s - screenshot\n")

    cap = cv2.VideoCapture(0)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect and track faces
        tracked_faces = detector.detect(frame)

        # Visualize
        output = detector.visualize(frame, tracked_faces)

        # Show info
        num_faces = len(tracked_faces)
        cv2.putText(output, f"Tracked Faces: {num_faces}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show active IDs
        if tracked_faces:
            ids_text = f"Active IDs: {', '.join(map(str, sorted(tracked_faces.keys())))}"
            cv2.putText(output, ids_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(output, "q: quit | s: screenshot",
                   (10, output.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('MediaPipe Face Detector + Tracker', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"tracked_face_{frame_count}.jpg"
            cv2.imwrite(filename, output)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n[OK] Face tracking test completed")
