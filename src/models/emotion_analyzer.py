"""
Emotion Analyzer - Integrated System
Combines face detection, tracking, and AU extraction for emotion analysis
"""

import cv2
import numpy as np
from typing import Dict, Optional
from .face_detector import FaceDetector
from .au_extractor import AUExtractor
from .au_storage import AUStorageManager
from .emotion_classifier import EmotionClassifier
from .ml_emotion_classifier import MLEmotionClassifier
from src.utils.landmark_storage import LandmarkStorage
from src.utils.config_loader import get_recording_config, get_emotion_classification_config


class EmotionAnalyzer:
    """
    Integrated emotion analysis system combining:
    - Face detection (MediaPipe Blaze Face)
    - Face tracking (Centroid + IoU with stability)
    - AU extraction (MediaPipe Face Mesh)
    - AU storage (per-ID history)
    """

    def __init__(
        self,
        enable_tracking: bool = True,
        min_detection_confidence: float = 0.5,
        max_history_seconds: float = 60.0,
        fps: int = 30,
        show_landmarks: bool = False,
        landmark_mode: str = 'simple',
        recording_sample_rate: Optional[int] = None,
        use_ml_classifier: bool = False,
        emotion_model_path: Optional[str] = None
    ):
        """
        Initialize emotion analyzer

        Args:
            enable_tracking: Enable face tracking for persistent IDs
            min_detection_confidence: Minimum confidence for detection
            max_history_seconds: Maximum AU history to keep per face
            fps: Expected frame rate
            show_landmarks: Show facial landmarks for debugging
            landmark_mode: 'simple' (AU only), 'full' (all 478), 'mesh' (with connections)
            recording_sample_rate: Save every Nth frame when recording (None = use config.yaml)
            use_ml_classifier: Use ML-based classifier instead of rule-based
            emotion_model_path: Path to trained ML model (None = try default path)
        """
        print("Initializing Emotion Analyzer...")

        # Load config from config.yaml
        recording_config = get_recording_config()
        emotion_config = get_emotion_classification_config()

        # Use provided sample_rate or fall back to config
        if recording_sample_rate is None:
            recording_sample_rate = recording_config.get('sample_rate', 5)

        output_dir = recording_config.get('output_dir', 'data/recordings')

        # Use config values if parameters not explicitly provided
        if use_ml_classifier is False and emotion_config.get('use_ml_classifier'):
            use_ml_classifier = True

        if emotion_model_path is None:
            emotion_model_path = emotion_config.get('model_path', 'models/emotion_classifier.pkl')

        # Face detection and tracking
        self.face_detector = FaceDetector(
            enable_tracking=enable_tracking,
            min_detection_confidence=min_detection_confidence
        )

        # AU extraction
        self.au_extractor = AUExtractor(
            min_detection_confidence=min_detection_confidence
        )

        # AU storage
        self.au_storage = AUStorageManager(
            max_history_seconds=max_history_seconds,
            fps=fps
        )

        # Emotion classifier (ML or rule-based)
        smoothing_window = emotion_config.get('smoothing_window', 5)

        if use_ml_classifier:
            model_type = emotion_config.get('model_type', 'random_forest')

            self.emotion_classifier = MLEmotionClassifier(
                model_type=model_type,
                model_path=emotion_model_path,
                smoothing_window=smoothing_window
            )
            print("[OK] Using ML-based emotion classifier")
        else:
            self.emotion_classifier = EmotionClassifier(
                smoothing_window=smoothing_window
            )
            print("[OK] Using rule-based emotion classifier")

        # Landmark storage for recording
        self.landmark_storage = LandmarkStorage(
            output_dir=output_dir,
            sample_rate=recording_sample_rate
        )
        self.is_recording = False

        self.show_landmarks = show_landmarks
        self.landmark_mode = landmark_mode  # 'simple', 'full', 'mesh'

        # Performance tracking
        self.frame_times = []

        print("[OK] Emotion Analyzer initialized")

    def analyze(self, image: np.ndarray) -> Dict[int, Dict]:
        """
        Analyze emotions in image

        Args:
            image: Input image (BGR)

        Returns:
            Dictionary {face_id: {'bbox': [...], 'confidence': float, 'aus': {...}, 'landmarks': [...]}}
        """
        import time
        start_time = time.time()

        # Detect and track faces
        tracked_faces = self.face_detector.detect(image)

        if not tracked_faces or len(tracked_faces) == 0:
            return {}

        # Extract AUs for each tracked face
        results = {}

        for face_id, detection in tracked_faces.items():
            bbox = detection['bbox']
            confidence = detection['confidence']

            # Extract landmarks first
            landmarks = self.au_extractor.extract_landmarks(image, bbox)

            if landmarks is not None:
                # Extract AUs from landmarks
                aus = self.au_extractor.extract_aus(image, bbox, face_id)

                if aus:
                    # Store AU measurements
                    self.au_storage.update(face_id, aus)

                    # Classify emotion from AUs
                    emotion, emotion_confidence, emotion_scores = self.emotion_classifier.classify(aus, face_id)

                    # Record to landmark storage if recording
                    if self.is_recording:
                        self.landmark_storage.add_frame(
                            timestamp=start_time,
                            face_id=face_id,
                            landmarks=landmarks,
                            bbox=bbox,
                            aus=aus,
                            emotion=emotion,
                            emotion_intensity=emotion_confidence
                        )

                    # Combine results
                    results[face_id] = {
                        'bbox': bbox,
                        'confidence': confidence,
                        'aus': aus,
                        'landmarks': landmarks,
                        'emotion': emotion,
                        'emotion_confidence': emotion_confidence,
                        'emotion_scores': emotion_scores
                    }

        # Track performance
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 30:  # Keep last 30 frames
            self.frame_times.pop(0)

        return results

    def get_statistics(self, face_id: int, window_seconds: float = 5.0) -> Dict:
        """Get AU statistics for a face ID"""
        return self.au_storage.get_statistics(face_id, window_seconds)

    def get_latest_aus(self, face_id: int) -> Optional[Dict]:
        """Get latest AU measurement for a face ID"""
        measurement = self.au_storage.get_latest(face_id)
        if measurement:
            return {
                'au1': measurement.au1,
                'au4': measurement.au4,
                'au6': measurement.au6,
                'au12': measurement.au12,
                'au15': measurement.au15
            }
        return None

    def visualize(self, image: np.ndarray, results: Dict[int, Dict]) -> np.ndarray:
        """
        Visualize detection results with AU values and landmarks

        Args:
            image: Input image
            results: Output from analyze()

        Returns:
            Image with visualization
        """
        output = image.copy()

        if not results:
            return output

        for face_id, data in results.items():
            bbox = data['bbox']
            confidence = data['confidence']
            aus = data.get('aus', {})
            landmarks = data.get('landmarks', None)
            emotion = data.get('emotion', 'Unknown')
            emotion_confidence = data.get('emotion_confidence', 0.0)

            x1, y1, x2, y2 = bbox

            # Color based on emotion (instead of ID)
            color = self.emotion_classifier.get_emotion_color(emotion)
            emoji = self.emotion_classifier.get_emotion_emoji(emotion)

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

            # Draw emotion label with intensity (larger, above bbox)
            emotion_label = f"{emotion} {emoji}"
            cv2.putText(output, emotion_label, (x1, y1 - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            # Draw intensity bar
            bar_width = 150
            bar_height = 20
            bar_x = x1
            bar_y = y1 - 45

            # Background bar (gray)
            cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (100, 100, 100), -1)

            # Filled bar (colored by emotion)
            fill_width = int(bar_width * emotion_confidence)
            cv2.rectangle(output, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                         color, -1)

            # Bar border
            cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 2)

            # Intensity text
            intensity_text = f"{emotion_confidence:.0%}"
            cv2.putText(output, intensity_text, (bar_x + bar_width + 10, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw ID and confidence (smaller, below emotion)
            id_label = f"ID {face_id}: {confidence:.2f}"
            cv2.putText(output, id_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw landmarks if enabled
            if self.show_landmarks and landmarks is not None:
                self._draw_au_landmarks(output, landmarks, color)

            # Draw AU values next to face
            if aus:
                au_y_offset = y1 + 20
                for au_name in ['au1', 'au4', 'au6', 'au12', 'au15']:
                    if au_name in aus:
                        au_text = f"{au_name.upper()}: {aus[au_name]:.2f}"
                        cv2.putText(output, au_text, (x2 + 10, au_y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        au_y_offset += 20

        return output

    def _draw_au_landmarks(self, image: np.ndarray, landmarks: np.ndarray, color: tuple):
        """
        Draw facial landmarks based on landmark_mode

        Args:
            image: Image to draw on
            landmarks: Facial landmarks array (478, 3)
            color: Color for drawing

        Modes:
            'simple': Only AU-relevant landmarks (fastest)
            'full': All 478 landmarks (medium)
            'mesh': All 478 landmarks + connections (slowest)
        """
        # AU-relevant landmarks (always drawn)
        au_landmarks = {
            'AU1_inner_brow': self.au_extractor.LEFT_INNER_BROW + self.au_extractor.RIGHT_INNER_BROW +
                             self.au_extractor.LEFT_UPPER_EYELID + self.au_extractor.RIGHT_UPPER_EYELID,
            'AU4_outer_brow': self.au_extractor.LEFT_BROW_CENTER + self.au_extractor.RIGHT_BROW_CENTER +
                             self.au_extractor.LEFT_OUTER_BROW + self.au_extractor.RIGHT_OUTER_BROW,
            'AU6_eye': self.au_extractor.LEFT_EYE_UPPER + self.au_extractor.LEFT_EYE_LOWER +
                      self.au_extractor.RIGHT_EYE_UPPER + self.au_extractor.RIGHT_EYE_LOWER +
                      self.au_extractor.LEFT_CHEEK + self.au_extractor.RIGHT_CHEEK,
            'AU12_mouth': self.au_extractor.MOUTH_LEFT_CORNER + self.au_extractor.MOUTH_RIGHT_CORNER +
                         self.au_extractor.UPPER_LIP_TOP[:3] + self.au_extractor.LOWER_LIP_BOTTOM[:3],
            'AU15_lips': self.au_extractor.MOUTH_LEFT_CORNER + self.au_extractor.MOUTH_RIGHT_CORNER +
                        self.au_extractor.LOWER_LIP_OUTER[:5]
        }

        au_colors = {
            'AU1_inner_brow': (255, 0, 0),      # Blue - AU1
            'AU4_outer_brow': (0, 0, 255),      # Red - AU4
            'AU6_eye': (0, 255, 255),           # Yellow - AU6
            'AU12_mouth': (0, 255, 0),          # Green - AU12
            'AU15_lips': (255, 0, 255)          # Magenta - AU15
        }

        # Simple mode: Only AU landmarks with full eye contours
        if self.landmark_mode == 'simple':
            # Draw eye contours (full outline)
            # Left eye contour (from MediaPipe Face Mesh)
            left_eye_contour = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            # Right eye contour
            right_eye_contour = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

            # Draw left eye contour
            eye_color = (0, 255, 255)  # Yellow for eyes
            for i in range(len(left_eye_contour)):
                start_idx = left_eye_contour[i]
                end_idx = left_eye_contour[(i + 1) % len(left_eye_contour)]
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_pt = (int(landmarks[start_idx, 0]), int(landmarks[start_idx, 1]))
                    end_pt = (int(landmarks[end_idx, 0]), int(landmarks[end_idx, 1]))
                    cv2.line(image, start_pt, end_pt, eye_color, 2)

            # Draw right eye contour
            for i in range(len(right_eye_contour)):
                start_idx = right_eye_contour[i]
                end_idx = right_eye_contour[(i + 1) % len(right_eye_contour)]
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_pt = (int(landmarks[start_idx, 0]), int(landmarks[start_idx, 1]))
                    end_pt = (int(landmarks[end_idx, 0]), int(landmarks[end_idx, 1]))
                    cv2.line(image, start_pt, end_pt, eye_color, 2)

            # Draw other AU landmarks as points
            for au_name, indices in au_landmarks.items():
                if au_name == 'AU6_eye':  # Skip AU6 eye points since we drew contours
                    continue
                au_color = au_colors.get(au_name, color)
                for idx in indices:
                    if idx < len(landmarks):
                        x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
                        cv2.circle(image, (x, y), 3, au_color, -1)
            return

        # Full and mesh modes: Draw all 478 landmarks
        for idx in range(len(landmarks)):
            x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])

            # Color-code different regions
            if idx in range(0, 17):  # Jaw
                point_color = (255, 200, 200)  # Light blue
            elif idx in range(17, 27):  # Right eyebrow
                point_color = (255, 0, 0)  # Blue
            elif idx in range(27, 36):  # Left eyebrow
                point_color = (255, 0, 0)  # Blue
            elif idx in range(36, 48):  # Eyes
                point_color = (0, 255, 255)  # Yellow
            elif idx in range(48, 68):  # Mouth/lips
                point_color = (255, 0, 255)  # Magenta
            elif idx in range(68, 83):  # Inner mouth
                point_color = (255, 100, 255)  # Light magenta
            else:  # Other facial features
                point_color = (0, 255, 0)  # Green

            cv2.circle(image, (x, y), 1, point_color, -1)

        # Highlight AU-relevant landmarks with larger circles
        for au_name, indices in au_landmarks.items():
            au_color = au_colors.get(au_name, color)
            for idx in indices:
                if idx < len(landmarks):
                    x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
                    cv2.circle(image, (x, y), 3, au_color, -1)

        # Mesh mode: Add face mesh connections
        if self.landmark_mode == 'mesh':
            self._draw_face_mesh_connections(image, landmarks)

    def _draw_face_mesh_connections(self, image: np.ndarray, landmarks: np.ndarray):
        """
        Draw face mesh connections (lines between landmarks)

        Args:
            image: Image to draw on
            landmarks: Facial landmarks array
        """
        # MediaPipe Face Mesh key connections
        # Simplified connections for cleaner visualization
        connections = [
            # Face oval
            *[(i, i+1) for i in range(10, 16)],
            *[(i, i+1) for i in range(234, 249)],
            *[(i, i+1) for i in range(127, 142)],
            *[(i, i+1) for i in range(356, 371)],

            # Left eye
            (33, 133), (133, 173), (173, 157), (157, 158),
            (158, 159), (159, 160), (160, 161), (161, 246),
            (246, 33),

            # Right eye
            (263, 362), (362, 398), (398, 384), (384, 385),
            (385, 386), (386, 387), (387, 388), (388, 466),
            (466, 263),

            # Lips outer
            (61, 146), (146, 91), (91, 181), (181, 84),
            (84, 17), (17, 314), (314, 405), (405, 321),
            (321, 375), (375, 291), (291, 61),

            # Eyebrows
            *[(i, i+1) for i in [70, 63, 105, 66, 107]],  # Right
            *[(i, i+1) for i in [336, 296, 334, 293, 300]],  # Left

            # Nose
            (168, 6), (6, 197), (197, 195), (195, 5),
        ]

        # Draw connections
        for connection in connections:
            if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]

                start_point = (int(start[0]), int(start[1]))
                end_point = (int(end[0]), int(end[1]))

                cv2.line(image, start_point, end_point, (100, 100, 100), 1)

    def export_data(self) -> Dict:
        """Export all AU data"""
        return self.au_storage.export_to_dict()

    def clear_storage(self):
        """Clear all AU storage"""
        self.au_storage.clear_all()

    def get_fps(self) -> float:
        """Get average FPS from recent frames"""
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_avg_processing_time(self) -> float:
        """Get average processing time per frame (ms)"""
        if not self.frame_times:
            return 0.0
        return (sum(self.frame_times) / len(self.frame_times)) * 1000.0

    def start_recording(self):
        """Start recording landmark data"""
        self.landmark_storage.start_session()
        self.is_recording = True
        print("[Recording] Started")

    def stop_recording(self):
        """Stop recording and export data"""
        if not self.is_recording:
            print("[Recording] Not currently recording")
            return

        self.is_recording = False
        print("[Recording] Stopped")

        # Export all formats
        self.landmark_storage.export_all()

        # Show statistics
        stats = self.landmark_storage.get_statistics()
        print("\n[Recording Statistics]")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    def toggle_recording(self):
        """Toggle recording on/off"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()


# Test
if __name__ == "__main__":
    print("=" * 70)
    print(" " * 20 + "Emotion Analyzer Test")
    print("=" * 70)

    analyzer = EmotionAnalyzer(enable_tracking=True, show_landmarks=True)

    print("\nEmotion Analyzer with AU extraction and landmark visualization")
    print("Tracking faces with AU measurements")
    print("Landmark colors:")
    print("  Blue - AU1 (Inner Brow)")
    print("  Red - AU4 (Outer Brow)")
    print("  Yellow - AU6 (Cheek + Eye)")
    print("  Green - AU12 (Mouth Corners)")
    print("  Magenta - AU15 (Upper/Lower Lip)")
    print("  Gray - Nose Bridge (reference)")
    print("\nLandmark modes:")
    print("  simple: Only AU landmarks (fastest)")
    print("  full: All 478 landmarks (medium)")
    print("  mesh: All landmarks + connections (slowest)")
    print("\nControls:")
    print("  q - quit | s - screenshot | c - clear AU storage")
    print("  l - toggle landmarks | m - cycle landmark mode")
    print("  r - toggle recording (saves landmark data)\n")

    cap = cv2.VideoCapture(0)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Analyze emotions
        results = analyzer.analyze(frame)

        # Visualize
        output = analyzer.visualize(frame, results)

        # Show info
        num_faces = len(results)
        cv2.putText(output, f"Tracked Faces: {num_faces}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show active IDs
        if results:
            ids_text = f"Active IDs: {', '.join(map(str, sorted(results.keys())))}"
            cv2.putText(output, ids_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show performance metrics
        fps = analyzer.get_fps()
        proc_time = analyzer.get_avg_processing_time()
        cv2.putText(output, f"FPS: {fps:.1f} | Processing: {proc_time:.1f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show landmark status and mode
        landmark_status = "ON" if analyzer.show_landmarks else "OFF"
        cv2.putText(output, f"Landmarks: {landmark_status} | Mode: {analyzer.landmark_mode}",
                   (10, output.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show recording status
        recording_status = "RECORDING" if analyzer.is_recording else "Not Recording"
        recording_color = (0, 0, 255) if analyzer.is_recording else (255, 255, 255)
        cv2.putText(output, f"[{recording_status}]",
                   (10, output.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, recording_color, 2)

        cv2.putText(output, "q: quit | s: screenshot | c: clear | l: toggle | m: mode | r: record",
                   (10, output.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Emotion Analyzer', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Stop recording if active before quitting
            if analyzer.is_recording:
                analyzer.stop_recording()
            break
        elif key == ord('s'):
            filename = f"emotion_analysis_{frame_count}.jpg"
            cv2.imwrite(filename, output)
            print(f"Screenshot saved: {filename}")
        elif key == ord('c'):
            analyzer.clear_storage()
            print("AU storage cleared")
        elif key == ord('l'):
            analyzer.show_landmarks = not analyzer.show_landmarks
            print(f"Landmarks: {'ON' if analyzer.show_landmarks else 'OFF'}")
        elif key == ord('m'):
            # Cycle through landmark modes: simple -> full -> mesh -> simple
            modes = ['simple', 'full', 'mesh']
            current_idx = modes.index(analyzer.landmark_mode)
            next_idx = (current_idx + 1) % len(modes)
            analyzer.landmark_mode = modes[next_idx]
            print(f"Landmark mode: {analyzer.landmark_mode}")
        elif key == ord('r'):
            analyzer.toggle_recording()

    cap.release()
    cv2.destroyAllWindows()

    # Export data
    export_data = analyzer.export_data()
    print(f"\n[OK] Emotion analyzer test completed")
    print(f"Total faces tracked: {len(export_data)}")
    for face_id, measurements in export_data.items():
        print(f"  Face ID {face_id}: {len(measurements)} AU measurements")
