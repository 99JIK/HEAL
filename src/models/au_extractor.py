"""
AU (Action Unit) Extractor using MediaPipe Face Mesh
Extracts FACS Action Units for emotion analysis
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import mediapipe as mp


class AUExtractor:
    """
    Extract Action Units from facial landmarks using MediaPipe Face Mesh

    Target AUs:
    - AU0: Neutral (baseline reference)
    - AU1: Inner Brow Raiser
    - AU4: Brow Lowerer
    - AU6: Cheek Raiser
    - AU12: Lip Corner Puller
    - AU15: Lip Corner Depressor
    """

    # MediaPipe Face Mesh landmark indices (478 landmarks total)
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    # AU1: Inner Brow Raiser - 내측 눈썹 올림
    LEFT_INNER_BROW = [70, 63, 105, 66, 107]  # 왼쪽 내측 눈썹
    RIGHT_INNER_BROW = [336, 296, 334, 293, 300]  # 오른쪽 내측 눈썹
    LEFT_UPPER_EYELID = [159, 145, 158]  # 왼쪽 위 눈꺼풀 (눈썹-눈 거리 측정용)
    RIGHT_UPPER_EYELID = [386, 374, 385]  # 오른쪽 위 눈꺼풀

    # AU4: Brow Lowerer - 눈썹 찡그림 (눈썹 내림)
    LEFT_OUTER_BROW = [46, 53, 52, 65]  # 왼쪽 외측 눈썹
    RIGHT_OUTER_BROW = [276, 283, 282, 295]  # 오른쪽 외측 눈썹
    LEFT_BROW_CENTER = [105, 66, 107]  # 왼쪽 눈썹 중앙
    RIGHT_BROW_CENTER = [334, 296, 300]  # 오른쪽 눈썹 중앙

    # AU6: Cheek Raiser - 뺨 올림 (눈 찌푸림)
    LEFT_EYE_UPPER = [159, 145, 158, 157, 173, 133]  # 왼쪽 위 눈꺼풀
    LEFT_EYE_LOWER = [144, 145, 153, 154, 155, 133]  # 왼쪽 아래 눈꺼풀
    RIGHT_EYE_UPPER = [386, 374, 385, 384, 398, 362]  # 오른쪽 위 눈꺼풀
    RIGHT_EYE_LOWER = [373, 374, 380, 381, 382, 362]  # 오른쪽 아래 눈꺼풀
    LEFT_CHEEK = [50, 101, 119, 118]  # 왼쪽 뺨 (눈 아래)
    RIGHT_CHEEK = [280, 330, 348, 347]  # 오른쪽 뺨

    # AU12: Lip Corner Puller - 입꼬리 올림 (미소)
    MOUTH_LEFT_CORNER = [61, 62, 76, 77]  # 왼쪽 입꼬리
    MOUTH_RIGHT_CORNER = [291, 292, 306, 307]  # 오른쪽 입꼬리
    UPPER_LIP_TOP = [0, 267, 269, 270, 409, 37, 39, 40, 185]  # 윗입술 상단
    LOWER_LIP_BOTTOM = [17, 314, 405, 321, 375, 84, 181, 91, 146]  # 아랫입술 하단
    MOUTH_CENTER = [13, 14]  # 입 중앙 (참조점)

    # AU15: Lip Corner Depressor - 입꼏리 내림 (찡그림)
    CHIN = [152, 175, 199, 200, 421, 418]  # 턱 (입꼬리 내림 측정용)
    LOWER_LIP_OUTER = [57, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 287]  # 아랫입술 외곽

    # Nose bridge (reference for normalization) - 얼굴 크기 정규화용
    NOSE_BRIDGE = [6, 168, 197, 195]  # 코 브리지
    NOSE_TIP = [1, 2]  # 코 끝

    # Face contour for size normalization - 얼굴 윤곽
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initialize AU extractor with MediaPipe Face Mesh

        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        print("Loading MediaPipe Face Mesh for AU extraction...")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,  # Support multiple faces
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Load AU definitions from config
        from src.utils.config_loader import get_au_definitions
        self.au_definitions = get_au_definitions()
        self.enabled_aus = list(self.au_definitions.keys())

        print(f"[OK] AU extractor loaded (Face Mesh)")
        print(f"[INFO] Enabled AUs: {', '.join(self.enabled_aus)}")

    def extract_landmarks(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from face region

        Args:
            image: Full input image (BGR)
            bbox: Face bounding box [x1, y1, x2, y2]

        Returns:
            Landmarks array (478, 3) with normalized coordinates, or None if failed
        """
        x1, y1, x2, y2 = bbox

        # Crop face region
        face_img = image[y1:y2, x1:x2]

        if face_img.size == 0:
            return None

        # Convert to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Process with Face Mesh
        results = self.face_mesh.process(face_rgb)

        if not results.multi_face_landmarks:
            return None

        # Get first face landmarks (already matched by bbox)
        landmarks = results.multi_face_landmarks[0]

        # Convert to numpy array with absolute coordinates
        h, w = face_img.shape[:2]
        points = np.array([
            [lm.x * w + x1, lm.y * h + y1, lm.z]
            for lm in landmarks.landmark
        ])

        return points

    def compute_distance(self, landmarks: np.ndarray, idx1: int, idx2: int) -> float:
        """Compute Euclidean distance between two landmark points"""
        return np.linalg.norm(landmarks[idx1, :2] - landmarks[idx2, :2])

    def compute_mean_distance(self, landmarks: np.ndarray, indices1: list, indices2: list) -> float:
        """Compute mean distance between two groups of landmarks"""
        distances = []
        for i in indices1:
            for j in indices2:
                distances.append(self.compute_distance(landmarks, i, j))
        return np.mean(distances)

    def normalize_by_face_size(self, value: float, landmarks: np.ndarray) -> float:
        """Normalize measurement by face size (eye distance)"""
        # Use inter-eye distance as reference (more stable than nose)
        left_eye_center = np.mean([landmarks[i, :2] for i in [33, 133]], axis=0)
        right_eye_center = np.mean([landmarks[i, :2] for i in [263, 362]], axis=0)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        if eye_distance > 0:
            return value / eye_distance
        return 0.0

    def extract_au1_inner_brow_raiser(self, landmarks: np.ndarray) -> float:
        """
        AU1: Inner Brow Raiser - 내측 눈썹 올림
        Measure vertical distance between inner eyebrows and upper eyelids
        """
        # Get inner brow positions (y-coordinate)
        left_brow_y = np.mean([landmarks[i, 1] for i in self.LEFT_INNER_BROW])
        right_brow_y = np.mean([landmarks[i, 1] for i in self.RIGHT_INNER_BROW])

        # Get upper eyelid positions
        left_eyelid_y = np.mean([landmarks[i, 1] for i in self.LEFT_UPPER_EYELID])
        right_eyelid_y = np.mean([landmarks[i, 1] for i in self.RIGHT_UPPER_EYELID])

        # Calculate distance (in image coords, smaller y = higher position)
        left_dist = left_eyelid_y - left_brow_y  # Higher brow = larger value
        right_dist = right_eyelid_y - right_brow_y
        avg_dist = (left_dist + right_dist) / 2.0

        # Normalize by face size
        normalized = self.normalize_by_face_size(avg_dist, landmarks)

        # Scale to 0-1 range
        # Based on observation: neutral ~0.38-0.40, lowered(frown) ~0.33, raised >0.42
        # Higher value = brow is raised (more distance)
        au1_intensity = max(0.0, (normalized - 0.38) * 10.0)
        return min(1.0, au1_intensity)

    def extract_au4_brow_lowerer(self, landmarks: np.ndarray) -> float:
        """
        AU4: Brow Lowerer - 눈썹 찡그림 (눈썹 내림)
        Measure inward and downward movement of eyebrows
        """
        # Measure brow-to-eye distance (should decrease when lowered)
        left_brow_y = np.mean([landmarks[i, 1] for i in self.LEFT_BROW_CENTER])
        right_brow_y = np.mean([landmarks[i, 1] for i in self.RIGHT_BROW_CENTER])

        left_eyelid_y = np.mean([landmarks[i, 1] for i in self.LEFT_UPPER_EYELID])
        right_eyelid_y = np.mean([landmarks[i, 1] for i in self.RIGHT_UPPER_EYELID])

        # Distance between brow and eyelid
        left_dist = left_eyelid_y - left_brow_y
        right_dist = right_eyelid_y - right_brow_y
        avg_dist = (left_dist + right_dist) / 2.0

        # Normalize
        normalized = self.normalize_by_face_size(avg_dist, landmarks)

        # Smaller distance = more AU4 (inverted scale)
        # Based on observation: neutral ~0.40-0.42, frown/angry ~0.35-0.36
        # Lower value = brow is lowered (less distance)
        # Increased sensitivity: higher threshold and steeper scale
        au4_intensity = max(0.0, (0.395 - normalized) * 20.0)
        return min(1.0, au4_intensity)

    def extract_au6_cheek_raiser(self, landmarks: np.ndarray) -> float:
        """
        AU6: Cheek Raiser - 뺨 올림 (눈 찌푸림)
        Measure eye narrowing caused by cheek raising (crow's feet)
        """
        # Measure eye height (vertical opening)
        # Left eye
        left_eye_upper_y = np.mean([landmarks[i, 1] for i in self.LEFT_EYE_UPPER])
        left_eye_lower_y = np.mean([landmarks[i, 1] for i in self.LEFT_EYE_LOWER])
        left_eye_height = abs(left_eye_lower_y - left_eye_upper_y)

        # Right eye
        right_eye_upper_y = np.mean([landmarks[i, 1] for i in self.RIGHT_EYE_UPPER])
        right_eye_lower_y = np.mean([landmarks[i, 1] for i in self.RIGHT_EYE_LOWER])
        right_eye_height = abs(right_eye_lower_y - right_eye_upper_y)

        # Average eye height
        avg_eye_height = (left_eye_height + right_eye_height) / 2.0

        # Also measure cheek elevation
        left_cheek_y = np.mean([landmarks[i, 1] for i in self.LEFT_CHEEK])
        right_cheek_y = np.mean([landmarks[i, 1] for i in self.RIGHT_CHEEK])
        avg_cheek_y = (left_cheek_y + right_cheek_y) / 2.0

        # Combine: smaller eye height + higher cheek = more AU6
        eye_normalized = self.normalize_by_face_size(avg_eye_height, landmarks)

        # Typical neutral eye height: ~0.04-0.05, squinted: <0.03
        au6_intensity = max(0.0, (0.045 - eye_normalized) * 40.0)
        return min(1.0, au6_intensity)

    def extract_au12_lip_corner_puller(self, landmarks: np.ndarray) -> float:
        """
        AU12: Lip Corner Puller - 입꼬리 올림 (미소)
        Measure upward and outward movement of mouth corners
        """
        # Get mouth corner positions
        left_corner_pos = np.mean([landmarks[i, :2] for i in self.MOUTH_LEFT_CORNER], axis=0)
        right_corner_pos = np.mean([landmarks[i, :2] for i in self.MOUTH_RIGHT_CORNER], axis=0)

        # Get upper and lower lip references
        upper_lip_y = np.mean([landmarks[i, 1] for i in self.UPPER_LIP_TOP])
        lower_lip_y = np.mean([landmarks[i, 1] for i in self.LOWER_LIP_BOTTOM])
        lip_height = lower_lip_y - upper_lip_y

        # Calculate mouth corner height relative to lip center
        mouth_center_y = (upper_lip_y + lower_lip_y) / 2.0
        corner_avg_y = (left_corner_pos[1] + right_corner_pos[1]) / 2.0

        # Upward pull: corners should be above mouth center
        corner_lift = mouth_center_y - corner_avg_y  # Positive = lifted

        # Measure mouth width (smile widens mouth)
        mouth_width = np.linalg.norm(left_corner_pos - right_corner_pos)

        # Normalize measurements
        lift_normalized = self.normalize_by_face_size(corner_lift, landmarks)
        width_normalized = self.normalize_by_face_size(mouth_width, landmarks)

        # Combine lift and width
        # Based on observation:
        # - lift: neutral ~0.01, smile ~0.04, sad ~-0.01
        # - width: neutral ~0.7, smile ~0.8, sad ~0.6
        lift_component = max(0.0, (lift_normalized - 0.02) * 25.0)
        width_component = max(0.0, (width_normalized - 0.72) * 8.0)

        au12_intensity = (lift_component * 0.7 + width_component * 0.3)
        return min(1.0, max(0.0, au12_intensity))

    def extract_au15_lip_corner_depressor(self, landmarks: np.ndarray) -> float:
        """
        AU15: Lip Corner Depressor - 입꼬리 내림 (찡그림)
        Measure downward pull of mouth corners
        """
        # Get mouth corner positions
        left_corner_pos = np.mean([landmarks[i, :2] for i in self.MOUTH_LEFT_CORNER], axis=0)
        right_corner_pos = np.mean([landmarks[i, :2] for i in self.MOUTH_RIGHT_CORNER], axis=0)

        # Get mouth references
        upper_lip_y = np.mean([landmarks[i, 1] for i in self.UPPER_LIP_TOP])
        lower_lip_y = np.mean([landmarks[i, 1] for i in self.LOWER_LIP_BOTTOM])
        mouth_center_y = (upper_lip_y + lower_lip_y) / 2.0

        # Corner height
        corner_avg_y = (left_corner_pos[1] + right_corner_pos[1]) / 2.0

        # Downward depression: corners below mouth center
        corner_depression = corner_avg_y - mouth_center_y  # Positive = depressed

        # Normalize
        depression_normalized = self.normalize_by_face_size(corner_depression, landmarks)

        # Based on observation:
        # - neutral: lift ~0.01, width ~0.7
        # - smile: lift ~0.04, width ~0.8
        # - sad: lift ~-0.01, width ~0.6
        # - frown: lift ~0.03, width ~0.7
        # AU15 is essentially the opposite of AU12 (sad vs happy)

        # Use both depression and mouth width narrowing
        mouth_width = np.linalg.norm(left_corner_pos - right_corner_pos)
        width_normalized = self.normalize_by_face_size(mouth_width, landmarks)

        # Sad expression: narrow mouth (width < 0.68) and low/negative lift
        width_component = max(0.0, (0.70 - width_normalized) * 12.0)

        # For depression: use absolute value comparison with smile lift
        # If lift is negative (below center), it's depression
        lift_value = mouth_center_y - corner_avg_y  # This is the lift (AU12 metric)
        lift_norm = self.normalize_by_face_size(lift_value, landmarks)

        # Depression: negative lift or very low lift
        # Neutral is around 0.01, so anything below that is depression
        depression_component = max(0.0, (0.005 - lift_norm) * 20.0)

        au15_intensity = (depression_component * 0.5 + width_component * 0.5)
        return min(1.0, max(0.0, au15_intensity))

    def extract_aus(self, image: np.ndarray, bbox: np.ndarray, face_id: int) -> Optional[Dict[str, float]]:
        """
        Extract all enabled AUs from a face (dynamically loaded from config)

        Args:
            image: Input image (BGR)
            bbox: Face bounding box [x1, y1, x2, y2]
            face_id: Face ID for baseline tracking

        Returns:
            Dictionary with AU measurements or None if extraction failed
            Only returns AUs that are enabled in config.yaml
        """
        landmarks = self.extract_landmarks(image, bbox)

        if landmarks is None:
            return None

        # AU extraction method mapping
        au_method_map = {
            'au1': self.extract_au1_inner_brow_raiser,
            'au4': self.extract_au4_brow_lowerer,
            'au6': self.extract_au6_cheek_raiser,
            'au12': self.extract_au12_lip_corner_puller,
            'au15': self.extract_au15_lip_corner_depressor
        }

        # Dynamically extract only enabled AUs
        aus = {}
        for au_id in self.enabled_aus:
            if au_id in au_method_map:
                aus[au_id] = au_method_map[au_id](landmarks)
            else:
                # AU is enabled in config but not implemented yet
                print(f"[WARNING] AU {au_id} is enabled but not implemented")
                aus[au_id] = 0.0

        return aus

    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# Test
if __name__ == "__main__":
    print("=" * 70)
    print(" " * 25 + "AU Extractor Test")
    print("=" * 70)

    extractor = AUExtractor()

    print("\nAU Extractor initialized")
    print("Target AUs: AU1, AU4, AU6, AU12, AU15")
    print("\nPress 'q' to quit\n")

    cap = cv2.VideoCapture(0)

    # Simple face detector for testing
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face with Haar Cascade (simple test)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            bbox = np.array([x, y, x + w, y + h])

            # Extract AUs
            aus = extractor.extract_aus(frame, bbox, face_id=0)

            if aus:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display AU values
                y_offset = 30
                for au_name, au_value in aus.items():
                    text = f"{au_name.upper()}: {au_value:.3f}"
                    cv2.putText(frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 30

        cv2.imshow('AU Extractor Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[OK] AU extractor test completed")
