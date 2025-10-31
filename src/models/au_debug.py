"""
AU Extractor Debug Tool
Real-time visualization and debugging for AU extraction
"""

import cv2
import numpy as np
from src.models.au_extractor import AUExtractor


def main():
    print("=" * 70)
    print(" " * 20 + "AU Extractor Debug Tool")
    print("=" * 70)
    print("\nThis tool shows:")
    print("  - Raw normalized distance values")
    print("  - Final AU intensity values")
    print("  - Key facial landmarks")
    print("\nPress 'q' to quit\n")

    extractor = AUExtractor()
    cap = cv2.VideoCapture(0)

    # Simple face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            bbox = np.array([x, y, x + w, y + h])

            # Extract landmarks
            landmarks = extractor.extract_landmarks(frame, bbox)

            if landmarks is not None:
                # Draw face box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # ===== AU1: Inner Brow Raiser =====
                left_brow_y = np.mean([landmarks[i, 1] for i in extractor.LEFT_INNER_BROW])
                right_brow_y = np.mean([landmarks[i, 1] for i in extractor.RIGHT_INNER_BROW])
                left_eyelid_y = np.mean([landmarks[i, 1] for i in extractor.LEFT_UPPER_EYELID])
                right_eyelid_y = np.mean([landmarks[i, 1] for i in extractor.RIGHT_UPPER_EYELID])

                au1_dist = ((left_eyelid_y - left_brow_y) + (right_eyelid_y - right_brow_y)) / 2.0
                au1_norm = extractor.normalize_by_face_size(au1_dist, landmarks)
                au1_value = extractor.extract_au1_inner_brow_raiser(landmarks)

                # ===== AU4: Brow Lowerer =====
                left_brow_center_y = np.mean([landmarks[i, 1] for i in extractor.LEFT_BROW_CENTER])
                right_brow_center_y = np.mean([landmarks[i, 1] for i in extractor.RIGHT_BROW_CENTER])

                au4_dist = ((left_eyelid_y - left_brow_center_y) + (right_eyelid_y - right_brow_center_y)) / 2.0
                au4_norm = extractor.normalize_by_face_size(au4_dist, landmarks)
                au4_value = extractor.extract_au4_brow_lowerer(landmarks)

                # ===== AU6: Cheek Raiser =====
                left_eye_upper_y = np.mean([landmarks[i, 1] for i in extractor.LEFT_EYE_UPPER])
                left_eye_lower_y = np.mean([landmarks[i, 1] for i in extractor.LEFT_EYE_LOWER])
                right_eye_upper_y = np.mean([landmarks[i, 1] for i in extractor.RIGHT_EYE_UPPER])
                right_eye_lower_y = np.mean([landmarks[i, 1] for i in extractor.RIGHT_EYE_LOWER])

                au6_eye_height = ((left_eye_lower_y - left_eye_upper_y) + (right_eye_lower_y - right_eye_upper_y)) / 2.0
                au6_norm = extractor.normalize_by_face_size(au6_eye_height, landmarks)
                au6_value = extractor.extract_au6_cheek_raiser(landmarks)

                # ===== AU12: Lip Corner Puller =====
                left_corner_pos = np.mean([landmarks[i, :2] for i in extractor.MOUTH_LEFT_CORNER], axis=0)
                right_corner_pos = np.mean([landmarks[i, :2] for i in extractor.MOUTH_RIGHT_CORNER], axis=0)
                upper_lip_y = np.mean([landmarks[i, 1] for i in extractor.UPPER_LIP_TOP])
                lower_lip_y = np.mean([landmarks[i, 1] for i in extractor.LOWER_LIP_BOTTOM])
                mouth_center_y = (upper_lip_y + lower_lip_y) / 2.0
                corner_avg_y = (left_corner_pos[1] + right_corner_pos[1]) / 2.0

                au12_lift = mouth_center_y - corner_avg_y
                au12_lift_norm = extractor.normalize_by_face_size(au12_lift, landmarks)

                mouth_width = np.linalg.norm(left_corner_pos - right_corner_pos)
                au12_width_norm = extractor.normalize_by_face_size(mouth_width, landmarks)
                au12_value = extractor.extract_au12_lip_corner_puller(landmarks)

                # ===== AU15: Lip Corner Depressor =====
                au15_depression = corner_avg_y - mouth_center_y
                au15_norm = extractor.normalize_by_face_size(au15_depression, landmarks)
                au15_value = extractor.extract_au15_lip_corner_depressor(landmarks)

                # Display debug info
                y_offset = 30
                font = cv2.FONT_HERSHEY_SIMPLEX

                # AU1
                text = f"AU1 (InnerBrowRaise): {au1_value:.3f} | norm={au1_norm:.4f}"
                cv2.putText(frame, text, (10, y_offset), font, 0.5, (0, 255, 255), 1)
                y_offset += 25

                # AU4
                text = f"AU4 (BrowLower): {au4_value:.3f} | norm={au4_norm:.4f}"
                cv2.putText(frame, text, (10, y_offset), font, 0.5, (0, 255, 255), 1)
                y_offset += 25

                # AU6
                text = f"AU6 (CheekRaise): {au6_value:.3f} | norm={au6_norm:.4f}"
                cv2.putText(frame, text, (10, y_offset), font, 0.5, (0, 255, 255), 1)
                y_offset += 25

                # AU12
                text = f"AU12 (LipPull): {au12_value:.3f} | lift={au12_lift_norm:.4f} width={au12_width_norm:.4f}"
                cv2.putText(frame, text, (10, y_offset), font, 0.5, (0, 255, 255), 1)
                y_offset += 25

                # AU15
                text = f"AU15 (LipDepress): {au15_value:.3f} | norm={au15_norm:.4f}"
                cv2.putText(frame, text, (10, y_offset), font, 0.5, (0, 255, 255), 1)
                y_offset += 25

                # Draw key landmarks for visualization
                # Brows (AU1, AU4)
                for i in extractor.LEFT_INNER_BROW + extractor.RIGHT_INNER_BROW:
                    pt = (int(landmarks[i, 0]), int(landmarks[i, 1]))
                    cv2.circle(frame, pt, 2, (255, 0, 0), -1)  # Blue for inner brow

                for i in extractor.LEFT_BROW_CENTER + extractor.RIGHT_BROW_CENTER:
                    pt = (int(landmarks[i, 0]), int(landmarks[i, 1]))
                    cv2.circle(frame, pt, 2, (255, 100, 0), -1)  # Cyan for brow center

                # Eyes (AU6)
                for i in extractor.LEFT_EYE_UPPER + extractor.RIGHT_EYE_UPPER:
                    pt = (int(landmarks[i, 0]), int(landmarks[i, 1]))
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)  # Green for upper eye

                for i in extractor.LEFT_EYE_LOWER + extractor.RIGHT_EYE_LOWER:
                    pt = (int(landmarks[i, 0]), int(landmarks[i, 1]))
                    cv2.circle(frame, pt, 2, (0, 200, 0), -1)  # Dark green for lower eye

                # Mouth corners (AU12, AU15)
                for i in extractor.MOUTH_LEFT_CORNER + extractor.MOUTH_RIGHT_CORNER:
                    pt = (int(landmarks[i, 0]), int(landmarks[i, 1]))
                    cv2.circle(frame, pt, 3, (0, 0, 255), -1)  # Red for mouth corners

        cv2.imshow('AU Debug Tool', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[OK] Debug session completed")


if __name__ == "__main__":
    main()
