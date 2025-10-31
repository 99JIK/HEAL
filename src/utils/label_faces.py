"""
Face Image Labeling Tool
Interactive tool for labeling face images with emotions for CNN training
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
import json
from typing import Dict, Optional
import argparse


class FaceLabelingTool:
    """
    Interactive tool for labeling face images with emotions

    Usage:
        1. Load face images from a directory
        2. View each face and press keys to label:
           - 'n' or '0': Neutral
           - 'h' or '1': Happy
           - 's' or '2': Sad
           - 'u': Undo (go back)
           - 'q': Quit and save
        3. Labeled faces are copied to emotion-specific directories
    """

    def __init__(self, input_dir: str, output_dir: str = "data/labeled_faces"):
        """
        Initialize labeling tool

        Args:
            input_dir: Directory containing face images to label
            output_dir: Output directory for labeled faces
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Emotion labels
        self.emotions = ["Neutral", "Happy", "Sad"]
        self.emotion_keys = {
            'n': 0, '0': 0,  # Neutral
            'h': 1, '1': 1,  # Happy
            's': 2, '2': 2   # Sad
        }

        # Emotion colors (BGR)
        self.emotion_colors = {
            0: (200, 200, 200),  # Neutral: Gray
            1: (0, 255, 0),      # Happy: Green
            2: (255, 0, 0)       # Sad: Blue
        }

        # Load face images
        self.image_files = self._load_images()
        self.current_idx = 0
        self.labels = {}  # {filename: emotion_idx}

        # Create output directories
        self._create_output_dirs()

        # Load existing labels if available
        self._load_existing_labels()

        print(f"[OK] Loaded {len(self.image_files)} face images from {input_dir}")
        print(f"[INFO] Output directory: {output_dir}")

    def _load_images(self):
        """Load all image files from input directory"""
        image_files = []

        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(self.input_dir.glob(ext)))
            image_files.extend(list(self.input_dir.glob(ext.upper())))

        # Sort by filename
        image_files.sort()

        return image_files

    def _create_output_dirs(self):
        """Create output directories for each emotion"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for emotion in self.emotions:
            emotion_dir = self.output_dir / emotion
            emotion_dir.mkdir(exist_ok=True)

    def _load_existing_labels(self):
        """Load existing labels from previous session"""
        label_file = self.output_dir / "labels.json"

        if label_file.exists():
            with open(label_file, 'r') as f:
                self.labels = json.load(f)
            print(f"[INFO] Loaded {len(self.labels)} existing labels")

    def _save_labels(self):
        """Save labels to JSON file"""
        label_file = self.output_dir / "labels.json"

        with open(label_file, 'w') as f:
            json.dump(self.labels, f, indent=2)

        print(f"[Saved] Labels saved to {label_file}")

    def _copy_labeled_image(self, image_file: Path, emotion_idx: int):
        """Copy labeled image to emotion-specific directory"""
        emotion = self.emotions[emotion_idx]
        emotion_dir = self.output_dir / emotion

        # Create unique filename if already exists
        dst_path = emotion_dir / image_file.name
        counter = 1
        while dst_path.exists():
            stem = image_file.stem
            suffix = image_file.suffix
            dst_path = emotion_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        shutil.copy2(image_file, dst_path)

    def _draw_instructions(self, frame: np.ndarray, emotion_idx: Optional[int] = None):
        """Draw instructions and current emotion on frame"""
        h, w = frame.shape[:2]

        # Add padding for instructions
        padding = 150
        display = np.zeros((h + padding, w, 3), dtype=np.uint8)
        display[padding:, :] = frame

        # Draw instructions
        y = 30
        cv2.putText(display, "Face Image Labeling Tool", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y += 35
        cv2.putText(display, f"Image {self.current_idx + 1}/{len(self.image_files)}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y += 30
        cv2.putText(display, "Keys: [N]eutral | [H]appy | [S]ad | [U]ndo | [Q]uit",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Draw current emotion label if set
        if emotion_idx is not None:
            emotion = self.emotions[emotion_idx]
            color = self.emotion_colors[emotion_idx]

            cv2.putText(display, f"Label: {emotion}", (10, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return display

    def run(self):
        """Run interactive labeling session"""
        if len(self.image_files) == 0:
            print("[ERROR] No face images found!")
            print(f"Please add face images to: {self.input_dir}")
            return

        print("\n" + "=" * 70)
        print(" " * 20 + "Starting Labeling Session")
        print("=" * 70)
        print("\nInstructions:")
        print("  [N] or [0] - Label as Neutral")
        print("  [H] or [1] - Label as Happy")
        print("  [S] or [2] - Label as Sad")
        print("  [U] - Undo (go back to previous image)")
        print("  [Q] - Quit and save progress")
        print("\n" + "=" * 70)

        while self.current_idx < len(self.image_files):
            image_file = self.image_files[self.current_idx]
            filename = image_file.name

            # Load image
            image = cv2.imread(str(image_file))

            if image is None:
                print(f"[Warning] Failed to load: {filename}")
                self.current_idx += 1
                continue

            # Resize for display if too large
            max_size = 800
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h))

            # Check if already labeled
            existing_label = self.labels.get(filename)

            # Draw display
            display = self._draw_instructions(image, existing_label)

            cv2.imshow("Face Labeling", display)

            # Wait for key press
            key = cv2.waitKey(0) & 0xFF

            # Handle key press
            if key == ord('q'):
                # Quit
                print("\n[INFO] Quitting...")
                break

            elif key == ord('u'):
                # Undo - go back
                if self.current_idx > 0:
                    self.current_idx -= 1
                    print(f"[Undo] Back to image {self.current_idx + 1}")
                continue

            elif chr(key).lower() in self.emotion_keys:
                # Label emotion
                emotion_idx = self.emotion_keys[chr(key).lower()]
                emotion = self.emotions[emotion_idx]

                # Save label
                self.labels[filename] = emotion_idx

                # Copy to emotion directory
                self._copy_labeled_image(image_file, emotion_idx)

                print(f"[{self.current_idx + 1}/{len(self.image_files)}] "
                      f"{filename} -> {emotion}")

                # Move to next image
                self.current_idx += 1

        cv2.destroyAllWindows()

        # Save labels
        self._save_labels()

        # Print summary
        print("\n" + "=" * 70)
        print(" " * 20 + "Labeling Summary")
        print("=" * 70)

        label_counts = {i: 0 for i in range(len(self.emotions))}
        for label_idx in self.labels.values():
            label_counts[label_idx] += 1

        print(f"\nTotal labeled: {len(self.labels)}/{len(self.image_files)}")
        for i, emotion in enumerate(self.emotions):
            count = label_counts[i]
            percentage = 100 * count / len(self.labels) if len(self.labels) > 0 else 0
            print(f"  {emotion}: {count} ({percentage:.1f}%)")

        print(f"\nLabeled faces saved to: {self.output_dir}")
        print("=" * 70)


def extract_faces_from_recording(recording_file: str, output_dir: str = "data/faces_to_label"):
    """
    Extract face images from recording for labeling

    Args:
        recording_file: Path to video recording
        output_dir: Output directory for extracted faces
    """
    print(f"\n[Extracting Faces] from {recording_file}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Import face detector
    from src.models.face_detector import FaceDetector

    detector = FaceDetector()
    cap = cv2.VideoCapture(recording_file)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {recording_file}")
        return

    frame_count = 0
    face_count = 0
    sample_rate = 10  # Extract every 10th frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Sample frames
        if frame_count % sample_rate != 0:
            continue

        # Detect faces
        face_bboxes = detector.detect(frame)

        # Save each face
        for bbox in face_bboxes:
            x1, y1, x2, y2 = bbox
            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            # Save face image
            face_filename = output_path / f"face_{face_count:04d}.jpg"
            cv2.imwrite(str(face_filename), face_img)
            face_count += 1

    cap.release()

    print(f"[OK] Extracted {face_count} faces to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label face images for CNN training")

    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/faces_to_label",
        help="Directory containing face images to label"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/labeled_faces",
        help="Output directory for labeled faces"
    )

    parser.add_argument(
        "--extract",
        type=str,
        default=None,
        help="Extract faces from video file before labeling"
    )

    args = parser.parse_args()

    # Extract faces from video if specified
    if args.extract:
        extract_faces_from_recording(args.extract, args.input_dir)

    # Run labeling tool
    tool = FaceLabelingTool(args.input_dir, args.output_dir)
    tool.run()
