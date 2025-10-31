"""
Data Labeling Tool
Label recorded landmark data with emotion labels for ML training
"""

import cv2
import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from src.utils.landmark_storage import LandmarkVisualizer


class DataLabelingTool:
    """
    Interactive tool for labeling emotion data
    Loads recorded sessions and allows frame-by-frame labeling
    """

    def __init__(self, data_dir: str = "data/recordings"):
        """
        Initialize data labeling tool

        Args:
            data_dir: Directory containing recorded JSON files
        """
        self.data_dir = Path(data_dir)
        self.visualizer = LandmarkVisualizer()

        # Emotion labels
        self.emotions = ["Neutral", "Happy", "Sad"]
        self.emotion_keys = {
            ord('0'): "Neutral",
            ord('1'): "Happy",
            ord('2'): "Sad"
        }

        # Current session data
        self.current_session = None
        self.current_frame_idx = 0
        self.labels = {}  # {frame_idx: emotion_label}

        # Colors for visualization
        self.emotion_colors = {
            "Neutral": (200, 200, 200),
            "Happy": (0, 255, 0),
            "Sad": (255, 0, 0),
            None: (255, 255, 255)  # Unlabeled
        }

    def list_sessions(self) -> List[str]:
        """List all recorded sessions"""
        json_files = list(self.data_dir.glob("landmarks_*.json"))
        return [f.stem for f in sorted(json_files)]

    def load_session(self, session_file: str):
        """Load a recorded session"""
        filepath = self.data_dir / f"{session_file}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Session file not found: {filepath}")

        self.current_session = self.visualizer.load_json(str(filepath))
        self.current_frame_idx = 0
        self.labels = {}

        print(f"\n[Loaded] Session: {session_file}")
        print(f"  Total frames: {self.current_session['total_frames']}")
        print(f"  Start time: {self.current_session['start_time']}")

    def render_current_frame(self, canvas_size=(600, 600)) -> np.ndarray:
        """Render current frame with labeling UI"""
        if self.current_session is None:
            return np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

        frames = self.current_session['frames']
        if self.current_frame_idx >= len(frames):
            self.current_frame_idx = len(frames) - 1

        frame_data = frames[self.current_frame_idx]

        # Render landmark visualization
        image = self.visualizer.render_frame(frame_data, canvas_size)

        # Get current label
        current_label = self.labels.get(self.current_frame_idx, None)

        # Draw labeling UI
        self._draw_ui(image, current_label, self.current_frame_idx, len(frames))

        return image

    def _draw_ui(self, image: np.ndarray, current_label: Optional[str], frame_idx: int, total_frames: int):
        """Draw labeling UI overlay"""
        h, w = image.shape[:2]

        # Semi-transparent overlay at bottom
        overlay = image.copy()
        cv2.rectangle(overlay, (0, h - 150), (w, h), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Frame info
        frame_text = f"Frame: {frame_idx + 1} / {total_frames}"
        cv2.putText(image, frame_text, (10, h - 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Current label
        label_text = f"Label: {current_label if current_label else 'UNLABELED'}"
        label_color = self.emotion_colors.get(current_label, (255, 255, 255))
        cv2.putText(image, label_text, (10, h - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

        # Instructions
        instructions = [
            "0: Neutral  |  1: Happy  |  2: Sad",
            "Left/Right: Navigate  |  S: Save  |  Q: Quit"
        ]

        y_offset = h - 55
        for instruction in instructions:
            cv2.putText(image, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        # Progress bar
        progress = (frame_idx + 1) / total_frames
        bar_width = w - 20
        bar_height = 10
        bar_x = 10
        bar_y = h - 160

        # Background
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)

        # Progress
        filled_width = int(bar_width * progress)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height),
                     (0, 255, 0), -1)

        # Labeled frames indicator
        labeled_count = len(self.labels)
        label_percent = (labeled_count / total_frames * 100) if total_frames > 0 else 0
        label_stat = f"Labeled: {labeled_count}/{total_frames} ({label_percent:.1f}%)"
        cv2.putText(image, label_stat, (w - 250, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def label_frame(self, emotion: str):
        """Label current frame with emotion"""
        if emotion not in self.emotions:
            print(f"[Warning] Unknown emotion: {emotion}")
            return

        self.labels[self.current_frame_idx] = emotion
        print(f"[Labeled] Frame {self.current_frame_idx}: {emotion}")

        # Auto-advance to next unlabeled frame
        self.next_unlabeled_frame()

    def next_frame(self):
        """Move to next frame"""
        if self.current_session is None:
            return

        total_frames = len(self.current_session['frames'])
        self.current_frame_idx = min(self.current_frame_idx + 1, total_frames - 1)

    def prev_frame(self):
        """Move to previous frame"""
        self.current_frame_idx = max(self.current_frame_idx - 1, 0)

    def next_unlabeled_frame(self):
        """Jump to next unlabeled frame"""
        if self.current_session is None:
            return

        total_frames = len(self.current_session['frames'])

        # Search forward
        for i in range(self.current_frame_idx + 1, total_frames):
            if i not in self.labels:
                self.current_frame_idx = i
                return

        # If no unlabeled frames ahead, wrap around
        for i in range(0, self.current_frame_idx):
            if i not in self.labels:
                self.current_frame_idx = i
                return

        # All frames labeled
        print("[Info] All frames labeled!")

    def save_labels(self, output_file: Optional[str] = None):
        """Save labels to CSV file"""
        if self.current_session is None:
            print("[Error] No session loaded")
            return

        if output_file is None:
            session_id = self.current_session['session_id']
            output_file = self.data_dir / f"labels_{session_id}.csv"

        # Collect labeled data
        labeled_data = []

        for frame_idx, emotion in self.labels.items():
            frame_data = self.current_session['frames'][frame_idx]

            # Extract AU values
            aus = frame_data['aus']

            labeled_data.append({
                'frame_idx': frame_idx,
                'timestamp': frame_data['timestamp'],
                'face_id': frame_data['face_id'],
                'au1': aus['au1'],
                'au4': aus['au4'],
                'au6': aus['au6'],
                'au12': aus['au12'],
                'au15': aus['au15'],
                'emotion_label': emotion
            })

        # Save to CSV
        with open(output_file, 'w', newline='') as f:
            if labeled_data:
                fieldnames = list(labeled_data[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(labeled_data)

        print(f"\n[Saved] Labels saved to: {output_file}")
        print(f"  Total labeled frames: {len(labeled_data)}")

    def run_interactive(self, session_file: str):
        """Run interactive labeling session"""
        print("=" * 70)
        print(" " * 20 + "Data Labeling Tool")
        print("=" * 70)

        # Load session
        self.load_session(session_file)

        print("\nControls:")
        print("  0 - Label as Neutral")
        print("  1 - Label as Happy")
        print("  2 - Label as Sad")
        print("  Left Arrow - Previous frame")
        print("  Right Arrow - Next frame")
        print("  S - Save labels")
        print("  Q - Quit\n")

        window_name = "Data Labeling Tool"
        cv2.namedWindow(window_name)

        while True:
            # Render frame
            image = self.render_current_frame()
            cv2.imshow(window_name, image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_labels()
            elif key in self.emotion_keys:
                emotion = self.emotion_keys[key]
                self.label_frame(emotion)
            elif key == 81 or key == 2:  # Left arrow
                self.prev_frame()
            elif key == 83 or key == 3:  # Right arrow
                self.next_frame()

        cv2.destroyAllWindows()

        # Ask to save before quitting
        if self.labels:
            print("\nDo you want to save labels before quitting? (y/n)")
            # Auto-save for now
            self.save_labels()


# Test
if __name__ == "__main__":
    tool = DataLabelingTool()

    # List available sessions
    sessions = tool.list_sessions()

    if not sessions:
        print("[Error] No recorded sessions found in data/recordings/")
        print("        Record a session first using EmotionAnalyzer (press 'r' to record)")
    else:
        print("\nAvailable sessions:")
        for i, session in enumerate(sessions):
            print(f"  {i}: {session}")

        print(f"\nUsing latest session: {sessions[-1]}")

        # Run interactive labeling
        tool.run_interactive(sessions[-1])

        print("\n[OK] Labeling session completed")
