"""
Landmark Viewer
View and export images from saved landmark JSON files
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional
from src.utils.landmark_storage import LandmarkVisualizer


class LandmarkViewer:
    """
    View saved landmark data as images
    Navigate through frames and export images
    """

    def __init__(self):
        self.visualizer = LandmarkVisualizer()
        self.current_session = None
        self.current_frame_idx = 0

    def load_session(self, json_file: str):
        """Load landmark JSON file"""
        self.current_session = self.visualizer.load_json(json_file)
        self.current_frame_idx = 0

        print(f"\n[Loaded] {json_file}")
        print(f"  Session ID: {self.current_session['session_id']}")
        print(f"  Total frames: {self.current_session['total_frames']}")
        print(f"  Start time: {self.current_session['start_time']}")

    def render_current_frame(self, canvas_size=(600, 600)) -> np.ndarray:
        """Render current frame"""
        if self.current_session is None:
            return np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

        frames = self.current_session['frames']
        if self.current_frame_idx >= len(frames):
            self.current_frame_idx = len(frames) - 1

        frame_data = frames[self.current_frame_idx]

        # Render landmark visualization
        image = self.visualizer.render_frame(frame_data, canvas_size)

        # Add frame info
        h, w = image.shape[:2]
        frame_text = f"Frame: {self.current_frame_idx + 1} / {len(frames)}"
        cv2.putText(image, frame_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

    def next_frame(self):
        """Move to next frame"""
        if self.current_session is None:
            return

        total_frames = len(self.current_session['frames'])
        self.current_frame_idx = min(self.current_frame_idx + 1, total_frames - 1)

    def prev_frame(self):
        """Move to previous frame"""
        self.current_frame_idx = max(self.current_frame_idx - 1, 0)

    def export_frame(self, output_path: Optional[str] = None):
        """Export current frame as image"""
        if self.current_session is None:
            print("[Error] No session loaded")
            return

        image = self.render_current_frame()

        if output_path is None:
            session_id = self.current_session['session_id']
            output_path = f"landmark_frame_{session_id}_{self.current_frame_idx:04d}.png"

        cv2.imwrite(output_path, image)
        print(f"[Saved] Frame exported to: {output_path}")

    def export_all_frames(self, output_dir: str = "exported_frames"):
        """Export all frames as images"""
        if self.current_session is None:
            print("[Error] No session loaded")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        session_id = self.current_session['session_id']
        total_frames = len(self.current_session['frames'])

        print(f"\n[Exporting] {total_frames} frames...")

        for i in range(total_frames):
            self.current_frame_idx = i
            image = self.render_current_frame()

            filename = output_path / f"frame_{session_id}_{i:04d}.png"
            cv2.imwrite(str(filename), image)

            if (i + 1) % 10 == 0:
                print(f"  Exported {i + 1}/{total_frames} frames...")

        print(f"[Done] All frames exported to: {output_dir}/")

    def run_interactive(self, json_file: str):
        """Run interactive viewer"""
        print("=" * 70)
        print(" " * 20 + "HEAL Landmark Viewer")
        print("=" * 70)

        # Load session
        self.load_session(json_file)

        print("\nControls:")
        print("  Left Arrow  - Previous frame")
        print("  Right Arrow - Next frame")
        print("  S - Save current frame")
        print("  A - Export all frames")
        print("  Q - Quit\n")

        window_name = "HEAL Landmark Viewer"
        cv2.namedWindow(window_name)

        while True:
            # Render frame
            image = self.render_current_frame()
            cv2.imshow(window_name, image)

            # Handle keyboard input - use waitKeyEx for arrow keys on Windows
            key = cv2.waitKeyEx(30)

            # Regular ASCII keys
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                self.export_frame()
            elif key == ord('a') or key == ord('A'):
                self.export_all_frames()
            # Arrow keys (special key codes)
            elif key == 2424832 or key == 2621440:  # Left arrow or  Down arrow (Windows)
                self.prev_frame()
            elif key == 2555904 or key == 2490368:  # Right arrow or Up arrow (Windows)
                self.next_frame()
            # Debug: print key code if not recognized
            elif key != -1 and key != 255:
                print(f"Key pressed: {key}")

        cv2.destroyAllWindows()


# Main
if __name__ == "__main__":
    import sys

    viewer = LandmarkViewer()

    # Check for command line argument
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Find latest recording
        recordings_dir = Path("data/recordings")
        json_files = list(recordings_dir.glob("landmarks_*.json"))

        if not json_files:
            print("[Error] No landmark JSON files found in data/recordings/")
            print("        Record data first using EmotionAnalyzer (press 'r' to record)")
            sys.exit(1)

        # Use latest file
        json_file = str(sorted(json_files)[-1])
        print(f"Using latest recording: {json_file}")

    # Run interactive viewer
    viewer.run_interactive(json_file)

    print("\n[OK] Viewer closed")
