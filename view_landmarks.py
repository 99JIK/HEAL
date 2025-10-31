"""
Landmark Viewer - Main Application
Interactive viewer for saved landmark JSON files
"""

import cv2
import sys
from pathlib import Path
from src.utils.landmark_viewer import LandmarkViewer


def main():
    print("=" * 70)
    print(" " * 20 + "HEAL Landmark Viewer")
    print("=" * 70)

    # Find all recording files
    recordings_dir = Path("data/recordings")

    if not recordings_dir.exists():
        print(f"\n[Error] Recording directory not found: {recordings_dir}")
        print("        Create recordings first using EmotionAnalyzer (press 'r' to record)")
        return

    json_files = sorted(list(recordings_dir.glob("landmarks_*.json")))

    if not json_files:
        print(f"\n[Error] No landmark JSON files found in {recordings_dir}/")
        print("        Record data first using EmotionAnalyzer (press 'r' to record)")
        return

    # Display menu
    print(f"\nFound {len(json_files)} recording(s):\n")

    for i, file in enumerate(json_files):
        # Get file info
        file_size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  [{i+1}] {file.name}")
        print(f"      Size: {file_size_mb:.2f} MB")

    # User selection
    print(f"\n{'─' * 70}")

    if len(sys.argv) > 1:
        # File path provided as argument
        selected_file = sys.argv[1]
        print(f"Loading: {selected_file}")
    else:
        # Interactive selection
        try:
            choice = input(f"\nSelect recording (1-{len(json_files)}, or 'q' to quit): ").strip()

            if choice.lower() == 'q':
                print("Cancelled.")
                return

            choice_idx = int(choice) - 1

            if choice_idx < 0 or choice_idx >= len(json_files):
                print("[Error] Invalid selection")
                return

            selected_file = str(json_files[choice_idx])

        except (ValueError, KeyboardInterrupt):
            print("\n[Cancelled]")
            return

    # Launch viewer
    print(f"\n{'=' * 70}")
    print("Controls:")
    print("  ← → (Arrow Keys) - Navigate frames")
    print("  S - Save current frame as image")
    print("  A - Export ALL frames as images")
    print("  Q - Quit")
    print("=" * 70)

    viewer = LandmarkViewer()

    try:
        viewer.run_interactive(selected_file)
    except KeyboardInterrupt:
        print("\n\n[Interrupted]")
    except Exception as e:
        print(f"\n[Error] {e}")

    print("\nViewer closed")


if __name__ == "__main__":
    main()
