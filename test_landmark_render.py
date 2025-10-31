"""
Quick test to render a frame from landmarks JSON
"""

import cv2
from src.utils.landmark_storage import LandmarkVisualizer

# Load the JSON file
visualizer = LandmarkVisualizer()
data = visualizer.load_json("data/recordings/landmarks_20251031_181016.json")

print(f"Session ID: {data['session_id']}")
print(f"Total frames: {data['total_frames']}")

# Render the first frame
if data['total_frames'] > 0:
    frame_data = data['frames'][0]

    print(f"\nFrame 0:")
    print(f"  Timestamp: {frame_data['timestamp']}")
    print(f"  Face ID: {frame_data['face_id']}")
    print(f"  Emotion: {frame_data['emotion']} ({frame_data['emotion_intensity']:.0%})")

    # Render to image
    image = visualizer.render_frame(frame_data, canvas_size=(600, 600))

    # Save
    output_file = "rendered_landmark_frame.png"
    cv2.imwrite(output_file, image)
    print(f"\n[Saved] Image saved to: {output_file}")

    # Show
    cv2.imshow("Rendered Landmark", image)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\n[OK] Test completed")
