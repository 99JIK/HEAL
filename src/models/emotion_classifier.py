"""
Emotion Classifier based on Action Units (AU)
Dynamically loads emotion definitions from config.yaml
"""

import numpy as np
from typing import Dict, Tuple, Optional
from src.utils.config_loader import get_emotion_definitions


class EmotionClassifier:
    """
    Config-driven emotion classifier from Action Unit measurements

    Emotions and AU rules are loaded dynamically from config.yaml
    Supports adding new emotions without code changes
    """

    def __init__(self, smoothing_window: int = 5):
        """
        Initialize emotion classifier with config-based definitions

        Args:
            smoothing_window: Number of recent predictions to smooth
        """
        self.smoothing_window = smoothing_window

        # Load emotion definitions from config
        self.emotion_definitions = get_emotion_definitions()

        # Emotion history per face ID
        self.emotion_history = {}  # {face_id: [recent_emotions]}

        # Extract emotion labels and colors
        self.emotions = list(self.emotion_definitions.keys())
        self.emotion_colors = {
            emotion: tuple(data['color'])
            for emotion, data in self.emotion_definitions.items()
        }

        print(f"[INFO] Emotion Classifier loaded with emotions: {', '.join(self.emotions)}")

    def classify(self, aus: Dict[str, float], face_id: Optional[int] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify emotion from AU measurements using config-based rules

        Args:
            aus: Dictionary with AU measurements {'au1': float, 'au4': float, ...}
            face_id: Optional face ID for temporal smoothing

        Returns:
            Tuple of (emotion_label, intensity, emotion_scores)
            - emotion_label: Dominant emotion label
            - intensity: 0.0-1.0 (strength of the emotion)
            - emotion_scores: Dict with all emotion scores
        """
        emotion_scores = {}

        # Calculate emotion intensities based on config rules
        for emotion, definition in self.emotion_definitions.items():
            au_rules = definition.get('au_rules', {})

            if emotion == "Neutral":
                # Neutral is special - calculated after other emotions
                continue

            # Get threshold
            threshold = au_rules.get('threshold', 0.3)

            # Calculate weighted sum of primary AUs
            intensity = 0.0
            primary_aus = au_rules.get('primary', {})
            for au_id, weight in primary_aus.items():
                au_value = aus.get(au_id, 0.0)
                intensity += au_value * weight

            # Add secondary AUs if any are activated
            secondary_aus = au_rules.get('secondary', {})
            if secondary_aus:
                secondary_activated = any(aus.get(au_id, 0.0) > 0.2 for au_id in secondary_aus.keys())
                if secondary_activated:
                    for au_id, weight in secondary_aus.items():
                        au_value = aus.get(au_id, 0.0)
                        intensity += au_value * weight

            # Clip to [0, 1]
            emotion_scores[emotion] = min(1.0, max(0.0, intensity))

        # Calculate Neutral (inversely proportional to other emotions)
        if "Neutral" in self.emotion_definitions:
            neutral_rules = self.emotion_definitions["Neutral"].get('au_rules', {})
            neutral_threshold = neutral_rules.get('threshold', 0.15)

            # Sum all non-Neutral activations
            total_activation = sum(score for emo, score in emotion_scores.items() if emo != "Neutral")

            # If all AUs below threshold, it's Neutral
            if total_activation < neutral_threshold:
                emotion_scores["Neutral"] = 1.0
            else:
                emotion_scores["Neutral"] = max(0.0, 1.0 - total_activation)

        # Find dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        intensity = emotion_scores[dominant_emotion]

        # Apply temporal smoothing if face_id provided
        if face_id is not None:
            if face_id not in self.emotion_history:
                self.emotion_history[face_id] = []

            self.emotion_history[face_id].append(dominant_emotion)

            # Keep only recent history
            if len(self.emotion_history[face_id]) > self.smoothing_window:
                self.emotion_history[face_id].pop(0)

            # Vote for most common emotion in recent history
            if len(self.emotion_history[face_id]) >= 3:
                from collections import Counter
                emotion_counts = Counter(self.emotion_history[face_id])
                smoothed_emotion = emotion_counts.most_common(1)[0][0]

                # Use smoothed emotion if intensity is reasonable
                if emotion_scores[smoothed_emotion] > 0.15:
                    dominant_emotion = smoothed_emotion
                    intensity = emotion_scores[smoothed_emotion]

        return dominant_emotion, intensity, emotion_scores

    def get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """Get BGR color for emotion visualization"""
        return self.emotion_colors.get(emotion, (255, 255, 255))

    def get_emotion_emoji(self, emotion: str) -> str:
        """Get emoji representation of emotion"""
        emoji_map = {
            "Neutral": ":|",
            "Happy": ":)",
            "Sad": ":("
        }
        return emoji_map.get(emotion, "?")

    def clear_history(self, face_id: Optional[int] = None):
        """Clear emotion history for a face or all faces"""
        if face_id is not None:
            if face_id in self.emotion_history:
                del self.emotion_history[face_id]
        else:
            self.emotion_history.clear()


# Test
if __name__ == "__main__":
    print("=" * 70)
    print(" " * 20 + "Emotion Classifier Test")
    print("=" * 70)

    classifier = EmotionClassifier(smoothing_window=5)

    # Test cases (only 3 emotions)
    test_cases = [
        ("Neutral", {'au1': 0.0, 'au4': 0.0, 'au6': 0.0, 'au12': 0.0, 'au15': 0.0}),
        ("Happy (weak)", {'au1': 0.0, 'au4': 0.0, 'au6': 0.0, 'au12': 0.3, 'au15': 0.0}),
        ("Happy (strong)", {'au1': 0.0, 'au4': 0.0, 'au6': 0.6, 'au12': 0.8, 'au15': 0.0}),
        ("Sad (weak)", {'au1': 0.0, 'au4': 0.0, 'au6': 0.0, 'au12': 0.0, 'au15': 0.3}),
        ("Sad (strong)", {'au1': 0.4, 'au4': 0.3, 'au6': 0.0, 'au12': 0.0, 'au15': 0.6}),
    ]

    print("\nTest Cases:")
    print("-" * 70)

    for expected, aus in test_cases:
        emotion, intensity, scores = classifier.classify(aus)

        print(f"\nExpected: {expected}")
        print(f"AU values: AU1={aus['au1']:.1f} AU4={aus['au4']:.1f} AU6={aus['au6']:.1f} AU12={aus['au12']:.1f} AU15={aus['au15']:.1f}")
        print(f"Classified: {emotion} (intensity: {intensity:.2f})")
        print(f"All scores: Happy={scores['Happy']:.2f}, Sad={scores['Sad']:.2f}, Neutral={scores['Neutral']:.2f}")

        # Check if correct
        if expected.startswith(emotion):
            print("[OK] CORRECT")
        else:
            print(f"[X] INCORRECT (expected {expected})")

    print("\n" + "=" * 70)
    print("[OK] Emotion classifier test completed")
