"""
Ensemble Emotion Classifier
Combines AU-based and CNN-based emotion classifiers for improved accuracy
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

from emotion_classifier import EmotionClassifier
from ml_emotion_classifier import MLEmotionClassifier
from cnn_emotion_classifier import CNNEmotionClassifier
from src.utils.config_loader import get_emotion_classification_config


class EnsembleClassifier:
    """
    Ensemble classifier combining multiple emotion recognition methods

    Modes:
        1. AU-based only (rule-based or ML)
        2. CNN-based only (direct image classification)
        3. Ensemble (weighted combination of both)
    """

    def __init__(self):
        """
        Initialize ensemble classifier with config-based settings
        """
        # Load config
        config = get_emotion_classification_config()

        self.use_ml_classifier = config.get('use_ml_classifier', False)
        self.use_cnn_classifier = config.get('use_cnn_classifier', False)
        self.use_ensemble = config.get('use_ensemble', False)

        # Ensemble weights
        ensemble_weights = config.get('ensemble_weights', {'au_based': 0.6, 'cnn_based': 0.4})
        self.au_weight = ensemble_weights.get('au_based', 0.6)
        self.cnn_weight = ensemble_weights.get('cnn_based', 0.4)

        # Normalize weights
        total_weight = self.au_weight + self.cnn_weight
        self.au_weight /= total_weight
        self.cnn_weight /= total_weight

        # Initialize classifiers
        self.au_classifier = None
        self.cnn_classifier = None

        # Initialize AU-based classifier
        if self.use_ml_classifier:
            # ML-based AU classifier
            model_path = config.get('model_path', 'models/emotion_classifier.pkl')
            model_type = config.get('model_type', 'random_forest')
            smoothing_window = config.get('smoothing_window', 5)

            if Path(model_path).exists():
                self.au_classifier = MLEmotionClassifier(
                    model_type=model_type,
                    model_path=model_path,
                    smoothing_window=smoothing_window
                )
                print(f"[OK] AU-based ML classifier loaded")
            else:
                print(f"[Warning] AU-based ML model not found: {model_path}")
                print("         Falling back to rule-based classifier")
                self.au_classifier = EmotionClassifier(smoothing_window=smoothing_window)
        else:
            # Rule-based AU classifier
            smoothing_window = config.get('smoothing_window', 5)
            self.au_classifier = EmotionClassifier(smoothing_window=smoothing_window)
            print(f"[OK] AU-based rule classifier loaded")

        # Initialize CNN-based classifier
        if self.use_cnn_classifier or self.use_ensemble:
            cnn_model_path = config.get('cnn_model_path', 'models/cnn_emotion.pth')
            cnn_model_type = config.get('cnn_model_type', 'mobilenet_v2')
            cnn_input_size = config.get('cnn_input_size', 224)
            smoothing_window = config.get('smoothing_window', 5)

            if Path(cnn_model_path).exists():
                self.cnn_classifier = CNNEmotionClassifier(
                    model_type=cnn_model_type,
                    model_path=cnn_model_path,
                    input_size=cnn_input_size,
                    smoothing_window=smoothing_window
                )
                print(f"[OK] CNN-based classifier loaded")
            else:
                print(f"[Warning] CNN model not found: {cnn_model_path}")
                if self.use_cnn_classifier and not self.use_ensemble:
                    print("         CNN mode requires trained model!")
                    print("         Falling back to AU-based only")
                    self.use_cnn_classifier = False
                elif self.use_ensemble:
                    print("         Ensemble mode falling back to AU-based only")
                    self.use_ensemble = False

        # Determine active mode
        if self.use_ensemble and self.cnn_classifier is not None:
            self.mode = "ensemble"
            print(f"[INFO] Ensemble mode: AU={self.au_weight:.2f}, CNN={self.cnn_weight:.2f}")
        elif self.use_cnn_classifier and self.cnn_classifier is not None:
            self.mode = "cnn_only"
            print(f"[INFO] CNN-only mode")
        else:
            self.mode = "au_only"
            print(f"[INFO] AU-only mode")

    def classify(
        self,
        aus: Optional[Dict[str, float]] = None,
        face_img: Optional[np.ndarray] = None,
        face_id: Optional[int] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify emotion using ensemble method

        Args:
            aus: AU measurements (required for AU-based and ensemble modes)
            face_img: Face image (required for CNN-based and ensemble modes)
            face_id: Optional face ID for temporal smoothing

        Returns:
            Tuple of (emotion_label, intensity, emotion_scores)
        """
        if self.mode == "au_only":
            # AU-based only
            if aus is None:
                raise ValueError("AU measurements required for AU-based classification")
            return self.au_classifier.classify(aus, face_id)

        elif self.mode == "cnn_only":
            # CNN-based only
            if face_img is None:
                raise ValueError("Face image required for CNN-based classification")
            return self.cnn_classifier.classify(face_img, face_id)

        elif self.mode == "ensemble":
            # Ensemble: combine both predictions
            if aus is None or face_img is None:
                raise ValueError("Both AU measurements and face image required for ensemble mode")

            # Get AU-based prediction
            au_emotion, au_intensity, au_scores = self.au_classifier.classify(aus, face_id)

            # Get CNN-based prediction
            cnn_emotion, cnn_intensity, cnn_scores = self.cnn_classifier.classify(face_img, face_id)

            # Combine scores with weights
            combined_scores = {}
            for emotion in au_scores.keys():
                au_score = au_scores.get(emotion, 0.0)
                cnn_score = cnn_scores.get(emotion, 0.0)
                combined_scores[emotion] = (
                    self.au_weight * au_score +
                    self.cnn_weight * cnn_score
                )

            # Get dominant emotion
            dominant_emotion = max(combined_scores, key=combined_scores.get)
            intensity = combined_scores[dominant_emotion]

            return dominant_emotion, intensity, combined_scores

    def get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """Get BGR color for emotion visualization"""
        if self.au_classifier:
            return self.au_classifier.get_emotion_color(emotion)
        elif self.cnn_classifier:
            return self.cnn_classifier.get_emotion_color(emotion)
        return (255, 255, 255)

    def get_emotion_emoji(self, emotion: str) -> str:
        """Get emoji representation of emotion"""
        if self.au_classifier:
            return self.au_classifier.get_emotion_emoji(emotion)
        elif self.cnn_classifier:
            return self.cnn_classifier.get_emotion_emoji(emotion)
        return "?"

    def clear_history(self, face_id: Optional[int] = None):
        """Clear emotion history for a face or all faces"""
        if self.au_classifier:
            self.au_classifier.clear_history(face_id)
        if self.cnn_classifier:
            self.cnn_classifier.clear_history(face_id)

    def get_mode_info(self) -> Dict[str, any]:
        """Get information about current mode"""
        return {
            'mode': self.mode,
            'use_ml_classifier': self.use_ml_classifier,
            'use_cnn_classifier': self.use_cnn_classifier,
            'use_ensemble': self.use_ensemble,
            'au_weight': self.au_weight if self.mode == 'ensemble' else None,
            'cnn_weight': self.cnn_weight if self.mode == 'ensemble' else None
        }


# Test
if __name__ == "__main__":
    print("=" * 70)
    print(" " * 20 + "Ensemble Classifier Test")
    print("=" * 70)

    # Create ensemble classifier
    classifier = EnsembleClassifier()

    # Print mode info
    mode_info = classifier.get_mode_info()
    print(f"\nMode: {mode_info['mode']}")
    print(f"  use_ml_classifier: {mode_info['use_ml_classifier']}")
    print(f"  use_cnn_classifier: {mode_info['use_cnn_classifier']}")
    print(f"  use_ensemble: {mode_info['use_ensemble']}")

    if mode_info['mode'] == 'ensemble':
        print(f"  au_weight: {mode_info['au_weight']:.2f}")
        print(f"  cnn_weight: {mode_info['cnn_weight']:.2f}")

    # Test classification based on mode
    print("\n" + "=" * 70)
    print("Testing classification...")
    print("-" * 70)

    test_aus = {'au1': 0.0, 'au4': 0.0, 'au6': 0.6, 'au12': 0.8, 'au15': 0.0}

    if mode_info['mode'] == 'au_only':
        # AU-based only
        emotion, intensity, scores = classifier.classify(aus=test_aus)
        print(f"\nAU-based classification:")
        print(f"  AU values: AU12={test_aus['au12']:.1f}, AU6={test_aus['au6']:.1f}")
        print(f"  Emotion: {emotion} (intensity: {intensity:.2f})")
        print(f"  Scores: {', '.join([f'{k}={v:.2f}' for k, v in scores.items()])}")

    elif mode_info['mode'] == 'cnn_only':
        # CNN-based only (requires face image)
        print("\nCNN-based classification requires face image")
        print("  Skipping test (use with real face images)")

    elif mode_info['mode'] == 'ensemble':
        # Ensemble (requires both)
        print("\nEnsemble classification requires both AU and face image")
        print("  Skipping test (use with real data)")

    print("\n" + "=" * 70)
    print("[OK] Ensemble classifier test completed")
