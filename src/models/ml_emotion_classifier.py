"""
ML-based Emotion Classifier
Train and use machine learning models for emotion classification from AU measurements
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import json


class MLEmotionClassifier:
    """
    Machine Learning based emotion classifier
    Trains on AU measurements to predict emotions
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        model_path: Optional[str] = None,
        smoothing_window: int = 5
    ):
        """
        Initialize ML emotion classifier

        Args:
            model_type: Type of ML model ('random_forest', 'svm', 'mlp')
            model_path: Path to pre-trained model (None = use rule-based fallback)
            smoothing_window: Number of recent predictions to smooth
        """
        self.model_type = model_type
        self.model_path = model_path
        self.smoothing_window = smoothing_window

        # Model and scaler
        self.model = None
        self.scaler = StandardScaler()

        # Emotion history per face ID
        self.emotion_history = {}  # {face_id: [recent_emotions]}

        # Emotion labels (only 3)
        self.emotions = ["Neutral", "Happy", "Sad"]
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        self.idx_to_emotion = {i: e for i, e in enumerate(self.emotions)}

        # Emotion colors for visualization (BGR)
        self.emotion_colors = {
            "Neutral": (200, 200, 200),    # Gray
            "Happy": (0, 255, 0),          # Green
            "Sad": (255, 0, 0),            # Blue
        }

        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            print(f"[OK] Loaded pre-trained model from {model_path}")
        else:
            print(f"[Warning] No pre-trained model found. Using rule-based fallback.")
            print("         Train a model using train_model() or data_labeling tool.")

    def _extract_features(self, aus: Dict[str, float]) -> np.ndarray:
        """
        Extract feature vector from AU measurements

        Args:
            aus: Dictionary with AU measurements

        Returns:
            Feature vector (5 features: AU1, AU4, AU6, AU12, AU15)
        """
        features = np.array([
            aus.get('au1', 0.0),
            aus.get('au4', 0.0),
            aus.get('au6', 0.0),
            aus.get('au12', 0.0),
            aus.get('au15', 0.0)
        ])
        return features

    def _rule_based_classify(self, aus: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Rule-based fallback classification (same as original EmotionClassifier)

        Args:
            aus: Dictionary with AU measurements

        Returns:
            Tuple of (emotion_label, intensity, emotion_scores)
        """
        # Extract AU values
        au1 = aus.get('au1', 0.0)
        au4 = aus.get('au4', 0.0)
        au6 = aus.get('au6', 0.0)
        au12 = aus.get('au12', 0.0)
        au15 = aus.get('au15', 0.0)

        # Calculate emotion intensities
        emotion_scores = {}

        # Happy
        happy_intensity = au12 * 0.7
        if au6 > 0.2:
            happy_intensity += au6 * 0.3
        emotion_scores["Happy"] = min(1.0, happy_intensity)

        # Sad
        sad_intensity = au15 * 0.5
        if au4 > 0.2 or au1 > 0.2:
            sad_intensity += (au4 + au1) * 0.25
        emotion_scores["Sad"] = min(1.0, sad_intensity)

        # Neutral
        total_activation = emotion_scores["Happy"] + emotion_scores["Sad"]
        emotion_scores["Neutral"] = max(0.0, 1.0 - total_activation)

        # Find dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        intensity = emotion_scores[dominant_emotion]

        return dominant_emotion, intensity, emotion_scores

    def classify(self, aus: Dict[str, float], face_id: Optional[int] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify emotion from AU measurements using ML model or rule-based fallback

        Args:
            aus: Dictionary with AU measurements {'au1': float, 'au4': float, ...}
            face_id: Optional face ID for temporal smoothing

        Returns:
            Tuple of (emotion_label, intensity, emotion_scores)
        """
        # Use ML model if available, otherwise fall back to rules
        if self.model is not None:
            features = self._extract_features(aus).reshape(1, -1)
            features_scaled = self.scaler.transform(features)

            # Get prediction and probabilities
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]

            # Convert to emotion label and scores
            dominant_emotion = self.idx_to_emotion[prediction]
            emotion_scores = {
                self.emotions[i]: float(probabilities[i])
                for i in range(len(self.emotions))
            }
            intensity = emotion_scores[dominant_emotion]
        else:
            # Fall back to rule-based classification
            dominant_emotion, intensity, emotion_scores = self._rule_based_classify(aus)

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

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train ML model on labeled data

        Args:
            X: Feature matrix (N samples, 5 features: AU1, AU4, AU6, AU12, AU15)
            y: Labels (N samples, emotion indices 0=Neutral, 1=Happy, 2=Sad)
            save_path: Path to save trained model
            test_size: Test set size ratio
            random_state: Random seed

        Returns:
            Training results dictionary
        """
        print(f"\n[Training] Starting {self.model_type} model training...")
        print(f"  Training samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create model
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        elif self.model_type == "svm":
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=random_state
            )
        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Train
        print(f"\n[Training] Fitting {self.model_type}...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        print(f"\n[Results]")
        print(f"  Train accuracy: {train_score:.3f}")
        print(f"  Test accuracy: {test_score:.3f}")

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, n_jobs=-1
        )
        print(f"  CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        # Classification report
        y_pred = self.model.predict(X_test_scaled)
        print(f"\n[Classification Report]")
        print(classification_report(y_test, y_pred, target_names=self.emotions))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n[Confusion Matrix]")
        print(f"  Predicted ->")
        print(f"  Actual ↓    {' '.join([f'{e:8s}' for e in self.emotions])}")
        for i, emotion in enumerate(self.emotions):
            print(f"  {emotion:8s}  {' '.join([f'{cm[i,j]:8d}' for j in range(len(self.emotions))])}")

        # Save model
        if save_path:
            self.save_model(save_path)
            print(f"\n[Saved] Model saved to {save_path}")

        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, target_names=self.emotions, output_dict=True),
            'confusion_matrix': cm.tolist()
        }

        return results

    def save_model(self, path: str):
        """Save trained model and scaler"""
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'emotions': self.emotions,
            'emotion_to_idx': self.emotion_to_idx
        }

        joblib.dump(model_data, path)

    def load_model(self, path: str):
        """Load trained model and scaler"""
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.emotions = model_data['emotions']
        self.emotion_to_idx = model_data['emotion_to_idx']
        self.idx_to_emotion = {i: e for i, e in enumerate(self.emotions)}

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
    print(" " * 20 + "ML Emotion Classifier Test")
    print("=" * 70)

    # Create dummy training data
    print("\nGenerating synthetic training data...")
    np.random.seed(42)

    # Generate data for each emotion
    n_samples = 100

    # Neutral: low AU activations
    X_neutral = np.random.rand(n_samples, 5) * 0.2
    y_neutral = np.zeros(n_samples, dtype=int)

    # Happy: high AU12, high AU6
    X_happy = np.random.rand(n_samples, 5) * 0.3
    X_happy[:, 2] = np.random.rand(n_samples) * 0.5 + 0.3  # AU6
    X_happy[:, 3] = np.random.rand(n_samples) * 0.5 + 0.4  # AU12
    y_happy = np.ones(n_samples, dtype=int)

    # Sad: high AU15, high AU4, high AU1
    X_sad = np.random.rand(n_samples, 5) * 0.3
    X_sad[:, 0] = np.random.rand(n_samples) * 0.4 + 0.2  # AU1
    X_sad[:, 1] = np.random.rand(n_samples) * 0.4 + 0.2  # AU4
    X_sad[:, 4] = np.random.rand(n_samples) * 0.5 + 0.3  # AU15
    y_sad = np.full(n_samples, 2, dtype=int)

    # Combine
    X = np.vstack([X_neutral, X_happy, X_sad])
    y = np.hstack([y_neutral, y_happy, y_sad])

    print(f"  Total samples: {len(X)}")
    print(f"  Neutral: {len(y_neutral)}, Happy: {len(y_happy)}, Sad: {len(y_sad)}")

    # Train model
    classifier = MLEmotionClassifier(model_type="random_forest")
    results = classifier.train_model(X, y, save_path="models/emotion_classifier.pkl")

    # Test classification
    print("\n" + "=" * 70)
    print("Testing classification on new samples:")
    print("-" * 70)

    test_cases = [
        ("Neutral", {'au1': 0.0, 'au4': 0.0, 'au6': 0.0, 'au12': 0.0, 'au15': 0.0}),
        ("Happy", {'au1': 0.0, 'au4': 0.0, 'au6': 0.6, 'au12': 0.8, 'au15': 0.0}),
        ("Sad", {'au1': 0.4, 'au4': 0.3, 'au6': 0.0, 'au12': 0.0, 'au15': 0.6}),
    ]

    for expected, aus in test_cases:
        emotion, intensity, scores = classifier.classify(aus)

        print(f"\nExpected: {expected}")
        print(f"Classified: {emotion} (confidence: {intensity:.2f})")
        print(f"All scores: Happy={scores['Happy']:.2f}, Sad={scores['Sad']:.2f}, Neutral={scores['Neutral']:.2f}")

        if expected == emotion:
            print("[OK] CORRECT")
        else:
            print(f"[X] INCORRECT")

    print("\n" + "=" * 70)
    print("[OK] ML Emotion classifier test completed")
