"""
CNN-based Emotion Classifier
Direct image-to-emotion classification using deep learning
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2


class CNNEmotionClassifier:
    """
    CNN-based emotion classifier for direct image-to-emotion classification
    Uses transfer learning with pre-trained models
    """

    def __init__(
        self,
        model_type: str = "mobilenet_v2",
        model_path: Optional[str] = None,
        input_size: int = 224,
        smoothing_window: int = 5
    ):
        """
        Initialize CNN emotion classifier

        Args:
            model_type: Type of CNN model ('mobilenet_v2', 'resnet18', 'efficientnet_b0')
            model_path: Path to pre-trained model weights (None = use ImageNet weights only)
            input_size: Input image size (default 224x224)
            smoothing_window: Number of recent predictions to smooth
        """
        self.model_type = model_type
        self.model_path = model_path
        self.input_size = input_size
        self.smoothing_window = smoothing_window

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # Emotion labels (only 3)
        self.emotions = ["Neutral", "Happy", "Sad"]
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        self.idx_to_emotion = {i: e for i, e in enumerate(self.emotions)}
        self.num_classes = len(self.emotions)

        # Emotion colors for visualization (BGR)
        self.emotion_colors = {
            "Neutral": (200, 200, 200),    # Gray
            "Happy": (0, 255, 0),          # Green
            "Sad": (255, 0, 0),            # Blue
        }

        # Emotion history per face ID
        self.emotion_history = {}  # {face_id: [recent_emotions]}

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()

        # Load pre-trained weights if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            print(f"[OK] Loaded pre-trained CNN model from {model_path}")
        else:
            print(f"[Warning] No pre-trained CNN model found. Using ImageNet weights only.")
            print("         Train a model using train_cnn.py or labeling tool.")

    def _build_model(self) -> nn.Module:
        """
        Build CNN model with transfer learning

        Returns:
            PyTorch model
        """
        if self.model_type == "mobilenet_v2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            # Replace classifier
            model.classifier[1] = nn.Linear(model.last_channel, self.num_classes)

        elif self.model_type == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Replace fc layer
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif self.model_type == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            # Replace classifier
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model

    def _preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for CNN input

        Args:
            face_img: Face image (BGR numpy array)

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_img = Image.fromarray(face_rgb)

        # Apply transforms
        tensor = self.transform(pil_img)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor

    def classify(
        self,
        face_img: np.ndarray,
        face_id: Optional[int] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify emotion from face image using CNN

        Args:
            face_img: Face image (BGR numpy array)
            face_id: Optional face ID for temporal smoothing

        Returns:
            Tuple of (emotion_label, intensity, emotion_scores)
        """
        # Preprocess image
        input_tensor = self._preprocess_face(face_img).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Convert to emotion scores
        emotion_scores = {
            self.emotions[i]: float(probabilities[i])
            for i in range(len(self.emotions))
        }

        # Get dominant emotion
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

    def train_model(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train CNN model on labeled face images

        Args:
            train_loader: PyTorch DataLoader for training data
            val_loader: PyTorch DataLoader for validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save trained model

        Returns:
            Training results dictionary
        """
        print(f"\n[Training] Starting {self.model_type} training...")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        best_val_acc = 0.0
        train_losses = []
        val_accs = []

        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            # Training
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            train_losses.append(train_loss)

            # Validation
            val_acc = self._evaluate(val_loader)
            val_accs.append(val_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    self.save_model(save_path)
                    print(f"  [Saved] Best model (val_acc={val_acc:.2f}%)")

        print(f"\n[Training Complete]")
        print(f"  Best validation accuracy: {best_val_acc:.2f}%")

        results = {
            'train_losses': train_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }

        return results

    def _evaluate(self, data_loader) -> float:
        """Evaluate model on validation/test set"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.model.train()
        accuracy = 100 * correct / total
        return accuracy

    def save_model(self, path: str):
        """Save trained model"""
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'emotions': self.emotions,
            'emotion_to_idx': self.emotion_to_idx,
            'input_size': self.input_size
        }

        torch.save(checkpoint, path)

    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_type = checkpoint['model_type']
        self.emotions = checkpoint['emotions']
        self.emotion_to_idx = checkpoint['emotion_to_idx']
        self.idx_to_emotion = {i: e for i, e in enumerate(self.emotions)}
        self.input_size = checkpoint['input_size']

        self.model.eval()

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
    print(" " * 20 + "CNN Emotion Classifier Test")
    print("=" * 70)

    # Create classifier
    classifier = CNNEmotionClassifier(model_type="mobilenet_v2")

    # Test with a dummy face image
    print("\nTesting classification with dummy image...")
    dummy_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    emotion, intensity, scores = classifier.classify(dummy_face)

    print(f"\nClassified: {emotion} (confidence: {intensity:.2f})")
    print(f"All scores: Happy={scores['Happy']:.2f}, Sad={scores['Sad']:.2f}, Neutral={scores['Neutral']:.2f}")

    print("\n" + "=" * 70)
    print("[OK] CNN Emotion classifier test completed")
    print("Note: This classifier needs training on labeled emotion data to work properly.")
