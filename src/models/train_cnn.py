"""
CNN Training Script
Train CNN-based emotion classifier on labeled face images
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import argparse

from cnn_emotion_classifier import CNNEmotionClassifier


class EmotionDataset(Dataset):
    """
    Dataset for emotion classification from face images

    Expected directory structure:
        data/labeled_faces/
            Neutral/
                face_0001.jpg
                face_0002.jpg
                ...
            Happy/
                face_0001.jpg
                face_0002.jpg
                ...
            Sad/
                face_0001.jpg
                face_0002.jpg
                ...
    """

    def __init__(self, data_dir: str, transform=None, split: str = 'train'):
        """
        Initialize dataset

        Args:
            data_dir: Root directory containing emotion subdirectories
            transform: Image transforms
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split

        # Emotion labels
        self.emotions = ["Neutral", "Happy", "Sad"]
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}

        # Load image paths and labels
        self.samples = self._load_samples()

        print(f"[{split.upper()}] Loaded {len(self.samples)} samples")
        self._print_class_distribution()

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their labels"""
        samples = []

        for emotion in self.emotions:
            emotion_dir = self.data_dir / emotion

            if not emotion_dir.exists():
                print(f"[Warning] Directory not found: {emotion_dir}")
                continue

            # Get all image files
            image_files = list(emotion_dir.glob("*.jpg")) + \
                         list(emotion_dir.glob("*.png")) + \
                         list(emotion_dir.glob("*.jpeg"))

            label = self.emotion_to_idx[emotion]

            for img_path in image_files:
                samples.append((str(img_path), label))

        return samples

    def _print_class_distribution(self):
        """Print class distribution"""
        label_counts = {}
        for _, label in self.samples:
            emotion = self.emotions[label]
            label_counts[emotion] = label_counts.get(emotion, 0) + 1

        print(f"  Class distribution:")
        for emotion in self.emotions:
            count = label_counts.get(emotion, 0)
            percentage = 100 * count / len(self.samples) if len(self.samples) > 0 else 0
            print(f"    {emotion}: {count} ({percentage:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def train_cnn_classifier(
    data_dir: str,
    model_type: str = "mobilenet_v2",
    input_size: int = 224,
    batch_size: int = 32,
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    save_path: str = "models/cnn_emotion.pth"
):
    """
    Train CNN emotion classifier

    Args:
        data_dir: Directory containing labeled face images
        model_type: CNN architecture type
        input_size: Input image size
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        val_split: Validation set ratio
        save_path: Path to save trained model
    """
    print("=" * 70)
    print(" " * 20 + "CNN Emotion Classifier Training")
    print("=" * 70)

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = EmotionDataset(data_dir, transform=train_transform, split='full')

    if len(full_dataset) == 0:
        print("[ERROR] No training data found!")
        print(f"Please create labeled face images in: {data_dir}")
        print("Expected structure:")
        print("  data/labeled_faces/")
        print("    Neutral/")
        print("      face_0001.jpg")
        print("      ...")
        print("    Happy/")
        print("      face_0001.jpg")
        print("      ...")
        print("    Sad/")
        print("      face_0001.jpg")
        print("      ...")
        return

    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Update transforms for validation set
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"\nDataset split:")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Batch size: {batch_size}")

    # Create classifier
    classifier = CNNEmotionClassifier(
        model_type=model_type,
        input_size=input_size
    )

    # Train
    results = classifier.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_path=save_path
    )

    print("\n" + "=" * 70)
    print("[OK] Training completed")
    print(f"[OK] Model saved to: {save_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN emotion classifier")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/labeled_faces",
        help="Directory containing labeled face images"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "resnet18", "efficientnet_b0"],
        help="CNN architecture type"
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input image size"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )

    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation set ratio"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="models/cnn_emotion.pth",
        help="Path to save trained model"
    )

    args = parser.parse_args()

    # Train
    train_cnn_classifier(
        data_dir=args.data_dir,
        model_type=args.model_type,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        save_path=args.save_path
    )
