"""
Train Emotion Classification Model
Train ML model from labeled data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import argparse
from src.models.ml_emotion_classifier import MLEmotionClassifier


def load_labeled_data(data_dir: str = "data/recordings") -> tuple:
    """
    Load all labeled CSV files from data directory

    Args:
        data_dir: Directory containing labeled CSV files

    Returns:
        Tuple of (X, y, metadata)
        - X: Feature matrix (N, 5) with AU measurements
        - y: Label array (N,) with emotion indices
        - metadata: Dict with data info
    """
    data_dir = Path(data_dir)
    label_files = list(data_dir.glob("labels_*.csv"))

    if not label_files:
        raise FileNotFoundError(
            f"No labeled data found in {data_dir}\n"
            "Please label data first using:\n"
            "  python -m src.utils.data_labeling_tool"
        )

    print(f"\n[Loading] Found {len(label_files)} labeled files")

    all_data = []

    for file in label_files:
        print(f"  Loading: {file.name}")
        df = pd.read_csv(file)
        all_data.append(df)

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"\n[Data] Total samples: {len(combined_df)}")

    # Check emotion distribution
    emotion_counts = combined_df['emotion_label'].value_counts()
    print(f"\n[Distribution]")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(combined_df)) * 100
        print(f"  {emotion}: {count} ({percentage:.1f}%)")

    # Extract features (AU measurements)
    X = combined_df[['au1', 'au4', 'au6', 'au12', 'au15']].values

    # Convert emotion labels to indices
    emotion_to_idx = {"Neutral": 0, "Happy": 1, "Sad": 2}
    y = combined_df['emotion_label'].map(emotion_to_idx).values

    metadata = {
        'total_samples': len(combined_df),
        'num_files': len(label_files),
        'emotion_distribution': emotion_counts.to_dict()
    }

    return X, y, metadata


def train_model(
    data_dir: str = "data/recordings",
    model_type: str = "random_forest",
    output_path: str = "models/emotion_classifier.pkl",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train emotion classification model

    Args:
        data_dir: Directory containing labeled CSV files
        model_type: Type of ML model ('random_forest', 'svm', 'mlp')
        output_path: Path to save trained model
        test_size: Test set size ratio
        random_state: Random seed
    """
    print("=" * 70)
    print(" " * 15 + "Emotion Model Training")
    print("=" * 70)

    # Load data
    X, y, metadata = load_labeled_data(data_dir)

    # Check if we have enough samples
    if len(X) < 30:
        print(f"\n[Warning] Only {len(X)} samples found.")
        print("          Recommend at least 100 samples per emotion for good performance.")
        print("          Label more data using data_labeling_tool.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return

    # Create classifier
    classifier = MLEmotionClassifier(model_type=model_type)

    # Train
    results = classifier.train_model(
        X, y,
        save_path=output_path,
        test_size=test_size,
        random_state=random_state
    )

    print(f"\n{'=' * 70}")
    print("[SUCCESS] Model training completed!")
    print(f"  Model saved to: {output_path}")
    print(f"  Test accuracy: {results['test_accuracy']:.3f}")
    print(f"\nYou can now use this model by setting model_path in EmotionAnalyzer:")
    print(f"  analyzer = EmotionAnalyzer(emotion_model_path='{output_path}')")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train emotion classification model")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/recordings",
        help="Directory containing labeled CSV files"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "svm", "mlp"],
        help="Type of ML model to train"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/emotion_classifier.pkl",
        help="Path to save trained model"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size ratio (0.0-1.0)"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        output_path=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )
