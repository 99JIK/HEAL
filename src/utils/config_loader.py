"""
Configuration Loader
Load settings from config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config.yaml (relative to project root)

    Returns:
        Configuration dictionary
    """
    # Find config file from project root
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent  # Go up to HEAL/
    config_file = project_root / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_recording_config() -> Dict[str, Any]:
    """
    Get recording-specific configuration

    Returns:
        Dictionary with recording settings:
        - output_dir: str
        - sample_rate: int
    """
    config = load_config()
    return config.get('recording', {
        'output_dir': 'data/recordings',
        'sample_rate': 5
    })


def get_face_detection_config() -> Dict[str, Any]:
    """
    Get face detection configuration

    Returns:
        Dictionary with face detection settings
    """
    config = load_config()
    face_detection = config.get('face_detection', {})
    return {
        'min_detection_confidence': face_detection.get('min_detection_confidence', 0.5)
    }


def get_emotion_classification_config() -> Dict[str, Any]:
    """
    Get emotion classification configuration

    Returns:
        Dictionary with emotion classification settings:
        - use_ml_classifier: bool
        - model_path: str
        - model_type: str
        - smoothing_window: int
        - emotion_definitions: dict
    """
    config = load_config()
    return config.get('emotion_classification', {
        'use_ml_classifier': False,
        'model_path': 'models/emotion_classifier.pkl',
        'model_type': 'random_forest',
        'smoothing_window': 5,
        'emotion_definitions': {}
    })


def get_au_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Get AU definitions from config

    Returns:
        Dictionary of AU definitions with metadata
        {
            'au1': {'name': '...', 'description': '...', 'enabled': True},
            ...
        }
    """
    config = load_config()
    au_extraction = config.get('au_extraction', {})
    au_definitions = au_extraction.get('au_definitions', {})

    # Filter only enabled AUs
    enabled_aus = {
        au_id: au_data
        for au_id, au_data in au_definitions.items()
        if au_data.get('enabled', False)
    }

    return enabled_aus


def get_emotion_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Get emotion definitions from config

    Returns:
        Dictionary of emotion definitions with AU rules
        {
            'Happy': {
                'color': [0, 255, 0],
                'description': '...',
                'au_rules': {...}
            },
            ...
        }
    """
    config = load_config()
    emotion_config = config.get('emotion_classification', {})
    emotion_definitions = emotion_config.get('emotion_definitions', {})

    # Filter only enabled emotions (enabled by default if not specified)
    enabled_emotions = {
        emotion_id: emotion_data
        for emotion_id, emotion_data in emotion_definitions.items()
        if emotion_data.get('enabled', True)  # Default to enabled
    }

    return enabled_emotions


def get_au_extraction_config() -> Dict[str, Any]:
    """
    Get AU extraction configuration

    Returns:
        Dictionary with AU extraction settings:
        - min_detection_confidence: float
        - min_tracking_confidence: float
        - au_definitions: dict
    """
    config = load_config()
    au_config = config.get('au_extraction', {
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'au_definitions': {}
    })

    return au_config


# Test
if __name__ == "__main__":
    print("=" * 70)
    print(" " * 20 + "Config Loader Test")
    print("=" * 70)

    # Load full config
    config = load_config()
    print("\nFull config loaded successfully")
    print(f"Keys: {list(config.keys())}")

    # Load recording config
    recording_config = get_recording_config()
    print("\nRecording config:")
    for key, value in recording_config.items():
        print(f"  {key}: {value}")

    # Load face detection config
    face_config = get_face_detection_config()
    print("\nFace detection config:")
    for key, value in face_config.items():
        print(f"  {key}: {value}")

    # Load AU definitions
    au_defs = get_au_definitions()
    print(f"\nEnabled AUs ({len(au_defs)}):")
    for au_id, au_data in au_defs.items():
        print(f"  {au_id}: {au_data['name']} - {au_data['description']}")

    # Load emotion definitions
    emotion_defs = get_emotion_definitions()
    print(f"\nEnabled Emotions ({len(emotion_defs)}):")
    for emotion, data in emotion_defs.items():
        print(f"  {emotion}: {data['description']}")
        au_rules = data.get('au_rules', {})
        primary = au_rules.get('primary', {})
        if primary:
            print(f"    Primary AUs: {list(primary.keys())}")

    print("\n[OK] Config loader test completed")
