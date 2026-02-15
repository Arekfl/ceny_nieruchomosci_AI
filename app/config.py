"""
Configuration and utilities for the FastAPI application
"""
import os
from pathlib import Path
from datetime import datetime
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# Model paths
MODEL_PATH = MODELS_DIR / "price_model.joblib"
ENCODERS_PATH = MODELS_DIR / "label_encoders.joblib"
FEATURES_PATH = MODELS_DIR / "features.joblib"

# Model metadata
MODEL_METADATA = {
    "model_type": "Regression",
    "algorithm": "Random Forest Regressor",
    "features_used": ["Area (m²)", "Number of rooms", "year_const", "Heating", "Building material", "Building type", "Market", "voivodeship", "city"],
    "training_samples": 19344,
    "test_r2_score": 0.6364,
    "test_rmse": 148161.29,
    "test_mae": 112492.14,
    "last_updated": datetime.now().isoformat()
}


def load_model():
    """Load the trained model from disk"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_encoders():
    """Load label encoders from disk"""
    if not ENCODERS_PATH.exists():
        raise FileNotFoundError(f"Encoders file not found at {ENCODERS_PATH}")
    return joblib.load(ENCODERS_PATH)


def load_features():
    """Load feature list from disk"""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found at {FEATURES_PATH}")
    return joblib.load(FEATURES_PATH)


# Global model cache
_model = None
_encoders = None
_features = None


def get_model():
    """Get the loaded model (cached)"""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def get_encoders():
    """Get the loaded encoders (cached)"""
    global _encoders
    if _encoders is None:
        _encoders = load_encoders()
    return _encoders


def get_features():
    """Get the loaded features (cached)"""
    global _features
    if _features is None:
        _features = load_features()
    return _features
