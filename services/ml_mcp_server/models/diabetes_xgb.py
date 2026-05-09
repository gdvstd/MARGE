"""GBDT ensemble classifier on the Pima Indians Diabetes Database (UCI / OpenML).

Binary task: tested_positive (1) vs tested_negative (0) for diabetes from
8 metabolic and demographic features.
Refactored to utilize the generalized DynamicMLAgent Factory pattern.
"""

from pathlib import Path
from typing import Any

# Assuming ml_agent_factory.py is accessible
from .ml_agent_factory import AgentConfig, DynamicMLAgent

# --- 1. Define Paths and Constants ---
_ARTIFACT_PATH = Path(__file__).parent.parent / "artifacts" / "diabetes_dynamic.joblib"

_FEATURE_NAMES = [
    "preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"
]

# --- 2. Define the Agent Configuration ---
_DIABETES_CONFIG = AgentConfig(
    agent_name="predict_diabetes_risk",
    description=(
        "Binary classifier for type-2 diabetes risk from 8 metabolic and "
        "demographic features (pregnancies, plasma glucose, blood pressure, "
        "triceps skin fold, serum insulin, BMI, genetic pedigree function, "
        "age). Trained dynamically via Agent Factory."
    ),
    version="0.2.0-factory",
    artifact_path=_ARTIFACT_PATH,
    feature_names=_FEATURE_NAMES,
    target_classes=["low_risk", "diabetic_risk"],  # 0 = low risk, 1 = diabetic risk
    trained_on_desc="OpenML 'diabetes' v1 (Pima Indians, n=768) - Dynamic Ensemble",
    n_splits=5
)

# --- 3. Create the Server-Compatible Model Class ---
class DiabetesClassifier(DynamicMLAgent):
    """Dynamic binary classifier for type-2 diabetes risk."""

    def __init__(self) -> None:
        # Initialize the generalized factory agent
        super().__init__(_DIABETES_CONFIG)

    def sample_inputs(self) -> dict[str, Any]:
        """Sample high-risk case from the Pima Indians dataset."""
        # Providing a static dictionary representation of a true positive case
        return {
            "preg": 6.0,
            "plas": 148.0,
            "pres": 72.0,
            "skin": 35.0,
            "insu": 0.0,
            "mass": 33.6,
            "pedi": 0.627,
            "age": 50.0
        }