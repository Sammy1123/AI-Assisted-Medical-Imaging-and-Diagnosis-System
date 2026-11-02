"""Central configuration for the Medical Imaging backend."""
import os

# Paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
ROOT_DIR   = os.path.dirname(BASE_DIR)                                    # project root
DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODEL_DIR  = os.path.join(ROOT_DIR, "models", "weights")                  # fixed path
AUDIT_LOG_PATH = os.path.join(ROOT_DIR, "audit_log.jsonl")

# Model
IMAGE_SIZE = 224
NUM_CLASSES = 1          # binary: pneumonia vs normal
MODEL_BACKBONE = "densenet121"

# Inference thresholds
POSITIVE_THRESHOLD = 0.5

# DISCLAIMER
DISCLAIMER = (
    "FOR EDUCATION / RESEARCH ONLY — NOT FOR CLINICAL USE. "
    "This system has not been validated for clinical diagnosis."
)
