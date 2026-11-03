"""Central configuration for the Medical Imaging backend."""
import os

# Paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/
ROOT_DIR   = os.path.dirname(BASE_DIR)                                                        # project root
DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODEL_DIR  = os.path.join(ROOT_DIR, "models", "weights")                  # fixed path
AUDIT_LOG_PATH = os.path.join(ROOT_DIR, "audit_log.jsonl")

# Model
IMAGE_SIZE = 224
NUM_CLASSES = 1          # binary: pneumonia vs normal
MODEL_BACKBONE = "densenet121"

# Inference thresholds
POSITIVE_THRESHOLD = 0.5

# CORS — restrict to local development by default.
# Override with a comma-separated list in the CORS_ALLOW_ORIGINS env var for deployment.
_env_origins = os.environ.get("CORS_ALLOW_ORIGINS", "")
CORS_ALLOW_ORIGINS = [
    o.strip() for o in _env_origins.split(",") if o.strip()
] or [
    "http://localhost:3000",      # React dev server
    "http://127.0.0.1:3000",
]

# DISCLAIMER
DISCLAIMER = (
    "FOR EDUCATION / RESEARCH ONLY — NOT FOR CLINICAL USE. "
    "This system has not been validated for clinical diagnosis."
)
