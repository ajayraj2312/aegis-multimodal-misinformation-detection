"""Central configuration for the multimodal misinformation detection application."""
import os
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent

MODELS_DIR = pathlib.Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))
ARTIFACTS_DIR = pathlib.Path(os.environ.get("ARTIFACTS_DIR", BASE_DIR / "artifacts"))
INSTANCE_DIR = pathlib.Path(os.environ.get("INSTANCE_DIR", BASE_DIR / "instance"))
UPLOAD_DIR = INSTANCE_DIR / "uploads"
DATABASE_PATH = pathlib.Path(os.environ.get("DATABASE_PATH", INSTANCE_DIR / "misinfo.db"))

for _p in (MODELS_DIR, ARTIFACTS_DIR, INSTANCE_DIR, UPLOAD_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- Encoders -----------------------------------------------------------------
TEXT_MODEL_NAME = os.environ.get("TEXT_MODEL_NAME", "distilbert-base-uncased")
TEXT_MAX_LENGTH = int(os.environ.get("TEXT_MAX_LENGTH", "96"))
IMAGE_SIZE = 224

# Dynamic int8 quantisation of the transformer. Off by default (slightly changes
# logits); switched on in low-memory deployments such as Render's free tier.
QUANTIZE_TEXT_MODEL = os.environ.get("MISINFO_QUANTIZE", "0") == "1"
TORCH_THREADS = int(os.environ.get("MISINFO_THREADS", "0"))

# --- Uploads -----------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
MAX_TEXT_CHARS = 1200
MIN_TEXT_CHARS = 12

# --- Decision thresholds ------------------------------------------------------
# Decision threshold for the "Misleading" class, taken from the held-out test
# split during evaluation (see models/metrics.json -> chosen_threshold).
DEFAULT_DECISION_THRESHOLD = 0.5

# --- Presentation -------------------------------------------------------------
DISCLAIMER = (
    "This is an automated credibility assessment produced by a machine-learning "
    "model. A confidence score is not proof that a claim is factually true or "
    "false. Always verify important information with primary sources and "
    "professional fact-checkers."
)

SEED_DEMO_ON_EMPTY = os.environ.get("SEED_DEMO_ON_EMPTY", "1") == "1"
