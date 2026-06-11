"""Project-wide constants for CardioCare."""

RANDOM_SEED = 42

# UCI Cleveland Heart Disease column names
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN = "target"

# Clinical plausibility ranges for inference validation
CLINICAL_RANGES = {
    "age": (0, 120),
    "sex": (0, 1),
    "cp": (1, 4),
    "trestbps": (80, 250),
    "chol": (0, 600),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (60, 220),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (1, 3),
    "ca": (0, 3),
    "thal": (3, 7),
}

MISSING_THRESHOLD = 0.30

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

DEFAULT_DATA_PATH = "data/heart_disease.csv"
MODEL_PATH = "artifacts/model.pkl"
METADATA_PATH = "artifacts/model_metadata.json"
INFERENCE_LOG_PATH = "logs/inference.log"
MONITOR_OUTPUT_DIR = "artifacts/monitor"
