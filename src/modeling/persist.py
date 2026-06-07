import pickle
from pathlib import Path
from typing import Any

from src.config import BOOL_COLS, CAT_COLS, MODELS_DIR, NUM_COLS, TARGET

PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
MODEL_FILE = MODELS_DIR / "best_model.pkl"
METADATA_FILE = MODELS_DIR / "model_metadata.pkl"


def save_artifacts(
    preprocessor: Any,
    model: Any,
    model_name: str,
    *,
    models_dir: Path | None = None,
) -> dict:
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    preprocessor_path = models_dir / "preprocessor.pkl"
    model_path = models_dir / "best_model.pkl"
    metadata_path = models_dir / "model_metadata.pkl"

    with preprocessor_path.open("wb") as f:
        pickle.dump(preprocessor, f)
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model_name": model_name,
        "target": TARGET,
        "input_features": NUM_COLS + BOOL_COLS + CAT_COLS,
        "encoded_features": list(preprocessor.get_feature_names_out()),
        "preprocessor_path": str(preprocessor_path),
        "model_path": str(model_path),
    }
    with metadata_path.open("wb") as f:
        pickle.dump(metadata, f)

    return metadata


def load_artifacts(models_dir: Path | None = None) -> tuple[Any, Any, dict]:
    models_dir = models_dir or MODELS_DIR
    with (models_dir / "preprocessor.pkl").open("rb") as f:
        preprocessor = pickle.load(f)
    with (models_dir / "best_model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (models_dir / "model_metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)
    return preprocessor, model, metadata
