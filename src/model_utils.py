from __future__ import annotations

from pathlib import Path

import joblib

try:
	from .config import MODEL_DIR
except ImportError:
	from config import MODEL_DIR


def _resolve_model_path(model_name_or_path: str | Path) -> Path:
	path = Path(model_name_or_path)
	if not path.is_absolute():
		path = MODEL_DIR / path
	return path


def save_model(model, filename: str | Path) -> Path:
	"""Persist a trained model or pipeline to the model directory."""

	MODEL_DIR.mkdir(parents=True, exist_ok=True)
	filepath = _resolve_model_path(filename)
	filepath.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, filepath)
	return filepath


def load_model(filename: str | Path):
	"""Load a persisted model or pipeline from disk."""

	filepath = _resolve_model_path(filename)
	if not filepath.exists():
		raise FileNotFoundError(
			f"Model not found at {filepath}. Run src/train_models.py first."
		)
	return joblib.load(filepath)
