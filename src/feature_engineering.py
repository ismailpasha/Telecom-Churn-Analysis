from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
	from .config import PROCESSED_DATA_PATH, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
	from .data_preprocessing import clean_data, engineer_features, load_data, split_features_target
except ImportError:
	from config import PROCESSED_DATA_PATH, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
	from data_preprocessing import clean_data, engineer_features, load_data, split_features_target


class FeatureEngineer:
	def __init__(self, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
		self.test_size = test_size
		self.random_state = random_state

	def load_dataset(self, path: Path | None = None) -> pd.DataFrame:
		if path is None:
			path = PROCESSED_DATA_PATH if PROCESSED_DATA_PATH.exists() else None
		return load_data(path)

	def prepare_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		cleaned = clean_data(dataframe)
		engineered = engineer_features(cleaned)
		if TARGET_COLUMN not in engineered.columns:
			raise ValueError(f"Missing target column: {TARGET_COLUMN}")
		return engineered

	def prepare_dataset(self, dataframe: pd.DataFrame | None = None):
		if dataframe is None:
			dataframe = self.load_dataset()

		prepared = self.prepare_dataframe(dataframe)
		features, target = split_features_target(prepared)

		return train_test_split(
			features,
			target,
			test_size=self.test_size,
			random_state=self.random_state,
			stratify=target,
		)
