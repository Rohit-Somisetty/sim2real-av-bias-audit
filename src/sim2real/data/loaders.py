"""Reusable dataframe loaders with schema enforcement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from sim2real.data.schema import validate_frame_schema


class BaseLoader(ABC):
    """Base class for schema-enforcing loaders."""

    def __init__(
        self,
        path: Path | str,
        *,
        alias_map: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self.path = Path(path)
        self.alias_map = alias_map or {}

    def load(self) -> pd.DataFrame:
        df = self._read()
        if df.empty:
            raise ValueError(f"No rows found in {self.path}")
        df = self._apply_aliases(df)
        return validate_frame_schema(df)

    def _apply_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        renamed = df.copy()
        for canonical, aliases in self.alias_map.items():
            for alias in aliases:
                if alias in renamed.columns and canonical not in renamed.columns:
                    renamed = renamed.rename(columns={alias: canonical})
        return renamed

    @abstractmethod
    def _read(self) -> pd.DataFrame:
        """Return a raw dataframe without schema checks."""


class CsvFrameLoader(BaseLoader):
    """CSV loader with dtype hints and boolean parsing."""

    def __init__(self, path: Path | str, *, alias_map=None, **read_csv_kwargs) -> None:
        super().__init__(path, alias_map=alias_map)
        self.read_csv_kwargs = read_csv_kwargs

    def _read(self) -> pd.DataFrame:
        return pd.read_csv(self.path, **self.read_csv_kwargs)


class ParquetLoader(BaseLoader):
    """Parquet loader with schema validation."""

    def __init__(self, path: Path | str, *, alias_map=None, columns: Iterable[str] | None = None) -> None:
        super().__init__(path, alias_map=alias_map)
        self.columns = list(columns) if columns else None

    def _read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path, columns=self.columns)
