"""Schema validation tests."""

import pandas as pd
import pytest

from sim2real.config import GeneratorConfig, REQUIRED_COLUMNS
from sim2real.data.schema import SchemaError, validate_frame_schema
from sim2real.data.synthetic import generate_paired_logs


def test_validate_frame_schema_succeeds() -> None:
    df = generate_paired_logs(GeneratorConfig(trips_per_domain=1))
    validated = validate_frame_schema(df)
    assert set(REQUIRED_COLUMNS).issubset(validated.columns)


def test_validate_frame_schema_missing_column_raises() -> None:
    df = pd.DataFrame({col: [] for col in REQUIRED_COLUMNS if col != "domain"})
    with pytest.raises(SchemaError):
        validate_frame_schema(df)
