"""Shared test helpers: a synthetic model table and a writer fixture."""
import numpy as np
import pandas as pd
import pytest


def make_model_table(n: int = 900, seed: int = 5) -> pd.DataFrame:
    """Build a frame with the raw columns the experiment runners require."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.0, 1.0, size=n)
    leverage = rng.uniform(0.02, 0.5, size=n)
    logit = 1.0 - 5.0 * score + 3.0 * leverage
    target = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype("int64")

    return pd.DataFrame(
        {
            "sk_id_curr": np.arange(5000, 5000 + n),
            "target": target,
            "amt_income_total": rng.uniform(50000, 200000, size=n),
            "amt_credit": rng.uniform(20000, 400000, size=n),
            "amt_annuity": rng.uniform(5000, 30000, size=n),
            "days_birth": -rng.integers(7000, 25000, size=n).astype("float64"),
            "days_employed": -rng.integers(0, 14000, size=n).astype("float64"),
            "ext_source_1": score,
            "ext_source_2": np.clip(score + rng.normal(0, 0.1, n), 0.01, 0.99),
            "ext_source_3": rng.uniform(0.0, 1.0, size=n),
            "bureau_cnt": rng.integers(0, 10, size=n).astype("float64"),
            "bureau_debt_total": rng.uniform(0, 300000, size=n),
        }
    )


@pytest.fixture
def write_model_table(tmp_path):
    """Factory writing a synthetic model table into the test's temp folder."""

    def _write(name: str = "model_table.csv", n: int = 900, seed: int = 5):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        make_model_table(n=n, seed=seed).to_csv(path, index=False)
        return path

    return _write


@pytest.fixture(scope="session")
def model_table_builder():
    """Session-scoped access to the synthetic table builder."""
    return make_model_table
