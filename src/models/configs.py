"""Frozen model configurations shared by the holdout experiment runners.

Steps 8, 9 and 10 all claim to use "the same fixed configuration", so the
parameters live here instead of being re-typed in each script where they could
drift apart.
"""
from src.models.boosting import RiskBoostingModel
from src.models.logistic import RiskLogisticModel


LOGISTIC_PARAMETERS = {
    "n_bins": 5,
    "alpha": 0.5,
    "iv_threshold": 0.02,
    "C": 1.0,
    "max_iter": 2000,
}

BOOSTING_PARAMETERS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 100,
    "reg_lambda": 1.0,
    "n_jobs": 1,
}


def build_logistic_model(**overrides) -> RiskLogisticModel:
    """Step 7/8 logistic configuration."""
    return RiskLogisticModel(**{**LOGISTIC_PARAMETERS, **overrides})


def build_boosting_model(random_state: int = 42, **overrides) -> RiskBoostingModel:
    """Step 8 gradient-boosting configuration."""
    parameters = {**BOOSTING_PARAMETERS, "random_state": random_state}
    return RiskBoostingModel(**{**parameters, **overrides})
