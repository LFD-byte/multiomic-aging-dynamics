try:
    from .preprocessing.age import compute_precise_age
except Exception:
    compute_precise_age = None

try:
    from .preprocessing.covariate_adjust import adjust_for_covariates
except Exception:
    adjust_for_covariates = None

try:
    from .metrics.regression import regression_metrics
except Exception:
    regression_metrics = None

try:
    from .metrics.robustness import cross_validation
except Exception:
    cross_validation = None

try:
    from .training.nested_cv_oof import run_nested_cv_oof
except Exception:
    run_nested_cv_oof = None

__all__ = [
    "compute_precise_age",
    "adjust_for_covariates",
    "regression_metrics",
    "cross_validation",
    "run_nested_cv_oof",
]