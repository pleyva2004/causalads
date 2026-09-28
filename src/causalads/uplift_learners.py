"""CATE learners for the uplift stage. Everything is fit on train; val is used only for
LightGBM early stopping (and, upstream, for choosing between learners)."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier

N_JOBS = 4
LEARNERS = ["S-learner", "T-learner", "X-learner", "DR-learner", "Causal forest"]
RESPONSE = "Response model"
_BASE = dict(learning_rate=0.05, num_leaves=15, min_child_samples=100, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=N_JOBS, verbose=-1)


@dataclass
class Fold:
    X: np.ndarray
    t: np.ndarray  # any email (0/1)
    a: np.ndarray  # arm 0/1/2
    y: np.ndarray


def _model(binary: bool, seed: int, n_estimators: int = 2000) -> LGBMClassifier | LGBMRegressor:
    return (LGBMClassifier if binary else LGBMRegressor)(n_estimators=n_estimators, random_state=seed, **_BASE)


def _fit_es(binary: bool, seed: int, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray):
    m = _model(binary, seed)
    m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def _pred(m, X: np.ndarray) -> np.ndarray:
    return m.predict_proba(X)[:, 1] if isinstance(m, LGBMClassifier) else m.predict(X)


def _propensity() -> DummyClassifier:
    # Randomized design: the true propensity is constant. The class prior of each
    # cross-fitting fold (~2/3, or 1/3 per arm) is the known design probability up to
    # sampling noise; a covariate propensity model would only fit noise.
    return DummyClassifier(strategy="prior")


def _nuisance_y(seed: int, n_trees: int) -> LGBMRegressor:
    return _model(False, seed, max(n_trees, 50))


def fit_binary(tr: Fold, va: Fold, binary: bool, seed: int) -> dict:
    """Fit S/T/X/DR/causal-forest CATE learners for any-email vs none, plus the response
    model P(y | X, emailed) and the control-outcome baseline mu0 (used by BLP).
    Returns {'predict': {name: fn(X) -> array}, 'cf': fitted CausalForestDML}."""
    s = _fit_es(binary, seed, np.c_[tr.X, tr.t], tr.y, np.c_[va.X, va.t], va.y)
    n_trees = int(s.best_iteration_ or s.n_estimators)  # val-selected complexity reused for econml nuisances

    t1, v1, t0, v0 = tr.t == 1, va.t == 1, tr.t == 0, va.t == 0
    mu1 = _fit_es(binary, seed, tr.X[t1], tr.y[t1], va.X[v1], va.y[v1])
    mu0 = _fit_es(binary, seed, tr.X[t0], tr.y[t0], va.X[v0], va.y[v0])

    # X-learner stage 2 on imputed individual effects; early stopping on val imputations.
    d1, d1v = tr.y[t1] - _pred(mu0, tr.X[t1]), va.y[v1] - _pred(mu0, va.X[v1])
    d0, d0v = _pred(mu1, tr.X[t0]) - tr.y[t0], _pred(mu1, va.X[v0]) - va.y[v0]
    tau1 = _fit_es(False, seed, tr.X[t1], d1, va.X[v1], d1v)
    tau0 = _fit_es(False, seed, tr.X[t0], d0, va.X[v0], d0v)
    g = float(tr.t.mean())  # propensity; Kunzel et al. weight tau0 by e(x)

    dr = DRLearner(model_propensity=_propensity(), model_regression=_nuisance_y(seed, n_trees),
                   model_final=LGBMRegressor(n_estimators=150, learning_rate=0.03, num_leaves=7,
                                             min_child_samples=500, subsample=0.8, subsample_freq=1,
                                             n_jobs=N_JOBS, verbose=-1, random_state=seed),
                   cv=3, random_state=seed)
    dr.fit(tr.y, tr.t, X=tr.X)

    cf = CausalForestDML(model_y=_nuisance_y(seed, n_trees), model_t=_propensity(),
                         discrete_treatment=True, cv=3, n_estimators=500, min_samples_leaf=50,
                         max_samples=0.45, honest=True, n_jobs=N_JOBS, random_state=seed)
    cf.fit(tr.y, tr.t, X=tr.X)

    def s_cate(X: np.ndarray) -> np.ndarray:
        n = len(X)
        return _pred(s, np.c_[X, np.ones(n)]) - _pred(s, np.c_[X, np.zeros(n)])

    predict = {
        "S-learner": s_cate,
        "T-learner": lambda X: _pred(mu1, X) - _pred(mu0, X),
        "X-learner": lambda X: g * _pred(tau0, X) + (1 - g) * _pred(tau1, X),
        "DR-learner": lambda X: np.ravel(dr.effect(X)),
        "Causal forest": lambda X: np.ravel(cf.const_marginal_effect(X)),
        RESPONSE: lambda X: _pred(mu1, X),
        "mu0": lambda X: _pred(mu0, X),
    }
    return {"predict": predict, "cf": cf, "n_trees": n_trees}


def fit_multiarm(tr: Fold, seed: int, n_trees: int) -> CausalForestDML:
    """One causal forest over the 3-level arm; const_marginal_effect -> [Mens-No, Womens-No]."""
    cf = CausalForestDML(model_y=_nuisance_y(seed, n_trees), model_t=_propensity(),
                         discrete_treatment=True, cv=3, n_estimators=500, min_samples_leaf=50,
                         max_samples=0.45, honest=True, n_jobs=N_JOBS, random_state=seed)
    cf.fit(tr.y, tr.a, X=tr.X)
    return cf
