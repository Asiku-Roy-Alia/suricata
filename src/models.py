"""Model components for the hybrid IDS.

Contains:
  * CalibratedOneClassSVM: wraps SGDOneClassSVM so its raw decision function is
    mapped to a calibrated probability via Platt scaling on a held-out set
    containing both benign and attack samples. This addresses the heterogeneous
    base-learner problem flagged in the proposal evaluation, where the
    unbounded OCSVM output cannot be meaningfully combined with a supervised
    classifier inside a stacking meta-learner.
  * HybridStackedClassifier: the top-level hybrid model. Trains a supervised
    linear SVM and the calibrated anomaly detector, then a Logistic Regression
    meta-learner on their out-of-fold predictions.
  * Baselines: linear SVC alone and Isolation Forest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDOneClassSVM
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# Calibrated One-Class SVM
# ---------------------------------------------------------------------------
class CalibratedOneClassSVM(BaseEstimator, ClassifierMixin):
    """SGDOneClassSVM with a sigmoid calibrator fitted on a mixed validation set.

    During fit() we train the one-class detector on benign data only, exactly
    as the methodology requires. During calibrate() we then fit a sigmoid from
    the raw decision function to a binary attack-probability using a held-out
    validation set that contains both benign and attack samples. The sigmoid is
    equivalent to Platt scaling and produces outputs in [0, 1] that can be
    meaningfully combined with the supervised base learner inside the stack.
    """

    def __init__(self, nu: float = 0.1, random_state: int = 42, max_iter: int = 2000):
        self.nu = nu
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X_benign, y=None):
        """Fit the anomaly detector on benign rows only."""
        self.detector_ = SGDOneClassSVM(
            nu=self.nu,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=1e-4,
        )
        self.detector_.fit(X_benign)
        self.is_calibrated_ = False
        self.classes_ = np.array([0, 1])
        return self

    def raw_score(self, X) -> np.ndarray:
        """Return the unbounded decision function. Higher means more benign."""
        return self.detector_.decision_function(X)

    def calibrate(self, X_val, y_val_binary, method: str = "sigmoid") -> "CalibratedOneClassSVM":
        """Fit the calibrator on held-out mixed data.

        y_val_binary uses the same convention as the rest of the project:
        1 for attack, 0 for benign. Because the raw score is higher for benign,
        we negate it so that higher always means more anomalous.
        """
        anomaly_score = -self.raw_score(X_val)

        if method == "sigmoid":
            # Platt-style calibration: fit a logistic function from score to P(attack).
            self.calibrator_ = LogisticRegression(C=1.0, max_iter=1000)
            self.calibrator_.fit(anomaly_score.reshape(-1, 1), y_val_binary)
        elif method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(anomaly_score, y_val_binary)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self.calibration_method_ = method
        self.is_calibrated_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not getattr(self, "is_calibrated_", False):
            raise RuntimeError("Call .calibrate(X_val, y_val_binary) before predict_proba().")
        anomaly_score = -self.raw_score(X)
        if self.calibration_method_ == "sigmoid":
            p_attack = self.calibrator_.predict_proba(anomaly_score.reshape(-1, 1))[:, 1]
        else:
            p_attack = self.calibrator_.predict(anomaly_score)
            p_attack = np.clip(p_attack, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - p_attack, p_attack])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Calibrated Linear SVC
# ---------------------------------------------------------------------------
def build_calibrated_linear_svc(C: float, class_weight: str, max_iter: int, seed: int, cv: int = 3):
    """LinearSVC wrapped in CalibratedClassifierCV to expose probabilities."""
    base = LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        dual="auto",
        random_state=seed,
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=cv)


# ---------------------------------------------------------------------------
# Hybrid stacked classifier
# ---------------------------------------------------------------------------
@dataclass
class HybridArtifacts:
    """Trained components of the hybrid model."""

    supervised: object
    anomaly: CalibratedOneClassSVM
    meta: LogisticRegression
    best_params: Dict[str, object] = field(default_factory=dict)


class HybridStackedClassifier:
    """Stacked hybrid: supervised + calibrated anomaly detector + LR meta-learner.

    The stacking strategy uses out-of-fold predictions from the supervised
    component for the meta-learner, while the anomaly detector's calibrated
    probabilities come from a single held-out validation pass (the detector is
    not refit inside CV since it trains on benign data only). This is a common
    practical simplification that preserves calibration quality.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.cfg = config
        self.logger = logger
        self.artifacts_: Optional[HybridArtifacts] = None

    def _search_supervised(self, X_train, y_train, cv: int):
        """Tiny grid search over C for the supervised base learner."""
        best = None
        for C in self.cfg["models"]["supervised"]["C_grid"]:
            model = build_calibrated_linear_svc(
                C=C,
                class_weight=self.cfg["models"]["supervised"]["class_weight"],
                max_iter=self.cfg["models"]["supervised"]["max_iter"],
                seed=self.cfg["seed"],
                cv=cv,
            )
            # Use StratifiedKFold to score
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.cfg["seed"])
            from sklearn.metrics import f1_score
            scores = []
            for tr_idx, va_idx in skf.split(X_train, y_train):
                m = clone(model)
                m.fit(X_train[tr_idx], y_train[tr_idx])
                preds = m.predict(X_train[va_idx])
                scores.append(f1_score(y_train[va_idx], preds, average="macro"))
            score = float(np.mean(scores))
            self.logger.info("  supervised C=%.2f -> macro F1 %.4f", C, score)
            if best is None or score > best[0]:
                best = (score, C)
        self.logger.info("Best supervised C: %.2f (macro F1 %.4f)", best[1], best[0])
        final = build_calibrated_linear_svc(
            C=best[1],
            class_weight=self.cfg["models"]["supervised"]["class_weight"],
            max_iter=self.cfg["models"]["supervised"]["max_iter"],
            seed=self.cfg["seed"],
            cv=cv,
        )
        final.fit(X_train, y_train)
        return final, best[1]

    def _search_anomaly(self, X_benign_train, X_val, y_val):
        """Small search over nu for the anomaly detector."""
        from sklearn.metrics import roc_auc_score
        best = None
        for nu in self.cfg["models"]["anomaly"]["nu_grid"]:
            detector = CalibratedOneClassSVM(nu=nu, random_state=self.cfg["seed"])
            detector.fit(X_benign_train)
            detector.calibrate(X_val, y_val, method="sigmoid")
            probs = detector.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, probs)
            self.logger.info("  anomaly nu=%.3f -> ROC-AUC %.4f", nu, score)
            if best is None or score > best[0]:
                best = (score, nu, detector)
        self.logger.info("Best anomaly nu: %.3f (ROC-AUC %.4f)", best[1], best[0])
        return best[2], best[1]

    def fit(self, X_train, y_train, X_val, y_val, cv: int = 3) -> HybridArtifacts:
        """Train the full stack."""
        self.logger.info("Training supervised component")
        supervised, best_C = self._search_supervised(X_train, y_train, cv)

        self.logger.info("Training anomaly component on benign rows")
        benign_mask = (y_train == 0)
        X_benign = X_train[benign_mask]
        max_benign = self.cfg["models"]["anomaly"]["max_benign_rows"]
        if len(X_benign) > max_benign:
            rng = np.random.default_rng(self.cfg["seed"])
            idx = rng.choice(len(X_benign), size=max_benign, replace=False)
            X_benign = X_benign[idx]
            self.logger.info("Subsampled benign training set to %d rows", max_benign)

        anomaly, best_nu = self._search_anomaly(X_benign, X_val, y_val)

        self.logger.info("Training meta-learner on stacked features")
        sup_proba = supervised.predict_proba(X_val)[:, 1]
        ano_proba = anomaly.predict_proba(X_val)[:, 1]
        stack_val = np.column_stack([sup_proba, ano_proba])

        best_meta = None
        for C in self.cfg["models"]["meta_learner"]["C_grid"]:
            meta = LogisticRegression(
                C=C,
                max_iter=self.cfg["models"]["meta_learner"]["max_iter"],
                random_state=self.cfg["seed"],
            )
            # 3-fold CV on the validation stack
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(meta, stack_val, y_val, cv=3, scoring="f1_macro")
            score = float(np.mean(scores))
            self.logger.info("  meta C=%.2f -> macro F1 %.4f", C, score)
            if best_meta is None or score > best_meta[0]:
                best_meta = (score, C)

        self.logger.info("Best meta C: %.2f (macro F1 %.4f)", best_meta[1], best_meta[0])
        meta_final = LogisticRegression(
            C=best_meta[1],
            max_iter=self.cfg["models"]["meta_learner"]["max_iter"],
            random_state=self.cfg["seed"],
        )
        meta_final.fit(stack_val, y_val)

        self.artifacts_ = HybridArtifacts(
            supervised=supervised,
            anomaly=anomaly,
            meta=meta_final,
            best_params={"C_supervised": best_C, "nu_anomaly": best_nu, "C_meta": best_meta[1]},
        )
        return self.artifacts_

    def predict_proba(self, X) -> np.ndarray:
        if self.artifacts_ is None:
            raise RuntimeError("Call fit() first.")
        sup = self.artifacts_.supervised.predict_proba(X)[:, 1]
        ano = self.artifacts_.anomaly.predict_proba(X)[:, 1]
        stack = np.column_stack([sup, ano])
        return self.artifacts_.meta.predict_proba(stack)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------
def fit_linear_svc_baseline(X_train, y_train, config: dict, logger: logging.Logger):
    """Plain calibrated linear SVC baseline."""
    logger.info("Fitting Linear SVC baseline")
    model = build_calibrated_linear_svc(
        C=1.0,
        class_weight=config["models"]["supervised"]["class_weight"],
        max_iter=config["models"]["supervised"]["max_iter"],
        seed=config["seed"],
    )
    model.fit(X_train, y_train)
    return model


def fit_isolation_forest_baseline(X_train, y_train, config: dict, logger: logging.Logger):
    """Isolation Forest baseline trained on benign rows only."""
    logger.info("Fitting Isolation Forest baseline on benign rows only")
    benign = X_train[y_train == 0]
    max_rows = min(100_000, len(benign))
    if len(benign) > max_rows:
        rng = np.random.default_rng(config["seed"])
        idx = rng.choice(len(benign), size=max_rows, replace=False)
        benign = benign[idx]
    model = IsolationForest(
        n_estimators=config["models"]["isolation_forest"]["n_estimators"],
        contamination=config["models"]["isolation_forest"]["contamination"],
        random_state=config["seed"],
        n_jobs=-1,
    )
    model.fit(benign)
    return model


def isolation_forest_predict(model, X) -> np.ndarray:
    """Convert IF output to binary labels: 1 for attack, 0 for benign."""
    raw = model.predict(X)
    return (raw == -1).astype(int)


def isolation_forest_score(model, X) -> np.ndarray:
    """Return an attack-likeness score in [0, 1] for IF."""
    s = -model.score_samples(X)  # higher score = more anomalous
    s_min, s_max = float(s.min()), float(s.max())
    if s_max - s_min < 1e-9:
        return np.zeros_like(s)
    return (s - s_min) / (s_max - s_min)
