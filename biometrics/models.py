"""Per-user verification models, score normalization, and quality-based fusion weights."""

import json
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def model_name(spec):
    """Build a deterministic model identifier from sorted configuration fields."""
    return "_".join(f"{k}-{v}" for k, v in sorted(spec.items()))


class Verifier:
    """Train one genuine-versus-other-user classifier for each enrolled identity."""
    def __init__(self, spec, seed=42):
        """Store the classifier specification and reproducibility seed."""
        self.spec = spec
        self.seed = seed

    def fit(self, X, y):
        """Fit the training scaler and one binary classifier per enrolled user."""
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.users = np.unique(y)
        self.models = []
        for user in self.users:
            binary = (y == user).astype(int)
            kind = self.spec["kind"]
            if kind == "logistic":
                m = LogisticRegression(
                    C=self.spec["C"],
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=self.seed,
                )
            elif kind == "svm":
                m = SVC(
                    C=self.spec["C"],
                    gamma=self.spec["gamma"],
                    kernel="rbf",
                    class_weight="balanced",
                )
            elif kind == "knn":
                if self.spec["k"] > len(X):
                    raise ValueError("k exceeds number of training samples")
                m = KNeighborsClassifier(
                    n_neighbors=self.spec["k"], weights=self.spec["weights"]
                )
            else:
                raise ValueError(f"Unknown classifier {kind}")
            m.fit(Z, binary)
            self.models.append(m)
        return self

    def raw_scores(self, X):
        """Return a sample-by-user matrix of margins or genuine-class probabilities."""
        Z = self.scaler.transform(X)
        result = []
        for m in self.models:
            result.append(
                m.decision_function(Z)
                if hasattr(m, "decision_function")
                else m.predict_proba(Z)[:, list(m.classes_).index(1)]
            )
        return np.column_stack(result)

    def calibrate(self, X):
        # Per-user score means and standard deviations are estimated from validation features.
        """Fit per-user score normalization parameters from validation features."""
        R = self.raw_scores(X)
        self.location = R.mean(axis=0)
        self.scale = np.maximum(R.std(axis=0), 1e-6)
        return self

    def scores(self, X):
        # A sigmoid maps standardized scores to [0, 1] for score-level fusion.
        # These similarities are not calibrated identity probabilities.
        """Return normalized similarities with columns ordered by enrolled user ID."""
        return expit((self.raw_scores(X) - self.location) / self.scale)


class QualityEstimator:
    """Estimate modality quality from training references and observable sample features."""

    def fit(self, face_quality, key_X):
        """Fit face-quality percentiles and robust keystroke references from training samples."""
        self.face_ref = np.sort(face_quality)
        self.key_center = np.median(key_X, axis=0)
        self.key_scale = np.maximum(
            1.4826 * np.median(abs(key_X - self.key_center), axis=0), 0.01
        )
        return self

    def face(self, raw):
        """Map image-quality values to training-reference percentiles bounded to [0.05, 1]."""
        return np.clip(
            np.searchsorted(self.face_ref, raw, side="right") / len(self.face_ref),
            0.05,
            1,
        )

    def key(self, X):
        """Convert robust timing deviations into quality scores bounded to [0.05, 1]."""
        z = np.abs((X - self.key_center) / self.key_scale)
        # Large timing deviations reduce this heuristic quality score.
        # The score does not measure emotion, fatigue, or timestamp reliability.
        return np.clip(np.exp(-np.quantile(z, 0.9, axis=1) / 3), 0.05, 1)


def adaptive_weight(qface, qkey, prior, power):
    """Normalize quality-adjusted modality weights and return the face contribution."""
    a = prior * np.asarray(qface) ** power
    b = (1 - prior) * np.asarray(qkey) ** power
    return a / np.maximum(a + b, 1e-12)
