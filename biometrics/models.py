"""Binary one-vs-rest verification models with explicit claimed-user scores."""

import json
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def model_name(spec):
    return "_".join(f"{k}-{v}" for k, v in sorted(spec.items()))


class Verifier:
    def __init__(self, spec, seed=42):
        self.spec = spec
        self.seed = seed

    def fit(self, X, y):
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
        # Label-free per-claimed-user score standardization, fitted on validation ONLY.
        R = self.raw_scores(X)
        self.location = R.mean(axis=0)
        self.scale = np.maximum(R.std(axis=0), 1e-6)
        return self

    def scores(self, X):
        # Logistic squashing after z-normalization: continuous comparable similarity scales.
        # This is score normalization, not an estimate of posterior match probability.
        return expit((self.raw_scores(X) - self.location) / self.scale)


class QualityEstimator:
    """Observable quality proxies; no illumination labels or injected noise level."""

    def fit(self, face_quality, key_X):
        self.face_ref = np.sort(face_quality)
        self.key_center = np.median(key_X, axis=0)
        self.key_scale = np.maximum(
            1.4826 * np.median(abs(key_X - self.key_center), axis=0), 0.01
        )
        return self

    def face(self, raw):
        return np.clip(
            np.searchsorted(self.face_ref, raw, side="right") / len(self.face_ref),
            0.05,
            1,
        )

    def key(self, X):
        z = np.abs((X - self.key_center) / self.key_scale)
        # High population timing outliers -> lower quality, a heuristic rather than
        # a validated measurement of emotion, fatigue, or timestamp reliability.
        return np.clip(np.exp(-np.quantile(z, 0.9, axis=1) / 3), 0.05, 1)


def adaptive_weight(qface, qkey, prior, power):
    a = prior * np.asarray(qface) ** power
    b = (1 - prior) * np.asarray(qkey) ** power
    return a / np.maximum(a + b, 1e-12)
