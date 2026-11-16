import numpy as np
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from .features import FEATURE_COLUMNS


def train_models(df):
    X = df[FEATURE_COLUMNS].values
    y = df["is_fraud"].values

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    iso = IsolationForest(n_estimators=120, contamination=0.03, random_state=42)
    iso.fit(Xs)

    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
    weights = np.where(y == 1, 12.0, 1.0)
    clf.fit(X, y, sample_weight=weights)

    return {"iso": iso, "clf": clf, "scaler": scaler}


def score_transaction(features: dict, models) -> tuple[float, dict]:
    X = np.array([[features[c] for c in FEATURE_COLUMNS]])
    Xs = models["scaler"].transform(X)

    iso_raw = -models["iso"].score_samples(Xs)[0]
    iso_norm = float(np.clip((iso_raw + 0.4) / 0.9, 0, 1))
    proba = float(models["clf"].predict_proba(X)[0][1])

    risk = float(np.clip(0.6 * proba + 0.4 * iso_norm, 0, 1))
    return risk, {"supervised": proba, "unsupervised": iso_norm}