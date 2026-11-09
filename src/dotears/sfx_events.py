"""Research tools for short SFX windows; classifications remain unconfirmed hypotheses."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from dotears.errors import ConfigurationError, DependencyError, MediaError

LABELS = {"whoosh", "notification", "ui_click", "other", "background"}
HEAD_SCHEMA = "dotears.sfx-linear-head/1"


def suppress_prefix_background(samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Wiener-style magnitude mask estimated from the first 100 ms, without boosting gain.

    This assumes the prefix is background-dominated. It is not source separation and may
    remove a foreground event already present in the prefix. The caller preserves raw audio.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if (
        samples.ndim != 1
        or not np.isfinite(samples).all()
        or not 0.2 * sample_rate <= len(samples) <= 10 * sample_rate
        or sample_rate != 16000
        or np.max(np.abs(samples)) > 1.00001
    ):
        raise ConfigurationError("Foreground probe needs finite 16 kHz mono audio, 0.2–10 s.")
    from scipy.ndimage import median_filter
    from scipy.signal import istft, stft

    _, times, spectrum = stft(
        samples, fs=sample_rate, nperseg=512, noverlap=384, boundary="zeros", padded=True
    )
    # Frame centers <=84 ms keep the complete 32 ms analysis window inside the prefix.
    prefix = (times >= 0.016) & (times <= 0.084)
    power = np.abs(spectrum) ** 2
    noise = np.quantile(power[:, prefix], 0.8, axis=1, keepdims=True)
    mask = np.sqrt(np.maximum(0.0, 1 - noise / np.maximum(power, 1e-20)))
    mask = np.clip(median_filter(mask, size=(3, 3), mode="nearest"), 0.05, 1.0)
    _, output = istft(spectrum * mask, fs=sample_rate, nperseg=512, noverlap=384)
    output = output[: len(samples)]
    # Bound the final peak without normalizing quiet residues upwards.
    output *= min(1.0, 1 / max(float(np.max(np.abs(output))), 1.0))
    if len(output) != len(samples) or not np.isfinite(output).all():
        raise MediaError("Foreground mask returned invalid audio.")
    return output.astype(np.float32)


def training_indices(rows: list[dict], query: dict) -> np.ndarray:
    """Exclude every derivative of the query family/source, plus identical PCM inputs."""
    indices = []
    for index, row in enumerate(rows):
        same_family = bool(query.get("family")) and row["family"] == query["family"]
        same_source = bool(query.get("source_sha256")) and (
            row["source_sha256"] == query["source_sha256"]
        )
        same_audio = row["audio_sha256"] == query["audio_sha256"]
        if not (same_family or same_source or same_audio):
            indices.append(index)
    return np.array(indices, dtype=int)


def balanced_source_weights(
    rows: list[dict], *, label_key: str = "human_label", allowed_labels: set[str] | None = None
) -> np.ndarray:
    """Equal class, family and source weight; synthetic views do not multiply evidence."""
    by_label = defaultdict(set)
    by_family = defaultdict(set)
    counts = Counter()
    labels = LABELS if allowed_labels is None else allowed_labels
    for row in rows:
        label, family, source = row[label_key], row["family"], row["source_sha256"]
        if label not in labels or not family or not source:
            raise ConfigurationError("Invalid training label or source family.")
        by_label[label].add(family)
        by_family[(label, family)].add(source)
        counts[(label, family, source)] += 1
    if not rows:
        raise ConfigurationError("An event classifier needs labeled training examples.")
    weights = []
    for row in rows:
        label, family, source = row[label_key], row["family"], row["source_sha256"]
        weights.append(
            1
            / (
                len(by_label[label])
                * len(by_family[(label, family)])
                * counts[(label, family, source)]
            )
        )
    values = np.array(weights, dtype=np.float64)
    return values / values.sum() * len(rows)


def fit_linear_head(features: np.ndarray, rows: list[dict]) -> dict:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(rows) or not np.isfinite(features).all():
        raise ConfigurationError("Training features and annotations do not match.")
    if len({row["human_label"] for row in rows}) < 3:
        raise ConfigurationError("The experimental multiclass head needs at least three classes.")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise DependencyError(
            "Event head training requires scikit-learn and threadpoolctl."
        ) from exc

    weights = balanced_source_weights(rows)
    with threadpool_limits(limits=4):
        scaler = StandardScaler().fit(features, sample_weight=weights)
        scaled = scaler.transform(features)
        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-4)
        classifier.fit(scaled, [row["human_label"] for row in rows], sample_weight=weights)
    if int(classifier.n_iter_.max()) >= 1000:
        raise MediaError("Event head did not converge under the frozen iteration budget.")
    return {
        "schema_version": HEAD_SCHEMA,
        "status": "research_only_not_calibrated",
        "classes": classifier.classes_.tolist(),
        "feature_count": features.shape[1],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_.tolist(),
        "intercepts": classifier.intercept_.tolist(),
        "iterations": classifier.n_iter_.tolist(),
        "training_examples": len(rows),
        "training_families": sorted({row["family"] for row in rows}),
        "training_source_hashes": sorted({row["source_sha256"] for row in rows}),
        "training_audio_hashes": sorted({row["audio_sha256"] for row in rows}),
        "sample_weighting": "equal_class_then_family_then_source_then_view",
        "C": 1.0,
    }


def predict_linear_head(head: dict, features: np.ndarray) -> list[dict]:
    if head.get("schema_version") != HEAD_SCHEMA:
        raise ConfigurationError("Unsupported SFX event head.")
    features = np.asarray(features, dtype=np.float64)
    classes = head["classes"]
    dimensions = head["feature_count"]
    mean, scale = np.array(head["mean"]), np.array(head["scale"])
    coefficients, bias = np.array(head["coefficients"]), np.array(head["intercepts"])
    if (
        features.ndim != 2
        or features.shape[1] != dimensions
        or mean.shape != (dimensions,)
        or scale.shape != (dimensions,)
        or coefficients.shape != (len(classes), dimensions)
        or bias.shape != (len(classes),)
        or len(set(classes)) != len(classes)
        or not 3 <= len(classes) <= len(LABELS)
        or set(classes) - LABELS
        or np.any(scale <= 0)
        or not all(
            np.isfinite(value).all() for value in (features, mean, scale, coefficients, bias)
        )
    ):
        raise ConfigurationError("Invalid event head parameters or input features.")
    scores = ((features - mean) / scale) @ coefficients.T + bias
    results = []
    for values in scores:
        order = np.argsort(-values, kind="stable")
        rankings = [{"label": classes[i], "raw_score": float(values[i])} for i in order]
        results.append(
            {
                "label": "unknown",
                "confidence": None,
                "status": "experimental_short_event_hypotheses_only",
                "best_match": rankings[0],
                "rankings": rankings,
                "score_kind": "uncalibrated_linear_decision_score_not_probability",
                "margin_to_second": float(values[order[0]] - values[order[1]]),
                "abstention_reason": "Independent real-mixture validation is still required.",
            }
        )
    return results
