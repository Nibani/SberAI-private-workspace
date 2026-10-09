"""Frozen projection tracking and retrospective L2 mean segmentation.

Boundaries are zero-based starts of new segments, excluding the terminal T.
No function interprets a label crossing or a fitted segment as an economic event.
"""
from __future__ import annotations

import numpy as np


def _finite(value, ndim, name):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if a.ndim != ndim or any(n == 0 for n in a.shape) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}D array")
    return a


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def projected_tracking(reference, monthly, centers, graph_labels, window=6):
    """Compare identical nearest-center maps q(reference) and q(month).

    Final-tail persistence counts an entity only when its final ``window``
    labels agree and differ from its projected baseline. Ties choose the first
    center. Graph labels are retained verbatim and never used as the baseline.
    """
    ref = _finite(reference, 2, "reference")
    x = _finite(monthly, 3, "monthly")
    c = _finite(centers, 2, "centers")
    g = _finite(graph_labels, 1, "graph_labels")
    window = _positive_int(window, "window")
    if (x.shape[0] != ref.shape[0] or x.shape[2] != ref.shape[1]
            or c.shape[1] != ref.shape[1] or g.shape != (len(ref),)
            or window > x.shape[1] or np.any(g != np.floor(g))
            or np.any(g < 0) or np.any(g >= len(c))):
        raise ValueError("Incompatible tracking shapes, labels or window")
    rd = np.square(ref[:, None] - c[None]).sum(axis=-1)
    md = np.square(x[:, :, None] - c[None, None]).sum(axis=-1)
    b = rd.argmin(axis=-1)
    labels = md.argmin(axis=-1)
    tail = labels[:, -window:]
    mask = np.all(tail == tail[:, :1], axis=1) & (tail[:, 0] != b)
    ordered = np.sort(md, axis=-1)
    margin = ordered[:, :, 1] - ordered[:, :, 0] if len(c) > 1 else np.zeros(labels.shape)
    return {"projected_baseline": b, "projected_labels": labels,
            "graph_labels": g.astype(int), "projection_disagreement": b != g,
            "persistent_changes": int(mask.sum()), "persistent_change_mask": mask,
            "boundary_crossed": labels != b[:, None],
            "switch_counts": np.count_nonzero(np.diff(labels, axis=1), axis=1),
            "distance_margin": margin}


def detect_mean_changes(signal, penalty, min_size=3):
    """Exact penalized L2 segmentation (ruptures PELT, jump=1).

    A persistent boundary has a following segment of at least six observations.
    This is batch retrospective segmentation; it has no online detection delay.
    """
    x = _finite(signal, 2, "signal")
    min_size = _positive_int(min_size, "min_size")
    if (isinstance(penalty, (bool, np.bool_))
            or not isinstance(penalty, (int, float, np.integer, np.floating))
            or not np.isfinite(penalty) or penalty <= 0 or len(x) < min_size):
        raise ValueError("Positive finite penalty and sufficient observations required")
    import ruptures as rpt
    if np.all(x == x[:1]):
        ends = [len(x)]
    else:
        ends = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(x).predict(pen=float(penalty))
    boundaries = ends[:-1]
    starts = [0] + boundaries
    segments = [{"start": a, "end": b, "length": b-a,
                 "mean": x[a:b].mean(axis=0).tolist()} for a, b in zip(starts, ends)]
    events = []
    for j, boundary in enumerate(boundaries):
        before, after = segments[j], segments[j+1]
        effect = np.asarray(after["mean"]) - np.asarray(before["mean"])
        events.append({"boundary": int(boundary), "segment_length": after["length"],
                       "effect_vector": effect.tolist(), "effect_norm": float(np.linalg.norm(effect)),
                       "persistent_profile_candidate": after["length"] >= 6,
                       "censored": after["length"] < 6})
    return {"boundaries": [int(b) for b in boundaries],
            "persistent_boundaries": [e["boundary"] for e in events if e["persistent_profile_candidate"]],
            "segments": segments, "events": events, "penalty": float(penalty),
            "min_size": min_size, "retrospective": True}


def fit_calendar(monthly, calibration_months=12):
    """Freeze common calendar offsets from the first complete annual cohort.

    Per-entity training means are removed, then the cohort mean is taken for
    each month. With one year this estimate also contains common training noise.
    """
    x = _finite(monthly, 3, "monthly")
    if calibration_months != 12 or x.shape[1] < 12:
        raise ValueError("A complete twelve-month calibration calendar is required")
    train = x[:, :12]
    return (train - train.mean(axis=1, keepdims=True)).mean(axis=0)


def temporal_channels(monthly, calendar):
    """Frozen calendar correction plus transductive fixed-cohort median channel."""
    x = _finite(monthly, 3, "monthly")
    s = _finite(calendar, 2, "calendar")
    if s.shape != (12, x.shape[2]) or x.shape[1] % 12:
        raise ValueError("Complete calendar cycles and matching dimensions required")
    absolute = x - s[np.arange(x.shape[1]) % 12][None]
    drift = np.median(absolute, axis=0)
    return {"absolute": absolute, "relative": absolute - drift[None], "common_drift": drift}


def train_variance(channel):
    """Pooled per-coordinate variance after entity centering, training df=11."""
    x = _finite(channel, 3, "channel")
    if x.shape[1] < 12:
        raise ValueError("Twelve training months required")
    train = x[:, :12]
    residual = train - train.mean(axis=1, keepdims=True)
    return float(np.square(residual).sum() / (len(x) * x.shape[2] * 11))


def penalty_value(c, variance, d, t=24):
    if not np.isfinite(c) or c <= 0 or not np.isfinite(variance) or variance < 0:
        raise ValueError("Invalid penalty calibration")
    return max(1e-12, float(c*d*variance*np.log(t)))


def match_boundaries(predicted, truth, tolerance=1):
    """Maximum-cardinality 1-to-1 monotone matching, nearest cost among ties.

    Dynamic programming avoids a greedy match stealing the only eligible
    boundary of a later truth. Returns (predicted, truth) pairs.
    """
    p, q = sorted(predicted), sorted(truth)
    table = {}
    for i in range(len(p)+1):
        for j in range(len(q)+1):
            if i == 0 or j == 0:
                table[i,j] = (0, 0, [])
                continue
            options = [table[i-1,j], table[i,j-1]]
            if abs(p[i-1]-q[j-1]) <= tolerance:
                n, cost, pairs = table[i-1,j-1]
                options.append((n+1, cost+abs(p[i-1]-q[j-1]), pairs+[(p[i-1],q[j-1])]))
            table[i,j] = max(options, key=lambda item: (item[0], -item[1]))
    return table[len(p),len(q)][2]
