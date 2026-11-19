"""Real dataset loading and audited identity/split handling"""

from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from scipy.io import loadmat

SPLITS = ("train", "validation", "test")


def face_representation(image):
    """64x64 grayscale crop -> 16x16 low-frequency orthonormal DCT (256 values).

    Fixed transform, no learned preprocessing. Intensity retained so illumination
    remains a measurable condition. Exposure quality is observable, not a label.
    """
    a = (
        np.asarray(
            image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), float
        )
        / 255
    )
    features = dctn(a, norm="ortho")[:16, :16].ravel()
    # Penalize under/overexposure, low contrast, and strong side-light imbalance.
    exposure = np.exp(-(((a.mean() - 0.5) / 0.3) ** 2))
    contrast = min(a.std() / 0.18, 1.0)
    unclipped = np.mean((a > 0.03) & (a < 0.97))
    symmetry = np.exp(-abs(a[:, :32].mean() - a[:, 32:].mean()) / 0.25)
    quality = max(0.01, float(exposure * contrast * unclipped * symmetry))
    return features, quality


def prepare_sof(image_root, metadata_mat, out, seed=42, gap=1, part="2"):
    """Read actual SoF MATLAB structs. Only original *_o.jpg images are accepted.

    Each identity/illumination stratum is ordered by imageSequence; allocate
    50/25/25 percent, purging `gap` last frames from train and validation.
    This is a within-collection split, NOT independent recording sessions.
    """
    if gap < 1:
        raise ValueError("Use a sequence gap of at least one original image")
    root = Path(image_root).resolve()
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = loadmat(metadata_mat, simplify_cells=True)["metadata"]
    if isinstance(metadata, dict):
        metadata = [metadata]
    files = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.stem.endswith("_o"):
            key = tuple(p.stem.split("_")[:2])
            if key in files:
                raise ValueError(f"Ambiguous original image: {key}")
            files[key] = p
    rows = []
    missing = []
    seen = {}
    for r in metadata:
        if str(r["part"]) != str(part):
            continue
        key = (str(r["userId"]), str(r["imageSequence"]).zfill(5))
        if key not in files:
            missing.append(key)
            continue
        p = files[key]
        # Hash decoded pixels: catches renamed originals and recompression-free copies.
        with Image.open(p) as im:
            digest = hashlib.sha256(np.asarray(im.convert("RGB")).tobytes()).hexdigest()
        if digest in seen:
            continue
        seen[digest] = str(p)
        x, y, w, h = np.asarray(r["faceRect"], float).ravel()
        # MATLAB rectangles use 1-based coordinates; PIL uses 0-based coordinates.
        rows.append(
            dict(
                path=os.path.relpath(p, out.parent),
                user_id=key[0],
                sequence=int(key[1]),
                illumination="good" if int(r["illuminationQuality"]) == 1 else "poor",
                x=x - 1,
                y=y - 1,
                width=w,
                height=h,
                group_id=digest,
                source_part=str(part),
            )
        )
    if missing:
        raise ValueError(
            f"{len(missing)} original images missing; first: {missing[:3]}"
        )
    if not rows:
        raise ValueError("No SoF original images matched metadata.")
    df = pd.DataFrame(rows)
    chunks = []
    for (_, _), group in df.groupby(["user_id", "illumination"], sort=True):
        group = group.sort_values("sequence").copy()
        n = len(group)
        a = n // 2
        b = a + n // 4
        for split, lo, hi in [
            ("train", 0, a - gap),
            ("validation", a, b - gap),
            ("test", b, n),
        ]:
            if hi > lo:
                g = group.iloc[lo:hi].copy()
                g["split"] = split
                chunks.append(g)
    manifest = pd.concat(chunks, ignore_index=True)
    # Suppress cross-condition immediate neighbors in different partitions too.
    # Remove training first, then validation, preserving held-out test observations.
    rank = {"train": 0, "validation": 1, "test": 2}
    drop = set()
    for _, g in manifest.groupby("user_id"):
        g = g.sort_values("sequence")
        for i, ri in g.iterrows():
            for j, rj in g.loc[
                g.sequence.between(ri.sequence - gap, ri.sequence + gap)
            ].iterrows():
                if ri["split"] != rj["split"] and rank[ri["split"]] < rank[rj["split"]]:
                    drop.add(i)
    manifest = manifest.drop(index=list(drop))
    manifest.to_csv(out, index=False)
    counts = (
        manifest.groupby(["user_id", "split", "illumination"])
        .size()
        .rename("n")
        .reset_index()
    )
    counts.to_csv(out.with_name(out.stem + "_counts.csv"), index=False)
    return manifest


def load_faces(manifest_path):
    path = Path(manifest_path).resolve()
    df = pd.read_csv(path, dtype={"user_id": str, "group_id": str})
    required = {
        "path",
        "user_id",
        "split",
        "illumination",
        "group_id",
        "x",
        "y",
        "width",
        "height",
    }
    if required - set(df):
        raise ValueError(f"Face manifest lacks {required-set(df)}")
    if not set(df.split) <= set(SPLITS):
        raise ValueError("Unknown face split")
    if not set(df.illumination) <= {"good", "poor"}:
        raise ValueError("Illumination must be good or poor")
    if df.groupby("group_id").split.nunique().max() > 1:
        raise ValueError("Face source group crosses splits")
    features = []
    qualities = []
    hash_splits = {}
    for r in df.itertuples():
        p = Path(r.path)
        p = p if p.is_absolute() else path.parent / p
        with Image.open(p) as im:
            im = im.convert("RGB")
            digest = hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
            if digest in hash_splits:
                raise ValueError(
                    f"Duplicate decoded face image: {p}; remove duplicates before running"
                )
            hash_splits[digest] = r.split
            x, y, w, h = map(float, (r.x, r.y, r.width, r.height))
            if not np.all(np.isfinite([x, y, w, h])) or min(w, h) <= 0:
                raise ValueError(f"Invalid face box: {p}")
            box = (
                max(0, int(np.floor(x))),
                max(0, int(np.floor(y))),
                min(im.width, int(np.ceil(x + w))),
                min(im.height, int(np.ceil(y + h))),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Face box outside image: {p}")
            f, q = face_representation(im.crop(box))
            features.append(f)
            qualities.append(q)
    X = np.asarray(features)
    q = np.asarray(qualities)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite face features")
    return df.reset_index(drop=True), X, q


def load_cmu(path):
    df = pd.read_csv(path, dtype={"subject": str})
    required = {"subject", "sessionIndex", "rep"}
    if required - set(df):
        raise ValueError(f"CMU CSV lacks {required-set(df)}")
    cols = [c for c in df if c.startswith(("H.", "DD.", "UD."))]
    if len(cols) != 31:
        raise ValueError(f"Expected 31 CMU timing columns, found {len(cols)}")
    if df.duplicated(["subject", "sessionIndex", "rep"]).any():
        raise ValueError("Duplicate CMU attempt identifiers")
    df = df.sort_values(["subject", "sessionIndex", "rep"]).reset_index(drop=True)
    df["split"] = np.where(
        df.sessionIndex <= 4,
        "train",
        np.where(df.sessionIndex <= 6, "validation", "test"),
    )
    if not df.sessionIndex.isin(range(1, 9)).all():
        raise ValueError("CMU sessions must be 1..8")
    X = df[cols].to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite keystroke values")
    validate_timings(X, cols)
    return df, X, cols


def timing_dependencies(cols):
    """Match full key names: the official CMU key 'Shift.r' contains a dot."""
    keys = [c[2:] for c in cols if c.startswith("H.")]
    for dd in (c for c in cols if c.startswith("DD.")):
        matches = [(a, b) for a in keys for b in keys if dd == "DD." + a + "." + b]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous timing digraph: {dd}")
        a, b = matches[0]
        ud = "UD." + a + "." + b
        if ud not in cols:
            raise ValueError(f"Missing dependent feature: {ud}")
        yield cols.index(dd), cols.index("H." + a), cols.index(ud)


def validate_timings(X, cols):
    for c in cols:
        if c.startswith(("H.", "DD.")) and np.any(X[:, cols.index(c)] < 0):
            raise ValueError(f"Negative hold/down-down timing: {c}")
    for dd, h, ud in timing_dependencies(cols):
        if not np.allclose(X[:, dd], X[:, h] + X[:, ud], atol=3e-4):
            raise ValueError(f"Inconsistent dependent timing: {cols[dd]}")


def degrade_timings(X, cols, noise_ms, rng):
    """Perturb hold and keydown intervals; rebuild dependent UD = DD - H.

    Noise is in seconds (ms/1000). Positive intervals clipped at 1 ms. Negative
    UD is legitimate overlapping typing and is retained. Same random draw scaled
    by severity in the experiment driver, creating paired severity conditions.
    """
    if noise_ms == 0:
        return X.copy()
    Y = X.copy()
    sigma = noise_ms / 1000
    for j, c in enumerate(cols):
        if c.startswith(("H.", "DD.")):
            Y[:, j] = np.maximum(0.001, X[:, j] + rng.normal(0, sigma, len(X)))
    for dd, h, ud in timing_dependencies(cols):
        Y[:, ud] = Y[:, dd] - Y[:, h]
    return Y


def user_mapping(faces, keys, n_users, seed):
    eligible = []
    for u, g in faces.groupby("user_id"):
        counts = g.groupby(["split", "illumination"]).size()
        if (
            all(counts.get((s, i), 0) >= 1 for s in SPLITS for i in ("good", "poor"))
            and sum(g.split == "train") >= 8
        ):
            eligible.append(u)
    kusers = [u for u, g in keys.groupby("subject") if set(g.split) == set(SPLITS)]
    n = min(n_users, len(eligible), len(kusers))
    if n < 3:
        raise ValueError(
            f"Only {n} usable users. Need >=3 with both lighting labels in each face split. Inspect manifest counts."
        )
    rng = np.random.default_rng(seed)
    # Keep face cohort fixed across seeds; vary artificial CMU subset and identity mapping.
    eligible = sorted(eligible)[:n]
    chosen = rng.choice(sorted(kusers), n, replace=False)
    return pd.DataFrame(
        {"pseudo_user": np.arange(n), "face_user": eligible, "keystroke_user": chosen}
    )
