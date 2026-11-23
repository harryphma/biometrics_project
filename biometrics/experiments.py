"""RQ1 classifier comparisons, RQ2 fixed fusion, and RQ3 quality-adaptive fusion."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from .data import load_faces, load_cmu, user_mapping, degrade_timings
from .models import Verifier, QualityEstimator, adaptive_weight, model_name
from .evaluation import eer, far_threshold, rates_at, measure, plot_hw3


def json_write(path, data):
    """Write indented JSON, converting NumPy scalar values to native Python values."""
    Path(path).write_text(
        json.dumps(
            data, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)
        )
    )


def flatten_scores(matrix, actual):
    """Flatten sample-by-user scores and label matching actual and claimed IDs as genuine."""
    claims = np.broadcast_to(np.arange(matrix.shape[1]), matrix.shape)
    return (claims == np.asarray(actual)[:, None]).astype(int).ravel(), matrix.ravel()


def thresholds(y, s, cfg):
    """Derive EER and target-FAR operating thresholds from validation trials."""
    return {
        "val_eer": eer(y, s, cfg["num_thresholds"])[1],
        "target": far_threshold(y, s, cfg["target_far"]),
    }


def mapped_data(frame, X, mapping, source, column, quality=None):
    """Filter a modality to mapped users while preserving feature and metadata alignment."""
    lookup = dict(zip(mapping[source], mapping.pseudo_user))
    mask = frame[column].isin(lookup)
    f = frame.loc[mask].copy().reset_index(drop=True)
    f["pseudo_user"] = f[column].map(lookup).astype(int)
    return {
        "df": f,
        "X": X[mask.to_numpy()],
        "q": quality[mask.to_numpy()] if quality is not None else None,
    }


def indices(data, split):
    """Return feature-row indices belonging to the requested sample partition."""
    return np.flatnonzero(data["df"].split.to_numpy() == split)


def make_pairs(F, K, split, seed, max_pairs, balanced=False):
    """Pair face and keystroke observations from the same pseudo-user and partition.

    Samples are drawn without replacement within each pool. Each pair is later
    compared with every claimed user to form genuine and impostor attempts.
    Balanced sampling uses equal good- and poor-illumination counts per user and
    reuses keystroke probes across illumination groups for paired comparisons.
    """
    rng = np.random.default_rng(seed)
    records = []
    f = F["df"]
    k = K["df"]
    for u in sorted(f.pseudo_user.unique()):
        fs = f.index[(f.split == split) & (f.pseudo_user == u)].to_numpy()
        ks = k.index[(k.split == split) & (k.pseudo_user == u)].to_numpy()
        if balanced:
            good = fs[f.loc[fs, "illumination"].to_numpy() == "good"]
            poor = fs[f.loc[fs, "illumination"].to_numpy() == "poor"]
            n = min(len(good), len(poor), len(ks), max_pairs)
            if n < 1:
                raise ValueError(
                    f"User {u} lacks paired lighting observations in {split}"
                )
            kk = rng.choice(ks, n, replace=False)
            for light, pool in [("good", good), ("poor", poor)]:
                for j, (fi, ki) in enumerate(
                    zip(rng.choice(pool, n, replace=False), kk)
                ):
                    records.append(
                        {
                            "actual_user": u,
                            "face_index": fi,
                            "key_index": ki,
                            "lighting": light,
                            "pair_id": f"{u}_{j}",
                        }
                    )
        else:
            n = min(len(fs), len(ks), max_pairs)
            for j, (fi, ki) in enumerate(
                zip(rng.choice(fs, n, replace=False), rng.choice(ks, n, replace=False))
            ):
                records.append(
                    {
                        "actual_user": u,
                        "face_index": fi,
                        "key_index": ki,
                        "lighting": f.loc[fi, "illumination"],
                        "pair_id": f"{u}_{j}",
                    }
                )
    return pd.DataFrame(records)


def trial_frame(pairs, face_scores, key_scores, qf, qk, n_users):
    """Expand observation pairs into claimed-user trials with labels, scores, and qualities."""
    rows = pairs.loc[pairs.index.repeat(n_users)].reset_index(drop=True)
    rows["claimed_user"] = np.tile(np.arange(n_users), len(pairs))
    rows["label"] = (rows.actual_user == rows.claimed_user).astype(int)
    fi = rows.face_index.to_numpy(int)
    ki = rows.key_index.to_numpy(int)
    c = rows.claimed_user.to_numpy(int)
    rows["face_score"] = face_scores[fi, c]
    rows["key_score"] = key_scores[ki, c]
    rows["face_quality"] = qf[fi]
    rows["key_quality"] = qk[ki]
    return rows


def save_result(
    out,
    rq,
    name,
    validation,
    test,
    val_scores,
    test_scores,
    cfg,
    seed,
    extra=None,
    plot=False,
    locked=None,
):
    """Evaluate and save one comparison using validation-derived or supplied thresholds.

    Outputs include pooled metrics, labeled trial scores, per-user metrics,
    optional HW3 plots, and the operating thresholds used for testing.
    """
    yv = validation.label.to_numpy()
    yt = test.label.to_numpy()
    th = thresholds(yv, val_scores, cfg) if locked is None else locked
    metric, curve = measure(yt, test_scores, th, cfg["num_thresholds"])
    vm, _ = measure(yv, val_scores, th, cfg["num_thresholds"])
    row = {
        "seed": seed,
        "RQ": rq,
        "system": name,
        **(extra or {}),
        **metric,
        "validation_EER": vm["EER"],
        "validation_target_FAR": vm["target_FAR"],
        "target_far_requested": cfg["target_far"],
        "target_far_met_on_test": metric["target_FAR"] <= cfg["target_far"],
    }
    target = out / rq / name
    target.mkdir(parents=True, exist_ok=True)
    curve.to_csv(target / "test_curve.csv", index=False)
    for split, frame, scores in [
        ("validation", validation, val_scores),
        ("test", test, test_scores),
    ]:
        t = frame.copy()
        t["score"] = scores
        t["accept_at_target"] = np.asarray(scores) >= th["target"]
        t.to_csv(target / f"{split}_trials.csv.gz", index=False, compression="gzip")
        np.savetxt(target / f"{split}_genuine.txt", np.asarray(scores)[t.label == 1])
        np.savetxt(target / f"{split}_impostor.txt", np.asarray(scores)[t.label == 0])
    # Per-user metrics complement pooled errors with an equally weighted mean user EER.
    user_rows = []
    for u, g in test.groupby("claimed_user"):
        idx = g.index.to_numpy()
        m, _ = measure(
            g.label.to_numpy(), np.asarray(test_scores)[idx], th, cfg["num_thresholds"]
        )
        user_rows.append({"claimed_user": int(u), **m})
    pd.DataFrame(user_rows).to_csv(target / "per_user.csv", index=False)
    row["macro_user_EER"] = float(np.mean([r["EER"] for r in user_rows]))
    if plot and cfg["plots"]:
        plot_hw3(yt, test_scores, name, target / "plots", cfg["num_thresholds"])
    return row, th


def rq1(F, K, cfg, out, seed):
    """Evaluate configured classifiers per modality and select models by validation EER."""
    chosen = {}
    rows = []
    selection = {}
    for modality, data in [("face", F), ("keystroke", K)]:
        tr = indices(data, "train")
        va = indices(data, "validation")
        te = indices(data, "test")
        labels = data["df"].pseudo_user.to_numpy()
        best = None
        candidates = []
        for spec in cfg["models"]:
            name = model_name(spec)
            model = (
                Verifier(spec, seed)
                .fit(data["X"][tr], labels[tr])
                .calibrate(data["X"][va])
            )
            yv, sv = flatten_scores(model.scores(data["X"][va]), labels[va])
            ve = eer(yv, sv, cfg["num_thresholds"])[0]
            candidates.append(
                {"modality": modality, "model": name, "validation_EER": ve}
            )
            # Model selection depends only on validation EER; test metrics are recorded separately.
            if best is None or ve < best[0]:
                best = (ve, model, name)
            th = thresholds(yv, sv, cfg)
            yt, st = flatten_scores(model.scores(data["X"][te]), labels[te])
            metrics, curve = measure(yt, st, th, cfg["num_thresholds"])
            row = {
                "seed": seed,
                "RQ": "RQ1",
                "system": modality + "_" + name,
                "modality": modality,
                "validation_EER": ve,
                "target_far_requested": cfg["target_far"],
                **metrics,
            }
            row["target_far_met_on_test"] = row["target_FAR"] <= cfg["target_far"]
            rows.append(row)
            path = out / "RQ1" / modality / name
            path.mkdir(parents=True, exist_ok=True)
            curve.to_csv(path / "test_curve.csv", index=False)
        chosen[modality] = best[1]
        selection[modality] = {
            "name": best[2],
            "validation_EER": best[0],
            "spec": best[1].spec,
        }
        for r in rows:
            if r["modality"] == modality:
                r["selected_on_validation"] = r["system"] == modality + "_" + best[2]
        for split, ii in [("validation", va), ("test", te)]:
            mat = best[1].scores(data["X"][ii])
            y, s = flatten_scores(mat, labels[ii])
            trial = pd.DataFrame(
                {
                    "source_index": np.repeat(ii, len(best[1].users)),
                    "actual_user": np.repeat(labels[ii], len(best[1].users)),
                    "claimed_user": np.tile(best[1].users, len(ii)),
                    "label": y,
                    "score": s,
                }
            )
            p = out / "RQ1" / modality
            trial.to_csv(
                p / f"selected_{split}_trials.csv.gz", index=False, compression="gzip"
            )
            if split == "test" and cfg["plots"]:
                plot_hw3(
                    y, s, modality + "_selected", p / "plots", cfg["num_thresholds"]
                )
        print(
            f"RQ1 {modality}: selected {best[2]}, validation EER={best[0]:.4f}",
            flush=True,
        )
    json_write(out / "selected_models.json", selection)
    return chosen, rows


def paired_bootstrap_difference(
    fixed, adaptive, threshold_fixed, threshold_adaptive, path, seed, n_boot=1000
):
    """Estimate adaptive-minus-fixed FAR and FRR differences by resampling claimed users.

    Each method uses its frozen operational threshold. Claimed users are sampled
    with replacement, preserving paired method differences within each user.
    The resulting intervals are exploratory because probes are shared across
    claims and repeated identity mappings reuse source participants.
    """
    per = []
    for u, g in fixed.groupby("claimed_user"):
        a = adaptive.loc[g.index]
        f = rates_at(g.label, g.score, threshold_fixed)
        r = rates_at(a.label, a.score, threshold_adaptive)
        per.append([r["FAR"] - f["FAR"], r["FRR"] - f["FRR"]])
    per = np.asarray(per)
    rng = np.random.default_rng(seed)
    samples = per[rng.integers(0, len(per), (n_boot, len(per)))].mean(axis=1)
    result = {
        k: {
            "mean_difference": float(per[:, j].mean()),
            "ci95_low": float(np.quantile(samples[:, j], 0.025)),
            "ci95_high": float(np.quantile(samples[:, j], 0.975)),
        }
        for j, k in enumerate(("FAR", "FRR"))
    }
    json_write(path, result)


def run_seed(face_df, face_X, face_q, key_df, key_X, key_cols, cfg, seed, root):
    """Execute RQ1, RQ2, and RQ3 for one reproducible artificial identity mapping.

    RQ1 selects modality classifiers, RQ2 compares fixed score weights, and RQ3
    compares fixed and adaptive fusion across illumination and timing-noise
    conditions. Parameter selection uses validation data; test data provide
    the reported performance estimates.
    """
    out = root / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    mapping = user_mapping(face_df, key_df, cfg["n_users"], seed)
    mapping.to_csv(out / "user_mapping.csv", index=False)
    n = len(mapping)
    F = mapped_data(face_df, face_X, mapping, "face_user", "user_id", face_q)
    K = mapped_data(key_df, key_X, mapping, "keystroke_user", "subject")
    for name, D in [("face", F), ("keystroke", K)]:
        D["df"].to_csv(out / f"{name}_sample_audit.csv", index=False)
        D["df"].groupby(["pseudo_user", "split"]).size().rename("samples").to_csv(
            out / f"{name}_split_counts.csv"
        )
    quality = QualityEstimator().fit(
        F["q"][indices(F, "train")], K["X"][indices(K, "train")]
    )
    qf = quality.face(F["q"])
    qk = quality.key(K["X"])
    chosen, rows = rq1(F, K, cfg, out, seed)
    # Stored models include training scalers, validation normalizers, and identity mappings.
    joblib.dump(
        {"models": chosen, "quality": quality, "mapping": mapping},
        out / "fitted_models.joblib",
    )
    fs = chosen["face"].scores(F["X"])
    ks = chosen["keystroke"].scores(K["X"])
    paired = {}
    for split in ("validation", "test"):
        p = make_pairs(F, K, split, seed + 100, cfg["max_pairs_per_user"])
        paired[split] = trial_frame(p, fs, ks, qf, qk, n)
    v, t = paired["validation"], paired["test"]
    fixed_candidates = []
    for w in cfg["fixed_face_weights"]:
        s = w * v.face_score.to_numpy() + (1 - w) * v.key_score.to_numpy()
        fixed_candidates.append((eer(v.label, s, cfg["num_thresholds"])[0], w))
    best_w = min(fixed_candidates)[1]
    for name, w in [("face_only", 1.0), ("keystroke_only", 0.0)] + [
        (f"fixed_face_{w:g}", w) for w in cfg["fixed_face_weights"]
    ]:
        vs = w * v.face_score.to_numpy() + (1 - w) * v.key_score.to_numpy()
        ts = w * t.face_score.to_numpy() + (1 - w) * t.key_score.to_numpy()
        row, _ = save_result(
            out,
            "RQ2",
            name,
            v,
            t,
            vs,
            ts,
            cfg,
            seed,
            {"face_weight": w, "selected_fixed": w == best_w},
            plot=w in (0, 0.5, 1, best_w),
        )
        rows.append(row)
    # RQ3 holds the identity mix and keystroke probes constant across illumination groups.
    stress = {}
    condition_scores = {}
    for split in ("validation", "test"):
        p = make_pairs(
            F, K, split, seed + 200, cfg["max_pairs_per_user"], balanced=True
        )
        stress[split] = {}
        for noise in cfg["timing_noise_ms"]:
            # Reinitializing the generator pairs random perturbations across noise levels.
            degraded = degrade_timings(
                K["X"], key_cols, noise, np.random.default_rng(seed + 300)
            )
            km = chosen["keystroke"].scores(degraded)
            kq = quality.key(degraded)
            for light in ("good", "poor"):
                pp = p.loc[p.lighting == light].reset_index(drop=True)
                trial = trial_frame(pp, fs, km, qf, kq, n)
                trial["timing_noise_ms"] = noise
                stress[split][(light, noise)] = trial
    # Adaptive power minimizes mean validation EER across stress conditions.
    # The prior face weight is the RQ2 validation-selected fixed weight.
    # Configured positive powers evaluate adaptation even when performance declines.
    adaptive_search = []
    for power in cfg["adaptive_powers"]:
        values = []
        for frame in stress["validation"].values():
            w = adaptive_weight(frame.face_quality, frame.key_quality, best_w, power)
            s = w * frame.face_score + (1 - w) * frame.key_score
            values.append(eer(frame.label, s, cfg["num_thresholds"])[0])
        adaptive_search.append(
            {"power": power, "mean_validation_condition_EER": float(np.mean(values))}
        )
    best_power = min(adaptive_search, key=lambda r: r["mean_validation_condition_EER"])[
        "power"
    ]
    pd.DataFrame(adaptive_search).to_csv(
        out / "adaptive_validation_search.csv", index=False
    )
    # Each method retains thresholds calibrated on the unperturbed RQ2 validation pool.
    # The same thresholds apply to every RQ3 test condition.
    baseline_vs = best_w * v.face_score + (1 - best_w) * v.key_score
    wf = adaptive_weight(v.face_quality, v.key_quality, best_w, best_power)
    baseline_as = wf * v.face_score + (1 - wf) * v.key_score
    locks = {
        "fixed": thresholds(v.label, baseline_vs, cfg),
        "adaptive": thresholds(v.label, baseline_as, cfg),
    }
    locks["face_only"] = thresholds(v.label, v.face_score, cfg)
    locks["keystroke_only"] = thresholds(v.label, v.key_score, cfg)
    json_write(
        out / "fusion_selection.json",
        {
            "fixed_face_weight": best_w,
            "adaptive_power": best_power,
            "thresholds": locks,
            "threshold_calibration": "unperturbed RQ2 validation pool",
            "n_users": n,
            "note": "Artificial cross-dataset identities; independent mappings reuse face subjects.",
        },
    )
    for condition, tframe in stress["test"].items():
        light, noise = condition
        vframe = stress["validation"][condition]
        frames = {}
        for method in ("fixed", "adaptive", "face_only", "keystroke_only"):
            scores = []
            for frame in (vframe, tframe):
                if method == "fixed":
                    w = np.full(len(frame), best_w)
                elif method == "face_only":
                    w = np.ones(len(frame))
                elif method == "keystroke_only":
                    w = np.zeros(len(frame))
                else:
                    w = adaptive_weight(
                        frame.face_quality, frame.key_quality, best_w, best_power
                    )
                scores.append(
                    np.asarray(w * frame.face_score + (1 - w) * frame.key_score)
                )
            name = f"{light}_noise_{noise:g}ms_{method}"
            row, _ = save_result(
                out,
                "RQ3",
                name,
                vframe,
                tframe,
                *scores,
                cfg,
                seed,
                {
                    "lighting": light,
                    "timing_noise_ms": noise,
                    "fusion": method,
                    "prior_face_weight": best_w,
                    "adaptive_power": best_power,
                },
                plot=method in ("fixed", "adaptive"),
                locked=locks[method],
            )
            row["mean_face_weight"] = float(np.mean(w))
            row["mean_face_quality"] = float(tframe.face_quality.mean())
            row["mean_keystroke_quality"] = float(tframe.key_quality.mean())
            rows.append(row)
            frames[method] = tframe.assign(score=scores[1])
        paired_bootstrap_difference(
            frames["fixed"],
            frames["adaptive"],
            locks["fixed"]["target"],
            locks["adaptive"]["target"],
            out / "RQ3" / f"{light}_noise_{noise:g}ms_paired_differences.json",
            seed,
        )
    print(
        f"RQ2 face weight={best_w}; RQ3 adaptive power={best_power}; users={n}",
        flush=True,
    )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    return rows


def run(cfg):
    """Load datasets, execute configured seeds, and save combined metrics and summaries."""
    root = Path(cfg["output"]).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {root}. Choose a new output path."
        )
    root.mkdir(parents=True, exist_ok=True)
    json_write(root / "config.json", cfg)
    fd, fx, fq = load_faces(cfg["face_manifest"])
    kd, kx, kc = load_cmu(cfg["cmu_csv"])
    all_rows = []
    with threadpool_limits(limits=1):
        for seed in cfg["seeds"]:
            all_rows.extend(run_seed(fd, fx, fq, kd, kx, kc, cfg, seed, root))
    results = pd.DataFrame(all_rows)
    results.to_csv(root / "all_metrics.csv", index=False)
    numeric = [
        "EER",
        "dprime",
        "val_eer_FAR",
        "val_eer_FRR",
        "target_FAR",
        "target_FRR",
        "validation_EER",
    ]
    results.groupby(["RQ", "system"])[numeric].agg(
        ["mean", "std", "min", "max"]
    ).to_csv(root / "summary_across_mappings.csv")
    (root / "STATUS.txt").write_text(
        "REAL DATA RUN unless input manifest explicitly identifies synthetic smoke fixtures.\n"
        "RQ1/RQ2 use original face images with mixed natural illumination. RQ3 uses balanced lighting strata and simulated interval noise.\n"
        "All rates are fractions. EER is the original HW3 500-threshold approximation.\n"
        "Target FAR is validation-calibrated, not guaranteed on test; always report actual test FAR alongside FRR.\n"
        "Repeated mappings reuse people and do not create independent cohorts. No claim of improvement is assumed.\n"
    )
    print(f"Results: {root}", flush=True)
    return results
