# Face and Keystroke Authentication: CAP4103 Project

Runnable experiments for SoF + CMU, implementing RQ1, RQ2, and RQ3.
The code uses actual original face images and the official CMU timing CSV.
No pretrained network, GPU, external service, or extra biometric dataset is required.

## Latest results

The full experiment completed for seeds 42, 43, and 44. All three runs
selected logistic regression (`C=1.0`) for face verification and
distance-weighted kNN (`k=9`) for keystroke verification by validation EER.
The 50:50 face-to-keystroke weight also won on validation in all three runs.
On the paired RQ2 test trials, mean EER across the three seeds was **7.74%**
for 50:50 fusion, **16.78%** for face only, and **11.50%** for keystroke only.

For the report and presentation, start with
`results/summary_across_mappings.csv` for averages across seeds and
`results/all_metrics.csv` for per-seed metrics. The `results/seed_*/` folders
contain model and fusion selections, split counts, and plots. Git ignores
these generated files. Use the team’s shared Google Drive folder for copies
of the results or regenerate them using the steps below.

For RQ3, report actual test FAR alongside test FRR for each lighting and
timing-noise condition. Each method’s threshold was calibrated to a 1% FAR
target on the validation set and held fixed across test conditions.

## Start here

The datasets are downloaded and extracted manually. The project does not need
an automatic download command. Use Python 3.10 or newer (tested on Python 3.12).
Run all commands from the project folder.

### 1. Set up Python

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Windows: create with py -3.12 -m venv .venv, then activate .venv\Scripts\activate
python -m pip install -r requirements.txt
```

If your Python 3.12 virtual environment is already active, skip creating and
activating it again.

### 2. Place the manually extracted datasets

| Project location | Contents |
|---|---|
| `data/images/` | Original SoF JPG images (2,662 images in the complete original archive) |
| `data/metadata/metadata.mat` | Extracted SoF MATLAB metadata file |
| `data/DSL-StrongPasswordData.csv` | Official CMU keystroke CSV |

Image subfolders inside `data/images/` are supported. Make sure `metadata.mat`
is directly inside `data/metadata/`, without an extra nested metadata folder.
The RAR archives are not needed after successful extraction.

### 3. Prepare the face manifest

```bash
python main.py prepare-sof --images data/images --metadata data/metadata/metadata.mat --out data/sof_manifest.csv
```

With the original dataset and default settings, this retains 1,768 samples after
selecting SoF part 2 and purging split boundaries. This is expected; it does not
mean images are missing. If `data/sof_manifest.csv` is already prepared and your
data and split settings have not changed, skip this step.

The metadata reader uses actual SoF fields (`userId`, `imageSequence`, `faceRect`,
`illuminationQuality`, `part`). Only original files ending `_o` are indexed.
The default cohort is SoF part 2, the 12-subject controlled-lighting subset.

### 4. Run the experiments

```bash
python main.py run
```

The run command reads `config/default.json`, runs every configured seed, and
saves everything in **`results/` by default**. No seed or output argument is
needed. Keep these settings in `config/default.json`:

```json
"output": "results",
"seeds": [42, 43, 44]
```

These are entries inside the existing JSON configuration, not a replacement for
the whole file. Other settings remain in place.

To hide the harmless warning about plot windows while still saving all figures:

```bash
python -W "ignore:FigureCanvasAgg is non-interactive:UserWarning" main.py run
```

For a faster numerical run without figures, use `python main.py run --no-plots`.
The program refuses to write into a nonempty output folder. To rerun into
`results/`, first move the previous folder to a backup location or delete its
contents if no longer needed. Keep the code, configuration, and `data/` folder.
Paths are relative to your current working directory.

### What is a seed?

A seed is a starting number for the random-number generator. It makes random
choices repeatable. In this project, seeds control CMU user selection, artificial
identity mappings, sample pairings, simulated timing noise, and bootstrap
resampling. The numbers 42, 43, and 44 have no special meaning; they define three
repeatable runs. They do not mean 42, 43, or 44 users.

The seed does not change the face cohort or the train/validation/test split rules.
With the same data, settings, code, and software environment, rerunning the same
seeds should reproduce numerical results. Different seeds can change the metrics.
Repeated mappings reuse the same face subjects and are not independent cohorts.
Choose the seed list before comparing results and report all configured runs.

## Research questions

**RQ1.** How do face recognition using SoF and keystroke authentication using
CMU compare in FAR, FRR, and EER across multiple classifiers and settings?

**RQ2.** Using the models selected through RQ1, how do equal-weight and
unequal-weight score fusion affect performance compared with either modality?

**RQ3.** Under varying face illumination and controlled keystroke timing
degradation, how does quality-adaptive fusion compare with the strongest fixed
weight from RQ2, measured by EER and FRR at a predefined FAR target?

## Experiment design

| Item | Implementation |
|---|---|
| Face feature | Metadata face crop -> 64x64 grayscale -> top-left 16x16 orthonormal 2D DCT = 256 features |
| Keystroke feature | All 31 official H, DD, UD timing columns, in seconds |
| Training | One binary genuine-vs-other-enrolled-users model per claimed identity |
| Classifiers | Logistic regression, RBF SVM, kNN; 10 settings per modality |
| Feature scaling | StandardScaler fit on training observations only |
| Score | SVM/logistic margin or kNN positive-class probability; larger means more similar |
| Score normalization | Per-claimed-user mean/SD fit on validation scores; sigmoid produces [0,1] scores |
| Identity pairing | Fixed artificial one-to-one mapping from 12 SoF users to 12 randomly selected CMU users |
| Attempts | Genuine if actual pseudo-user equals claimed user; otherwise impostor |
| RQ2 weights | Face weights .25, .50, .75; face-only and keystroke-only controls |
| RQ3 conditions | Good/poor natural SoF illumination x 0/10/30 ms simulated interval noise |
| Adaptive rule | Prior fixed weight multiplied by observable quality raised to a validation-selected power |
| Selection | Classifier, fixed weights, and adaptive power chosen from validation data only |
| Thresholds | Validation EER threshold and validation-calibrated 1% FAR threshold, frozen for testing |
| Evaluation | Original HW3 Evaluator: 500 thresholds from -.1 to 1.1, EER, d-prime, distributions, ROC, linear-axis DET |

`config/default.json` contains every experimental setting. The timing-noise
levels are hypotheses for a sensitivity experiment, not measured device errors.
You can change the FAR target *before* running; do not choose it after seeing
test results. Normalized scores are not calibrated probabilities.

### Split and leakage controls

- SoF: within each identity/illumination stratum, order original images by
  `imageSequence`, allocate approximately 50% training, 25% validation, 25% test,
  and purge the last original from training and validation. Also remove lower
  priority observations that are immediate sequence neighbors of another split.
  The purge width is configurable with `prepare-sof --gap`.
- Original-image hashes prevent exact duplicate images crossing partitions.
  `group_id` must belong to a single split. Synthetic variants are excluded.
- CMU: sessions 1-4 train, 5-6 validate, 7-8 test (200/100/100 per selected user).
  A session is never shared between partitions. Negative UD intervals are valid.
- All users remain enrolled across splits. This measures verification of enrolled
  users, not generalization to previously unseen identities.
- Validation is used for score normalization, model selection, weighting, and
  thresholds. Its reported error is a tuning result; the test split is held out.
  No claim of nested cross-validation is made.
- RQ1 uses every available probe for each modality, so their probe counts differ.
  RQ2 includes modality-only controls on the **same paired probe subset** for a
  fair fusion comparison. These control EERs may differ from RQ1's full-pool EERs.
- A paired observation uses both modalities from one actual pseudo-user and is
  compared to every claimed pseudo-user. This includes genuine attempts and
  attempts where one enrolled user impersonates another. It does not evaluate
  unknown outsiders, spoofing, or mixed-identity cross-modality attacks.
- RQ2 samples without replacement within each user/split. RQ3 balances good and
  poor face counts per identity and pairs the same keystroke attempts to both
  lighting conditions; the same random noise draw is scaled by each severity.
  Trials are reused intentionally across compared systems, not across splits.

### Adaptive weighting, explicitly

For a face score `sf` and key score `sk`, fixed fusion is:

```
s = w0 * sf + (1 - w0) * sk
```

Let `qf` and `qk` be observable quality proxies in [.05,1]. Adaptive fusion is:

```
a = w0 * qf**power
b = (1 - w0) * qk**power
w_face = a / (a + b)
s = w_face * sf + (1 - w_face) * sk
```

- `w0` is the best RQ2 fixed face weight by validation EER.
- `power` is chosen from .5, 1, 2, 4 using mean validation EER across the RQ3
  conditions. It is frozen before inspecting test results.
- Face quality combines exposure, contrast, unclipped pixels, and left/right
  intensity balance; a training-reference percentile scales this value.
- Keystroke quality decreases for population timing outliers measured using the
  training median and median absolute deviation. This heuristic can also flag
  legitimate unusual typing; it is NOT a ground-truth quality label.
- The weighting function receives no test identity, genuine/impostor label,
  illumination label, or injected-noise level. Illumination labels only define
  comparison groups.
- Keystroke degradation adds Gaussian noise to hold and down-down intervals,
  clips those positive intervals at 1 ms, then rebuilds `UD = DD - H`.
  This preserves feature dependencies, including dotted key names `Shift.r`.
  It simulates timing-interval disturbance, not fatigue or typing mistakes.

RQ3 uses one frozen operating threshold per method, calibrated on RQ2's original
validation pool, across **all** stress conditions. It does not recalibrate a
threshold on each test condition to force the desired FAR.

## HW3 evaluation reuse

`vendor/hw3_original.py` is the earlier provided `main(1).py` HW3 implementation.
Its original `Evaluator` class is imported directly; the random-score `main()`
is not called. `biometrics/evaluation.py` passes genuine/impostor real scores
into `compute_rates()`, `get_EER()`, `get_dprime()`, and the original plotting
methods. No external HW3 file is missing or needs to be added for this version.

If your instructor expects a different approved HW3 revision, replace this file
with that version while retaining the same interface, then rerun the project.

## Results and metrics

The primary table is `results/all_metrics.csv`. All rates are fractions: `.05` = 5%.

| Field | Meaning |
|---|---|
| `validation_EER` | Tuning result; used to select models and fixed weights |
| `EER` | HW3 finite-grid approximation on the held-out test scores |
| `test_curve_EER_threshold` | Descriptive crossing on test ROC; NOT a deployment threshold |
| `dprime` | Absolute mean separation / pooled within-class SD, from HW3 |
| `val_eer_FAR`, `val_eer_FRR` | Test errors at the threshold selected for EER on validation |
| `target_threshold` | Frozen validation threshold meeting requested empirical FAR |
| `target_FAR`, `target_FRR` | Actual test errors at that frozen threshold |
| `target_far_met_on_test` | Whether actual test FAR is at or below the requested target |
| `TP`, `FN`, `FP`, `TN` with threshold prefix | Confusion counts, at each named operating threshold |
| `macro_user_EER` | Mean per-claimed-user HW3 EER, for paired RQ2/RQ3 trials |

Do not call `target_FRR` “FRR at exactly 1% test FAR” unless the actual test FAR
matches that value. Test FAR can exceed 1% under distribution shift. EER is a
threshold-swept description, so it differs from errors at the fixed operational
threshold. kNN and small samples produce tied scores and coarse EER estimates.

Inside `results/`, every run also saves:

- `summary_across_mappings.csv`: mean/SD/range across the configured mapping seeds.
- `seed_*/user_mapping.csv`: original identities -> pseudo-identities.
- `seed_*/*_sample_audit.csv`, `*_split_counts.csv`: exact observations/splits.
- `seed_*/selected_models.json`, `fusion_selection.json`: all choices and thresholds.
- `seed_*/adaptive_validation_search.csv`: selection evidence for adaptive power.
- `seed_*/RQ1/.../test_curve.csv`: all 20 classifier-setting curves per seed.
- `seed_*/RQ2/...` and `RQ3/...`: compressed trial CSVs, genuine/impostor TXT scores,
  ROC data, per-user metrics, and original HW3 figures for selected comparisons.
- `seed_*/RQ3/*_paired_differences.json`: exploratory 95% claimed-user-cluster
  bootstrap intervals for adaptive-minus-fixed FAR/FRR at frozen thresholds.
- `seed_*/fitted_models.joblib`: models, training scalers, validation normalizers,
  quality estimator, identity mapping.
- `config.json`: experiment settings including seeds.

RQ3 includes face-only and keystroke-only diagnostic rows. Use face-only good vs
poor rows at zero noise to check whether lighting actually affected this model.
Use keystroke-only rows to check the injected-noise effect before interpreting
adaptive fusion. It is valid for fusion to perform worse or show no benefit.

## Rubric mapping

| Rubric criterion | Code |
|---|---|
| Correct sample IDs; train/test separation | `data.py`: metadata readers, session/sequence splits, duplicate checks; mapping/sample audit CSVs |
| Feature representations | `data.py`: face crop/DCT and CMU timing features |
| Claimed-user matching score | `models.py`: binary per-user `Verifier` and normalized similarities |
| Genuine/impostor distinction + HW3 | `experiments.py`: trial labels; `evaluation.py` imports original HW3 class |
| Fusion rule | `models.py`: `adaptive_weight`; `experiments.py`: weighted score combinations |
| RQ1 implementation | `experiments.py`: `rq1` model/parameter comparisons |
| RQ2 implementation | `experiments.py`: paired single-modality controls + .25/.5/.75 weights |
| RQ3 implementation | `experiments.py`: balanced lighting/noise conditions, adaptive selection, frozen thresholds |

## Limits to state in your report

1. The 12 people in SoF part 2 have few well-lit examples; sequence purging can
   reduce these further. This is a small course experiment.
2. Face images come from one underlying collection and include video-derived
   frames. Sequence-aware splitting reduces immediate-neighbor overlap but does
   not establish independent face recording sessions or eliminate all visual
   similarity. A larger purge or manual session grouping may exclude users.
3. Artificial mappings do not represent the same real humans across modalities.
   Repeating mappings is a sensitivity analysis, not independent cohorts.
4. Training on other enrolled users supplies background negatives; that differs
   from a genuine-only deployment. All identities are known during enrollment.
5. Quality proxies and interval noise are deliberately simple experiments.
   Lighting groups may also differ in expression/pose. No causal lighting-only,
   real fatigue, fairness, security, or deployment claim follows automatically.
6. Bootstrap intervals cluster claimed users only. Shared probes across claims,
   reused source users, and very few subjects limit statistical conclusions.
7. The code does not guarantee an improvement or a submission grade. The final
   five-page report, ethics statement (at least three credible citations), and
   interpretation must use the actual output and follow the assignment format.

## Files and sources

- `main.py`: commands; `config/default.json`: reproducible settings.
- `biometrics/data.py`: input/feature/split handling.
- `biometrics/models.py`: matching and quality weighting.
- `biometrics/evaluation.py`: HW3 adapter and fixed-threshold rates.
- `biometrics/experiments.py`: experiment progression and outputs.
- `tests/test_core.py`: threshold ties, known errors, split leakage,
  preprocessing isolation, dependent timing features, quality-weight behavior.
- `vendor/hw3_original.py`: earlier HW3 source, retained for direct reuse.

Official SoF dataset and requested citation:
https://sites.google.com/view/sof-dataset
Afifi, M., & Abdelhamed, A. (2019). AFIF4: Deep gender classification based on an
AdaBoost-based fusion of isolated facial features and foggy faces.
Journal of Visual Communication and Image Representation.
https://arxiv.org/abs/1706.04277

Official CMU dataset and original paper:
https://www.cs.cmu.edu/~keystroke/
Killourhy, K. S., & Maxion, R. A. (2009). Comparing anomaly-detection algorithms
for keystroke dynamics. IEEE/IFIP International Conference on Dependable Systems
& Networks, 125-134. Use the publication link on the official benchmark page.

Libraries: https://scikit-learn.org/stable/ and https://docs.scipy.org/doc/scipy/

Use the datasets under their creators' stated academic-use terms. Raw biometric
images and dataset archives are not distributed in this code package. The run
outputs contain biometric scores, original subject IDs, and fitted models; keep
those access-controlled when sharing your project.
