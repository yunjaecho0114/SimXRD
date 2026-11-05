# Production Dataset v1 — Structure-Level Strain-Labeled XRD

## 1. Purpose and status

This document describes the **locally generated SimXRD production dataset v1** for predicting lattice-axis-length RMS strain from synthetic XRD patterns. It accompanies the small, runnable examples in [`../README.md`](../README.md), but **the 30,000-pattern dataset and its source crystal database are not distributed here**.

The dataset was generated and checked on the HANS research environment. The checks establish internal consistency of this particular simulation dataset; they do not demonstrate accuracy on experimental diffraction data.

**Prediction task**

- **Perturbed-only baseline (Model A):** perturbed XRD → scalar strain RMS.
- **Reference-guided baseline (Model C):** (ideal XRD, perturbed XRD) → scalar strain RMS.
- Both XRD channels use 3,500 intensity points. The ideal channel is a *binned stick pattern*, while the perturbed channel is a *continuous simulated profile*.

## 2. Dataset at a glance

| Property | Production v1 |
|---|---:|
| Distinct crystal structures | 1,000 |
| Ideal patterns | 1,000 |
| Perturbed patterns per structure | 30 |
| Perturbed patterns in total | 30,000 |
| Training structures / patterns | 800 / 24,000 |
| Validation structures / patterns | 100 / 3,000 |
| Test structures / patterns | 100 / 3,000 |
| 2θ range | 10.00° through 79.98° |
| Grid step | 0.02° |
| Points per XRD pattern | 3,500 |
| Intensity normalization | Maximum = 100 |
| Numeric array dtype | `float64` |

**Split policy:** splitting is performed at the **structure level**. All 30 perturbations belonging to a structure remain in one of train, validation, or test. A split by individual XRD rows would leak the same structure across partitions and is **not** the intended evaluation.

## 3. Controlled simulation conditions

The initial dataset isolates lattice stretching/compression by keeping other available simulation controls fixed.

| Control | Setting |
|---|---|
| Lattice stretch/compression | Varied (sampled within configured bounds) |
| Lattice shear bound | `0.0` |
| Grain size | Fixed **10 nm** (passed internally as **100 Å**) |
| Thermal displacement | `0.0 Å` |
| Preferred-orientation randomness | `0.0` |
| Background ratio | `0.0` |
| Noise ratio | `0.0` |
| Zero shift | `0.0°` |
| XRD coordinate mode | `reciprocal` (2θ axis) |
| XRD engine | Pymatgen-based SimXRD |

Calling this a *controlled strain-only dataset* means that **lattice stretch is the only varied simulation parameter** among these controls. It does not mean all instrument and material physics are perfectly modeled or that the label represents diffraction microstrain broadening.

## 4. Actual strain label — not the requested perturbation bound

Let `a0, b0, c0` and `a1, b1, c1` be the conventional-cell axis lengths before and after deformation, respectively:

```text
strain_a = (a1 - a0) / a0
strain_b = (b1 - b0) / b0
strain_c = (c1 - c0) / c0

strain_rms = sqrt((strain_a**2 + strain_b**2 + strain_c**2) / 3)
```

The **ground-truth regression target is `strain_rms` computed from the realized lattice geometry**, not from `stretch_bound`. `strain_a`, `strain_b`, and `strain_c` can have signs; their RMS is nonnegative and loses directional sign information. Changes in conventional-cell angles (`delta_alpha`, `delta_beta`, `delta_gamma`, in degrees) are also recorded as metadata, but are not included in the v1 scalar regression label.

This particular RMS is **not** the norm of a general strain tensor, a microstrain broadening parameter, or a universally measurable material strain without additional model assumptions.

### Per-structure strain sampling

Each structure receives 30 perturbations, allocated by a rejection-sampling schedule:

| Group | Acceptance range for realized RMS | Samples / structure | Total samples |
|---|---|---:|---:|
| `zero` | Exactly `0` | 1 | 1,000 |
| `very_low` | `[0, 0.001)` | 5 | 5,000 |
| `low` | `[0.001, 0.002)` | 5 | 5,000 |
| `low_mid` | `[0.002, 0.004)` | 5 | 5,000 |
| `medium` | `[0.004, 0.006)` | 5 | 5,000 |
| `high` | `[0.006, 0.008)` | 5 | 5,000 |
| `very_high` | `[0.008, 0.010)` | 4 | 4,000 |
| **Total** | | **30** | **30,000** |

The generation configuration specifies a base seed of `20261101` and up to **500 attempts** per group sample. A trial deformation is accepted only when its **calculated** RMS falls in the target range. The recorded `random_seed` and `sampling_attempt` allow individual trials to be traced under a compatible engine/environment. Note that the `zero` group is an explicitly exact-zero condition, distinct from the range-based groups.

## 5. Local merged dataset schema

The validated merged dataset on HANS is located under:

```text
db_expansion/dataset_v1/production_1000x30_v1/merged/
```

The following files are described for reference; **they are not part of this Git collaboration package**.

| File | Shape / format | Meaning |
|---|---|---|
| `ideal_xrd.npy` | `(1000, 3500)` | One ideal reference per structure |
| `perturbed_xrd.npy` | `(30000, 3500)` | One perturbed profile per sample |
| `lattice_before.npy` | `(30000, 3, 3)` | Conventional lattice matrices before deformation |
| `lattice_after.npy` | `(30000, 3, 3)` | Conventional lattice matrices after deformation |
| `strain_rms.npy` | `(30000,)` | Actual scalar strain labels |
| `strain_a.npy` | `(30000,)` | Actual a-axis strain |
| `strain_b.npy` | `(30000,)` | Actual b-axis strain |
| `strain_c.npy` | `(30000,)` | Actual c-axis strain |
| `ideal_index.npy` | `(30000,)` | Index linking a perturbed row to its ideal row |
| `structure_id.npy` | `(30000,)` | Per-sample structure ID/index mapping |
| `split.npy` | `(30000,)` | Encoded data split: `0=train`, `1=validation`, `2=test` |
| `seed.npy` | `(30000,)` | Simulation random seeds |
| `samples.csv` | 30,000 sample records | Labels, group, seed, structure and split metadata |
| `structures.csv` | 1,000 structure records | Structure metadata and ideal-XRD mapping |
| `merge_manifest.json` | JSON | Merge/integrity metadata |

**Pairing rule:** for sample row `i`, the perturbed profile is `perturbed_xrd[i]` and its ideal reference is `ideal_xrd[ideal_index[i]]`. Do **not** assume that a row in the 30,000-row array has the same index as its ideal reference in the 1,000-row array.

### Important metadata fields

The underlying `samples.csv` includes sample and structure indices, `mp_id`, the textual split, `rms_group`, `group_index`, `group_sample_index`, `stretch_bound`, `shear_bound`, `random_seed`, `sampling_attempt`, `strain_rms`, `strain_a`, `strain_b`, `strain_c`, and conventional-cell angle differences. The metadata is useful for per-bin evaluation and reproducing the generation settings.

### Why both CSV files are included

`strain_plan_1000x30_v1.csv` and `samples.csv` contain identical
data in Production v1, but represent different stages of our workflow.

- **strain_plan_1000x30_v1.csv**: The sampling plan prepared before
  XRD generation, specifying deformation conditions and actual
  strain labels for each sample.
- **samples.csv**: The final sample metadata associated with the
  generated XRD dataset.

Both files are preserved to document the original dataset-generation
workflow and support reproducibility.

## 6. Loading paired inputs for machine learning

The following code applies **only when the local merged dataset is available**. It does not fetch or regenerate data. Run it from the repository root or change `root` accordingly.

```python
from pathlib import Path
import numpy as np

root = Path("db_expansion/dataset_v1/production_1000x30_v1/merged")

ideal = np.load(root / "ideal_xrd.npy", mmap_mode="r", allow_pickle=False)
perturbed = np.load(root / "perturbed_xrd.npy", mmap_mode="r", allow_pickle=False)
ideal_index = np.load(root / "ideal_index.npy", mmap_mode="r", allow_pickle=False)
y = np.load(root / "strain_rms.npy", mmap_mode="r", allow_pickle=False)
split = np.load(root / "split.npy", mmap_mode="r", allow_pickle=False)

assert ideal.shape == (1000, 3500)
assert perturbed.shape == (30000, 3500)
assert ideal_index.shape == y.shape == split.shape == (30000,)
assert np.all((ideal_index >= 0) & (ideal_index < len(ideal)))

# Demonstrate one two-channel input without materializing the full dataset.
i = int(np.flatnonzero(split == 0)[0])  # one training sample
x_pair = np.stack([ideal[int(ideal_index[i])], perturbed[i]], axis=0)
target = float(y[i])

print(x_pair.shape)  # (2, 3500)
print(target)         # actual lattice strain RMS

# For a perturbed-only baseline:
x_single = np.asarray(perturbed[i])[None, :]
print(x_single.shape)  # (1, 3500)
```

A PyTorch model may add a batch dimension to obtain `(batch, 2, 3500)` or `(batch, 1, 3500)` respectively. Intensity arrays have been max-normalized to 100; any further input or target scaling must be documented separately. For example, scaling the training target by 100 is a **model-training convention**, not a change to the saved physical label.

**Evaluation rule:** keep training, validation, and test rows isolated using `split.npy` and preserve the structure-level separation. Report overall MAE/RMSE and performance by realized strain-RMS group, including exactly-zero cases.

## 7. How the data was produced

The relevant implementation and setup files in the local research tree are:

```text
db_expansion/build_crystdb_inventory.py
db_expansion/select_1000_candidates_v1.py
db_expansion/build_smoke_3x30_v2.py
db_expansion/generate_xrd_dataset_v1.py
db_expansion/fast_xrd_mapping.py
db_expansion/safe_xrd_generation.py
db_expansion/dataset_v1/generation_config.json
```

Conceptually:

```text
Source crystal structures (ASE DB)
       ↓
Structure inventory and 1,000-structure selection
       ↓
Structure-level train/validation/test assignment
       ↓
30 realized-strain targets per structure (rejection sampling)
       ↓
Ideal XRD + once-deformed lattice + perturbed XRD
       ↓
Structure-by-structure chunks, validation, merged dataset
```

**Reproduction limitation:** the production generator reads a source structure DB, a final 1,000-structure selection/split CSV, a precomputed `strain_plan_1000x30_v1.csv`, and `generation_config.json`. A separate standalone script that recreates the complete 30,000-row sampling-plan CSV was **not identified** in the code review. The smoke-test script demonstrates the sampling algorithm for three structures, but is **not** itself a complete production sampling-plan generator. Reconstructing the entire 30,000-pattern dataset therefore requires additional inputs and a reconstruction procedure; the three collaboration demos alone are not sufficient.

The optimized mapping and axial-divergence safety utilities temporarily patch engine functions at the process level. Do not call them concurrently from multiple threads.

## 8. Validation and interpretation caveats

Validated dataset characteristics include consistent array shapes, finite simulated XRD values, structure-level split integrity, and consistency between saved strain labels and lattice geometry. These are **simulation/data integrity** checks, not experimental generalization results.

- Absolute lattice strain here is **not** inferred from a measured sample; labels are generated from known synthetic deformations.
- With shear disabled, changes in lattice angles should be negligible up to numerical tolerance.
- The simulated grain-size peak-width model still warrants separate physical/angular-unit consistency tests before quantitative grain-size inference.
- Different software versions or hardware can produce small numeric differences even at identical seeds.
- External simulated XRD or experimental XRD requires an explicitly designed evaluation procedure; no external accuracy claim is made here.

## 9. Sharing and attribution

The full original `CrystDB.db`, strain-plan CSV, production arrays, and lab-internal outputs are **excluded from Git**. Do not upload these files or credentials. The original SimXRD software is MIT-licensed; preserve its copyright/license notices. Any redistribution of third-party structure data or laboratory-developed materials requires separate authorization.

See the [collaboration overview](../README.md) for the tested small demo scripts and original SimXRD attribution.

### Production configuration provenance

The supplied `generation_config.json` preserves configuration
metadata from the original dataset-development stage.

Two fields contain historical status values:

- `status: draft_controlled_baseline` reflects the initial
  configuration stage. Production dataset v1 was subsequently
  generated and passed integrity checks on November 3, 2026.
- `safety.axial_divergence_fix:
  validated_in_memory_not_yet_integrated` is an older status
  description. The current generation pipeline uses
  `db_expansion/safe_xrd_generation.py` for numerical-safety
  handling, including axial-divergence protection.

The original configuration file has intentionally not been
modified to preserve dataset-generation provenance.

These statements describe the validated HANS research workflow;
the complete production dataset and its validation records are
not distributed with this collaboration package.
