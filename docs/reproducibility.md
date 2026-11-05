# Reproducibility and verification

**English** | [简体中文](reproducibility.zh-CN.md)

[Overview](../README.md) · [Method](method.md) · [Data](data.md)

This is an independent release of the submitted ReST implementation. It was
prepared from the local research code at commit `9fa628fb` and checked against
the final manuscript titled *Reliable Spatiotemporal Generalization for Snow
Water Equivalent Forecasting*. Experimental alternatives outside that manuscript
were omitted. The original research checkout was retained.

## Verified locally

- The complete ReST model loads the research implementation's
  weights after renaming the MPE/SDEF prefixes. Predictions, raw predictions,
  base/trend/residual terms, routing weights, and anchors match exactly on four
  input conditions: ordinary observations, trailing gaps, absent SWE history,
  and entirely missing dynamic history.
- The seven study splits produce exactly the manuscript's counts, totaling
  590,988 windows at 399 active stations. Analogue indices match the original
  hydrological-day construction, including leap-year behavior.
- Temporal training-only min/max statistics match a saved paper checkpoint.
  A sample of 16 actual Temporal validation windows produces bitwise-identical
  predictions using the trained seed-52 weights. This is a sample comparison,
  not a full-dataset accuracy rerun.
- The three released Temporal models load with `weights_only=True`, with no
  optimizer, machine paths, or training logs in their exported artifacts.

The tests exercise checkpoint roundtrips, finite gradients, mask isolation, causal windows, complete
target filtering, split membership, train-only static normalization, invalid
input handling, and inference batch independence. `rest-demo` performs a
synthetic forward/backward pass.

A separate synthetic seven-file fixture also passed a complete one-epoch
command-line training, checkpoint reload, and evaluation cycle. Its numerical
errors are not manuscript results.

## Scope and limits

The public training entry point uses a standard shuffled PyTorch DataLoader.
It preserves the model, loss, data splits, and hyperparameters; it does not
reproduce historical multi-machine job scheduling or the exact order of packed
training batches. Newly trained weights therefore need not match archived
weights bit for bit, even with the same seed. Exact scientific scores also
depend on the archived processed arrays and the evaluation aggregation.

This release was not used to retrain all 21 main experiments. The manuscript's
18 external baselines, module ablations, the 42-condition paired missingness-plan archive, and
figure-generation pipelines are outside this focused model release. Training
augmentation is included; it is distinct from the fixed evaluation perturbations
used for the manuscript's robustness tables.

The separate [data-v1.0.0 archive](https://github.com/team-tgq/ReST/releases/tag/data-v1.0.0) now supplies primary processed
arrays, splits, publication result tables and numerical figure inputs. All window
identities and seven split counts were checked; 32 sampled input/target records
match the original archive exactly. Recalculation from the external prediction
pairs gives station-equal MAE 14.5145572 mm (reported as 14.515 mm).
See the [data release guide](data-release.md) and its numerical-validation record.
Historical Alaska forcing inputs and fixed evaluation missingness plans remain
unavailable in this release; a Zenodo DOI is pending. Plot source values do not
imply that the original plotting pipelines or comparative model code are included.
