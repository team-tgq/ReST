# Manuscript to implementation

| Manuscript component | Implementation |
|---|---|
| §2.1: 30/30/75 historical positions | `data.history_indices` |
| §2.2: variable patches and process groups | `process.PhysicsGuidedHydroProcessEncoder`, `mpe.MultiScaleProcessEncoder` |
| §2.3: reliability state and context adaptation | `amrm.AdaptiveMissingnessReliabilityModule` |
| §2.3: trailing-gap correction | `reliability.GapAwareSWEStateAnchor` |
| Missing Short SWE: cross-variable anchor | `reliability.SWEFreeCrossVariableAnchor` |
| §2.4: historical standardization and output-scale recovery | `sdef._masked_standardize`, `ScaleConsistentDecomposedExpertForecasting` |
| §2.4: Top-2 combination of five experts | `sdef._ProcessRouter`, `_ProcessExperts` |
| §2.5: masked MAE + 0.01 balancing + 0.1 anchor loss | `train.run_epoch`, `model.ReST.forward` |

The model uses 128-dimensional tokens, four attention heads, one within-variable
temporal layer, dropout 0.1, and 192-dimensional expert hidden layers. The main
model has 4,209,294 parameters. Dynamic channels follow `DEFAULT_FEATURE_NAMES`
in `config.py`; changing their order changes the method.

## Input and output contracts

Each `ObservedHistory` contains three aligned `[batch, positions, 15]` tensors:
`normalized`, `raw`, and `observed`. Raw physical values must remain available
for anchors and sample-wise standardization. Missing values have mask zero;
observed zero SWE is a valid snow-free observation. The five static attributes
are elevation, latitude, longitude, land-cover code, and southness, in this order.
The offline prior has five components in the order listed in `data.PRIOR_NAMES`.

Short covers `t-29` through `t`; Long samples `t-210, t-203, ..., t-7`.
Analogue positions are arranged from the fifth previous hydrological year to
the first previous year, with offsets −7 through +7 around the same hydrological
day (October 1 origin). This differs from simply replacing the calendar year
around leap days. Missing early years are padded with mask zero.

Historical z-scores use only observed values independently for each sample,
scale, and variable. If fewer than two values are observed or the scale is
degenerate, the implementation uses mean 0 and scale 1. Dataset-level min-max
statistics are separately fitted on coordinates visible to training windows.

The observed issue-day SWE is the anchor when available. A trailing gap uses
the last observed SWE plus a bounded correction. Without Short SWE, a cross-variable
anchor can use the remaining non-snow process tokens, static attributes, and
prior. A learned standardized baseline provides a final fallback. Trends and
residuals can be negative; only final inference SWE is clamped at zero.

Training augmentation modifies only Short values and masks together: SWE tail
dropout probability 0.30, at most seven positions; exogenous block probability
0.15 per variable, with five-position blocks. It leaves targets, Long, and
Analogue unchanged and is disabled by `eval()`.

## Public implementation scope

Only the complete ReST model is published. External baselines, module ablations,
replacement encoders, and alternative prediction heads are excluded. The public
training entry point does not expose a model-variant selector.

The inference and training computation graphs were retained from the submitted
implementation. Public names use ReST, MPE, and SDEF. Historical checkpoint key
prefixes `mpcde` and `sc_dpef` were renamed to `mpe` and `sdef` during the verified
weight export; the exported checkpoints load directly with this package.
