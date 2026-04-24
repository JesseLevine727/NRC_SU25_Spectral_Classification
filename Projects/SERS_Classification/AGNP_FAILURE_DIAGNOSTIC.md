# AgNP Failure Diagnostic

## Summary

After canonicalizing `bt -> benzenethiol`, `pSERS` is not a real failure. The remaining substrate-agnostic failure is specifically `AgNP`.

Best canonical Siamese run:

- Feature: `derivative_1`
- Loss: triplet, margin `0.2`
- Prototype mode: `substrate_balanced`
- Mean accuracy: `0.854`
- Mean macro F1: `0.686`

Per held-out substrate:

| Held-out substrate | Accuracy | Interpretation |
|---|---:|---|
| Ag | 0.920 | mostly correct |
| AgNP | 0.380 | failure |
| Au | 0.960 | mostly correct |
| AuNP | 1.000 | correct |
| PICO | 0.867 | partly benzenethiol/pyridine confusion |
| pSERS | 1.000 | correct |

## What Fails In AgNP

`Workspace/siamese_canonical_derivative1_triplet_confusions/AgNP.csv`:

| True | Pred 4np | Pred benzenethiol | Pred pyridine |
|---|---:|---:|---:|
| 4np | 0 | 25 | 0 |
| pyridine | 0 | 6 | 19 |

So the dominant failure is:

```text
4np on AgNP -> benzenethiol, 25/25 spectra
```

## Prototype-Distance Evidence

Diagnostics were generated in `Workspace/agnp_diagnostics/`.

The diagnostic bundle includes five checks:

- Average spectra for `4np_AgNP`, `4np_PICO`, `4np_pSERS`, `benzenethiol_Ag`, `benzenethiol_Au`, `benzenethiol_PICO`, `benzenethiol_pSERS`, and `pyridine_AgNP`.
- Prototype-distance tables for held-out `AgNP` and held-out `pSERS`.
- PCA of the first-derivative input representation.
- PCA of the trained Siamese embeddings for the held-out `AgNP` and `pSERS` folds.
- Raw-file and label audit for `4np` on `AgNP`.

For the AgNP-held-out model, AgNP `4np` spectra are much closer to the `benzenethiol` prototype than to the `4np` prototype:

| True/pred group | dist to 4np | dist to benzenethiol | dist to pyridine |
|---|---:|---:|---:|
| true 4np, pred benzenethiol | 0.635 | 0.268 | 0.680 |
| true pyridine, pred benzenethiol | 0.878 | 0.348 | 0.444 |
| true pyridine, pred pyridine | 0.905 | 0.565 | 0.225 |

For the pSERS-held-out model, all classes are cleanly separated:

| True/pred group | dist to 4np | dist to benzenethiol | dist to pyridine |
|---|---:|---:|---:|
| true 4np, pred 4np | 0.174 | 1.170 | 1.141 |
| true benzenethiol, pred benzenethiol | 1.122 | 0.080 | 1.092 |
| true pyridine, pred pyridine | 1.088 | 1.009 | 0.179 |

## Input Feature vs Embedding

The failure is not obvious in the first-derivative input PCA. In derivative-feature PCA, the AgNP `4np` centroid is closer to the training `4np` centroid than to `benzenethiol`:

```text
AgNP 4np distance to train 4np centroid:          0.243
AgNP 4np distance to train benzenethiol centroid: 0.704
AgNP 4np distance to train pyridine centroid:     0.859
```

But after training the AgNP-held-out Siamese embedding, AgNP `4np` moves closer to `benzenethiol`:

```text
AgNP 4np distance to train benzenethiol centroid: 0.242
AgNP 4np distance to train 4np centroid:          0.624
AgNP 4np distance to train pyridine centroid:     0.669
```

Interpretation:

```text
The network/prototype geometry is creating the AgNP 4np -> benzenethiol collapse.
The derivative input representation itself still contains a usable 4np signal.
```

## Raw File/Label Audit

The expected raw files exist:

```text
Workspace/SERs/4-NP - 632nm/AgNP/*.txt
```

The file audit found 25 AgNP 4np map spectra, each with 1024 rows and two columns. There is no obvious file-count or shape issue.

However, the AgNP 4np map is heterogeneous. Many spectra have maximum intensity around `1094-1096 cm^-1`, while others peak near `1570-1585 cm^-1`. This suggests spatial/map heterogeneity or substrate-dependent enhancement differences within the AgNP 4np measurement.

## Practical Conclusion

The problem is deeper than simply "not enough data", but data coverage explains why the model cannot resolve it robustly:

- When `AgNP` is held out, `4np` training data only comes from `PICO` and `pSERS`.
- AgNP `4np` appears to occupy an embedding region that the trained model associates with `benzenethiol`.
- The first-derivative input still separates AgNP `4np` better than the learned embedding, so future work should regularize the embedding rather than just add more CNN capacity.
- The practical failure mode is representation collapse under leave-one-substrate-out training, not a missing-file or obvious label-count issue.

Recommended next improvement before collecting more data:

```text
Use a hybrid decision rule or auxiliary loss that preserves derivative-feature chemical geometry,
because the Siamese embedding currently distorts AgNP 4np toward benzenethiol.
```

But if AgNP remains the target substrate, the cleanest experimental fix is still to add more `4np` coverage on metallic substrates and/or repeat AgNP 4np maps to determine whether this map is representative.
