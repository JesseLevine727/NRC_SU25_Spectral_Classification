# P13 design-freeze memo

**Protocol:** `nato-sers-p13-v1-locked`

**Approval:** project owner, 2026-09-04

**Status:** locked before any P13 outcome calculation; classical execution is
authorized under the frozen registries.

## What was approved

P13 tests whether analyte identity remains predictively recoverable for a named
SERS substrate family when one acquisition instrument is completely excluded
from model development. It is a post-P03 secondary amendment motivated by the
field-trial purpose; it is not a retroactive change to P03.

The evaluation domain is station × substrate family × held instrument. Each
analyte within it is a class-support cell. The independent unit is the physical
master, and technical-repeat probabilities are averaged within each
master–substrate–instrument view before primary scoring.

The primary endpoint is three-class balanced accuracy. A confirmatory domain
supports portability only when both `LCB95(held BA) >= 0.60` and
`UCB95(source BA - held BA) <= 0.10`. Substrate-level support requires every
confirmatory domain for that substrate to pass both bounds.

Confirmatory eligibility requires at least three held masters and three
training masters per analyte in every outer split, two source instruments per
analyte in every outer split, and three paired held/source masters per analyte.
Exploratory eligibility requires at least two held and two training masters per
analyte; its source-instrument and pairing limitations remain explicit.

The primary model is source-only `C-SELECTED`; fixed RBF SVM is the main
sensitivity and the remaining frozen P03 classical families are secondary.
`PP-U-MIN` is primary, with universal SG and arPLS paired sensitivities. The
five repeated four-fold P02 master splits are reused.

Inference uses a 10,000-resample physical-master-clustered hierarchical
bootstrap with 95% BCa intervals where stable and percentile intervals
otherwise. Substrate claims use an intersection-union rule; individual
secondary cell claims use Holm correction. Terminal failures remain visible
and receive common-endpoint and chance-performance sensitivities.

The field log remains corroborating: nonblank `Y` and blank `N` are analyzed as
separate detection and specificity endpoints, ambiguous `M` is excluded, and
missing entries remain missing with complete-case and best/worst bounds.

## Metadata-only support result

The frozen audit found 34 observed evaluation domains:

- 13 confirmatory;
- three exploratory low-support; and
- 18 unsupported by design.

The confirmatory domains are:

- CWA / H-SERS H-Kit: Mira-2, Pendar-2, RMX-1;
- pills / pSERS Metrohm silver: Agilent-1, Agilent-3, Mira-3, Pendar-1,
  Pendar-3; and
- surfaces / pSERS Metrohm silver: Agilent-3, Mira-1, Pendar-2, Pendar-3,
  RMX-2.

The exploratory domains are CWA / H-SERS H-Kit / Agilent-3 and pills / NRC
Canadian SERS / Agilent-1 and Mira-3. GaN/polymer cannot support three-class
classification in the relevant station/substrate combinations because the
observed support is confined to one analyte class.

The crossover audit found 34 observed analyte-specific two-substrate ×
two-instrument blocks: eight confirmatory, seven exploratory low-support, and
19 descriptive singletons.

## Reproducibility record

Input SHA-256:

- primary manifest:
  `db1f298a76aeb9962db004776a9f41d6c9afe5b76c39aa9277a24848108d5f90`
- P02 master splits:
  `d92da67742dd74693da518395c06dd1f33c16145d5d786a557166c8a6cb05558`

Frozen output SHA-256:

- domain support registry:
  `07a4f017cb7caaee6f18007601738c4f6d183cea5d7cb30f6eab1a15f0022fd8`
- crossover support registry:
  `ced0ac652651c4a85c9f596f825b0692ec300fda2c9b1769c585282125f5692c`

The generator is `scripts/build_p13_support_freeze.py`; the complete structured
summary is `registries/p13_support_freeze_summary.json`. P03 outcomes were known
before this amendment. No P13 predictive, crossover-effect, or field-log
outcome informed the decisions above.

## Next action

Build and validate the outcome-blind P13 classical execution manifest, then run
`EXP-P13-C01` through `EXP-P13-C04`. Do not change the locked rules after P13
outcome access; corrections require a versioned amendment and deviation entry.
