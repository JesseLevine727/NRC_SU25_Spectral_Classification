# P13 substrate-portability public release

This directory contains the disclosure-safe aggregate release for the locked
NATO SERS P13 classical substrate-portability study.

Start with [P13_RESULTS.md](P13_RESULTS.md). It explains the primary bounded
decision, fixed-model and preprocessing sensitivities, crossover evidence,
field-log corroboration, terminal failures, limitations, and next research
decision.

## Contents

- `tables/`: aggregate domain, interval, claim, multiplicity, preprocessing,
  procedure-comparison, crossover, field-log, and failure tables.
- `semantic/`: byte-stable source tables for F45–F47.
- `p13_figure_manifest.csv`: semantic and rendered-artifact hashes.
- `release_manifest.json`: run identity, protected-state hashes, public file
  hashes, and publication boundary.
- `../../plan/figures/`: native TikZ, vector PDF, PNG review copies, and
  standalone interactive HTML for F45–F47.

The public release excludes observation identifiers, physical-master IDs,
fold membership, row-level predictions, model binaries, calibration ledgers,
and source paths. Unsupported and terminal-failure endpoints remain visible in
the aggregate denominators.

The study used 598 stored spectra but treated the 69 physical masters as the
independent units. It does not claim universal substrate or instrument
independence.
