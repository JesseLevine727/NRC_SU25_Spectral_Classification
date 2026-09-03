# Supervisor figure disclosure review

**Review date:** 2026-08-07

**Release scope:** four descriptive NATO SERS figures supporting the supervisor update

## Approved content

- instrument-level robust spectral summaries;
- de-identified PCA coordinates colored by instrument or chemical class;
- de-identified physical-sample PCA, UMAP, and t-SNE views colored by chemical class;
- aggregate cluster-association values by preprocessing representation;
- PNG previews, vector PDFs, native TikZ sources, and standalone HTML versions.

## Excluded content

- source spectra and processed row-level spectra;
- figure data tables and representation arrays;
- observation or physical-sample identifiers;
- filenames, acquisition paths, notes, timestamps, and source-record keys;
- private workspace paths or source-organization identifiers.

## Review findings

- The interactive PCA hover text contains only instrument and chemical labels.
- The physical-sample maps contain no sample identifiers or join keys.
- The HTML files are standalone and do not load private files.
- The PDFs contain generic TeX/pdfTeX metadata and no private paths.
- The figures support descriptive statements only and do not report predictive performance.

Release was authorized by the user after the private/public boundary was explained. The underlying figure tables remain private.
