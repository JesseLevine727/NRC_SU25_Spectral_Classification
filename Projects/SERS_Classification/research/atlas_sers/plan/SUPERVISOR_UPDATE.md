# ATLAS SERS research update

I have been organizing and examining the ATLAS field-trial SERS dataset to determine which research questions it can reliably address. The working dataset has 598 spectra from 69 physical samples, collected using 10 instruments across 7 chemical classes. The main preliminary finding is that spectra still group strongly by instrument after common scaling, which could obscure chemical differences. Background correction helps in some cases but can also remove useful chemical structure.

I am therefore setting up a controlled comparison of preprocessing and classification methods aimed at identifying chemicals on instruments and samples not used during method development. The next step is to define the training and independent test groups in advance, benchmark classical machine-learning methods, and then compare them with deep-learning methods designed to handle instrument differences. Once this workflow is stable, I will begin examining the cell dataset in parallel.

## Research questions

1. Can chemicals be identified on instruments and physical samples not used during training?
2. Which preprocessing best reduces background and noise while preserving useful chemical peaks?
3. Should preprocessing be universal or adapted according to instrument family or measured spectrum quality?
4. Does classical machine learning or deep learning provide better chemical classification on an unseen instrument?
5. How much do a few calibration spectra from a new instrument improve performance?
6. Can the method flag unfamiliar chemicals or unreliable classifications?

## Immediate next step

Freeze the independent training and test groups, benchmark the classical methods, and then evaluate instrument-aware deep-learning approaches under the same conditions.
