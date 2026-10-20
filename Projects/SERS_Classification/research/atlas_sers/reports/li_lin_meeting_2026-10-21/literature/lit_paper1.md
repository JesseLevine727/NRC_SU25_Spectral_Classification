# Paper 1 literature check: unseen-instrument SERS identification

Checked 2026-10-20. **Verified via:** CR = Crossref DOI record (title, authors, journal, volume, pages); PM = PubMed/Europe PMC abstract read; SS = Semantic Scholar abstract read; AX = arXiv record; NP = NeurIPS proceedings listing.

## Key references (slide)
1. **Fornasaro S, Alsamad F, Baia M, et al. (2020).** Surface Enhanced Raman Spectroscopy for Quantitative Analysis: Results of a Large-Scale European Multi-Instrument Interlaboratory Study. *Anal. Chem.* 92, 4053–4064. doi:10.1021/acs.analchem.9b05658. The first SERS interlaboratory study (15 labs); the standard citation for SERS reproducibility.
2. **Guo S, Beleites C, Neugebauer U, et al. (2020).** Comparability of Raman Spectroscopic Configurations: A Large Scale Cross-Laboratory Study. *Anal. Chem.* 92(24), 15745–15756. doi:10.1021/acs.analchem.0c02696. 35 devices: the same samples give different spectra on each setup.
3. **Mahmud S, Sayem FR, Hassan M, et al. (2026).** Deep learning-based cross-device standardization of surface-enhanced Raman spectroscopy for enhanced bacterial recognition. *Spectrochim. Acta A* 347, 126931. doi:10.1016/j.saa.2025.126931. Closest prior work: SERS on 4 portable instruments plus 1 lab instrument.
4. **Guo S, Bocklitz T, Neugebauer U, Popp J (2017).** Common mistakes in cross-validating classification models. *Anal. Methods* 9(30), 4410–4417. doi:10.1039/c7ay01363a. Split at the highest hierarchical level (independent replicates); supports splitting by physical sample.
5. **Baek S-J, Park A, Ahn Y-J, Choo J (2015).** Baseline correction using asymmetrically reweighted penalized least squares smoothing. *Analyst* 140(1), 250–257. doi:10.1039/c4an01061b. The original arPLS paper, behind our ~6-point gain.
6. **Mirsafavi R, Moskovits M, Meinhart C (2020).** Detection and classification of fentanyl and its precursors by surface-enhanced Raman spectroscopy. *Analyst* 145(9), 3440–3446. doi:10.1039/c9an02568e. SERS with hierarchical PLS-DA for fentanyl, 4-ANPP and NPP, which have near-identical spectra.

## Full verified list

### 1. Multi-instrument and interlaboratory reproducibility
- Fornasaro 2020 and Guo 2020: see above [CR+PM].
- **Bell SEJ, Charron G, Cortés E, et al. (2020).** Towards Reliable and Quantitative Surface-Enhanced Raman Scattering (SERS): From Key Parameters to Good Analytical Practice. *Angew. Chem. Int. Ed.* 59(14), 5454–5462. doi:10.1002/anie.201908154 [CR]. Community good-practice paper for SERS.
- **Guicheteau JA, Farrell ME, Christesen SD, et al. (2013).** Surface-Enhanced Raman Scattering (SERS) Evaluation Protocol for Nanometallic Surfaces. *Appl. Spectrosc.* 67(4), 396–403. doi:10.1366/12-06846 [CR]. Substrate evaluation protocol from the US Army ECBC; a precursor to the trial's question.
- **Hakonen A, Wu K, Stenbæk Schmidt M, et al. (2018).** Detecting forensic substances using commercially available SERS substrates and handheld Raman spectrometers. *Talanta* 189, 649–652. doi:10.1016/j.talanta.2018.07.009 [CR+SS]. Commercial substrates on handheld instruments (cyclosarin, RDX, amphetamine). No ML.
- **Lange C, Borisyak M, Kögler M, et al. (2025).** Comparing machine learning methods on Raman spectra from eight different spectrometers. *Spectrochim. Acta A* 334, 125861. doi:10.1016/j.saa.2025.125861 [CR+PM]. One CNN pooled over 8 instruments beats PLS. Regression only; no instrument is unseen.

### 2. Calibration and model transfer
- **Wang Y, Veltkamp DJ, Kowalski BR (1991).** Multivariate instrument standardization. *Anal. Chem.* 63(23), 2750–2756. doi:10.1021/ac00023a016 [CR]. Introduced DS and PDS.
- **Workman JJ (2018).** A Review of Calibration Transfer Practices and Instrument Differences in Spectroscopy. *Appl. Spectrosc.* 72(3), 340–365. doi:10.1177/0003702817736064 [CR].
- **Rodriguez JD, Westenberger BJ, Buhse LF, Kauffman JF (2011).** Standardization of Raman spectra for transfer of spectral libraries across different instruments. *Analyst* 136(20), 4232. doi:10.1039/c1an15636e [CR+PM]. Five 785 nm instruments, including handheld units.
- **Mishra P, Passos D (2021).** Deep calibration transfer: Transferring deep learning models between infrared spectroscopy instruments. *Infrared Phys. Technol.* 117, 103863. doi:10.1016/j.infrared.2021.103863 [CR+SS]. Classical transfer methods don't suit deep-learning models; includes handheld NIR.
- **Lai J, Li M, Chen S, et al. (2025).** Calibration Transfer of Deep Learning Models among Multiple Raman Spectrometers via Low-Rank Adaptation. *Anal. Chem.* 97(35), 19009–19018. doi:10.1021/acs.analchem.5c01846 [CR+PM]. Beats PDS, but needs transfer samples from the target instrument.
- **Li Y, Wang Z, Liang P, et al. (2026).** A robust model transfer approach for portable Raman spectrometers: Enabling interoperable analysis across measurement environments. *Spectrochim. Acta A* 360, 127981. doi:10.1016/j.saa.2026.127981 [CR+PM]. Maps handheld spectra to benchtop using standard materials.
- **Zhang Z, Xu Y, Meng S, et al. (2026).** Novel Deep-Learning Unsupervised Domain Adaptation Method for Mitigating Batch, Strain, and Instrument Variations to Enhance Raman Spectroscopy-Based Bacterial Pathogen Identification. *Anal. Chem.* 98(12), 9014–9026. doi:10.1021/acs.analchem.5c07113 [CR+PM].
- **She H, Wang F, Bai X, et al. (2026).** Cross-Device Risk Calibration for Surface-Enhanced Raman Spectroscopy-Based Urologic Disease Classification. *Anal. Chem.* 98(31), 23023–23031. doi:10.1021/acs.analchem.6c02588 [CR+PM]. Cross-device SERS shift handled by few-shot alignment, which uses target data.
- **Li B, Schmidt MN, Alstrøm TS (2022).** Raman spectrum matching with contrastive representation learning. *Analyst* 147(10), 2238–2246. doi:10.1039/d2an00403h [CR+AX].
- **Khosla P, Teterwak P, Wang C, et al. (2020).** Supervised Contrastive Learning. *NeurIPS 33*; arXiv:2004.11362 [AX+NP]. Source of our SupCon loss.

### 3. Fentanyl, 4-ANPP and CWA simulants
- **Haddad A, Comanescu MA, Green O, Kubic TA, Lombardi JR (2018).** Detection and Quantitation of Trace Fentanyl in Heroin by Surface-Enhanced Raman Spectroscopy. *Anal. Chem.* 90(21), 12678–12685. doi:10.1021/acs.analchem.8b02909 [CR].
- **Wang L, Vendrell-Dones MO, Deriu C, et al. (2021).** Multivariate Analysis Aided Surface-Enhanced Raman Spectroscopy (MVA-SERS) Multiplex Quantitative Detection of Trace Fentanyl in Illicit Drug Mixtures Using a Handheld Raman Spectrometer. *Appl. Spectrosc.* 75(10), 1225–1236. doi:10.1177/00037028211032930 [CR+PM]. Handheld SERS with PCA and SPLS-DA.
- **Smith M, Logan M, Bazley M, et al. (2021).** A Semi-quantitative method for the detection of fentanyl using surface-enhanced Raman scattering (SERS) with a handheld Raman instrument. *J. Forensic Sci.* 66(2), 505–519. doi:10.1111/1556-4029.14610 [CR+PM]. Handheld SORS; telling analogues apart was hard.
- **Cooman T, Ott CE, Arroyo LE (2023).** Evaluation and classification of fentanyl-related compounds using EC-SERS and machine learning. *J. Forensic Sci.* 68(5), 1520–1526. doi:10.1111/1556-4029.15285 [CR+PM]. CNN: 98% in cross-validation, 85% on casework samples.
- **Wilson NG, Raveendran J, Docoslis A (2021).** Portable identification of fentanyl analogues in drugs using surface-enhanced Raman scattering. *Sens. Actuators B* 330, 129303. doi:10.1016/j.snb.2020.129303 [CR+SS].
- **Hakonen A, Andersson PO, Stenbæk Schmidt M, et al. (2015).** Explosive and chemical threat detection by surface-enhanced Raman scattering: A review. *Anal. Chim. Acta* 893, 1–13. doi:10.1016/j.aca.2015.04.010 [CR].
- **Wu J, Li F, Zhou JW, et al. (2025).** A Raman spectroscopy algorithm based on convolutional neural networks and multilayer perceptrons: qualitative and quantitative analyses of chemical warfare agent simulants. *Analyst* 150(9), 1823–1836. doi:10.1039/d5an00075k [CR+PM]. Normal Raman, not SERS.
- **Tay L-L, Poirier S, Ghaemi A, et al. (2021).** Paper-based surface-enhanced Raman spectroscopy sensors for field applications. *J. Raman Spectrosc.* 52(2), 563–572. doi:10.1002/jrs.6017 [CR]. NRC inkjet-printed SERS read on a handheld analyzer; possibly the trial's NRC substrate.
- **Mozaffari MH, Tay L-L (2022).** Overfitting One-Dimensional convolutional neural networks for Raman spectra identification. *Spectrochim. Acta A* 272, 120961. doi:10.1016/j.saa.2022.120961 [CR+PM]. NRC 1D-CNN work for handheld spectrometers.

### 4. Leakage, validation, classical ML vs deep learning
- **Esbensen KH, Geladi P (2010).** Principles of Proper Validation: use and abuse of re-sampling for validation. *J. Chemom.* 24(3–4), 168–187. doi:10.1002/cem.1310 [CR].
- **Kapoor S, Narayanan A (2023).** Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4(9), 100804. doi:10.1016/j.patter.2023.100804 [CR].
- **Guo S, Popp J, Bocklitz T (2021).** Chemometric analysis in Raman spectroscopy from experimental design to machine learning–based modeling. *Nat. Protoc.* 16(12), 5426–5459. doi:10.1038/s41596-021-00620-3 [CR]. Best-practice protocol.
- **Beleites C, Neugebauer U, Bocklitz T, et al. (2013).** Sample size planning for classification models. *Anal. Chim. Acta* 760, 25–33. doi:10.1016/j.aca.2012.11.007 [CR]. Relevant with only 69 independent samples.
- **Grinsztajn L, Oyallon E, Varoquaux G (2022).** Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 35* (Datasets & Benchmarks); arXiv:2207.08815 [AX+NP].
- **Sineesh A, Kamsali AR (2026).** Benchmarking deep learning models for Raman spectroscopy across open-source datasets. *Digital Discovery* 5(8), 3314–3327. doi:10.1039/d6dd00044d [CR+AX]. 5 deep-learning and 2 classical models; includes a distribution-shift test.
- **Koddenbrock M, Lange C, Legner R, et al. (2026).** RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy. arXiv:2605.02003 (journal-ref: NeurIPS 2026) [AX]. 77 datasets; appendix: "Classical methods remain competitive in the small-data regime."

### 5. Preprocessing
- **Savitzky A, Golay MJE (1964).** Smoothing and Differentiation of Data by Simplified Least Squares Procedures. *Anal. Chem.* 36(8), 1627–1639. doi:10.1021/ac60214a047 [CR].
- **Liland KH, Almøy T, Mevik B-H (2010).** Optimal Choice of Baseline Correction for Multivariate Calibration of Spectra. *Appl. Spectrosc.* 64(9), 1007–1016. doi:10.1366/000370210792434350 [CR+SS]. Tune the baseline method to the downstream model; fits arPLS hurting some instruments.
- **Bocklitz T, Walter A, Hartmann K, et al. (2011).** How to pre-process Raman spectra for reliable and stable models? *Anal. Chim. Acta* 704(1–2), 47–56. doi:10.1016/j.aca.2011.06.043 [CR+SS].

### 6. Deep learning for Raman (context)
- **Ho C-S, Jean N, Hogan CA, et al. (2019).** Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning. *Nat. Commun.* 10, 4927. doi:10.1038/s41467-019-12898-9 [CR+PM]. Confirmed (30 pathogens).
- **Liu J, Osadchy M, Ashton L, et al. (2017).** Deep convolutional neural networks for Raman spectrum recognition: a unified solution. *Analyst* 142(21), 4067–4074. doi:10.1039/c7an01371j [CR].
- **Lussier F, Thibault V, Charron B, Wallace GQ, Masson J-F (2020).** Deep learning and artificial intelligence methods for Raman and surface-enhanced Raman scattering. *TrAC* 124, 115796. doi:10.1016/j.trac.2019.115796 [CR+SS].

## Closest prior work and novelty gap
- **Mahmud 2026** differs from Paper 1 in three ways: bacteria, a single AgNR substrate, and a learned mapping from portable to lab spectra. Its abstract doesn't say whether instruments were held out or splits grouped by sample.
- **Lai 2025, She 2026, Zhang 2026 and Li 2026** all use data from the target instrument.
- **Lange 2025** pools every instrument in training.
- **Guo 2020 and Fornasaro 2020** characterize variability but don't benchmark identification.
- **Hakonen 2018** has no ML.

**Plausible novelty.** No verified paper combines all four of the following:
1. Identification on a fully unseen handheld instrument, with splits grouped by sample.
2. Threat analytes across 10 instruments from 4 vendors and 4 substrate families.
3. Classical ML vs CNN under the same leakage controls.
4. An equivalence test of the substrate instrument-independence claim.

**Caveats.**
- Classical ≈ CNN is consistent with Grinsztajn and RamanBench, so it confirms rather than surprises.
- The contrastive gain (≤2 points) is a negative result.
- The arPLS gain echoes Liland and Bocklitz.
- 69 samples means wide confidence intervals (Beleites).

The contribution is the protocol plus a rare dataset, not a new method. The trial appears to be NATO-SET-292, "Enhanced Raman Spectroscopy for Defense Applications": six nations including Canada, a lab field trial on March 7 at Aberdeen Proving Ground (army.mil/article/275093, 2 Apr 2024, which I opened). Check for a NATO technical report before submitting.

## Venue recommendations
1. **Analyst (RSC). First choice.** Published Baek, Liu, Mirsafavi, Rodriguez, Li 2022 and Wu.
2. **Analytical Chemistry. Reach.** Published Fornasaro, Guo 2020, Lai and She. A benchmark with negative results is a harder sell.
3. **Applied Spectroscopy.** Published Workman, Wang 2021 (handheld fentanyl), Guicheteau and Liland. Its readers value instrument comparisons.
4. **Analytica Chimica Acta.** Strong on validation (Bocklitz, Beleites, Hakonen 2015).
5. **Journal of Raman Spectroscopy.** Specialist journal; NRC's Tay 2021 appeared there. Lower impact.
6. **Digital Discovery.** Only if code and data are open; Sineesh 2026 appeared there.

Less suitable:
- **Talanta:** OK for the forensic angle (Hakonen 2018).
- **Spectrochimica Acta A:** fast but lower prestige, and the closest competitors (Mahmud, Lange) are there.
- **Chemometrics and Intelligent Laboratory Systems:** narrow SERS readership.

**Data descriptors.**
- **Scientific Data** publishes Raman datasets (Flanagan AR, Glavin FG 2025, *Sci. Data* 12, 498, doi:10.1038/s41597-025-04848-6 [CR]). It requires public deposition, so NATO/DND release clearance comes first.
- **Data in Brief** publishes SERS datasets (Kundalevich A et al. 2024, *Data Brief* 57, 110965, doi:10.1016/j.dib.2024.110965 [CR]). Lower bar; can accompany an Elsevier research paper.

## Unverified
- No public NATO-SET-292 report or task-group paper found. A TÜBİTAK closing-meeting page (Oct 2024) appeared in search results but redirected when opened.
- No ML-on-SERS study found for **4-nitrophenol** or **paraoxon** specifically.
- Sineesh 2026 numbers (RF 69.2% vs CNN 79.7%) come from a search snippet only.
- Mahmud 2026's split design: full text not accessible.
- Chen M et al. 2023, *Vib. Spectrosc.* 127, 103567 (doi:10.1016/j.vibspec.2023.103567): exists per Crossref; abstract not read.
- Origin of the GaN/polymer substrate and identity of the NRC substrate: not confirmed.
