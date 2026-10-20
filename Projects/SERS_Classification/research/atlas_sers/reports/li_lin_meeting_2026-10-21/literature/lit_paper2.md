# Paper 2: verified literature check (crossed SERS dataset + disentangled VAEs)

All checks done 2026-10-20. Verification tags: **[CR]** = I resolved the DOI on the Crossref API and the title, authors, venue and year matched. **[PMLR] / [JMLR] / [NeurIPS]** = the official proceedings page. **[ICLR]** = the iclr.cc virtual page. **[MLA]** = the ML Anthology page that links to OpenReview (openreview.net blocked automated access). **[arXiv]** = the arXiv API record and its venue note.

## Key references (for the slide)

- **K1.** Locatello F, Bauer S, Lucic M, et al. Challenging common assumptions in the unsupervised learning of disentangled representations. *ICML 2019*, PMLR 97:4114–4124. https://proceedings.mlr.press/v97/locatello19a.html [PMLR]. **Why it matters:** without supervision or inductive biases, unsupervised disentanglement is impossible, so the factors must be labelled and crossed.
- **K2.** Khemakhem I, Kingma DP, Monti RP, Hyvärinen A. Variational autoencoders and nonlinear ICA: a unifying framework. *AISTATS 2020*, PMLR 108:2207–2217. https://proceedings.mlr.press/v108/khemakhem20a.html [PMLR]. **Why it matters:** latents become identifiable once an observed auxiliary variable is available. Here that variable is the analyte, substrate and instrument labels.
- **K3.** Montero ML, Ludwig CJH, Costa RP, et al. The role of disentanglement in generalisation. *ICLR 2021*. https://iclr.cc/virtual/2021/poster/3173 [ICLR]. **Why it matters:** VAEs generalised only weakly to held-out factor combinations, so the compositional test must be designed in.
- **K4.** Lotfollahi M, Klimovskaia Susmelj A, De Donno C, et al. Predicting cellular responses to complex perturbations in high-throughput screens. *Mol Syst Biol* 2023, 19, e11517. doi:10.15252/msb.202211517 [CR]. **Why it matters:** a working template, the compositional perturbation autoencoder (CPA), which predicts unseen drug × cell-type × dose combinations.
- **K5.** Fornasaro S, Alsamad F, Baia M, et al. Surface enhanced Raman spectroscopy for quantitative analysis: results of a large-scale European multi-instrument interlaboratory study. *Anal Chem* 2020, 92, 4053–4064. doi:10.1021/acs.analchem.9b05658 [CR + full text]. **Why it matters:** 15 labs and 6 Ag/Au SERS methods, but one analyte (adenine). The authors say the results cannot be extrapolated to other substrates or analytes.
- **K6.** Mahmud S, Sayem F, Hassan M, et al. Deep learning-based cross-device standardization of surface-enhanced Raman spectroscopy for enhanced bacterial recognition. *Spectrochim Acta A* 2026, 347, 126931. doi:10.1016/j.saa.2025.126931 [CR]. **Why it matters:** the closest SERS ML study (20 analytes, 5 instruments, one AgNR substrate). It covers instrument shift only.

## Full verified list

### 1. VAEs and autoencoders on Raman/SERS
- Thrift WJ, Ronaghi S, Samad M, et al. Deep learning analysis of vibrational spectra of bacterial lysate for rapid antimicrobial susceptibility testing. *ACS Nano* 2020, 14, 15336–15348. doi:10.1021/acsnano.0c05693 [CR]. A VAE on SERS spectra exposed features linked to antibiotic response.
- He C, Zhu S, Wu X, et al. Accurate tumor subtype detection with Raman spectroscopy via variational autoencoder and machine learning. *ACS Omega* 2022, 7, 10458–10468. doi:10.1021/acsomega.1c07263 [CR].
- Georgiev D, Fernández-Galiana Á, Vilms Pedersen S, et al. Hyperspectral unmixing for Raman spectroscopy via physics-constrained autoencoders. *PNAS* 2024, 121, e2407439121. doi:10.1073/pnas.2407439121 [CR]. Autoencoders recover the spectra of pure components, which is the same task as goal (a).
- Horgan CC, Jensen M, Nagelkerke A, et al. High-throughput molecular imaging via deep-learning-enabled Raman spectroscopy. *Anal Chem* 2021, 93, 15850–15860. doi:10.1021/acs.analchem.1c02178 [CR]. Deep-learning denoising.
- Lussier F, Thibault V, Charron B, et al. Deep learning and artificial intelligence methods for Raman and surface-enhanced Raman scattering. *TrAC* 2020, 124, 115796. doi:10.1016/j.trac.2019.115796 [CR]. Review from the Masson group, Université de Montréal.

### 2. Disentanglement on spectra and other scientific signals
- Abram KJ, McCloskey D. In search of disentanglement in tandem mass spectrometry datasets. *Biomolecules* 2023, 13, 1343. doi:10.3390/biom13091343 [CR]. VAEs disentangle MS/MS factors, including **instrument type**. This is the nearest methodological analogue.
- Shuai W, Tian X, Zuo E, et al. Disentangled global and local features of multi-source data variational autoencoder… *Artif Intell Med* 2025, 160, 103053. doi:10.1016/j.artmed.2024.103053 [CR]. Separates shared Raman features from source-specific ones.
- Liu Y, Vasudevan RK, Kelley KP, et al. Decoding the shift-invariant data: applications for band-excitation scanning probe microscopy. *Mach Learn Sci Technol* 2021, 2, 045028. doi:10.1088/2632-2153/ac28de [CR]. A VAE separates peak shift from the other factors.

### 3. Identifiability theory
- Kingma DP, Welling M. Auto-encoding variational Bayes. *ICLR 2014*. https://arxiv.org/abs/1312.6114 [arXiv; the ICLR 2014 listing is confirmed in the University of Amsterdam repository].
- Higgins I, Matthey L, Pal A, et al. β-VAE: learning basic visual concepts with a constrained variational framework. *ICLR 2017*. https://openreview.net/forum?id=Sy2fzU9gl [MLA].
- Sohn K, Lee H, Yan X. Learning structured output representation using deep conditional generative models. *NeurIPS 2015* (vol. 28) [NeurIPS]. The conditional VAE.
- Hyvärinen A, Pajunen P. Nonlinear independent component analysis: existence and uniqueness results. *Neural Netw* 1999, 12, 429–439. doi:10.1016/S0893-6080(98)00140-3 [CR]. The seminal non-identifiability result.
- Locatello F, Tschannen M, Bauer S, et al. Disentangling factors of variation using few labels. *ICLR 2020*. https://arxiv.org/abs/1905.01258 [arXiv]. A few labels are enough.
- Locatello F, Poole B, Rätsch G, et al. Weakly-supervised disentanglement without compromises. *ICML 2020*, PMLR 119:6348–6359 [PMLR]. Pairs of samples that differ in only some factors make the factors identifiable.
- Bouchacourt D, Tomioka R, Nowozin S. Multi-level variational autoencoder: learning disentangled representations from grouped observations. *AAAI* 2018, 32(1). doi:10.1609/aaai.v32i1.11867 [CR]. Here the groups would be replicates.
- Träuble F, Creager E, Kilbertus N, et al. On disentangled representations learned from correlated data. *ICML 2021*, PMLR 139:10401–10412 [PMLR]. Correlated factors end up entangled.
- Lachapelle S, Mahajan D, Mitliagkas I, Lacoste-Julien S. Additive decoders for latent variables identification and Cartesian-product extrapolation. *NeurIPS 2023* (vol. 36) [NeurIPS]. Structured decoders can provably generate unseen combinations (Mila).
- K1, K2 and K3 above also belong here.

### 4. Instrument and substrate translation or standardization
- Wang Y, Veltkamp DJ, Kowalski BR. Multivariate instrument standardization. *Anal Chem* 1991, 63, 2750–2756. doi:10.1021/ac00023a016 [CR]. Seminal piecewise direct standardization (PDS).
- Workman JJ. A review of calibration transfer practices and instrument differences in spectroscopy. *Appl Spectrosc* 2018, 72, 340–365. doi:10.1177/0003702817736064 [CR].
- Zhu J-Y, Park T, Isola P, Efros AA. Unpaired image-to-image translation using cycle-consistent adversarial networks. *ICCV* 2017, 2242–2251. doi:10.1109/ICCV.2017.244 [CR]. CycleGAN.
- Mishra P, Passos D. Deep calibration transfer: transferring deep learning models between infrared spectroscopy instruments. *Infrared Phys Technol* 2021, 117, 103863. doi:10.1016/j.infrared.2021.103863 [CR].
- Wang T, Yang Y, Lu H, et al. Functional regression for SERS spectrum transformation across diverse instruments. *Analyst* 2025, 150, 460–469. doi:10.1039/d4an01177e [CR].
- Guo S, Beleites C, Neugebauer U, et al. Comparability of Raman spectroscopic configurations: a large scale cross-laboratory study. *Anal Chem* 2020, 92, 15745–15756. doi:10.1021/acs.analchem.0c02696 [CR]. Across 35 devices, peak positions, intensities, widths and noise all differed.
- K6 above, a seq-to-seq network that maps portable-instrument spectra onto a laboratory reference.
- No peer-reviewed CycleGAN study for Raman/SERS was found.

### 5. Batch-effect and nuisance-removal analogues
- Leek JT, Scharpf RB, Bravo HC, et al. Tackling the widespread and critical impact of batch effects in high-throughput data. *Nat Rev Genet* 2010, 11, 733–739. doi:10.1038/nrg2825 [CR]. Batch effects that are confounded with the biology cannot be corrected afterwards.
- Lopez R, Regier J, Cole MB, et al. Deep generative modeling for single-cell transcriptomics. *Nat Methods* 2018, 15, 1053–1058. doi:10.1038/s41592-018-0229-2 [CR]. scVI.
- Ganin Y, Ustinova E, Ajakan H, et al. Domain-adversarial training of neural networks. *JMLR* 2016, 17(59):1–35. https://jmlr.org/papers/v17/15-239.html [JMLR + PDF].
- Piran Z, Cohen N, Hoshen Y, Nitzan M. Disentanglement of single-cell data with biolord. *Nat Biotechnol* 2024, 42, 1678–1683. doi:10.1038/s41587-023-02079-x [CR].
- K4 above (CPA).

### 6. Public data for pretraining
- Lafuente B, Downs RT, Yang H, Stone N. The power of databases: the RRUFF project. In *Highlights in Mineralogical Crystallography*, De Gruyter 2015, 1–30. doi:10.1515/9783110417104-003 [CR].
- De Gelder J, De Gussem K, Vandenabeele P, Moens L. Reference database of Raman spectra of biological molecules. *J Raman Spectrosc* 2007, 38, 1133–1147. doi:10.1002/jrs.1734 [CR].
- Ho C-S, Jean N, Hogan CA, et al. Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning. *Nat Commun* 2019, 10, 4927. doi:10.1038/s41467-019-12898-9 [CR]. Covers 30 pathogens; the data are linked from github.com/csho33/bacteria-ID.
- Flanagan AR, Glavin FG. Open-source Raman spectra of chemical compounds for active pharmaceutical ingredient development. *Sci Data* 2025, 12, 498. doi:10.1038/s41597-025-04848-6 [CR]. 3,510 spectra of 32 compounds.
- Mahmud S. Multi-Instrument SERS Dataset. Zenodo 2026. doi:10.5281/zenodo.23372983 [Zenodo]. 29,545 SERS spectra from 5 spectrometers at 785 nm, one AgNR substrate, CC BY 4.0. The files need a Zenodo login.
- Sineesh A, Kamsali A. Benchmarking deep learning models for Raman spectroscopy across open-source datasets. *Digital Discovery* 2026, 5, 3314–3327. doi:10.1039/d6dd00044d [CR].
- No public multi-substrate × multi-instrument SERS dataset was found.

### 7. Sample size and model capacity
- Baum EB, Haussler D. What size net gives valid generalization? *Neural Comput* 1989, 1, 151–160. doi:10.1162/neco.1989.1.1.151 [CR].
- Zhang C, Bengio S, Hardt M, et al. Understanding deep learning requires rethinking generalization. *ICLR 2017*. https://arxiv.org/abs/1611.03530 [arXiv]. Deep networks can memorise random labels.
- Belkin M, Hsu D, Ma S, Mandal S. Reconciling modern machine-learning practice and the classical bias–variance trade-off. *PNAS* 2019, 116, 15849–15854. doi:10.1073/pnas.1903070116 [CR]. Double descent.
- Beleites C, Neugebauer U, Bocklitz T, et al. Sample size planning for classification models. *Anal Chim Acta* 2013, 760, 25–33. doi:10.1016/j.aca.2012.11.007 [CR]. The spectroscopy-specific guidance (from the Popp group).

### 8. Dataset design and reporting
- Guo S, Popp J, Bocklitz T. Chemometric analysis in Raman spectroscopy from experimental design to machine learning–based modeling. *Nat Protoc* 2021, 16, 5426–5459. doi:10.1038/s41596-021-00620-3 [CR].
- Bell SEJ, Charron G, Cortés E, et al. Towards reliable and quantitative surface-enhanced Raman scattering (SERS): from key parameters to good analytical practice. *Angew Chem Int Ed* 2020, 59, 5454–5462. doi:10.1002/anie.201908154 [CR].
- Guicheteau JA, Farrell ME, Christesen SD, et al. Surface-enhanced Raman scattering (SERS) evaluation protocol for nanometallic surfaces. *Appl Spectrosc* 2013, 67, 396–403. doi:10.1366/12-06846 [CR]. A US defence protocol for comparing substrates.
- Choquette SJ, Etz ES, Hurst WS, et al. Relative intensity correction of Raman spectrometers: NIST SRMs 2241 through 2243… *Appl Spectrosc* 2007, 61, 117–129. doi:10.1366/000370207779947585 [CR].
- Wang C, Xiao L, Dai C, et al. A statistical approach of background removal and spectrum identification for SERS data. *Sci Rep* 2020, 10, 1460. doi:10.1038/s41598-020-58061-z [CR]. Substrate and solvent backgrounds dominate the analyte signal.
- Artrith N, Butler KT, Coudert F-X, et al. Best practices in machine learning for chemistry. *Nat Chem* 2021, 13, 505–508. doi:10.1038/s41557-021-00716-z [CR].
- Gebru T, Morgenstern J, Vecchione B, et al. Datasheets for datasets. *Commun ACM* 2021, 64(12), 86–92. doi:10.1145/3458723 [CR].

## Closest prior work and novelty gap

I found **no crossed analyte × substrate × instrument SERS study that uses disentangled or structured generative models.** The nearest work:
- **K5 (Fornasaro 2020):** many labs and substrates, but one analyte and no ML.
- **Guo 2020:** many instruments, but conventional Raman rather than SERS.
- **Wang 2025 and K6 (Mahmud 2026):** 20 analytes on 5 instruments, but one substrate, using supervised translation.
- **Abram 2023:** a VAE disentangles instrument type, but for MS/MS.
- **K4 (CPA) and biolord:** compositional generalisation, but in single-cell biology.

**Caveat:** I searched the web and Crossref, not Scopus or Web of Science, and the Mahmud dataset only appeared on 14 Oct 2026. Rerun a Scopus search before submission.

## Dataset-design implications

- **Label every factor.** Analyte, substrate, instrument and preparation IDs are the auxiliary variables that make the latents identifiable (K1, K2).
- **Cross the factors.** Correlated factors become entangled (Träuble 2021), and confounded batch effects cannot be corrected afterwards (Leek 2010). This is exactly what went wrong with the NATO data.
- **Measure the same preparation on every instrument.** These grouped and paired observations act as weak supervision (Bouchacourt 2018; Locatello 2020). Size the study by counting independent preparations, not spectra (Beleites 2013).
- **Pre-register the held-out cells.** Every factor level must still appear in training, which is the Cartesian-product condition in Lachapelle 2023. Report the held-out results separately (K3).
- **Include physical anchors.** Neat Raman references are the ground truth for goal (a). Substrate blanks capture the dominant background (Wang 2020). NIST SRM intensity standards (Choquette 2007) cover the instrument differences documented in Guo 2020.

## Venue recommendations

1. **Analytical Chemistry:** the home of Fornasaro 2020, Guo 2020, Horgan 2021 and Wang 1991. Lead with the metrology and reference-spectrum recovery.
2. **Digital Discovery (RSC):** a data-centric chemistry-ML journal that published the 2026 Raman deep-learning benchmark (Sineesh & Kamsali).
3. **Machine Learning: Science and Technology (IOP):** the best fit if identifiability and compositional generalisation are the headline. It published the shift-invariant spectral VAE (Liu 2021).
4. **Nature Communications:** a stretch target if the model achieves zero-shot transfer to new substrates or instruments. Ho 2019 is the precedent.
5. **ACS Sensors:** suits a field-deployment framing. I found no direct precedent.
6. **Workshops (ICML/NeurIPS AI4Science, NeurIPS ML4PS):** non-archival, good for early feedback. Georgiev's unmixing work appeared at ICML 2024 AI4Science before PNAS [MLA].

Chemical Science and JCIM are weaker fits; I found no precedents in either.

**Scientific Data: yes.** Flanagan & Glavin 2025 is a Raman precedent there. A Data Descriptor gives the dataset a DOI that Paper 2 can cite. The alternative is the NeurIPS Datasets & Benchmarks track: Alberts M, Schilter O, Zipoli F, et al. Unraveling molecular structure: a multimodal spectroscopic dataset for chemistry. *NeurIPS* 2024 (vol. 37). doi:10.52202/079017-3996 [CR].

## Unverified (do not show)

- ASTM E1840 Raman-shift standard: the publisher page was blocked.
- The NIST EC-SERS drug spectral database (data.gov listing only).
- The exact spectrum count in Ho 2019.
- SERS reviews whose DOIs exist but which I did not read (Quarin 2025 *npj Biosensing*; Tan 2025 *ACS Cent Sci*).
- Preprints arXiv 2412.08131 and 2607.10196.
- A Raman/SERS CycleGAN paper.
