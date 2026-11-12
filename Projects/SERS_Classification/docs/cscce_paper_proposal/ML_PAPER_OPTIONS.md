# Every possible ML paper from the data we already have

2 November 2026. Internal scoping. Data-set letters follow `PAPER_SCOPING.md`:
- A: poster maps
- B: LabRAM thiols/acid
- C: XploRA thiols/acid
- D: handheld sensors
- NATO: field trial

## The constraint shared by every option

About 50 independent SERS sensors outside NATO, mostly one per chemical × substrate cell, over three instruments. With that few sensors no paper can claim one method *beats* another (with 0 errors in 11 held-out sensors the 95% upper error bound is still 28.5%). So a credible ML paper needs one of these:
- **a method contribution** with a clear mechanism, tested under a sealed design;
- **a negative result** with a preregistered protocol;
- **more independent units** (public data, or NATO).

## Options

| # | ML paper | Data | Novelty (as far as checked) | Main risk | Venue | Effort |
|---|---|---|---|---|---|---|
| M1 | "Count sensors, not spectra": sizing and validating networks on few-sensor SERS data (Option A) | A–D | **Checked.** Units, checklist and within/cross-domain reporting are known. New: substrate-held-out learning curve; Zhang–Gupta sizing on spectra | Not enough for a full paper; contribution 1 vs 3 contradiction; sensitive given Mozaffari & Tay 2022 | Short communication (*Anal. Methods*, *J. Raman Spectrosc.*) | 2–3 weeks |
| **M2** | **Background-swap augmentation:** teach networks substrate invariance by swapping measured substrate backgrounds between spectra during training | A (develop), B/C/D (sealed tests), Au blanks | **Not checked.** Generic spectral augmentation (offset, slope, baseline) is known, e.g. for NIR CNNs. Swapping *measured SERS substrate backgrounds*, with leave-one-substrate-type-out testing, may be new | Might not work. Small-data ranking limits. Needs a public set to generalise | *Chemom. Intell. Lab. Syst.*, *Analyst*, *J. Raman Spectrosc.*, *Digital Discovery* | Prototype on A: 1–2 days. Paper: 5–7 weeks |
| M3 | Domain-generalisation benchmark across SERS substrates (ERM vs CORAL/DANN/group-DRO/IRM vs derivative matching) | A–D | Not checked. DG benchmarks are common in ML; on SERS substrates, unclear | Likely "nothing beats plain training" (as in Gulrajani & Lopez-Paz 2021), and too few domains to tell methods apart | *Digital Discovery*, CILS | 4–6 weeks |
| M4 | Open-set identification across substrates: reject chemicals not in the library (conformal prediction at sensor level) | A–D (held-out chemicals act as unknowns) | Partly known (open-set Raman exists). The NATO open-world CNN already failed all 6 of its gates | Very few unknown chemicals and sensors | CILS, *Analyst* | 4–6 weeks |
| M5 | Knowledge-based network: a smooth background model plus band templates, with only a few learned parameters | A–D + Raman references | Not checked. **Overlaps Paper 2's physics-structured model** | Cannibalises Paper 2; uncertain result | – | – |
| M6 | NATO companion ML paper: the July preregistered studies (Siamese 0.632 vs classical 0.685; supervised contrastive 0.701; open-world CNN 0/6 gates; structured VAE unsuccessful) as a negative-results paper | NATO | Negative results with preregistration are publishable; not checked | **Same data set as Paper 1** and the Paper 2 motivation; must be agreed with Li-Lin | *Digital Discovery*, CILS | Largely done; 3–4 weeks writing |
| M7 | Matching SERS spectra to a normal-Raman library across substrates | A–D + Raman references | **Pre-empted in part** by CaPSim (Ju et al. 2023, *ACS Nano*) | Pilot found SERS ≠ Raman for 3 of 6 analytes | – | – |

## Recommendation

**M2 is the only real ML-method paper the existing data can support.** It turns our main finding into a method:
- substrate shift in SERS is mostly broad background;
- the derivative removes that background for library matching;
- background-swap augmentation teaches a network the same invariance, so the network stops learning substrate features (which is how the CNN failed on Ag and PICO).

Whether it works is unknown. That is why the first step is a cheap feasibility test, not a commitment.

**Fallbacks:**
- M1 as a short communication.
- M6, if Li-Lin agrees that a NATO companion paper is acceptable alongside Paper 1.
- Otherwise Option B (Technical Note) with the neural networks in the ESI.

## Step 1: M2 feasibility, before any decision (1–2 days)

1. On set A only (development), split each training spectrum into a smooth background (arPLS, as in the NATO preprocessing) and a residual (bands + noise).
2. During training, recombine each residual with a background drawn from another training spectrum, preferably from another substrate. Add the Au blanks as extra backgrounds.
3. Train the sized CNN / Siamese network with and without the swap. Use the same splits, seeds and recipe as the sizing sweep, and inner-fold selection only.
4. **Go / no-go:** on the held-back substrate type in the inner folds, does the swap close most of the gap between the CNN (about 10–12% error) and derivative matching (7.4%)? If not, stop.
5. If go, run a targeted novelty check (augmentation of Raman/SERS/NIR spectra, background or baseline augmentation, substrate invariance).
6. Then freeze the protocol and run the sealed tests on B, C and D.

## What every option still needs

- Metadata answers from Li-Lin (PICO, concentrations, laser power, batches).
- A decision on fentanyl (set D).
- NRC approval for data release.
