# Optional Image Generation Prompts

These are optional prompts for generating polished visual figures in browser ChatGPT. They are not required for the compiled LaTeX poster, which already uses repository-generated figures and diagrams.

## Sample Preparation Workflow

Create a clean scientific poster illustration showing the SERS sample preparation workflow. Use a 1:1 square aspect ratio suitable for a 4 ft by 4 ft conference poster. The visual should show four connected stages from left to right: commercial PICO and pSERS sensors; in-house Ag and Au nanoparticle substrates made by inkjet-printing colloidal nanoparticle solution onto filter paper; soaking the substrates in a diluted analyte solution for 1 hour; drying the substrates; Raman/SERS spectral acquisition with a laser and detector producing a spectrum. Use a modern NRC-style palette with deep blue, teal, warm gold, and off-white. Keep it scientifically precise, uncluttered, and vector-like. Do not add any extra experimental details not listed here. Include minimal labels only: "Commercial sensors", "Printed Ag/Au nanoparticle paper", "1 h analyte soak", "Dry", "SERS measurement".

## Siamese Metric Learning Concept Figure

Create a clean scientific schematic of Siamese metric learning for substrate-agnostic SERS classification. Use a 1:1 square aspect ratio suitable for a conference poster. Show three SERS spectra entering a shared Conv1D encoder: an anchor spectrum, a positive spectrum with the same chemical on a different substrate, and a negative spectrum from a different chemical. Show all three passing through the same encoder block into a 64-dimensional embedding space. In the embedding space, show clusters for 4NP, benzenethiol, and pyridine, with different marker shapes representing Ag, Au, PICO, and pSERS substrates. Make the same-chemical points close together across substrate markers and different chemicals separated. Include the triplet-loss concept visually with "pull same chemical together" and "push different chemical apart". Use deep blue, teal, gold, and red accents on an off-white background.

