# NRC SU25 Spectral Classification

This repository is organized around three main projects:

- [Projects/Standard_Raman_Classification](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/Standard_Raman_Classification)
- [Projects/SERS_Classification](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification)
- [Mixture_Classification](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification)

The supporting shared material lives in:

- [Data](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Data)
- [Docs](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Docs)
- [Archive](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Archive)

## Repo Map

### Standard Raman classification

- active home: [Projects/Standard_Raman_Classification](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/Standard_Raman_Classification)
- core scripts: [Projects/Standard_Raman_Classification/Scripts](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/Standard_Raman_Classification/Scripts)
- exploratory notebooks: [Projects/Standard_Raman_Classification/Notebooks](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/Standard_Raman_Classification/Notebooks)
- extended Siamese work: [Projects/Standard_Raman_Classification/ExtendingSiamese](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/Standard_Raman_Classification/ExtendingSiamese)

### SERS classification

- active home: [Projects/SERS_Classification](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification)
- current workspace: [Projects/SERS_Classification/Workspace](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/Workspace)

### Mixture classification

- active home: [Mixture_Classification](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification)
- active unmixing pipeline: [Mixture_Classification/Unmixing_Pipeline](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification/Unmixing_Pipeline)
- legacy Siamese pipeline: [Mixture_Classification/Legacy_Siamese_Pipeline](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification/Legacy_Siamese_Pipeline)
- result index: [Mixture_Classification/RESULTS_INDEX.md](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification/RESULTS_INDEX.md)

## Datasets

The top-level datasets are grouped under [Data](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Data):

- [Data/Jesse_Dataset](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Data/Jesse_Dataset): original pure/reference Raman data and related exports
- [Data/Jesse_Dataset_v2](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Data/Jesse_Dataset_v2): later mixed Raman/SERS dataset bundle
- [Data/Jesse_Dataset_Update](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Data/Jesse_Dataset_Update): updated Raman and SERS folders
- [Data/Jesse_Dataset_PT2](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Data/Jesse_Dataset_PT2): pt2 pure and mixture spectra
- [Data/Feb26_Spectra](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Data/Feb26_Spectra): February 26 spectra collection
- [Data/Test_Data](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Data/Test_Data): split query/reference CSVs used in earlier experiments

The active mixture pipeline also carries its own self-contained copy of the data it needs under [Mixture_Classification/Unmixing_Pipeline/Data](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification/Unmixing_Pipeline/Data).

## Results

If you need the most relevant current results first:

- mixture unmixing results: [Mixture_Classification/Unmixing_Pipeline/Results](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification/Unmixing_Pipeline/Results)
- mixture result guide: [Mixture_Classification/RESULTS_INDEX.md](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification/RESULTS_INDEX.md)
- legacy mixture Siamese outputs: [Mixture_Classification/Legacy_Siamese_Pipeline/Notebooks](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification/Legacy_Siamese_Pipeline/Notebooks)

## Notes

- `Mixture_Classification/Unmixing_Pipeline` is the active mixture-classification codebase.
- `Mixture_Classification/Legacy_Siamese_Pipeline` is preserved for comparison and historical reproducibility.
- `Archive` contains legacy loose notes and ad hoc material that are no longer part of the main workflow.
