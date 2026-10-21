# Phase 1 summary — inventory and loader

`load.py` ran cleanly and read every raw file in place. Counts were re-checked against the raw
folders: **275** Block A spectra, **780** Block B spectra, **265** reference spectra and **98**
background spectra. Every file is two tab-separated columns, 1,024 points, finite and with a
strictly increasing shift axis. `.ngc`/`.ngv` and averages were ignored (no scoped `.txt` was an
average). Outputs: `outputs/inventory.csv`, `spectra_blockA.npz`, `spectra_blockB.npz`,
`references.npz`, `backgrounds.npz`; figures: `figures/cell_means_blockA.png`,
`figures/cell_means_blockB.png`. In the inventory, `path` is relative to `Data/` and `map_id` is
a globally unique cell ID built from block/analyte/substrate/laser/raw map (e.g.
`A_4NP_Ag_632_map1`), because the raw label is “map1” in every folder.

## Maps and spectra per cell (entries: maps / spectra)

**Block A — 632 nm, one instrument.** Each cell is one 5 x 5 map (25 spots).

| Analyte | Ag | Au | PICO | pSERS |
|---|---:|---:|---:|---:|
| 4-nitrophenol | 1 / 25 | — | 1 / 25 | 1 / 25 |
| benzenethiol | 1 / 25 | 1 / 25 | 1 / 25 | 1 / 25 |
| pyridine | 1 / 25 | 1 / 25 | 1 / 25 | 1 / 25 |

**Block B — two lasers.** One map per cell: 7 x 7 (49 spots) at 632 nm, 9 x 9 (81 spots) at 785 nm.

| Analyte | Ag 632 | Ag 785 | Au 632 | Au 785 |
|---|---:|---:|---:|---:|
| 4-mercaptobenzoic acid | 1 / 49 | 1 / 81 | 1 / 49 | 1 / 81 |
| 2,6-pyridinecarboxylic acid | 1 / 49 | 1 / 81 | 1 / 49 | 1 / 81 |
| 4-mercaptobenzonitrile | 1 / 49 | 1 / 81 | 1 / 49 | 1 / 81 |

Total: 11 Block A maps and 12 Block B maps, one prepared substrate per cell.

## Common axes (1 cm⁻¹ grid; intersection of every file, no extrapolation)

| Set | Per-file range (cm⁻¹) | Common axis | Points |
|---|---|---|---:|
| Block A | 178.677–2447.08 | 179–2447 | 2269 |
| Block B | 186.85–2762.88 | 366–2453 | 2088 |
| References | 176.074–2762.24 | 365–2445 | 2081 |
| Backgrounds | 176.817–2445.73 (632); 425.605–1873.52 (785) | 426–1873 | 1448 |

Block B loses 632 nm data below 366 cm⁻¹ and 785 nm data above 2453 cm⁻¹ because the two lasers
measure different windows; the reference axis is similarly truncated by mixing laser classes.

## References and blanks found

| Analyte | Source | Laser(s) | Spectra (maps) | Matched to |
|---|---|---|---|---|
| benzenethiol | `Jesse_Dataset_v2/Pure_Raman_txt` | unknown | 27 (5 x 5 map + 2 singles) | Block A overlay / E1 |
| pyridine | `Jesse_Dataset_v2/Pure_Raman_txt` | unknown | 27 (5 x 5 map + 2 singles) | Block A overlay / E1 |
| 4-nitrophenol | `Jesse_Dataset_Update/Raman` | 632, 785 | 25 + 25 (5 x 5) | Block A (632) |
| 4-mercaptobenzoic acid | Block B `powder - non-SERs` | 632, 785 | 25 + 25 (5 x 5) | Block B (laser-matched) |
| 2,6-pyridinecarboxylic acid | Block B `powder - non-SERs` | 632, 785 | 36 (6 x 6) + 25 (5 x 5) | Block B (laser-matched) |
| 4-mercaptobenzonitrile | `Jesse_Dataset_Update/Raman` | 632, 785 | 25 + 25 (5 x 5) | Block B (laser-matched) |

Substrate-only blanks: AuNP at 632 nm (49 spots, one 7 x 7 map) and 785 nm (49 spots, one 7 x 7
map). They are inventoried and archived but **not subtracted** anywhere in this pilot.

## Problems and judgement calls

- **Missing cell:** 4-nitrophenol has no Au SERS cell in Block A (11 cells, not 12).
- **Unequal map sizes:** 25 (A), 49 (B 632), 81 (B 785) and 36 (2,6-PDA 632 reference). Block B
  map size is confounded with laser.
- **Spelling/axis normalisation:** AgNP/Ag → Ag, AuNP/Au → Au, PICO → PICO, pSERS/pSERs/pSERSs →
  pSERS; `632`/`633` both recorded as 632. The 2,6-PDA 785 filenames also use “pyrdine…”.
- **Laser unknown:** the `Pure_Raman_txt` BT and pyridine references carry no laser in name or
  folder; their 176–2445 cm⁻¹ axis is 632-class but is recorded as `unknown`.
- **Odd values:** two Block B 785 nm spectra (MBA/Ag, MBN/Ag) dip slightly below zero (min −66 and
  −50); per-spectrum min–max removes the offset. All block cells are single-map, so “cell mean”
  and “map mean” coincide throughout Phase 2.
