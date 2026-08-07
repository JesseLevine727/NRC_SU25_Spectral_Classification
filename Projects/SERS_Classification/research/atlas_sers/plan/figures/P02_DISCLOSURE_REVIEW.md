# P02 public figure disclosure review

F10 and F11 are approved public aggregate method figures. Their shared CSV
tables contain only figure metadata, conceptual role coordinates, public
station–instrument domain labels, aggregate master/class support, aggregate
family-support states, aggregate target-access feasibility fractions, and
declared QC gate counts.

They contain no spectrum, intensity, wavelength coordinate, observation UID,
physical master ID, source file identity, row-level QC value, numeric QC
cutpoint, chemical label by observation, native path, private provenance, fold
membership, calibration draw membership, or test outcome. F10 is a design
schematic. F11 reports metadata-only feasibility and explicitly contains no
predictive result.

The publication script rejects protected columns and requires the aggregate
CSV SHA-256 to be embedded in both native TikZ and standalone HTML before it
copies CSV/TikZ/PDF/PNG/HTML forms into the public tree.
