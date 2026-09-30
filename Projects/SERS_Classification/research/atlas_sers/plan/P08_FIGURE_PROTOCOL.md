# P08 figure delivery and disclosure rules

**Specified:** 2026-09-30. **State:** prospective reporting specification; no new scientific calculation or figure has been produced under this document.

The [figure plan](P08_FIGURE_PLAN.csv) pairs spectral views with domain scatter, paired effects, interactions and degradation curves. All formats must use one reviewed semantic table. A visually convincing plot cannot substitute for complete support, a registered comparison or a source-only selection procedure.

## 1. Formats and typography

Each figure requires native TikZ/PGFPlots, compiled vector PDF, a PNG review copy and a self-contained HTML view. No raster or pre-rendered plot may be embedded inside a nominal TikZ source. HTML must work offline, without a remote script or font request. Both renderers consume the same frozen plot-level data; neither may independently recalculate statistics.

Use black text with standard LaTeX roman or Times-compatible fonts, as requested for P08. This scoped choice supersedes the older figure guide's Helvetica-compatible preference for new P08 figures only. Record the actual font configuration in the build manifest. Keep the inherited minimum text size, stroke widths, final dimensions, redundant marker/line encodings and colorblind-safe palette. Colors distinguish data traces, not prose or labels.

TikZ and HTML must agree on data hash, population, aggregation, axes, units, scale, category order, colors, line/marker keys, estimates, intervals, missingness and claim scope. HTML hover may expose additional approved aggregate support information, but cannot disclose private records omitted from the visible plot. Every figure must include an accessible data table or approved data download and a caption defining its independent unit.

## 2. Public aggregates and private examples

The current P08 goal excludes raw data, row records, identities and checkpoints from its release. Earlier repository permissions do not automatically authorize new row-level P08 artifacts. This rule does not delete or reclassify historical releases.

For public P08-F01, compare MIN, SG and arPLS spectral aggregates within each station–instrument domain and analyte. Use the frozen 598-spectrum primary population, with each stored observation included once, not repeated by its cross-validation appearances. Explicitly label the four exploratory domains that do not enter the primary held comparison. Use identical membership across actions. First average stored views within each physical master and domain; then average those master-level curves equally. Publish the resulting mean curve and aggregate counts, not the constituent curves or master identifiers. Require at least two physical masters for a public spectral cell so that it is not an individual-master trace. A cell below that minimum is labelled unavailable for this public spectral view; it remains in all scientific analyses authorized by the protocol.

Do not normalize the aggregate again or vertically offset it without explicit labelling. The first trace is the MIN representation, not raw instrument counts. These curves illustrate whole-pipeline changes and are not the model's averaged inputs: classification still operates on the registered individual spectra. Aggregate curves can conceal variation across substrates and repeated measurements. They do not establish chemical peak preservation.

Private review may include individual before/after examples and dense interactive traces, in the same native TikZ/HTML output formats. For each station–instrument/analyte cell, sort the available master IDs lexicographically, then sort observation IDs within the first master and use its first observation. Record that choice before inspecting new classifier outcomes; use the same row for every action and do not replace an unfavorable example. This ordering is an illustrative sampling rule, not a claim of statistical representativeness. No private semantic table, embedded HTML data, hover payload or individual-trace source is copied into the public release.

P08-F06 publishes approved QC/action-bin summaries only. Individual QC coordinates, observation IDs and routing records remain private. P08-F07 uses domain/action preservation summaries under the existing diagnostic definitions; it is not a collection of identifiable row-level points. These display restrictions do not change model fitting, test denominators or statistical estimands.

## 3. Scientific display requirements

The comparison figures show every planned domain, including poor and unavailable results. Use paired domain scatter, dot/interval plots and zero-reference interaction views. Use a bar plot only where counts or proportions are clearer that way. Show spectrum-level and combined-probability endpoints separately; do not label the latter as averaged input spectra.

Family-aware results must distinguish structural MIN aliases from a supported transfer estimate, which is unavailable here. QC figures show both all-context operational results and the separately labelled supported CWA subset. Counts of contexts, masters, spectra and technical seeds must not be presented as interchangeable sample sizes.

Use the intervals, missing-cell rules and multiplicity families in the [statistical protocol](P08_STATISTICAL_PROTOCOL.md). Negative effects, weakest-domain changes and fallback burdens remain visible. An interval is conditional on the saved fits and observed support; no figure may imply clean-spectrum recovery or a passed superiority gate.

## 4. Release evidence

The figure manifest must bind the semantic-table hash, renderer/version, scientific input references, aggregation and disclosure review, font configuration and all output hashes. Compile the native source, inspect the vector PDF and PNG, open the offline HTML, and check its embedded payload. Compilation logs stay private when they contain workstation paths.

Publication requires explicit review that an apparent aggregate does not expose an individual trace or a forbidden identity. Hash equality alone does not establish visual or semantic correctness. Any unsupported panel is labelled, not silently dropped. Numerical table generation and rendering of new scientific results require the later execution scope; this readiness document authorizes neither.
