# Existing Real Failure Audit

- samples: `580`
- anchor exact match: `0.9069`
- anchor errors: `54`
- operational accepted count: `562`
- operational rejected count: `18`
- operational accepted exact match: `0.9093`
- accepted errors: `51`
- rejected but anchor-correct: `15`
- rejected and anchor-incorrect: `3`

## Top Failure Modes

- `6-mercapto-1-hexanol + pyridine` -> `benzenethiol + pyridine`: `36` samples
- `1-dodecanethiol + meoh` -> `1-dodecanethiol + diethylamine`: `9` samples
- `1-dodecanethiol + meoh` -> `1-undecanethiol + meoh`: `8` samples
- `1-dodecanethiol + meoh` -> `1-dodecanethiol + tris(2-ethylhexyl) phosphate`: `1` samples

## Worst True Pairs

- `6-mercapto-1-hexanol + pyridine`: anchor_exact=`0.000`, reject_rate=`0.000`, accepted_exact=`0.000`, mean_residual_rel=`0.229`, mean_minor_share=`0.286`
- `1-dodecanethiol + meoh`: anchor_exact=`0.886`, reject_rate=`0.114`, accepted_exact=`0.893`, mean_residual_rel=`0.106`, mean_minor_share=`0.198`
- `1-dodecanethiol + benzene`: anchor_exact=`1.000`, reject_rate=`0.000`, accepted_exact=`1.000`, mean_residual_rel=`0.145`, mean_minor_share=`0.488`
- `benzene + benzenethiol`: anchor_exact=`1.000`, reject_rate=`0.000`, accepted_exact=`1.000`, mean_residual_rel=`0.125`, mean_minor_share=`0.154`
- `benzenethiol + pyridine`: anchor_exact=`1.000`, reject_rate=`0.000`, accepted_exact=`1.000`, mean_residual_rel=`0.105`, mean_minor_share=`0.416`
- `6-mercapto-1-hexanol + n,n-dimethylformamide`: anchor_exact=`1.000`, reject_rate=`0.000`, accepted_exact=`1.000`, mean_residual_rel=`0.099`, mean_minor_share=`0.441`
- `6-mercapto-1-hexanol + benzene`: anchor_exact=`1.000`, reject_rate=`0.000`, accepted_exact=`1.000`, mean_residual_rel=`0.098`, mean_minor_share=`0.299`
- `benzene + etoh`: anchor_exact=`1.000`, reject_rate=`0.000`, accepted_exact=`1.000`, mean_residual_rel=`0.096`, mean_minor_share=`0.477`
- `etoh + meoh`: anchor_exact=`1.000`, reject_rate=`0.000`, accepted_exact=`1.000`, mean_residual_rel=`0.092`, mean_minor_share=`0.420`
- `n,n-dimethylformamide + pyridine`: anchor_exact=`1.000`, reject_rate=`0.000`, accepted_exact=`1.000`, mean_residual_rel=`0.083`, mean_minor_share=`0.355`
