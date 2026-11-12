"""Fixed cosine baseline (literature settings) on a Siamese data file: ranks for every query."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import siamese as s
DATA, OUT = sys.argv[1:3]
D = s.Data(DATA)
out = {}
for fw in [10.0, 16.0]:
    L = D.libx if fw == 10.0 else D.render_clean(fw)
    out[f"fixed_sqrt_cos_fwhm{int(fw)}"] = s.real_eval(D, D.qx, L)
json.dump(out, open(OUT, "w"), indent=1)
for k, v in out.items():
    r = np.array(v["full"]["ranks"])
    print(k, "full top1 %.3f top5 %.3f top20 %.3f" % ((r == 1).mean(), (r <= 5).mean(), (r <= 20).mean()))
