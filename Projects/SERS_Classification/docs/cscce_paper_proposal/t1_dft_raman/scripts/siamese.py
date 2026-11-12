"""Two-tower (measured / calculated) Siamese matcher for experimental Raman vs a DFT library.

sweep:    pretrain on pseudo-measured spectra made from QM9S calculated sticks, for a
          grid of depth x width; train vs validation retrieval curves; zero-shot real eval.
finetune: leave-one-compound-out fine-tuning of the measured tower on the real pairs,
          with a learning curve over the number of training compounds.

Augmentation settings come from the literature and the D1 plan (per-band shift sigma 8 cm^-1,
+-0.5% scale, +-10 cm^-1 offset, log-normal intensity sigma 0.7, width 6-20 cm^-1), not from
the real queries.
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
LO = float(os.environ.get("T1_LO", "400"))  # step 2 uses 500 (VibraCLIP spectra start there)
GRID = torch.arange(LO, 1801.0, 2.0, device=DEV)
G = len(GRID)


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, path):
        d = np.load(path)
        K = int((d["inten"] > 0).sum(1).max())
        self.nu = torch.tensor(d["nu3"][:, :K], device=DEV)
        self.inten = torch.tensor(d["inten"][:, :K], device=DEV)
        self.n = self.nu.shape[0]
        self.train_idx = torch.tensor(d["train_idx"], device=DEV)
        self.val_idx = torch.tensor(d["val_idx"], device=DEV)
        self.qx = torch.tensor(d["q_x"], device=DEV)
        self.qtrue = d["q_true"]
        self.qgroup = d["q_group"]
        self.qname = d["q_name"]
        self.qsource = d["q_source"]
        self.p3 = d["p3"]
        self.formula = d["formula"]
        self.qsplit = d["q_split"] if "q_split" in d else np.zeros(len(self.qtrue), int)
        self.qclass = d["q_class"] if "q_class" in d else np.array([CLASSES[n] for n in self.qname])
        assert self.qx.shape[1] == G, "query grid does not match T1_LO"
        self.libx = self.render_clean(10.0)

    def render_clean(self, fwhm):
        out = torch.empty((self.n, G), device=DEV)
        for lo in range(0, self.n, 4096):
            sl = slice(lo, min(self.n, lo + 4096))
            s = render(self.nu[sl], self.inten[sl], torch.full((self.nu[sl].shape[0], 1), fwhm, device=DEV))
            out[sl] = to_input(s)
        return out


def render(nu, inten, fwhm):
    g = (fwhm / 2.0) ** 2
    L = g[:, :, None] / ((GRID[None, None, :] - nu[:, :, None]) ** 2 + g[:, :, None])
    return (inten[:, :, None] * L).sum(1)


def to_input(s):
    s = s.clamp_min(0)
    x = s.sqrt()
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


def pseudo(D, idx, gen):
    nu, I = D.nu[idx], D.inten[idx]
    B, K = nu.shape
    m = (I > 0).float()

    def rn(*s):
        return torch.randn(*s, device=DEV, generator=gen)

    def ru(*s):
        return torch.rand(*s, device=DEV, generator=gen)

    nu_m = nu * (1 + 0.005 * rn(B, 1)) + (ru(B, 1) * 20 - 10) + 8 * rn(B, K)
    I_m = I * torch.exp(0.7 * rn(B, K)) * (ru(B, K) > 0.1).float() * m
    s = render(nu_m, I_m, 6 + 14 * ru(B, 1))
    s = s / (s.amax(1, keepdim=True) + 1e-8)
    c = LO + (1800 - LO) * ru(B, 2, 1)
    w = 100 + 300 * ru(B, 2, 1)
    a = -0.03 + 0.08 * ru(B, 2, 1)
    base = (a * torch.exp(-0.5 * ((GRID[None, None, :] - c) / w) ** 2)).sum(1)
    noise = (0.002 + 0.018 * ru(B, 1)) * rn(B, G)
    return to_input(s + base + noise)


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, c, stride):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 5, stride, 2)
        self.n1 = nn.GroupNorm(min(8, c), c)
        self.c2 = nn.Conv1d(c, c, 5, 1, 2)
        self.n2 = nn.GroupNorm(min(8, c), c)
        self.sk = nn.Identity() if stride == 1 else nn.Conv1d(c, c, 1, stride)

    def forward(self, x):
        return F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x))))) + self.sk(x))


class Tower(nn.Module):
    """1D CNN with a wavenumber coordinate channel and a positional (flattened) head.

    With skip=True the embedding is [x, beta * learned], so at beta -> 0 the similarity
    is exactly the fixed sqrt-intensity cosine and training can only add to it.
    """

    def __init__(self, width, depth, emb=256, skip=True):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(2, width, 9, 1, 4), nn.GroupNorm(min(8, width), width), nn.GELU())
        nstride = min(depth, 3)
        self.blocks = nn.Sequential(*[Block(width, 2 if i < nstride else 1) for i in range(depth)])
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * width, 512), nn.GELU(), nn.Linear(512, emb))
        self.skip = skip
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("coord", torch.linspace(-1, 1, G))

    def forward(self, x):
        h = self.blocks(self.stem(torch.stack([x, self.coord.expand_as(x)], 1)))
        z = F.normalize(self.head(self.pool(h)).float(), dim=1)
        if not self.skip:
            return z
        return F.normalize(torch.cat([x.float(), self.beta * z], 1), dim=1)


class TwoTower(nn.Module):
    def __init__(self, width, depth, skip=True):
        super().__init__()
        self.meas = Tower(width, depth, skip=skip)
        self.calc = Tower(width, depth, skip=skip)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def embed(tower, X, bs=1024):
    tower.eval()
    out = []
    for lo in range(0, X.shape[0], bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(tower(X[lo:lo + bs]))
    tower.train()
    return torch.cat(out)


@torch.no_grad()
def ranks_full(Q, E, true):
    """rank of E[true[i]] among all rows of E for query embedding Q[i]."""
    r = []
    true = torch.as_tensor(true, device=DEV)
    for lo in range(0, Q.shape[0], 1024):
        S = Q[lo:lo + 1024] @ E.T
        t = S.gather(1, true[lo:lo + 1024, None])
        r.append(1 + (S > t).sum(1))
    return torch.cat(r).cpu().numpy()


@torch.no_grad()
def real_eval(D, Qe, E):
    S = (Qe @ E.T).cpu().numpy()
    out = {}
    for name in ["full", "P3", "P2"]:
        rr = []
        for j, t in enumerate(D.qtrue):
            if name == "full":
                cols = np.arange(D.n)
            elif name == "P3":
                cols = np.union1d(D.p3, [t])
            else:
                cols = np.flatnonzero(D.formula == D.formula[t])
            s = S[j, cols]
            rr.append(int(1 + (s > S[j, t]).sum()))
        rr = np.array(rr)
        out[name] = dict(top1=float((rr == 1).mean()), top5=float((rr <= 5).mean()),
                         mrr=float((1 / rr).mean()), ranks=rr.tolist())
    return out


def summarise_ranks(r):
    r = np.asarray(r)
    return dict(top1=float((r == 1).mean()), top5=float((r <= 5).mean()), mrr=float((1 / r).mean()))


# ---------------------------------------------------------------- sweep
def sweep(args):
    D = Data(args.data)
    os.makedirs(args.out, exist_ok=True)
    fixed = {}
    for fw in [10.0, 16.0]:
        L = D.libx if fw == 10.0 else D.render_clean(fw)
        fixed[f"fixed_sqrt_cos_fwhm{int(fw)}"] = real_eval(D, D.qx, L)
        del L
    json.dump(fixed, open(os.path.join(args.out, "fixed_baseline.json"), "w"), indent=1)
    for k, v in fixed.items():
        print(k, {p: (round(v[p]["top1"], 3), round(v[p]["top5"], 3)) for p in v}, flush=True)

    g = torch.Generator(device=DEV)
    g.manual_seed(123)
    val_x = pseudo(D, D.val_idx, g)
    tr_eval_idx = D.train_idx[torch.randperm(len(D.train_idx), generator=torch.Generator().manual_seed(5))[:5000].to(DEV)]
    tr_x = pseudo(D, tr_eval_idx, g)
    val_lib_cols = D.val_idx

    for depth in args.depths:
        for width in args.widths:
            tag = f"d{depth}_w{width}_s{int(args.skip)}"
            torch.manual_seed(0)
            gen = torch.Generator(device=DEV)
            gen.manual_seed(1000 + depth * 1000 + width)
            M = TwoTower(width, depth, skip=bool(args.skip)).to(DEV)
            opt = torch.optim.AdamW(M.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
            curve = []
            t0 = time.time()
            run_loss = 0.0
            for step in range(0, args.steps + 1):
                if step > 0:
                    idx = D.train_idx[torch.randint(len(D.train_idx), (args.batch,), device=DEV, generator=gen)]
                    xm = pseudo(D, idx, gen)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        em, ec = M.meas(xm), M.calc(D.libx[idx])
                    logits = em @ ec.T * M.logit_scale.exp().clamp(max=100)
                    lab = torch.arange(len(idx), device=DEV)
                    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sched.step()
                    run_loss += loss.item()
                if step % args.eval_every == 0 or step == args.steps:
                    E = embed(M.calc, D.libx)
                    rv = ranks_full(embed(M.meas, val_x), E, D.val_idx.cpu().numpy())
                    rt = ranks_full(embed(M.meas, tr_x), E, tr_eval_idx.cpu().numpy())
                    Ev = E[val_lib_cols]
                    rv5k = ranks_full(embed(M.meas, val_x), Ev, np.arange(len(val_lib_cols)))
                    real = real_eval(D, embed(M.meas, D.qx), E)
                    rec = dict(step=step, loss=run_loss / max(1, args.eval_every if step else 1), sec=time.time() - t0,
                               train_full=summarise_ranks(rt), val_full=summarise_ranks(rv),
                               val_5k=summarise_ranks(rv5k),
                               real={p: {k: v for k, v in real[p].items() if k != "ranks"} for p in real})
                    curve.append(rec)
                    run_loss = 0.0
                    print(f"{tag} step {step} loss {rec['loss']:.3f} train130k top1 {rec['train_full']['top1']:.3f} "
                          f"val130k top1 {rec['val_full']['top1']:.3f} val5k {rec['val_5k']['top1']:.3f} | real full "
                          f"{real['full']['top1']:.3f}/{real['full']['top5']:.3f} P3 {real['P3']['top1']:.3f}/"
                          f"{real['P3']['top5']:.3f} P2 {real['P2']['top1']:.3f}  ({rec['sec']:.0f}s)", flush=True)
            final_real = real_eval(D, embed(M.meas, D.qx), embed(M.calc, D.libx))
            json.dump(dict(depth=depth, width=width, skip=int(args.skip), params=nparams(M), curve=curve, final_real=final_real),
                      open(os.path.join(args.out, f"sweep_{tag}.json"), "w"), indent=1)
            torch.save(M.state_dict(), os.path.join(args.out, f"ckpt_{tag}.pt"))
            del M, opt
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- fine-tune
CLASSES = {  # chemical class per compound, for leave-one-class-out splits
    "1,3-dimethyl-2-imidazolidinone": "amide", "N,N-dimethylformamide": "amide",
    "2-propanol": "alcohol", "methanol": "alcohol", "ethanol": "alcohol", "tert-butanol": "alcohol",
    "diethylene glycol": "alcohol",
    "4-methyl-2-pentanone": "ketone", "methyl isobutyl ketone (2nd bottle)": "ketone",
    "acetonitrile": "nitrile",
    "benzaldehyde": "aromatic", "toluene": "aromatic", "benzene": "aromatic", "pyridine": "aromatic",
    "aniline": "aromatic",
    "butyl acetate": "ester", "ethyl acetate": "ester", "propyl acetate": "ester",
    "cyclohexane": "alkane", "n-heptane": "alkane", "n-hexane": "alkane",
    "tert-butyl methyl ether": "ether",
    # added with our own B3LYP/TZVP calculations (not in QM9S)
    "acetone": "ketone", "ethylene glycol": "alcohol", "tetrahydrofuran": "ether", "2,2-dimethoxypropane": "ether",
    "formic acid": "acid", "acetic acid": "acid", "diethylamine": "amine", "isobutylamine": "amine",
}


def augment_real(x, gen):
    B = x.shape[0]
    shifts = torch.randint(-3, 4, (B,), device=DEV, generator=gen)
    out = torch.stack([torch.roll(x[i], int(shifts[i])) for i in range(B)])
    wig = 1 + 0.1 * F.avg_pool1d(torch.randn(B, 1, G, device=DEV, generator=gen), 51, 1, 25)[:, 0] * 7
    out = (out * wig.clamp(0.5, 1.5) + 0.005 * torch.randn(B, G, device=DEV, generator=gen)).clamp_min(0)
    return out / (out.norm(dim=1, keepdim=True) + 1e-8)


def finetune(args):
    D = Data(args.data)
    depth, width = args.depth, args.width
    M0 = TwoTower(width, depth, skip=bool(args.skip)).to(DEV)
    M0.load_state_dict(torch.load(os.path.join(args.out, f"ckpt_d{depth}_w{width}_s{int(args.skip)}.pt"), map_location=DEV))
    E = embed(M0.calc, D.libx)
    scale = M0.logit_scale.exp().clamp(max=100).detach()
    hgroup = (np.unique(D.qclass, return_inverse=True)[1] if args.by_class
              else D.qgroup)
    groups = np.unique(hgroup)
    rng = np.random.default_rng(7 + args.seed)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(77 + args.seed)
    results, curves = [], []
    for held in groups:
        others = groups[groups != held]
        held_q = np.flatnonzero(hgroup == held)
        held_lib = set(D.qtrue[held_q].tolist())
        for ntr in args.ntrain:
            reps = 1 if ntr >= len(others) else args.repeats
            for rep in range(reps):
                sub = others if ntr >= len(others) else rng.choice(others, ntr, replace=False)
                trq = np.flatnonzero(np.isin(hgroup, sub))
                tq = D.qtrue[trq]
                if args.shuffle:
                    # label-shuffle control: pair each training compound with another training
                    # compound's calculated spectrum (a derangement over compounds)
                    sub_sorted = np.unique(D.qgroup[trq])
                    while True:
                        perm = rng.permutation(len(sub_sorted))
                        if not np.any(perm == np.arange(len(sub_sorted))):
                            break
                    lib_of = {g: D.qtrue[np.flatnonzero(D.qgroup == g)[0]] for g in sub_sorted}
                    remap = {g: lib_of[sub_sorted[perm[i]]] for i, g in enumerate(sub_sorted)}
                    tq = np.array([remap[g] for g in D.qgroup[trq]])
                cand_true = np.unique(tq)
                tgt = torch.tensor(np.searchsorted(cand_true, tq), device=DEV)
                allowed = np.setdiff1d(np.arange(D.n), list(held_lib) + cand_true.tolist())
                A = copy.deepcopy(M0.meas)
                opt = torch.optim.AdamW(A.parameters(), lr=args.lr, weight_decay=1e-2)
                track = ntr >= len(others)
                for step in range(args.steps + 1):
                    if track and step % 25 == 0:
                        Qe = embed(A, D.qx)
                        rk = ranks_full(Qe, E, D.qtrue)
                        curves.append(dict(held=int(held), step=step, held_ranks=rk[held_q].tolist(),
                                           train_ranks=rk[trq].tolist()))
                    if step == args.steps:
                        break
                    neg = torch.tensor(rng.choice(allowed, 4096, replace=False), device=DEV)
                    cand = torch.cat([torch.tensor(cand_true, device=DEV), neg])
                    x = augment_real(D.qx[trq], gen)
                    sidx = D.train_idx[torch.randint(len(D.train_idx), (256,), device=DEV, generator=gen)]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        e = A(x)
                        es = A(pseudo(D, sidx, gen))
                    loss = F.cross_entropy(e @ E[cand].T * scale, tgt) + \
                        F.cross_entropy(es @ E[sidx].T * scale, torch.arange(256, device=DEV))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                Qe = embed(A, D.qx[held_q])
                real = real_eval_subset(D, Qe, E, held_q)
                for j, qi in enumerate(held_q):
                    results.append(dict(held=int(held), ntrain=int(ntr), rep=rep, query=int(qi),
                                        name=str(D.qname[qi]), source=str(D.qsource[qi]),
                                        **{f"rank_{p}": real[p][j] for p in real}))
        print("held-out compound", held, D.qname[held_q[0]],
              [(r["ntrain"], r["rank_full"]) for r in results if r["held"] == held], flush=True)
    tag = f"d{depth}_w{width}_s{int(args.skip)}" + (f"_shuffle{args.seed}" if args.shuffle else "") + \
        ("_byclass" if args.by_class else "") + (f"_seed{args.seed}" if args.by_class and not args.shuffle else "")
    json.dump(dict(results=results, curves=curves), open(os.path.join(args.out, f"finetune_{tag}.json"), "w"))


def transfer(args):
    """Fine-tune on the training spectra (TRAIN_SPLITS, default split 0) only; rank every other query once."""
    D = Data(args.data)
    depth, width = args.depth, args.width
    M0 = TwoTower(width, depth, skip=bool(args.skip)).to(DEV)
    M0.load_state_dict(torch.load(os.path.join(args.out, f"ckpt_d{depth}_w{width}_s{int(args.skip)}.pt"), map_location=DEV))
    E = embed(M0.calc, D.libx)
    scale = M0.logit_scale.exp().clamp(max=100).detach()
    rng = np.random.default_rng(7 + args.seed)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(77 + args.seed)
    train_splits = [int(v) for v in os.environ.get("TRAIN_SPLITS", "0").split(",")]
    trq = np.flatnonzero(np.isin(D.qsplit, train_splits))
    teq = np.flatnonzero(~np.isin(D.qsplit, train_splits))
    tq = D.qtrue[trq]
    if args.shuffle:
        comps = np.unique(D.qgroup[trq])
        while True:
            perm = rng.permutation(len(comps))
            if not np.any(perm == np.arange(len(comps))):
                break
        lib_of = {g: D.qtrue[np.flatnonzero(D.qgroup == g)[0]] for g in comps}
        remap = {g: lib_of[comps[perm[i]]] for i, g in enumerate(comps)}
        tq = np.array([remap[g] for g in D.qgroup[trq]])
    cand_true = np.unique(tq)
    tgt = torch.tensor(np.searchsorted(cand_true, tq), device=DEV)
    allowed = np.setdiff1d(np.arange(D.n), np.concatenate([D.qtrue[teq], cand_true]))
    A = copy.deepcopy(M0.meas)
    opt = torch.optim.AdamW(A.parameters(), lr=args.lr, weight_decay=1e-2)
    for step in range(args.steps):
        neg = torch.tensor(rng.choice(allowed, 4096, replace=False), device=DEV)
        cand = torch.cat([torch.tensor(cand_true, device=DEV), neg])
        x = augment_real(D.qx[trq], gen)
        sidx = D.train_idx[torch.randint(len(D.train_idx), (256,), device=DEV, generator=gen)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = A(x)
            es = A(pseudo(D, sidx, gen))
        loss = F.cross_entropy(e @ E[cand].T * scale, tgt) + \
            F.cross_entropy(es @ E[sidx].T * scale, torch.arange(256, device=DEV))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    allq = np.arange(len(D.qtrue))
    real = real_eval_subset(D, embed(A, D.qx), E, allq)
    zero = real_eval_subset(D, embed(M0.meas, D.qx), E, allq)
    results = [dict(query=int(qi), name=str(D.qname[qi]), split=int(D.qsplit[qi]), source=str(D.qsource[qi]),
                    rank_full=real["full"][j], rank_zeroshot=zero["full"][j]) for j, qi in enumerate(allq)]
    r1 = np.array([r["rank_full"] for r in results if r["split"] == D.qsplit.max()])
    print(f"transfer seed {args.seed} shuffle {args.shuffle}: test top1 {(r1 == 1).mean():.3f} top5 {(r1 <= 5).mean():.3f}", flush=True)
    tag = f"transfer_seed{args.seed}" + ("_shuffle" if args.shuffle else "")
    json.dump(dict(results=results), open(os.path.join(args.out, f"{tag}.json"), "w"))


@torch.no_grad()
def real_eval_subset(D, Qe, E, qidx):
    S = (Qe @ E.T).cpu().numpy()
    out = {"full": [], "P3": [], "P2": []}
    for j, qi in enumerate(qidx):
        t = D.qtrue[qi]
        for name in out:
            cols = np.arange(D.n) if name == "full" else (
                np.union1d(D.p3, [t]) if name == "P3" else np.flatnonzero(D.formula == D.formula[t]))
            s = S[j, cols]
            out[name].append(int(1 + (s > S[j, t]).sum()))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "finetune", "transfer"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--ntrain", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--shuffle", type=int, default=0)
    ap.add_argument("--by_class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"sweep": sweep, "finetune": finetune, "transfer": transfer}[a.cmd](a)
