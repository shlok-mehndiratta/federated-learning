"""
Core experiment runner for simulated coordinator outages.

This script evaluates what happens when a network outage forces two training
cohorts to train in isolation for a fixed number of steps (L) starting at step k,
compared against a reference model trained continuously on all data without an outage.

Terms used in code:
  * Outage start (onset_pct / k): percentage of total training steps before the outage.
  * Outage length (L): number of isolated training steps per cohort during the outage.
  * No-outage model (ref): trained continuously on all data for the same number of steps.
  * Merged model (acc_merge): simple average of weights from cohort A and cohort B.
  * Final layer recalibration (acc_cure2): retrains only the classification layer on 200 server images.
  * Full fine-tuning (acc_cure2_fullft): fine-tunes the entire network on the same 200 server images.
  * Diagnostic probe (acc_probe_converged): converged classifier on frozen features to test representation quality.

Reads: CIFAR-10 or CIFAR-100 dataset from --data_dir.
Writes: JSON files containing evaluation metrics and recovery curves to --out_dir.

Example:
  python experiments/run_outage_experiment.py --dataset cifar10 --sources label0.8 --onsets 2 --L 5000 --seeds 42 --out_dir runs/main --data_dir data/cache
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, copy, json, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fl_outage.models.resnet import ResNet20LN

STATS = {"cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
         "cifar100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])}
CORR = {"mild": dict(contrast=0.8, blur_sigma=0.6, noise=0.04),
        "severe": dict(contrast=0.6, blur_sigma=1.2, noise=0.08)}
SOURCES = {"iid": (0.0, None), "label0.8": (0.8, None), "label1.0": (1.0, None),
           "feature_mild": (0.0, "mild"), "feature_severe": (0.0, "severe")}


def set_determinism():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


# ----------------------------------------------------------------------------- data
def load_data(name, data_dir, dev):
    import torchvision
    D = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    tr, te = D(data_dir, train=True, download=True), D(data_dir, train=False, download=True)
    f = lambda d: torch.tensor(d.data).permute(0, 3, 1, 2).contiguous().to(dev)  # uint8 on GPU
    return f(tr), torch.tensor(tr.targets, device=dev), f(te), torch.tensor(te.targets, device=dev)


def make_splits(y, C, seed, server_pc, val_pc):
    """Server pool and validation split are removed before any client partitioning."""
    rng = np.random.RandomState(seed); y = y.cpu().numpy()
    server, val, pool = [], [], []
    for c in range(C):
        idx = rng.permutation(np.where(y == c)[0])
        server += list(idx[:server_pc]); val += list(idx[server_pc:server_pc + val_pc]); pool += list(idx[server_pc + val_pc:])
    return np.array(server), np.array(val), np.array(pool)


def cohort_split(y, pool, C, s, seed):
    """Per class c: cohort A gets (1+s)/2 of class c if c < C/2 else (1-s)/2; B gets the rest."""
    rng = np.random.RandomState(seed + 7); y = y.cpu().numpy()
    A, B = [], []
    for c in range(C):
        idx = rng.permutation(pool[y[pool] == c])
        fa = (1 + s) / 2 if c < C // 2 else (1 - s) / 2
        n = int(round(fa * len(idx))); A += list(idx[:n]); B += list(idx[n:])
    return np.array(A), np.array(B)


def calib_subset(y, server, per_class, C, seed):
    rng = np.random.RandomState(seed + 13); y = y.cpu().numpy(); out = []
    for c in range(C):
        out += list(rng.choice(server[y[server] == c], per_class, replace=False))
    return np.array(out)


class Pipe:
    def __init__(self, dataset, dev):
        m, s = STATS[dataset]
        self.mean = torch.tensor(m, device=dev).view(1, 3, 1, 1); self.std = torch.tensor(s, device=dev).view(1, 3, 1, 1)

    def norm(self, x01):
        return (x01 - self.mean) / self.std

    @staticmethod
    def to01(xu8):
        return xu8.float() / 255.0

    @staticmethod
    def augment(x, gen):
        B, dev = x.shape[0], x.device
        xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
        i = torch.randint(0, 9, (B,), device=dev, generator=gen); j = torch.randint(0, 9, (B,), device=dev, generator=gen)
        ar = torch.arange(32, device=dev)
        out = xp[torch.arange(B, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None],
                 (i[:, None] + ar)[:, None, :, None], (j[:, None] + ar)[:, None, None, :]]
        flip = torch.rand(B, device=dev, generator=gen) < 0.5
        return torch.where(flip[:, None, None, None], out.flip(3), out)

    @staticmethod
    def corrupt(x, level, gen):
        """Contrast reduction + Gaussian blur + Gaussian noise, all in [0,1] space."""
        p = CORR[level]
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mu) * p["contrast"] + mu
        sig = p["blur_sigma"]; r = 2
        t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        k1 = torch.exp(-t ** 2 / (2 * sig ** 2)); k1 = k1 / k1.sum(); k = (k1[:, None] * k1[None, :])
        x = F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), k.expand(3, 1, 2 * r + 1, 2 * r + 1).contiguous(), groups=3)
        x = x + p["noise"] * torch.randn(x.shape, device=x.device, generator=gen)
        return x.clamp(0, 1)


# ----------------------------------------------------------------------------- training
def lr_at(t, a, outage_end):
    """Absolute-step schedule shared by every arm: linear warm-up, constant, cooldown over the
    last cooldown_frac of the outage (applied identically to A, B and the no-outage reference)."""
    warm = int(0.02 * a.total_steps)
    if t < warm:
        return a.lr * (t + 1) / warm
    if outage_end is not None:
        cool_start = outage_end - int(a.cooldown_frac * a.L)
        if t >= cool_start:
            return a.lr * max(0.0, (outage_end - t) / max(1, outage_end - cool_start))
    return a.lr


def train_arm(model, opt, X, Y, idx, steps, start_t, outage_end, a, P, gen, bad_mask=None, level=None, lr_const=None):
    """bad_mask: bool tensor over dataset indices marking images that are corrupted (cohort B, feature skew)."""
    model.train(); n = len(idx)
    for i in range(steps):
        b = idx[torch.randint(0, n, (a.batch_size,), device=idx.device, generator=gen)]
        x = P.to01(X[b])
        if bad_mask is not None:
            m = bad_mask[b]
            if m.any():
                x = x.clone(); x[m] = P.corrupt(x[m], level, gen)
        x = P.augment(x, gen)
        lr = lr_const if lr_const is not None else lr_at(start_t + i, a, outage_end)
        for g in opt.param_groups: g["lr"] = lr
        loss = F.cross_entropy(model(P.norm(x)), Y[b])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def new_opt(model, a, state=None):
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
    if state is not None: opt.load_state_dict(copy.deepcopy(state))  # never share momentum buffers between arms
    return opt


def clone_sd(sd):
    return {k: v.detach().clone() for k, v in sd.items()}


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, X, P, bs=2500, feats=False):
    model.eval(); outs, fs = [], []
    h = model.fc.register_forward_hook(lambda m, i, o: fs.append(i[0])) if feats else None
    for i in range(0, len(X), bs):
        outs.append(model(P.norm(P.to01(X[i:i + bs]))))
    if h: h.remove()
    return (torch.cat(outs), torch.cat(fs)) if feats else torch.cat(outs)


def acc_loss(logits, y):
    return (logits.argmax(1) == y).float().mean().item(), F.cross_entropy(logits, y).item()


def mix(sdA, sdB, alpha, keys=None):
    return {k: ((1 - alpha) * sdA[k].float() + alpha * sdB[k].float()) if (keys is None or k in keys) else sdA[k].clone()
            for k in sdA}


def stage_of(name, n_blocks):
    if name.startswith("fc."): return 4
    if name.startswith("stages."): return 1 + int(name.split(".")[1]) // (n_blocks // 3)
    return 0


@torch.no_grad()
def stage_features(model, X, P, n_blocks, bs=2500):
    """Outputs of the stem and of each of the 3 residual stages (globally pooled), plus penultimate."""
    model.eval(); bounds = [n_blocks // 3 * i - 1 for i in (1, 2, 3)]
    outs = {k: [] for k in ["s0", "s1", "s2", "s3"]}
    for i in range(0, len(X), bs):
        x = P.norm(P.to01(X[i:i + bs]))
        h = model.init_act(model.init_norm(model.init_conv(x))); outs["s0"].append(h.flatten(1))
        for bi, blk in enumerate(model.stages):
            h = blk(h)
            if bi in bounds: outs[f"s{bounds.index(bi) + 1}"].append(h.flatten(1))
    return {k: torch.cat(v) for k, v in outs.items()}


def linear_cka(X, Y):
    """Linear CKA via n x n Gram matrices (works for very wide flattened activations)."""
    X = X.double(); Y = Y.double()
    K, Lm = X @ X.T, Y @ Y.T
    n = K.shape[0]; H = torch.eye(n, device=X.device, dtype=X.dtype) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ Lm @ H
    return (torch.sum(Kc * Lc) / torch.sqrt(torch.sum(Kc * Kc) * torch.sum(Lc * Lc))).item()


def fit_head(f, y, w0, b0, C, mode, gen=None, epochs=5, lr=0.05, bs=64):
    head = nn.Linear(f.shape[1], C).to(f.device); head.weight.data.copy_(w0); head.bias.data.copy_(b0)
    if mode == "lbfgs":
        opt = torch.optim.LBFGS(head.parameters(), lr=1.0, max_iter=300, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = F.cross_entropy(head(f), y) + 1e-4 * head.weight.pow(2).sum(); l.backward(); return l
        opt.step(closure)
    else:
        opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
        for _ in range(epochs):
            perm = torch.randperm(len(y), device=f.device, generator=gen)
            for i in range(0, len(y), bs):
                b = perm[i:i + bs]; opt.zero_grad(); F.cross_entropy(head(f[b]), y[b]).backward(); opt.step()
    return head


def probe_on(model, sd, X, Y, Xte, Yte, pool_t, P, C):
    """Converged head-only probe (warm start from the model's own head, L-BFGS) on clean client-pool features."""
    _, fpool = predict(model, X[pool_t], P, feats=True); _, fte = predict(model, Xte, P, feats=True)
    hp = fit_head(fpool, Y[pool_t], sd["fc.weight"].float(), sd["fc.bias"].float(), C, "lbfgs")
    with torch.no_grad():
        return (hp(fte).argmax(1) == Yte).float().mean().item()


# ----------------------------------------------------------------------------- one run
def get_trunk(a, X, Y, pool_t, P, dev, seed, onset_step, C):
    path = os.path.join(a.trunk_cache, f"trunk_{a.dataset}_seed{seed}_onset{onset_step}_wd{a.wd}_lr{a.lr}.pt")
    if os.path.exists(path):
        ck = torch.load(path, map_location=dev, weights_only=False); return ck["model"], ck["opt"]
    torch.manual_seed(seed)
    model = ResNet20LN(num_classes=C).to(dev); opt = new_opt(model, a)
    gen = torch.Generator(device=dev).manual_seed(seed * 1000 + 1)
    train_arm(model, opt, X, Y, pool_t, onset_step, 0, None, a, P, gen)
    ck = {"model": clone_sd(model.state_dict()), "opt": copy.deepcopy(opt.state_dict())}
    os.makedirs(a.trunk_cache, exist_ok=True); torch.save(ck, path)
    return ck["model"], ck["opt"]


def run_one(a, data, source, onset_pct, seed, dev):
    X, Y, Xte, Yte = data; C = 10 if a.dataset == "cifar10" else 100
    s, level = SOURCES[source]; P = Pipe(a.dataset, dev); t0 = time.time()
    spc, vpc, cpc = (100, 200, 20) if C == 10 else (10, 20, 2)
    server, val, pool = make_splits(Y, C, seed, spc, vpc)
    A_idx, B_idx = cohort_split(Y, pool, C, s, seed)
    cal = calib_subset(Y, server, cpc, C, seed)
    T = lambda v: torch.tensor(v, device=dev, dtype=torch.long)
    pool_t, A_t, B_t, val_t, cal_t = T(pool), T(A_idx), T(B_idx), T(val), T(cal)
    bad = None
    if level:
        bad = torch.zeros(len(Y), dtype=torch.bool, device=dev); bad[B_t] = True
    onset_step = int(round(onset_pct / 100 * a.total_steps)); end = onset_step + a.L

    trunk_sd, trunk_opt = get_trunk(a, X, Y, pool_t, P, dev, seed, onset_step, C)
    m_tr = ResNet20LN(num_classes=C).to(dev); m_tr.load_state_dict(trunk_sd)
    acc_onset = acc_loss(predict(m_tr, Xte, P), Yte)[0]

    arms = {}
    for name, idx, sid in [("A", A_t, 2), ("B", B_t, 3), ("ref", pool_t, 4)]:
        m = ResNet20LN(num_classes=C).to(dev); m.load_state_dict(trunk_sd); opt = new_opt(m, a, trunk_opt)
        gen = torch.Generator(device=dev).manual_seed(seed * 1000 + sid * 10 + int(onset_pct * 10))
        train_arm(m, opt, X, Y, idx, a.L, onset_step, end, a, P, gen, bad_mask=bad if name != "A" else None, level=level)
        arms[name] = clone_sd(m.state_dict())
    sdA, sdB, sdR = arms["A"], arms["B"], arms["ref"]
    sdM = mix(sdA, sdB, 0.5)
    mk = lambda sd: (lambda m: (m.load_state_dict(sd), m)[1])(ResNet20LN(num_classes=C).to(dev))
    mA, mB, mR, mM = mk(sdA), mk(sdB), mk(sdR), mk(sdM)
    L = {k: predict(m, Xte, P) for k, m in [("A", mA), ("B", mB), ("ref", mR), ("merge", mM)]}
    Lv = {k: predict(m, X[val_t], P) for k, m in [("A", mA), ("B", mB), ("merge", mM)]}
    yv = Y[val_t]
    r = dict(dataset=a.dataset, source=source, s=s, corruption=level, onset_pct=onset_pct, onset_step=onset_step,
             L=a.L, total_steps=a.total_steps, seed=seed, lr=a.lr, wd=a.wd, acc_onset=acc_onset,
             n_A=len(A_idx), n_B=len(B_idx), n_server=len(server), n_val=len(val), n_cal=len(cal))
    for k in L: r[f"acc_{k}"], r[f"loss_{k}"] = acc_loss(L[k], Yte)
    r["penalty"] = r["acc_ref"] - r["acc_merge"]

    # ---- interpolation paths (full, head-only, body-only)
    head_keys = {k for k in sdA if k.startswith("fc.")}; body_keys = set(sdA) - head_keys
    alphas = [round(0.1 * i, 1) for i in range(11)]
    for tag, keys, base in [("full", None, None), ("head_only", head_keys, sdM), ("body_only", body_keys, sdM)]:
        accs, losses = [], []
        for al in alphas:
            sd = mix(sdA, sdB, al) if keys is None else {k: (((1 - al) * sdA[k].float() + al * sdB[k].float()) if k in keys else base[k]) for k in sdA}
            ac, lo = acc_loss(predict(mk(sd), Xte, P), Yte); accs.append(ac); losses.append(lo)
        r[f"path_{tag}_acc"], r[f"path_{tag}_loss"] = accs, losses
    lin = lambda e0, e1: [(1 - al) * e0 + al * e1 for al in alphas]
    r["barrier_acc_endpoint"] = max(l - p for l, p in zip(lin(r["path_full_acc"][0], r["path_full_acc"][-1]), r["path_full_acc"]))
    r["barrier_loss_endpoint"] = max(p - l for l, p in zip(lin(r["path_full_loss"][0], r["path_full_loss"][-1]), r["path_full_loss"]))

    # ---- drift
    nb = len(mA.stages); diff = {k: sdA[k].float() - sdB[k].float() for k in sdA if sdA[k].dtype.is_floating_point}
    def D(keys):
        keys = [k for k in keys if k in diff]; n = sum(diff[k].numel() for k in keys)
        return float(torch.sqrt(sum(diff[k].pow(2).sum() for k in keys) / n)) if n else float("nan")
    r["d_pair"], r["d_body"], r["d_head"] = D(diff), D(body_keys), D(head_keys)
    r["rho"] = r["d_head"] / r["d_body"]
    for j in range(4): r[f"d_stage{j}"] = D([k for k in diff if stage_of(k, nb) == j])

    # ---- CKA (A vs B, merge vs ref) per stage + penultimate, on a fixed 2,000-image test subset
    sub = torch.arange(2000, device=dev)
    fA, fB, fM, fR = [stage_features(m, Xte[sub], P, nb) for m in (mA, mB, mM, mR)]
    pen = {k: predict(m, Xte[sub], P, feats=True)[1] for k, m in [("A", mA), ("B", mB), ("M", mM), ("R", mR)]}
    for st in ["s0", "s1", "s2", "s3"]:
        r[f"cka_AB_{st}"] = linear_cka(fA[st], fB[st]); r[f"cka_MR_{st}"] = linear_cka(fM[st], fR[st])
    r["cka_AB_pen"] = linear_cka(pen["A"], pen["B"]); r["cka_MR_pen"] = linear_cka(pen["M"], pen["R"])

    # ---- stage-wise partial merges: prefix (<= j) averaged, suffix kept per branch, softmax-ensembled
    pm = {}
    for j in range(5):
        keys = {k for k in sdA if stage_of(k, nb) <= j}
        hA = mk({k: (0.5 * (sdA[k].float() + sdB[k].float()) if k in keys else sdA[k]) for k in sdA})
        hB = mk({k: (0.5 * (sdA[k].float() + sdB[k].float()) if k in keys else sdB[k]) for k in sdA})
        p = 0.5 * (F.softmax(predict(hA, Xte, P), 1) + F.softmax(predict(hB, Xte, P), 1))
        pm[f"prefix_{j}"] = (p.argmax(1) == Yte).float().mean().item()
    r["stagewise_merge_acc"] = pm

    # ---- baselines and cures
    gen = torch.Generator(device=dev).manual_seed(seed * 1000 + 99)
    r["acc_ensemble_full"] = ((F.softmax(L["A"], 1) + F.softmax(L["B"], 1)).argmax(1) == Yte).float().mean().item()
    vA, vB = acc_loss(Lv["A"], yv)[0], acc_loss(Lv["B"], yv)[0]
    r["acc_best_branch"] = r["acc_A"] if vA >= vB else r["acc_B"]
    # Cure 3: merged body + both heads, softmax average
    _, fte = predict(mM, Xte, P, feats=True)
    hA_logits = fte @ sdA["fc.weight"].float().T + sdA["fc.bias"].float(); hB_logits = fte @ sdB["fc.weight"].float().T + sdB["fc.bias"].float()
    r["acc_cure3_decoupled"] = ((F.softmax(hA_logits, 1) + F.softmax(hB_logits, 1)).argmax(1) == Yte).float().mean().item()
    # Cure 1: logit adjustment with the server-known prior of the union (tau chosen on val)
    cnt = torch.bincount(Y[torch.cat([A_t, B_t])], minlength=C).float(); logpi = torch.log(cnt / cnt.sum())
    best_tau = max([0.0, 0.25, 0.5, 1.0, 1.5, 2.0], key=lambda tau: acc_loss(Lv["merge"] - tau * logpi, yv)[0])
    r["cure1_tau"], r["acc_cure1"] = best_tau, acc_loss(L["merge"] - best_tau * logpi, Yte)[0]
    # Cure 2: head-only on server calibration set (fixed protocol: 5 epochs), and epochs tuned on val
    _, fcal = predict(mM, X[cal_t], P, feats=True); _, fval = predict(mM, X[val_t], P, feats=True)
    w0, b0 = sdM["fc.weight"].float(), sdM["fc.bias"].float()
    h2 = fit_head(fcal, Y[cal_t], w0, b0, C, "sgd", gen, epochs=5)
    with torch.no_grad(): r["acc_cure2"] = (h2(fte).argmax(1) == Yte).float().mean().item()
    best = None
    for ep in [1, 3, 5, 10, 20, 40]:
        h = fit_head(fcal, Y[cal_t], w0, b0, C, "sgd", gen, epochs=ep)
        with torch.no_grad(): va = (h(fval).argmax(1) == yv).float().mean().item()
        if best is None or va > best[0]: best = (va, ep, h)
    with torch.no_grad(): r["acc_cure2_tuned"], r["cure2_tuned_epochs"] = (best[2](fte).argmax(1) == Yte).float().mean().item(), best[1]
    # Full-model fine-tune on the same 200 server images (5 epochs, augmented)
    mF = mk(sdM); optF = torch.optim.SGD(mF.parameters(), lr=a.fullft_lr, momentum=0.9)
    mF.train()
    for _ in range(5):
        perm = cal_t[torch.randperm(len(cal_t), device=dev, generator=gen)]
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; x = Pipe.augment(P.to01(X[b]), gen)
            optF.zero_grad(); F.cross_entropy(mF(P.norm(x)), Y[b]).backward(); optF.step()
    r["acc_cure2_fullft"] = acc_loss(predict(mF, Xte, P), Yte)[0]
    # Converged head-only probe on clean client-pool features (diagnostic, not deployable)
    _, fpool = predict(mM, X[pool_t], P, feats=True)
    hp = fit_head(fpool, Y[pool_t], w0, b0, C, "lbfgs")
    with torch.no_grad(): r["acc_probe_converged"] = (hp(fte).argmax(1) == Yte).float().mean().item()
    gap = r["acc_ref"] - r["acc_merge"]
    r["head_share"] = (r["acc_probe_converged"] - r["acc_merge"]) / gap if gap > 0.005 else None
    # Same converged probe on the no-outage reference backbone: removes the "converged head beats SGD head" bias.
    r["acc_probe_ref"] = probe_on(mR, sdR, X, Y, Xte, Yte, pool_t, P, C)

    # ---- recovery: resume joint training on all client data (constant lr), from merge / Cure 2 / ref
    sdC2 = clone_sd(sdM); sdC2["fc.weight"] = h2.weight.data.clone(); sdC2["fc.bias"] = h2.bias.data.clone()
    for tag, sd0 in [("merge", sdM), ("cure2", sdC2), ("ref", sdR)]:
        m = mk(sd0); opt = new_opt(m, a); g2 = torch.Generator(device=dev).manual_seed(seed * 1000 + 77)
        curve, done = [], 0
        for cp in a.recovery_checkpoints:
            train_arm(m, opt, X, Y, pool_t, cp - done, 0, None, a, P, g2, bad_mask=bad, level=level, lr_const=a.recovery_lr)
            done = cp; curve.append(acc_loss(predict(m, Xte, P), Yte)[0])
        r[f"recovery_{tag}"] = curve
    r["recovery_checkpoints"] = a.recovery_checkpoints
    r["seconds"] = time.time() - t0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--sources", nargs="+", default=["iid", "label0.8", "label1.0"])
    ap.add_argument("--onsets", nargs="+", type=float, default=[0.5, 1, 2, 5, 10, 25], help="% of total_steps")
    ap.add_argument("--L", type=int, default=5000, help="outage length in steps (fixed across onsets)")
    ap.add_argument("--total_steps", type=int, default=25000, help="reference budget for onset %% and warm-up")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--cooldown_frac", type=float, default=0.1)
    ap.add_argument("--fullft_lr", type=float, default=0.01)
    ap.add_argument("--recovery_lr", type=float, default=0.02)
    ap.add_argument("--recovery_checkpoints", nargs="+", type=int, default=[50, 100, 250, 500, 1000])
    ap.add_argument("--data_dir", default="./data/cache")
    ap.add_argument("--trunk_cache", default="./runs/trunk_cache")
    ap.add_argument("--out_dir", default="./runs/main")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(); os.makedirs(a.out_dir, exist_ok=True)
    data = load_data(a.dataset, a.data_dir, dev)
    for seed in a.seeds:
        for onset in a.onsets:
            for src in a.sources:
                path = os.path.join(a.out_dir, f"{a.dataset}_{src}_k{onset}_L{a.L}_seed{seed}.json")
                if os.path.exists(path): print("skip", path); continue
                r = run_one(a, data, src, onset, seed, dev); r["args"] = vars(a)
                json.dump(r, open(path, "w"), indent=1)
                print(f"[{a.dataset} {src:15s} k={onset:5.1f}% L={a.L} seed={seed}] ref={r['acc_ref']*100:.2f} "
                      f"merge={r['acc_merge']*100:.2f} pen={r['penalty']*100:.2f} cure2={r['acc_cure2']*100:.2f} "
                      f"fullft={r['acc_cure2_fullft']*100:.2f} probe={r['acc_probe_converged']*100:.2f} "
                      f"rec@1000 merge/cure2/ref={r['recovery_merge'][-1]*100:.1f}/{r['recovery_cure2'][-1]*100:.1f}/"
                      f"{r['recovery_ref'][-1]*100:.1f} | {r['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
