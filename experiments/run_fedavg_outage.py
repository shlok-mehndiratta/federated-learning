"""
Federated learning outage experiment (10 clients, 2 cohorts).

This script simulates a coordinator outage in a federated learning network where
10 clients train with local SGD and periodic FedAvg aggregation. During an outage,
the network splits into two isolated 5-client cohorts that continue aggregating locally.

Terms used in code:
  * Outage start (onset rounds k): global communication rounds before the outage.
  * Outage length (L): communication rounds conducted within each isolated cohort.
  * Recovery rounds (R): global communication rounds resumed after reconnection.
  * No-outage model (ref): global FedAvg continued without communication interruption.
  * Merged model (acc_naive): simple parameter average of the two cohort models upon reconnection.
  * Final layer recalibration (cure2): retrains only the classification layer on 200 server images.

Reads: CIFAR-10 dataset from --data_dir.
Writes: JSON files with communication telemetry and round-by-round accuracies to --out_dir.

Example:
  python experiments/run_fedavg_outage.py --sources label0.8 iid --onsets 2 25 --L 40 --R 10 --local_steps 20 --batch_size 128 --lr 0.03 --momentum 0.9 --seeds 42 --data_dir data/cache --out_dir runs/fedavg
"""
import argparse, json, os, sys, time
from copy import deepcopy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fl_outage.models.resnet import ResNet20LN

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
NUM_CLIENTS, NUM_CLASSES, SERVER_PER_CLASS, CAL_PER_CLASS = 10, 10, 100, 20


# ----------------------------------------------------------------------------- data
def load_cifar(data_dir, device, fake=False):
    if fake:  # smoke-test mode, no download
        g = torch.Generator().manual_seed(0)
        xtr = torch.rand(5000, 3, 32, 32, generator=g); ytr = torch.arange(5000) % 10
        xte = torch.rand(1000, 3, 32, 32, generator=g); yte = torch.arange(1000) % 10
    else:
        import torchvision
        tr = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
        te = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
        xtr = torch.tensor(tr.data).permute(0, 3, 1, 2).float() / 255.0; ytr = torch.tensor(tr.targets)
        xte = torch.tensor(te.data).permute(0, 3, 1, 2).float() / 255.0; yte = torch.tensor(te.targets)
    return xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)


def make_partitions(y, s, seed, within_alpha=None):
    """Server holds SERVER_PER_CLASS images/class (disjoint from clients). Remaining images are
    split between cohorts with label separation s (p_A(c)=(1+s)/C for c<5, mirrored for B),
    then IID across the 5 clients inside each cohort."""
    rng = np.random.RandomState(seed)
    y = y.cpu().numpy()
    server, cohort = [], {0: [], 1: []}
    for c in range(NUM_CLASSES):
        idx = rng.permutation(np.where(y == c)[0])
        server.extend(idx[:SERVER_PER_CLASS]); rest = idx[SERVER_PER_CLASS:]
        frac_A = (1 + s) / 2 if c < NUM_CLASSES // 2 else (1 - s) / 2
        n_A = int(round(frac_A * len(rest)))
        cohort[0].extend(rest[:n_A]); cohort[1].extend(rest[n_A:])
    clients = []
    for g in (0, 1):
        idx = rng.permutation(np.array(cohort[g]))
        if within_alpha is None:
            clients.extend(np.array_split(idx, NUM_CLIENTS // 2))
        else:  # Dirichlet(alpha) label skew across the 5 clients inside each cohort
            k = NUM_CLIENTS // 2
            for _try in range(50):  # resample until every client holds >= 256 images
                parts = [[] for _ in range(k)]
                for c in np.unique(y[idx]):
                    ic = rng.permutation(idx[y[idx] == c])
                    cuts = (np.cumsum(rng.dirichlet([within_alpha] * k)) * len(ic)).astype(int)[:-1]
                    for j, chunk in enumerate(np.split(ic, cuts)): parts[j].extend(chunk)
                if min(len(pp) for pp in parts) >= 256: break
            clients.extend([np.array(pp) for pp in parts])
    cal = []
    srv = np.array(server); ys = y[srv]
    for c in range(NUM_CLASSES):
        cal.extend(rng.choice(srv[ys == c], CAL_PER_CLASS, replace=False))
    return [torch.tensor(ci, dtype=torch.long) for ci in clients], torch.tensor(np.array(cal), dtype=torch.long)


def corrupt(x, gen):
    """Persistent synthetic covariate shift for cohort B: Gaussian blur (sigma=1) + pixel noise."""
    k = torch.tensor([1., 4., 6., 4., 1.], device=x.device); k = (k[:, None] * k[None, :]); k = k / k.sum()
    x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k.expand(3, 1, 5, 5).contiguous(), groups=3)
    return (x + 0.08 * torch.randn(x.shape, device=x.device, generator=gen)).clamp(0, 1)


def augment(x, gen):
    B = x.shape[0]; dev = x.device
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (B,), device=dev, generator=gen); j = torch.randint(0, 9, (B,), device=dev, generator=gen)
    ar = torch.arange(32, device=dev)
    rows = (i[:, None] + ar)[:, None, :, None]; cols = (j[:, None] + ar)[:, None, None, :]
    out = xp[torch.arange(B, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None], rows, cols]
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# ----------------------------------------------------------------------------- FL core
class Fed:
    def __init__(self, xtr, ytr, clients, corrupted_clients, args, device, seed):
        self.xtr, self.ytr, self.clients, self.bad = xtr, ytr, clients, set(corrupted_clients)
        self.a, self.dev = args, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.work = ResNet20LN(num_classes=10).to(device)

    def local(self, state, cid, lr):
        m = self.work; m.load_state_dict(state); m.train()
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=self.a.momentum, weight_decay=5e-4)
        idx = self.clients[cid]
        for _ in range(self.a.local_steps):
            b = idx[torch.randint(0, len(idx), (self.a.batch_size,), device=self.dev, generator=self.gen)]
            x = self.xtr[b]
            if cid in self.bad: x = corrupt(x, self.gen)
            with torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16, enabled=self.a.amp and self.dev.type == "cuda"):
                loss = F.cross_entropy(m(norm(augment(x, self.gen))).float(), self.ytr[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return {k: v.detach().clone() for k, v in m.state_dict().items()}

    def fedavg_round(self, state, cids, lr):
        acc = None
        sizes = torch.tensor([len(self.clients[c]) for c in cids], dtype=torch.float32)
        w = (sizes / sizes.sum()).tolist()
        for wi, c in zip(w, cids):
            s = self.local(state, c, lr)
            acc = {k: wi * v.float() for k, v in s.items()} if acc is None else {k: acc[k] + wi * s[k].float() for k in acc}
        return acc

    def run(self, state, cids, rounds, eval_fn=None, start_round=0):
        hist = []
        for i in range(rounds):
            lr = self.a.lr * min(1.0, (start_round + i + 1) / self.a.warmup_rounds)  # global-round warmup
            state = self.fedavg_round(state, cids, lr)
            if eval_fn: hist.append(eval_fn(state))
        return state, hist


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def logits_of(model, x, bs=2000):
    model.eval(); return torch.cat([model(norm(x[i:i + bs])) for i in range(0, len(x), bs)])


@torch.no_grad()
def feats_of(model, x, bs=2000):
    out = []
    h = model.fc.register_forward_hook(lambda m, i, o: out.append(i[0]))
    logits_of(model, x, bs); h.remove()
    return torch.cat(out)


def acc_state(state, xte, yte, dev):
    m = ResNet20LN(num_classes=10).to(dev); m.load_state_dict(state)
    return (logits_of(m, xte).argmax(1) == yte).float().mean().item()


def avg_states(a, b):
    return {k: 0.5 * (a[k].float() + b[k].float()) for k in a}


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = lambda A, B: (A.T @ B).pow(2).sum()
    return (hsic(X, Y) / torch.sqrt(hsic(X, X) * hsic(Y, Y))).item()


def head_fit(feats, y, w0, b0, epochs=None, lr=0.05, bs=64, gen=None, lbfgs=False):
    head = nn.Linear(feats.shape[1], 10).to(feats.device)
    head.weight.data.copy_(w0); head.bias.data.copy_(b0)
    if lbfgs:  # diagnostic ceiling: warm-started, full-batch, converged
        opt = torch.optim.LBFGS(head.parameters(), lr=1.0, max_iter=200, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = F.cross_entropy(head(feats), y) + 1e-4 * head.weight.pow(2).sum(); l.backward(); return l
        opt.step(closure)
    else:      # Cure 2 protocol: 5 epochs SGD on 200 server samples
        opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
        for _ in range(epochs):
            perm = torch.randperm(len(y), device=feats.device, generator=gen)
            for i in range(0, len(y), bs):
                b = perm[i:i + bs]; opt.zero_grad(); F.cross_entropy(head(feats[b]), y[b]).backward(); opt.step()
    return head


# ----------------------------------------------------------------------------- one run
def run_one(source, k, seed, args, data, dev):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed); np.random.seed(seed)
    s = {"iid": 0.0, "label0.8": 0.8, "label1.0": 1.0, "feature": 0.0}[source]
    clients, cal_idx = make_partitions(ytr, s, seed, args.within_alpha)
    bad = list(range(5, 10)) if source == "feature" else []
    fed = Fed(xtr, ytr, [c.to(dev) for c in clients], bad, args, dev, seed)
    ev = lambda st: acc_state(st, xte, yte, dev)
    A, B, ALL = list(range(5)), list(range(5, 10)), list(range(10))
    t0 = time.time()

    init = {kk: v.clone() for kk, v in ResNet20LN(num_classes=10).to(dev).state_dict().items()}
    w_k, _ = fed.run(init, ALL, k, start_round=0)                                   # 1. pre-outage FedAvg
    acc_onset = ev(w_k)
    w_A, _ = fed.run(w_k, A, args.L, start_round=k)                                 # 2a. isolated cohorts
    w_B, _ = fed.run(w_k, B, args.L, start_round=k)
    w_ref, _ = fed.run(w_k, ALL, args.L, start_round=k)                             # 2b. no-outage counterfactual
    w_mrg = avg_states(w_A, w_B)

    mA, mB, mM = [ResNet20LN(num_classes=10).to(dev) for _ in range(3)]
    mA.load_state_dict(w_A); mB.load_state_dict(w_B); mM.load_state_dict(w_mrg)
    lA, lB, lM = logits_of(mA, xte), logits_of(mB, xte), logits_of(mM, xte)
    acc = lambda l: (l.argmax(1) == yte).float().mean().item()
    ens = acc(0.5 * (F.softmax(lA, 1) + F.softmax(lB, 1)))

    # Cure 2: server-held calibration set (disjoint from all clients), head only
    cal = cal_idx.to(dev)
    f_cal = feats_of(mM, xtr[cal]); f_te = feats_of(mM, xte)
    h2 = head_fit(f_cal, ytr[cal], mM.fc.weight.data, mM.fc.bias.data, epochs=5, gen=fed.gen)
    cure2 = (h2(f_te).argmax(1) == yte).float().mean().item()
    w_cure2 = {kk: v.clone() for kk, v in w_mrg.items()}
    w_cure2["fc.weight"] = h2.weight.data.clone(); w_cure2["fc.bias"] = h2.bias.data.clone()

    # Diagnostic ceiling: converged head on ALL client data (not deployable; measures head share)
    all_idx = torch.cat(clients).to(dev)
    xa = xtr[all_idx]
    if source == "feature":  # probe sees the data as clients saw it
        nA = sum(len(clients[i]) for i in A); xa = torch.cat([xa[:nA], corrupt(xa[nA:], fed.gen)])
    hp = head_fit(feats_of(mM, xa), ytr[all_idx], mM.fc.weight.data, mM.fc.bias.data, lbfgs=True)
    probe = (hp(f_te).argmax(1) == yte).float().mean().item()

    # representation + drift telemetry (computed without any data except a public probe batch)
    pb = xte[:1000]
    cka = linear_cka(feats_of(mA, pb), feats_of(mB, pb))
    diff = {kk: (w_A[kk].float() - w_B[kk].float()) for kk in w_A}
    body = [kk for kk in diff if not kk.startswith("fc.")]
    n_b = sum(diff[kk].numel() for kk in body); n_h = sum(diff[kk].numel() for kk in ("fc.weight", "fc.bias"))
    d_body = torch.sqrt(sum(diff[kk].pow(2).sum() for kk in body) / n_b).item()
    d_head = torch.sqrt(sum(diff[kk].pow(2).sum() for kk in ("fc.weight", "fc.bias")) / n_h).item()
    d_pair = torch.sqrt(sum(v.pow(2).sum() for v in diff.values()) / (n_b + n_h)).item()

    # 4. recovery after reconnection
    _, rec_naive = fed.run(w_mrg, ALL, args.R, ev, start_round=k + args.L)
    _, rec_cure2 = fed.run(w_cure2, ALL, args.R, ev, start_round=k + args.L)
    _, rec_ref = fed.run(w_ref, ALL, args.R, ev, start_round=k + args.L)

    rec = dict(source=source, s=s, onset_rounds=k, seed=seed, L=args.L,
               acc_onset=acc_onset, acc_A=acc(lA), acc_B=acc(lB), acc_no_outage=ev(w_ref),
               acc_naive=acc(lM), acc_cure2=cure2, acc_ensemble=ens, acc_probe=probe,
               cka_penult=cka, d_pair=d_pair, d_body=d_body, d_head=d_head, head_body_ratio=d_head / d_body,
               recovery_naive=rec_naive, recovery_cure2=rec_cure2, recovery_no_outage=rec_ref,
               seconds=time.time() - t0)
    print(f"[{source:8s} k={k:2d} seed={seed}] no-outage={rec['acc_no_outage']*100:.2f} naive={rec['acc_naive']*100:.2f} "
          f"cure2={cure2*100:.2f} ens={ens*100:.2f} probe={probe*100:.2f} | rec@1 naive={rec_naive[0]*100:.2f} "
          f"cure2={rec_cure2[0]*100:.2f} | {rec['seconds']:.0f}s", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["label1.0", "label0.8", "feature", "iid"])
    ap.add_argument("--onsets", nargs="+", type=int, default=[2, 10], help="pre-outage FedAvg rounds k")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--L", type=int, default=20, help="outage length (rounds)")
    ap.add_argument("--R", type=int, default=5, help="post-reconnection recovery rounds")
    ap.add_argument("--local_steps", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--warmup_rounds", type=int, default=3)
    ap.add_argument("--momentum", type=float, default=0.9, help="client-side SGD momentum")
    ap.add_argument("--within_alpha", type=float, default=None, help="Dirichlet alpha across clients inside each cohort (None = IID)")
    ap.add_argument("--amp", type=int, default=1, help="bf16 autocast for local training")
    ap.add_argument("--data_dir", default="./data/cache")
    ap.add_argument("--out_dir", default="./runs/fedavg")
    ap.add_argument("--fake_data", action="store_true", help="smoke test without CIFAR download")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)
    data = load_cifar(args.data_dir, dev, fake=args.fake_data)
    for seed in args.seeds:
        for source in args.sources:
            for k in args.onsets:
                path = os.path.join(args.out_dir, f"fl_{source}_k{k}_seed{seed}.json")  # one config per out_dir
                if os.path.exists(path): print(f"skip {path}"); continue
                rec = run_one(source, k, seed, args, data, dev)
                rec["args"] = vars(args)
                json.dump(rec, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
