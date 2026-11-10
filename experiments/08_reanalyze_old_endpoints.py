"""
Re-analyze the EXISTING Stage 1 / Stage 3 branch endpoints (*_branchA.pt / *_branchB.pt) with the
corrected diagnostics, without any retraining (~1 min per pair):
  * stage-wise partial merges with the correct parameter->stage mapping and softmax ensembling
    (the old evaluate_cumulative_partial_merges mapped every body parameter to stage 0 and averaged
    logits, so every prefix equalled the full merge);
  * head-only and body-only interpolation paths;
  * converged head-only probe (warm start + L-BFGS) on all 50k clean training images;
  * full-output ensemble.
Usage (repo root):
  python experiments/08_reanalyze_old_endpoints.py --model_dirs runs/stage1_science runs/stage3_feature_skew
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, glob, importlib.util, json, sys
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
spec = importlib.util.spec_from_file_location("v2", os.path.join(HERE, "07_outage_v2.py"))
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
from fl_outage.models.resnet import ResNet20LN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", nargs="+", required=True)
    ap.add_argument("--data_dir", default="./data/cache")
    ap.add_argument("--out", default="runs/old_endpoints_reanalysis.json")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); v2.set_determinism()
    X, Y, Xte, Yte = v2.load_data("cifar10", a.data_dir, dev); P = v2.Pipe("cifar10", dev)
    files = sorted({f for d in a.model_dirs for f in glob.glob(os.path.join(d, "*_branchA.pt"))})
    print(f"{len(files)} endpoint pairs"); results = {}
    if os.path.exists(a.out): results = json.load(open(a.out))
    for fA in files:
        name = os.path.basename(fA).replace("_branchA.pt", "")
        fB = fA.replace("_branchA.pt", "_branchB.pt")
        if name in results or not os.path.exists(fB): continue
        sdA, sdB = torch.load(fA, map_location=dev, weights_only=False), torch.load(fB, map_location=dev, weights_only=False)
        mk = lambda sd: (lambda m: (m.load_state_dict(sd), m)[1])(ResNet20LN(num_classes=10).to(dev))
        nb = len(mk(sdA).stages); sdM = v2.mix(sdA, sdB, 0.5); mM = mk(sdM)
        r = {}
        LA, LB, LM = v2.predict(mk(sdA), Xte, P), v2.predict(mk(sdB), Xte, P), v2.predict(mM, Xte, P)
        r["acc_A"], r["acc_B"], r["acc_merge"] = [v2.acc_loss(l, Yte)[0] for l in (LA, LB, LM)]
        r["acc_ensemble_full"] = ((F.softmax(LA, 1) + F.softmax(LB, 1)).argmax(1) == Yte).float().mean().item()
        head = {k for k in sdA if k.startswith("fc.")}; body = set(sdA) - head
        for tag, keys in [("head_only", head), ("body_only", body)]:
            r[f"path_{tag}_acc"] = [v2.acc_loss(v2.predict(mk({k: (((1 - al) * sdA[k].float() + al * sdB[k].float()) if k in keys else sdM[k]) for k in sdA}), Xte, P), Yte)[0]
                                    for al in [0.1 * i for i in range(11)]]
        pm = {}
        for j in range(5):
            keys = {k for k in sdA if v2.stage_of(k, nb) <= j}
            hA = mk({k: (0.5 * (sdA[k].float() + sdB[k].float()) if k in keys else sdA[k]) for k in sdA})
            hB = mk({k: (0.5 * (sdA[k].float() + sdB[k].float()) if k in keys else sdB[k]) for k in sdA})
            p = 0.5 * (F.softmax(v2.predict(hA, Xte, P), 1) + F.softmax(v2.predict(hB, Xte, P), 1))
            pm[f"prefix_{j}"] = (p.argmax(1) == Yte).float().mean().item()
        r["stagewise_merge_acc"] = pm
        _, ftr = v2.predict(mM, X, P, feats=True); _, fte = v2.predict(mM, Xte, P, feats=True)
        hp = v2.fit_head(ftr, Y, sdM["fc.weight"].float(), sdM["fc.bias"].float(), 10, "lbfgs")
        with torch.no_grad(): r["acc_probe_converged"] = (hp(fte).argmax(1) == Yte).float().mean().item()
        results[name] = r; json.dump(results, open(a.out, "w"), indent=1)
        print(name, f"merge={r['acc_merge']*100:.2f} probe={r['acc_probe_converged']*100:.2f} ens={r['acc_ensemble_full']*100:.2f}",
              "prefix:", " ".join(f"{v*100:.1f}" for v in pm.values()), flush=True)


if __name__ == "__main__":
    main()
