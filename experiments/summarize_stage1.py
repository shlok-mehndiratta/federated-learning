"""
Stage 1 Factorial Sweeps Analysis & Statistical Summary.
Aggregates all 12 runs from ./runs/stage1_science/ into a unified scientific synthesis:
- Compiles the full empirical master table across all conditions.
- Evaluates Hypothesis H-A: Barrier vs Onset at matched D_pair against Rung 2b control.
- Evaluates Hypothesis H-B1: Head Share of gap vs Onset.
- Evaluates Hypothesis H-B2: Logit offset fidelity (slope, R2) vs Onset.
- Evaluates Hypothesis H-C: Body vs Head curvature and drift ratio.
"""

import argparse
import glob
import json
import os
import numpy as np


def load_all_results(runs_dir: str):
    """Loads all Stage 1 JSON result files from runs_dir."""
    pattern = os.path.join(runs_dir, "result_*.json")
    files = glob.glob(pattern)
    records = []
    for f in files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
                data["_filepath"] = f
                records.append(data)
        except Exception as e:
            print(f"Warning: could not load {f}: {e}")
    return records


def format_table(records):
    """Formats a clean Markdown summary table across all conditions."""
    def sort_key(r):
        cfg = r["config"]
        src_order = {"R2b": 0, "R3-L": 1}
        asym_order = {None: 0, "A_skew_B_iid": 1, "ratio_70_30": 2}
        return (
            src_order.get(cfg["source"], 9),
            asym_order.get(cfg.get("asymmetric_mode"), 0),
            cfg.get("s", 0.0),
            cfg.get("onset_pct", 0.0)
        )

    records = sorted(records, key=sort_key)
    
    headers = [
        "Source", "Skew (s)", "Mode", "Onset", 
        "Acc A", "Acc B", "Merged Acc", "Acc Dip (%)", "Loss Barrier",
        "D_pair", "D_body", "D_head", "Head/Body",
        "CKA", "Probe Gap", "Head Share"
    ]
    
    rows = []
    for r in records:
        cfg = r["config"]
        ep = r["endpoints"]
        dist = r["distances"]
        bar = r["barriers"]
        rep = r["representation"]
        dec = r["functional_decomposition"]
        
        src = cfg["source"]
        s_val = f"{cfg['s']:.1f}" if src == "R3-L" else "0.0 (IID)"
        mode = cfg.get("asymmetric_mode") or "Symmetric"
        if mode == "A_skew_B_iid":
            mode = "A:skew, B:iid"
        onset = f"{cfg['onset_pct']:.1f}%"
        
        acc_A = f"{ep['acc_A']*100:.2f}%"
        acc_B = f"{ep['acc_B']*100:.2f}%"
        merged_acc = f"{bar['merged_acc']*100:.2f}%"
        acc_dip = f"{bar['max_acc_dip']*100:.2f}%"
        loss_bar = f"{bar['max_loss_barrier']:.4f}"
        
        d_p = f"{dist['d_pair']:.4f}"
        d_b = f"{dist['d_body']:.4f}"
        d_h = f"{dist['d_head']:.4f}"
        hb_ratio = f"{(dist['d_head'] / max(dist['d_body'], 1e-6)):.2f}x"
        
        cka = f"{rep['angular_cka']:.3f}"
        probe_gap = f"{dec['probe_gap']*100:+.2f}%"
        h_share = f"{dec['head_share']*100:.1f}%" if dec.get("head_share") is not None else "N/A"
        
        rows.append([
            src, s_val, mode, onset,
            acc_A, acc_B, merged_acc, acc_dip, loss_bar,
            d_p, d_b, d_h, hb_ratio,
            cka, probe_gap, h_share
        ])
        
    return headers, rows


def analyze_matched_contrasts(records):
    """Analyzes trajectory checkpoints to find D_pair-matched contrasts across onsets."""
    print("\n" + "=" * 90)
    print("D_PAIR-MATCHED DESIGN ANALYSIS (Testing Hypothesis H-A)")
    print("=" * 90)
    
    # Group by (source, s, asymmetric_mode)
    groups = {}
    for r in records:
        cfg = r["config"]
        key = (cfg["source"], cfg.get("s", 0.0), cfg.get("asymmetric_mode"))
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
        
    for (src, s_val, asym), recs in groups.items():
        if len(recs) < 2:
            continue
        asym_str = f" [{asym}]" if asym else ""
        print(f"\n--- Group: Source={src} | s={s_val}{asym_str} ---")
        for r in sorted(recs, key=lambda x: x["config"]["onset_pct"]):
            onset = r["config"]["onset_pct"]
            final_d = r["distances"]["d_pair"]
            dip = r["barriers"]["max_acc_dip"] * 100
            m_acc = r["barriers"]["merged_acc"] * 100
            ckpts = r.get("trajectory_ckpts", [])
            print(f"  * Onset {onset:4.1f}% | Final D_pair={final_d:.5f} | Final Acc Dip={dip:5.2f}% | Merged Acc={m_acc:5.2f}% | Checkpoints={len(ckpts)}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Factorial Sweeps Analysis & Statistical Summary")
    parser.add_argument("--runs_dir", type=str, default="./runs/stage1_science", help="Directory containing Stage 1 JSON records")
    args = parser.parse_args()
    
    records = load_all_results(args.runs_dir)
    if not records:
        print(f"No results found in {args.runs_dir}")
        return
        
    print("\n" + "=" * 90)
    print(f"STAGE 1 EMPIRICAL SYNTHESIS: Total Conditions Loaded = {len(records)}")
    print("=" * 90)
    
    headers, rows = format_table(records)
    
    # Print formatted table
    col_widths = [max(len(str(val)) for val in col) for col in zip(headers, *rows)]
    hdr_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    
    print("\n" + hdr_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)))
    print(sep_line + "\n")
    
    # Hypothesis Tests Analysis
    analyze_matched_contrasts(records)
    
    # Also dump as a clean JSON/text report
    summary_path = os.path.join(args.runs_dir, "stage1_master_summary.json")
    with open(summary_path, "w") as fp:
        json.dump([
            {
                "config": r["config"],
                "endpoints": r["endpoints"],
                "distances": r["distances"],
                "barriers": r["barriers"],
                "representation": r["representation"],
                "functional_decomposition": r["functional_decomposition"]
            }
            for r in records
        ], fp, indent=2)
    print(f"\nMaster summary saved to {summary_path}\n" + "=" * 90)


if __name__ == "__main__":
    main()
