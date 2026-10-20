"""Cycle-duration distribution by speed regime (Reviewer 2, point 2).

Reads the stored training data, converts the stored period fraction to
seconds (period * 3.2 s window) and the stored speed fraction to m/s
(speed * 2.4), then reports n, median, IQR, min and max per speed bin, and
the same split by leg length (below / above the median leg length).

    python analyse_cycle_durations.py
    python analyse_cycle_durations.py --data-dir "gait reference 6joint"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
SPEED_BINS = np.array([0.0, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5])


def summarise(dur: np.ndarray, speed: np.ndarray, bins: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (speed >= lo) & (speed < hi)
        if not m.any():
            continue
        d = dur[m]
        rows.append(dict(speed_lo=lo, speed_hi=hi, n=int(m.sum()), median=np.median(d),
                         q1=np.percentile(d, 25), q3=np.percentile(d, 75), min=d.min(), max=d.max()))
    return rows


def to_tex(rows: list[dict], caption: str, label: str) -> str:
    lines = [r"\begin{table}[!htb]", r"  \centering", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
             r"  \footnotesize", r"  \begin{tabular}{@{}lccccc@{}}", r"    \toprule",
             r"    Speed [m/s] & $n$ & Median [s] & IQR [s] & Min [s] & Max [s] \\", r"    \midrule"]
    for r in rows:
        lines.append(f"    ${r['speed_lo']:.1f}$--${r['speed_hi']:.1f}$ & {r['n']} & {r['median']:.2f} & "
                     f"{r['q1']:.2f}--{r['q3']:.2f} & {r['min']:.2f} & {r['max']:.2f} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="gait reference phase 2")
    ap.add_argument("--out-dir", default="tables")
    args = ap.parse_args()
    d = REPO / args.data_dir
    X = np.load(d / "input_vector.npy")
    period = np.load(d / "period.npy").ravel()
    dur = period * 3.2
    speed = X[:, 0] * 2.4
    leg = X[:, 1:3].mean(axis=1)

    print(f"{len(dur)} cycles. Duration {dur.min():.2f} to {dur.max():.2f} s, median {np.median(dur):.2f} s.")
    rows_all = summarise(dur, speed, SPEED_BINS)
    hdr = f"{'speed':>9} {'n':>5} {'median':>7} {'IQR':>12} {'min':>5} {'max':>5}"
    print("\nAll cycles\n" + hdr)
    for r in rows_all:
        print(f"{r['speed_lo']:.1f}-{r['speed_hi']:.1f} {r['n']:5d} {r['median']:7.2f} "
              f"{r['q1']:5.2f}-{r['q3']:5.2f} {r['min']:5.2f} {r['max']:5.2f}")

    med_leg = np.median(leg)
    for name, m in (("short legs", leg <= med_leg), ("long legs", leg > med_leg)):
        print(f"\n{name} (leg {'<=' if name[0]=='s' else '>'} {med_leg:.3f} m, n={m.sum()})\n" + hdr)
        for r in summarise(dur[m], speed[m], SPEED_BINS):
            print(f"{r['speed_lo']:.1f}-{r['speed_hi']:.1f} {r['n']:5d} {r['median']:7.2f} "
                  f"{r['q1']:5.2f}-{r['q3']:5.2f} {r['min']:5.2f} {r['max']:5.2f}")

    out = REPO / args.out_dir
    out.mkdir(exist_ok=True)
    with (out / "cycle_durations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        w.writeheader(); w.writerows(rows_all)
    (out / "cycle_durations.tex").write_text(to_tex(
        rows_all, f"Gait-cycle duration by speed regime over the ${len(dur)}$ cycles.", "tab:durations"))
    print(f"\nwrote {out / 'cycle_durations.csv'} and .tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
