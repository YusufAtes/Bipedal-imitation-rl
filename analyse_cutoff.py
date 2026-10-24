"""Low-pass cutoff check (Reviewer 2, point 1).

1. Frequency response of the zero-phase 4th-order Butterworth used in the
   pipeline: prints the -3 dB and -20 dB frequencies of the forward-backward
   (filtfilt) response.
2. Energy fraction of the raw 100 Hz joint-angle signals above a set of
   frequencies, per joint and overall.
3. MSE between the encoded 32-sample window with and without the filter
   (and at other cutoffs), per joint and overall, in rad^2.

    python analyse_cutoff.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import butter, freqz

from fft_extract_cycles import FS, JOINT_NAMES, encode_cycle, iter_trials, trial_angles

REPO = Path(__file__).resolve().parent


def response_points(cutoff: float, order: int) -> tuple[float, float]:
    b, a = butter(order, cutoff / (0.5 * FS), btype="low")
    w, h = freqz(b, a, worN=20000, fs=FS)
    mag_db = 20 * np.log10(np.abs(h) ** 2 + 1e-300)   # filtfilt: magnitude squared
    f3 = w[np.argmax(mag_db <= -3.0)]
    f20 = w[np.argmax(mag_db <= -20.0)]
    return float(f3), float(f20)


def energy_above(q: np.ndarray, f_edges: list[float]) -> np.ndarray:
    """Fraction of mean-removed signal energy above each edge, per joint. q: (n,6) at 100 Hz."""
    x = q - q.mean(axis=0)
    P = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(x.shape[0], 1 / FS)
    tot = P.sum(axis=0)
    return np.array([P[f > fe].sum(axis=0) / tot for fe in f_edges])   # (len(edges), 6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--paper-cutoff", type=float, default=4.0)
    ap.add_argument("--cutoffs", nargs="+", type=float, default=[3.0, 4.0, 5.0, 6.0, 8.0])
    ap.add_argument("--edges", nargs="+", type=float, default=[3.0, 4.0, 5.0, 6.0, 8.0])
    args = ap.parse_args()

    f3, f20 = response_points(args.paper_cutoff, args.order)
    print(f"filtfilt Butterworth order {args.order}, design cutoff {args.paper_cutoff} Hz: "
          f"-3 dB at {f3:.2f} Hz, -20 dB at {f20:.2f} Hz")

    frac, mse = [], {c: [] for c in args.cutoffs}
    n_trials = 0
    for _folder, _f, c in iter_trials():
        q, s, e, *_ = trial_angles(c)
        n_trials += 1
        frac.append(energy_above(q[s:e], args.edges))
        raw = encode_cycle(q, s, e, cutoff=None)
        if raw is None:
            continue
        for cut in args.cutoffs:
            enc = encode_cycle(q, s, e, cutoff=cut, order=args.order)
            mse[cut].append(((enc[2] - raw[2]) ** 2).mean(axis=0))   # per joint, rad^2
    frac = np.mean(frac, axis=0)          # (edges, 6)

    print(f"\n{n_trials} trials. Fraction of cycle energy above f, per joint (%):")
    print(f"{'f [Hz]':>7} " + " ".join(f"{j:>7}" for j in JOINT_NAMES) + f" {'all':>7}")
    rows = []
    for fe, fr in zip(args.edges, frac):
        print(f"{fe:7.1f} " + " ".join(f"{100*v:7.3f}" for v in fr) + f" {100*fr.mean():7.3f}")
        rows.append(dict(kind="energy_above_pct", value=fe, **{j: 100 * v for j, v in zip(JOINT_NAMES, fr)}, all=100 * fr.mean()))

    print("\nMSE between filtered and unfiltered 32-sample windows, per joint (1e-4 rad^2):")
    print(f"{'cut [Hz]':>8} " + " ".join(f"{j:>7}" for j in JOINT_NAMES) + f" {'all':>7}")
    for cut in args.cutoffs:
        m = np.mean(mse[cut], axis=0)
        print(f"{cut:8.1f} " + " ".join(f"{1e4*v:7.3f}" for v in m) + f" {1e4*m.mean():7.3f}")
        rows.append(dict(kind="filtered_vs_raw_mse_1e-4", value=cut, **{j: 1e4 * v for j, v in zip(JOINT_NAMES, m)}, all=1e4 * m.mean()))

    out = REPO / "tables"
    out.mkdir(exist_ok=True)
    with (out / "cutoff_check.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "value", *JOINT_NAMES, "all"])
        w.writeheader(); w.writerows(rows)
        w.writerow(dict(kind="minus3dB_Hz", value=f3, all=f3))
        w.writerow(dict(kind="minus20dB_Hz", value=f20, all=f20))
    print(f"\nwrote {out / 'cutoff_check.csv'}")

    # ---- Table 1 of the paper: loss of the whole chain against the raw 100 Hz cycle
    # filter at 100 Hz -> crop cycle -> keep every 10th sample -> cubic spline back to 100 Hz
    from scipy.interpolate import CubicSpline
    chain_cutoffs = [2.0, 3.0, 4.0, 5.0, None]
    acc = {c: np.zeros(6) for c in chain_cutoffs}
    n_samples = 0
    for _folder, _f, c3 in iter_trials():
        q, s, e, *_ = trial_angles(c3)
        raw = q[s:e]; L = len(raw); n_samples += L
        for cut in chain_cutoffs:
            d = lowpass(q, cut, FS, args.order)[s:e][::10]
            up = CubicSpline(np.arange(len(d)) * 10, d, axis=0, extrapolate=True)(np.arange(L))
            acc[cut] += ((up - raw) ** 2).sum(axis=0)
    print("\nPreprocessing-chain loss vs raw 100 Hz cycles (filter, decimate to 10 Hz, cubic spline back):")
    print(f"{'cutoff':>7} {'MSE[1e-4 rad^2]':>15} {'RMS[deg]':>9} {'hips':>6} {'knees':>6} {'ankles':>6}")
    chain_rows = []
    for cut in chain_cutoffs:
        e6 = acc[cut]; mse = e6.sum() / (n_samples * 6)
        grp = [np.degrees(np.sqrt(e6[[i, i + 3]].sum() / (2 * n_samples))) for i in range(3)]
        chain_rows.append(dict(cutoff="none" if cut is None else cut, mse_1e4=1e4 * mse,
                               rms_deg=np.degrees(np.sqrt(mse)), hips_deg=grp[0], knees_deg=grp[1], ankles_deg=grp[2]))
        print(f"{str(chain_rows[-1]['cutoff']):>7} {1e4*mse:15.2f} {np.degrees(np.sqrt(mse)):9.2f} "
              + " ".join(f"{g:6.2f}" for g in grp))
    with (out / "preprocessing_loss.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chain_rows[0].keys()))
        w.writeheader(); w.writerows(chain_rows)
    print(f"wrote {out / 'preprocessing_loss.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
