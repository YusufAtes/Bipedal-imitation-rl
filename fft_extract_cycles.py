"""Extract one gait cycle per C3D trial and encode it as a 32-point DFT.

Script version of the cell in ``fft_datacreate.ipynb`` that produced the
six-joint training data of the paper, with three additions:

* the filter cutoff, order and window length are flags,
* the subject folder of every cycle is saved in ``subject_id.npy``,
* the cycle duration is saved in seconds in ``period_seconds.npy``.

Pipeline per trial (unchanged from the notebook):
  markers (100 Hz) -> six sagittal joint angles -> zero-phase Butterworth
  low-pass over the whole trial -> crop one cycle between two right foot
  strikes -> keep every tenth sample (10 Hz) -> repeat the cycle to fill a
  window of ``n_window`` samples -> rfft of the window.

Run from the repo root:
    python fft_extract_cycles.py --out-dir "gait reference 6joint"
    python fft_extract_cycles.py --cutoff none --out-dir "gait reference 6joint_nofilter"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

REPO = Path(__file__).resolve().parent
DATASET = REPO / "dataset"
FS = 100            # marker rate, Hz
DECIM = 10          # 100 -> 10 Hz
JOINT_NAMES = ["RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle"]


def lowpass(data: np.ndarray, cutoff: float | None, fs: int = FS, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0. ``cutoff=None`` returns the input."""
    if cutoff is None:
        return data
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, data, axis=0)


def _idx(labels: list[str], name: str) -> int:
    if name not in labels:
        raise RuntimeError(f"{name} not found")
    return labels.index(name)


def trial_angles(c) -> tuple[np.ndarray, int, int, float, float, float]:
    """Return (angles (n,6) at 100 Hz, cycle start, cycle end, speed, r_leg, l_leg).

    Angle formulas are copied verbatim from the notebook so the output matches
    the data the paper's model was trained on.
    """
    times = c["parameters"]["EVENT"]["TIMES"]["value"]
    contexts = c["parameters"]["EVENT"]["CONTEXTS"]["value"]
    ev_labels = c["parameters"]["EVENT"]["LABELS"]["value"]
    gait_start = gait_end = None
    for i, lab in enumerate(ev_labels):
        if lab == "Foot Strike1" and contexts[i] == "Right":
            gait_start = int(times[1, i] * 100)
        if lab == "Foot Strike2" and contexts[i] == "Right":
            gait_end = int(times[1, i] * 100 + 1)
    if gait_start is None or gait_end is None:
        raise RuntimeError("right foot-strike events missing")

    pts = c["data"]["points"]
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    names = ["R_FTC", "L_FTC", "R_FLE", "L_FLE", "R_FAL", "L_FAL",
             "SXS", "SJN", "R_FCC", "R_FM1", "L_FCC", "L_FM1"]
    jd = pts[:, [_idx(labels, n) for n in names], :]
    sxs = _idx(labels, "SXS")

    sub_labels = c["parameters"]["SUBJECT"]["LABELS"]["value"]
    sub_values = c["parameters"]["SUBJECT"]["VALUES"]["value"]
    r_leg = float(sub_values[_idx(sub_labels, "R_legLength")])
    l_leg = float(sub_values[_idx(sub_labels, "L_legLength")])

    n = jd.shape[2]
    forward = pts[0, sxs, -1] > pts[0, sxs, 0]
    speed = (abs(pts[0, sxs, -1] - pts[0, sxs, 0]) / n) / 10.0
    sgn = 1.0 if forward else -1.0

    x, z = jd[0], jd[2]
    q = np.zeros((n, 6))
    q[:, 0] = np.arctan2(sgn * (x[2] - x[0]), (-z[2] + z[0]))
    q[:, 1] = np.arctan2(sgn * (x[4] - x[2]), (-z[4] + z[2])) - q[:, 0]
    q[:, 2] = np.arctan2((z[9] - z[8]), sgn * (x[9] - x[8])) - q[:, 0] - q[:, 1]
    q[:, 3] = np.arctan2(sgn * (x[3] - x[1]), (-z[3] + z[1]))
    q[:, 4] = np.arctan2(sgn * (x[5] - x[3]), (-z[5] + z[3])) - q[:, 3]
    q[:, 5] = np.arctan2((z[11] - z[10]), sgn * (x[11] - x[10])) - q[:, 3] - q[:, 4]
    return q, gait_start, gait_end, float(speed), r_leg, l_leg


def encode_cycle(
    q100: np.ndarray, start: int, end: int,
    cutoff: float | None = 4.0, order: int = 4, n_window: int = 32,
) -> tuple[np.ndarray, int, np.ndarray] | None:
    """Filter, crop, decimate, tile and transform. Returns (rfft (bins,6,2), L, window)."""
    q = lowpass(q100, cutoff, FS, order)[start:end][::DECIM]
    L = q.shape[0]
    if L > n_window:
        return None                      # cycle longer than the window: dropped, as in the notebook
    w = q
    while w.shape[0] < n_window:
        w = np.concatenate((w, w), axis=0)
    w = w[:n_window]
    C = np.fft.rfft(w, n=n_window, axis=0)
    return np.stack((C.real, C.imag), axis=2), L, w


def iter_trials():
    """Yield (subject folder, file name, ezc3d handle) in the notebook's folder order."""
    from ezc3d import c3d
    # ezc3d cannot open paths with non-ASCII characters on Windows, so read
    # through a relative path from the repo root (the notebook did the same).
    os.chdir(REPO)
    for folder in os.listdir("dataset"):
        fdir = Path("dataset") / folder
        if not fdir.is_dir():
            continue
        for f in os.listdir(fdir):
            if f.endswith(".c3d") and "ST" not in f:
                yield folder, f, c3d(str(fdir / f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="4", help="Hz, or 'none' for no filter")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--n-window", type=int, default=32)
    ap.add_argument("--out-dir", default="gait reference 6joint")
    args = ap.parse_args()
    cutoff = None if args.cutoff.lower() == "none" else float(args.cutoff)

    X, C, L, subj, files = [], [], [], [], []
    dropped = 0
    for folder, fname, c in iter_trials():
        q, s, e, speed, r_leg, l_leg = trial_angles(c)
        enc = encode_cycle(q, s, e, cutoff, args.order, args.n_window)
        if enc is None:
            dropped += 1
            continue
        X.append([speed / 2.4, r_leg, l_leg])
        C.append(enc[0]); L.append(enc[1]); subj.append(folder); files.append(f"{folder}/{fname}")

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    X, C = np.asarray(X), np.asarray(C)
    Ls = np.asarray(L)
    np.save(out / "input_vector.npy", X)
    np.save(out / "output_fft_constants.npy", C)
    np.save(out / "period.npy", (Ls / args.n_window)[:, None])
    np.save(out / "period_seconds.npy", (Ls / DECIM)[:, None])
    np.save(out / "subject_id.npy", np.asarray(subj))
    (out / "meta.json").write_text(json.dumps({
        "cutoff_hz": cutoff, "order": args.order, "n_window": args.n_window,
        "fs": FS, "decimation": DECIM, "n_cycles": int(len(L)), "dropped": dropped,
        "joints": JOINT_NAMES, "files": files}, indent=1))
    print(f"[extract] {len(L)} cycles from {len(set(subj))} subjects, {dropped} dropped "
          f"(cycle longer than {args.n_window / DECIM:.1f} s) -> {out}")
    print(f"[extract] input {X.shape}, coefficients {C.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
