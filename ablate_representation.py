"""Representation ablation (Reviewer 2, points 1 and 3).

Cells
  cutoff  in {3, 4, 5, 6, 8, none} Hz        window 3.2 s at 10 Hz (N = 32)
  rate    in {5, 10, 20} Hz (N = 16, 32, 64) cutoff 4 Hz, window 3.2 s
  K       in {2, 4, 6, 8, 12, 16} bins kept  cutoff 4 Hz, N = 32

For every cell the six-joint network is retrained with the paper's recipe on
the subject-grouped split (seed 42) and scored on the five test subjects
against that cell's own encoding of the test cycles (MSE, DTW, S, all six
channels, same protocol as ``train_gaitgen_subject.py``). K cells also report
a "ceiling" row: the ground-truth code truncated to K bins and decoded, which
is the best any K-bin model could do. Independent parameter count and forward
time per query are reported for every cell.

    python ablate_representation.py            # ~15 min on CPU
"""
from __future__ import annotations

import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample

from analyse_gait_generators import _dtw, _fft_fidelity, _period_sample
from fft_extract_cycles import FS, iter_trials, lowpass, trial_angles
from gait_generator_net import SimpleFCNN
from train_gaitgen_subject import HP, SEED, load, split_subjects

REPO = Path(__file__).resolve().parent
OUT = REPO / "kfold_results" / "ablation"
CACHE = OUT / "raw_trials.pkl"
WINDOW_S = 3.2


# ----------------------------------------------------------------------------
# raw trials cached once
# ----------------------------------------------------------------------------
def raw_trials() -> list[dict]:
    if CACHE.exists():
        return pickle.load(CACHE.open("rb"))
    rows = []
    for folder, fname, c in iter_trials():
        q, s, e, speed, r_leg, l_leg = trial_angles(c)
        rows.append(dict(q=q, s=s, e=e, x=np.array([speed / 2.4, r_leg, l_leg], np.float32), subj=folder))
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, CACHE.open("wb"))
    return rows


def encode(rows, cutoff, rate, keep_idx=None):
    """Return X, C (n, bins, 6, 2), subj for one cell. Cycles longer than the window are dropped."""
    decim, n = FS // rate, int(round(WINDOW_S * rate))
    X, C, subj = [], [], []
    for r in rows:
        q = lowpass(r["q"], cutoff, FS, 4)[r["s"]:r["e"]][::decim]
        if q.shape[0] > n:
            continue
        w = q
        while w.shape[0] < n:
            w = np.concatenate((w, w), axis=0)
        F = np.fft.rfft(w[:n], n=n, axis=0)
        C.append(np.stack((F.real, F.imag), axis=2)); X.append(r["x"]); subj.append(r["subj"])
    X, C, subj = np.asarray(X, np.float32), np.asarray(C, np.float32), np.asarray(subj)
    if keep_idx is not None:
        mask = np.zeros(C.shape[1], bool); mask[keep_idx] = True
        C[:, ~mask] = 0.0
    return X, C, subj


def decode(C, n):
    """(bins,6,2) -> (3200, 6) at 1 kHz, as the controller decodes."""
    w = np.fft.irfft(C[..., 0] + 1j * C[..., 1], n=n, axis=0)
    return resample(w, 3200, axis=0)


# ----------------------------------------------------------------------------
# train one cell
# ----------------------------------------------------------------------------
def train_cell(X, C, tr, va, n, device):
    torch.manual_seed(SEED); np.random.seed(SEED)
    Y = C.reshape(len(C), -1)
    active = Y[tr].std(0) > 1e-8                     # drop structurally-zero / truncated outputs
    mu, sd = Y[tr].mean(0)[active], Y[tr].std(0)[active]
    Yn = (Y[:, active] - mu) / sd
    Xt, Yt = torch.tensor(X[tr]).to(device), torch.tensor(Yn[tr]).to(device)
    Xv = torch.tensor(X[va]).to(device)
    net = SimpleFCNN(3, int(active.sum()), HP["hidden"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50)
    mu_t, sd_t = torch.tensor(mu).to(device), torch.tensor(sd).to(device)

    def full(Pn):
        P = np.zeros((len(Pn), Y.shape[1]), np.float32); P[:, active] = Pn; return P.reshape(-1, *C.shape[1:])

    def v_fair():
        net.eval()
        with torch.no_grad():
            P = full((net(Xv) * sd_t + mu_t).cpu().numpy())
        net.train()
        return float(np.mean([((np.fft.irfft(p[..., 0] + 1j * p[..., 1], n=n, axis=0) -
                                np.fft.irfft(c[..., 0] + 1j * c[..., 1], n=n, axis=0)) ** 2).mean() for p, c in zip(P, C[va])]))

    best, best_state, wait, m = np.inf, None, 0, len(tr)
    for ep in range(1, HP["max_epochs"] + 1):
        perm = torch.randperm(m)
        for i in range(0, m, HP["batch"]):
            b = perm[i:i + HP["batch"]]
            loss = nn.functional.mse_loss(net(Xt[b]), Yt[b]); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        v = v_fair(); sched.step(v)
        if v < best - 1e-9:
            best, wait, best_state = v, 0, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
        if wait >= HP["patience"]:
            break
    net.load_state_dict(best_state); net.eval().cpu()
    n_params = sum(p.numel() for p in net.parameters())

    def predict(x):
        with torch.no_grad():
            return full((net(torch.tensor(x[None])).numpy() * sd + mu))[0]
    return predict, n_params, ep


def score(pred_fn, X, C, te, n):
    rows = []
    for i in te:
        gt = _period_sample(decode(C[i], n), 32)
        s = _period_sample(pred_fn(X[i], C[i]), 32)
        rows.append((float(np.mean((s - gt) ** 2)), _dtw(s, gt), _fft_fidelity(s, gt, 3)))
    return np.mean(rows, axis=0)


def main() -> int:
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = raw_trials()
    _, _, subj_ref = load()
    cells = ([("cutoff", c, dict(cutoff=c, rate=10)) for c in (3.0, 4.0, 5.0, 6.0, 8.0, None)]
             + [("rate", r, dict(cutoff=4.0, rate=r)) for r in (5, 20)]
             + [("K", k, dict(cutoff=4.0, rate=10, K=k)) for k in (2, 4, 6, 8, 12, 16)])
    if only:
        cells = [c for c in cells if c[0] == only]
    results = []
    OUT.mkdir(parents=True, exist_ok=True)
    csv_name = f"ablation_representation_{only}.csv" if only else "ablation_representation.csv"
    for kind, val, cfg in cells:
        t0 = time.time()
        rate, n = cfg["rate"], int(round(WINDOW_S * cfg["rate"]))
        keep = list(range(cfg["K"] + 1)) if "K" in cfg else None
        X, C, subj = encode(rows, cfg["cutoff"], rate, keep)
        tr, va, te = split_subjects(subj, SEED)
        bins = C.shape[1]
        n_indep = 6 * (2 * (len(keep) if keep else bins) - 2) if True else 0   # real DC and Nyquist, 2 per other bin
        if keep is not None and keep[-1] < bins - 1:
            n_indep = 6 * (2 * len(keep) - 1)                                   # Nyquist bin dropped: DC real + 2 per bin
        # ceiling for K cells: truncated GT vs full GT
        ceiling = None
        if keep is not None:
            _, Cfull, _ = encode(rows, cfg["cutoff"], rate, None)
            ceiling = score(lambda x, c, i=[0]: None, X, Cfull, te, n) if False else None
            vals = []
            for i in te:
                gt = _period_sample(decode(Cfull[i], n), 32); s = _period_sample(decode(C[i], n), 32)
                vals.append((float(np.mean((s - gt) ** 2)), _dtw(s, gt), _fft_fidelity(s, gt, 3)))
            ceiling = np.mean(vals, axis=0)
        predict, n_params, epochs = train_cell(X, C, tr, va, n, device)
        t1 = time.perf_counter()
        for _ in range(200):
            predict(X[te[0]])
        fwd_ms = (time.perf_counter() - t1) / 200 * 1e3
        m = score(lambda x, c: decode(predict(x), n), X, C, te, n)
        m_full = score(lambda x, c: decode(predict(x), n), X, Cfull, te, n) if keep is not None else m
        r = dict(kind=kind, value="none" if val is None else val, cutoff_hz=cfg["cutoff"], rate_hz=rate, N=n, bins=bins,
                 K=cfg.get("K", bins - 1), n_cycles=len(C), n_test=len(te), independent_params=n_indep, net_params=n_params,
                 mse=m[0], dtw=m[1], S=m[2], mse_vs_full=m_full[0], dtw_vs_full=m_full[1], S_vs_full=m_full[2], ceiling_mse=None if ceiling is None else ceiling[0],
                 ceiling_dtw=None if ceiling is None else ceiling[1], ceiling_S=None if ceiling is None else ceiling[2],
                 forward_ms=fwd_ms, epochs=epochs, train_s=time.time() - t0)
        results.append(r)
        print(f"{kind:6s} {str(r['value']):5s} N={n:3d} K={r['K']:2d} indep={n_indep:3d} cycles={len(C)} "
              f"MSE {m[0]:.4f} DTW {m[1]:.3f} S {m[2]:.3f}"
              + (f"  vs full GT: MSE {m_full[0]:.4f} DTW {m_full[1]:.3f} S {m_full[2]:.3f}  ceiling MSE {ceiling[0]:.4f} DTW {ceiling[1]:.3f} S {ceiling[2]:.3f}" if ceiling is not None else "")
              + f"  fwd {fwd_ms:.3f} ms  ({time.time()-t0:.0f}s)", flush=True)
        with (OUT / csv_name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    (OUT / "ablation_representation.json").write_text(json.dumps(results, indent=1, default=float))
    print("wrote", OUT / "ablation_representation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
