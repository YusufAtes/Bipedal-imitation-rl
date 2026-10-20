"""Retrain the six-joint gait generator on a subject-grouped split and
re-evaluate it against the baselines on the held-out subjects (R-13).

Data: ``gait reference 6joint/`` written by ``fft_extract_cycles.py``
(1121 cycles, 50 subjects, 17 x 6 x 2 coefficients per cycle).

Split (fixed seed): 5 test subjects, 5 validation subjects, 40 training
subjects. Hyperparameters are the paper's selected configuration: direct
prediction, L_freq on z-scored coefficients, 3 x 512 LeakyReLU(0.1), Adam
3e-4, weight decay 1e-5, batch 64, grad clip 1, ReduceLROnPlateau on V_fair,
patience 250, at most 4000 epochs. Early stopping and model selection use
V_fair = time-domain MSE on the validation subjects.

Evaluation protocol copies ``analyse_gait_generators.py`` (same
``_period_sample``, ``_dtw``, ``_fft_fidelity``), with every data-driven
baseline restricted to the training subjects. All six channels are scored.

    python train_gaitgen_subject.py            # train + evaluate
    python train_gaitgen_subject.py --eval-only
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
from scipy.signal import resample
from sklearn.model_selection import GroupShuffleSplit

from analyse_gait_generators import DEFAULT_SPEED_BINS, _dtw, _fft_fidelity, _period_sample
from gait_generator_net import SimpleFCNN
from gait_generators import build_generator

REPO = Path(__file__).resolve().parent
DATA = REPO / "gait reference 6joint"
OUT = REPO / "kfold_results" / "subject_split"
SEED = 42
N_TEST_SUBJ, N_VAL_SUBJ = 5, 5
HP = dict(hidden=512, lr=3e-4, wd=1e-5, batch=64, patience=250, max_epochs=4000)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load():
    X = np.load(DATA / "input_vector.npy").astype(np.float32)          # speed/2.4, r_leg, l_leg
    C = np.load(DATA / "output_fft_constants.npy").astype(np.float32)  # (N,17,6,2)
    subj = np.load(DATA / "subject_id.npy")
    return X, C, subj


def split_subjects(subj: np.ndarray, seed: int = SEED):
    """Return (train_idx, val_idx, test_idx), grouped by subject, fixed seed."""
    idx = np.arange(len(subj))
    gss = GroupShuffleSplit(n_splits=1, test_size=N_TEST_SUBJ, random_state=seed)
    trainval, test = next(gss.split(idx, groups=subj))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=N_VAL_SUBJ, random_state=seed)
    tr, va = next(gss2.split(trainval, groups=subj[trainval]))
    return trainval[tr], trainval[va], test


def window(C: np.ndarray) -> np.ndarray:
    """(17,6,2) coefficients -> (32,6) window."""
    return np.fft.irfft(C[..., 0] + 1j * C[..., 1], n=32, axis=0)


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def train(X, C, tr, va, device, out: Path = OUT, verbose: bool = True):
    torch.manual_seed(SEED); np.random.seed(SEED)
    Y = C.reshape(len(C), -1)                                   # (N, 204)
    mu, sd = Y[tr].mean(0), Y[tr].std(0)
    sd[sd < 1e-8] = 1.0                                         # the 12 structurally zero outputs
    Yn = (Y - mu) / sd
    Xt, Yt = torch.tensor(X[tr]).to(device), torch.tensor(Yn[tr]).to(device)
    Xv, Cv = torch.tensor(X[va]).to(device), C[va]
    net = SimpleFCNN(input_size=3, output_size=Y.shape[1], hidden_size=HP["hidden"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50)
    mu_t, sd_t = torch.tensor(mu).to(device), torch.tensor(sd).to(device)

    def v_fair() -> float:
        net.eval()
        with torch.no_grad():
            P = (net(Xv) * sd_t + mu_t).cpu().numpy().reshape(-1, 17, 6, 2)
        net.train()
        return float(np.mean([((window(p) - window(c)) ** 2).mean() for p, c in zip(P, Cv)]))

    best, best_ep, best_state, wait = np.inf, 0, None, 0
    n = len(tr); t0 = time.time()
    for ep in range(1, HP["max_epochs"] + 1):
        perm = torch.randperm(n)
        for i in range(0, n, HP["batch"]):
            b = perm[i:i + HP["batch"]]
            loss = nn.functional.mse_loss(net(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        v = v_fair(); sched.step(v)
        if v < best - 1e-9:
            best, best_ep, wait = v, ep, 0
            best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
        if verbose and (ep % 100 == 0 or ep == 1):
            print(f"  epoch {ep:4d}  train {loss.item():.5f}  V_fair {v:.5f}  best {best:.5f}@{best_ep}  "
                  f"lr {opt.param_groups[0]['lr']:.1e}  {time.time()-t0:.0f}s")
        if wait >= HP["patience"]:
            if verbose: print(f"  early stop at epoch {ep}")
            break
    net.load_state_dict(best_state)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model.pth")
    np.savez(out / "norm.npz", mu=mu, sd=sd)
    return net.cpu(), mu, sd, best, best_ep


# ----------------------------------------------------------------------------
# generators restricted to training subjects
# ----------------------------------------------------------------------------
def to_control_rate(w32: np.ndarray, dt: float = 1e-3) -> np.ndarray:
    """Same decode as FFTMLP2DGenerator._ifft: 32 samples at 10 Hz -> 3.2 s at 1/dt."""
    return resample(w32, int(32 * (1 / dt) / 10), axis=0)


class NewGaitGen:
    def __init__(self, net, mu, sd):
        self.net, self.mu, self.sd = net.eval(), mu, sd

    def predict(self, speed, legs):
        x = torch.tensor([speed / 2.4, legs[0], legs[1]], dtype=torch.float32)
        with torch.no_grad():
            C = (self.net(x).numpy() * self.sd + self.mu).reshape(17, 6, 2)
        return to_control_rate(window(C))


class Playback:
    """Nearest-speed cycle from the training subjects (6 channels)."""
    def __init__(self, X, C, tr):
        self.speeds, self.W = X[tr, 0] * 2.4, np.stack([window(c) for c in C[tr]])

    def predict(self, speed, legs):
        return to_control_rate(self.W[np.argmin(np.abs(self.speeds - speed))])


class Spline(Playback):
    """Mean of the 8 nearest-speed training cycles, periodic cubic spline, as cubic_spline.py."""
    def predict(self, speed, legs, k=8):
        nb = np.argsort(np.abs(self.speeds - speed))[:k]
        m = self.W[nb].mean(0)
        phase = np.linspace(0, 1, 33)
        cs = CubicSpline(phase, np.vstack([m, m[:1]]), bc_type="periodic", axis=0)
        return cs(np.linspace(0, 1, 3200, endpoint=False))


def evaluate(gens: dict, X, C, te) -> list[dict]:
    rows = []
    for i in te:
        speed, legs = float(X[i, 0]) * 2.4, (float(X[i, 1]), float(X[i, 2]))
        gt = _period_sample(window(C[i]), 32)
        for name, g in gens.items():
            s = _period_sample(np.asarray(g.predict(speed, legs))[:, :6], 32)
            rows.append(dict(generator=name, test_idx=int(i), speed=speed,
                             mse=float(np.mean((s - gt) ** 2)), dtw=_dtw(s, gt),
                             fft=_fft_fidelity(s, gt, 3)))
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for name in dict.fromkeys(r["generator"] for r in rows):
        rr = [r for r in rows if r["generator"] == name]
        out.append(dict(generator=name, n=len(rr), **{k: float(np.mean([r[k] for r in rr])) for k in ("mse", "dtw", "fft")}))
    return out


def multi_seed(seeds: list[int]) -> int:
    """Repeat the subject split with several seeds; report mean and sd per generator."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, C, subj = load()
    allrows = []
    for seed in seeds:
        tr, va, te = split_subjects(subj, seed)
        out = OUT / f"seed{seed}"
        net, mu, sd, best, best_ep = train(X, C, tr, va, device, out, verbose=False)
        gens = {"GaitGen (subject split, retrained)": NewGaitGen(net, mu, sd),
                "cubic_spline": Spline(X, C, tr), "raw_mocap": Playback(X, C, tr),
                "cpg_matsuoka": build_generator("cpg_matsuoka", dt=1e-3, tile_repeats=1)}
        summ = summarise(evaluate(gens, X, C, te))
        for r in summ:
            r.update(seed=seed, test_subjects=" ".join(sorted(set(subj[te].tolist()))), best_epoch=best_ep)
        allrows += summ
        print(f"seed {seed}: test {sorted(set(subj[te].tolist()))}  " +
              "  ".join(f"{r['generator'].split(' ')[0]} {r['mse']:.4f}" for r in summ), flush=True)
    with (OUT / "multi_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print(f"\n{len(seeds)} subject splits, mean +- sd over splits:")
    print(f"{'generator':38s} {'MSE':>16} {'DTW':>16} {'S':>16}")
    for name in dict.fromkeys(r["generator"] for r in allrows):
        rr = [r for r in allrows if r["generator"] == name]
        print(f"{name:38s} " + " ".join(f"{np.mean([r[k] for r in rr]):8.4f}+-{np.std([r[k] for r in rr]):6.4f}" for k in ("mse", "dtw", "fft")))
    wins = sum(1 for seed in seeds if min((r for r in allrows if r["seed"] == seed), key=lambda r: r["mse"])["generator"].startswith("GaitGen"))
    print(f"GaitGen has the lowest MSE on {wins} of {len(seeds)} splits")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="multi-split mode")
    args = ap.parse_args()
    if args.seeds:
        return multi_seed(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, C, subj = load()
    tr, va, te = split_subjects(subj)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_subjects.json").write_text(json.dumps(dict(
        seed=SEED, test=sorted(set(subj[te].tolist())), val=sorted(set(subj[va].tolist())),
        train=sorted(set(subj[tr].tolist())), n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te))), indent=1))
    print(f"split: train {len(tr)} cycles / {len(set(subj[tr]))} subj, val {len(va)} / {len(set(subj[va]))}, "
          f"test {len(te)} / {len(set(subj[te]))}  test subjects {sorted(set(subj[te]))}")

    if args.eval_only:
        net = SimpleFCNN(3, 204, HP["hidden"]); net.load_state_dict(torch.load(OUT / "model.pth"))
        nz = np.load(OUT / "norm.npz"); mu, sd = nz["mu"], nz["sd"]; best = best_ep = None
    else:
        net, mu, sd, best, best_ep = train(X, C, tr, va, device)
        print(f"best val V_fair {best:.5f} at epoch {best_ep}")

    gens = {
        "GaitGen (subject split, retrained)": NewGaitGen(net, mu, sd),
        "GaitGen (paper weights)": build_generator("fft_mlp_2d", dt=1e-3, tile_repeats=1),
        "cubic_spline": Spline(X, C, tr),
        "raw_mocap": Playback(X, C, tr),
        "cpg_matsuoka": build_generator("cpg_matsuoka", dt=1e-3, tile_repeats=1),
    }
    rows = evaluate(gens, X, C, te)
    with (OUT / "per_sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summ = summarise(rows)
    with (OUT / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    # per speed bin for the figure
    with (OUT / "per_speed.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["generator", "speed_bin_centre", "n_samples", "mse", "dtw", "fft_fidelity"])
        for name in gens:
            rr = [r for r in rows if r["generator"] == name]
            sp = np.array([r["speed"] for r in rr])
            for lo, hi in zip(DEFAULT_SPEED_BINS[:-1], DEFAULT_SPEED_BINS[1:]):
                m = (sp >= lo) & (sp < hi)
                if m.any():
                    w.writerow([name, (lo + hi) / 2, int(m.sum())] + [np.mean([r[k] for r, mm in zip(rr, m) if mm]) for k in ("mse", "dtw", "fft")])
    (OUT / "metrics.json").write_text(json.dumps(dict(best_val_vfair=best, best_epoch=best_ep, summary=summ), indent=1))

    print(f"\nHeld-out subjects ({len(te)} cycles, 6 channels), mean over test cycles:")
    print(f"{'generator':38s} {'MSE':>8} {'DTW':>7} {'S':>7}")
    for s in summ:
        print(f"{s['generator']:38s} {s['mse']:8.4f} {s['dtw']:7.3f} {s['fft']:7.3f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
