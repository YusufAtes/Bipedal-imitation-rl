"""Two parameter-matched baselines for the representation comparison
(Reviewer 2, point 4), trained on the same cycle split as Table 2.

* ``pca_mlp``: PCA fitted on the training windows in the time domain
  (32 samples x 4 joints = 128 values); the FFT-MLP trunk (3 x 512,
  LeakyReLU 0.1) predicts the 128 PCA scores from [v, l_r, l_l]; decode is
  the inverse PCA. Data-adaptive linear basis (Tilmanne & Dutoit 2010;
  Warmenhoven et al. 2021).
* ``gru_seq``: residual GRU decoder (Martinez et al. 2017). Hidden state 64,
  initialised by a linear map of [v, l_r, l_l]; 32 autoregressive steps, each
  emitting x_{t+1} = x_t + W h_t for the 4 joints; the first pose is also
  predicted from the condition. Trained with teacher forcing, evaluated
  free-running, so the first-to-last sample jump (periodicity error) is
  reported.

Data and split: ``gait reference phase 2`` (1121 cycles, 4 joints) and
``split_indices.npz`` (897 / 135 / 89), the files behind Table 2. Both
models use the paper's optimiser settings and early stopping on the
validation time-domain MSE (V_fair).

    python train_latent_baselines.py
Then:
    python analyse_gait_generators.py --out figs_demo/gait_gen_latent --extra pca_mlp gru_seq
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from gait_generator_net import SimpleFCNN

REPO = Path(__file__).resolve().parent
DATA = REPO / "gait reference phase 2"
OUT = REPO / "kfold_results" / "latent_baselines"
SEED = 42
HP = dict(hidden=512, lr=3e-4, wd=1e-5, batch=64, patience=250, max_epochs=4000)


def load():
    X = np.load(DATA / "input_vector.npy").astype(np.float32)
    C = np.load(DATA / "output_fft_constants.npy").astype(np.float32)        # (N,17,4,2)
    W = np.fft.irfft(C[..., 0] + 1j * C[..., 1], n=32, axis=1).astype(np.float32)  # (N,32,4)
    sp = np.load(DATA / "split_indices.npz")
    return X, W, sp["train"], sp["val"], sp["test"]


class GRUSeq(nn.Module):
    def __init__(self, n_joint=4, hidden=64, cond=3, steps=32):
        super().__init__()
        self.steps, self.h0 = steps, nn.Linear(cond, hidden)
        self.x0 = nn.Linear(cond, n_joint)
        self.cell = nn.GRUCell(n_joint + cond, hidden)
        self.out = nn.Linear(hidden, n_joint)

    def forward(self, c, teacher=None):
        h = torch.tanh(self.h0(c)); x = self.x0(c); xs = [x]
        for t in range(1, self.steps):
            inp = teacher[:, t - 1] if teacher is not None else x
            h = self.cell(torch.cat([inp, c], -1), h)
            x = inp + self.out(h)                       # residual step (Martinez et al.)
            xs.append(x)
        return torch.stack(xs, 1)                       # (B, steps, n_joint)


def fit(model, Xtr, Ytr, Xva, Yva, decode, forward):
    """Generic trainer: MSE on ``Ytr`` targets, early stop on time-domain val MSE."""
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50)
    best, best_state, wait, n = np.inf, None, 0, len(Xtr)
    for ep in range(1, HP["max_epochs"] + 1):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, HP["batch"]):
            b = perm[i:i + HP["batch"]]
            loss = nn.functional.mse_loss(forward(Xtr[b], Ytr[b]), Ytr[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(((decode(model, Xva) - Yva) ** 2).mean())
        sched.step(v)
        if v < best - 1e-9:
            best, wait, best_state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
        if wait >= HP["patience"]:
            break
    model.load_state_dict(best_state); model.eval()
    return best, ep


def main() -> int:
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    X, W, tr, va, te = load()
    Xt, Xv = torch.tensor(X[tr]), torch.tensor(X[va])
    Wv = torch.tensor(W[va])
    report = {}

    # ---- PCA + MLP -------------------------------------------------------
    t0 = time.time()
    pca = PCA(n_components=128, random_state=SEED).fit(W[tr].reshape(len(tr), -1))
    S = pca.transform(W.reshape(len(W), -1)).astype(np.float32)
    mu, sd = S[tr].mean(0), S[tr].std(0) + 1e-8
    St = torch.tensor((S[tr] - mu) / sd)
    comp, mean = torch.tensor(pca.components_, dtype=torch.float32), torch.tensor(pca.mean_, dtype=torch.float32)
    mu_t, sd_t = torch.tensor(mu), torch.tensor(sd)
    mlp = SimpleFCNN(3, 128, HP["hidden"])
    dec_pca = lambda m, x: ((m(x) * sd_t + mu_t) @ comp + mean).reshape(-1, 32, 4)
    best, ep = fit(mlp, Xt, St, Xv, Wv, dec_pca, lambda x, y: mlp(x))
    torch.save(mlp.state_dict(), OUT / "pca_mlp.pth")
    np.savez(OUT / "pca_mlp_basis.npz", components=pca.components_, mean=pca.mean_, mu=mu, sd=sd,
             explained=pca.explained_variance_ratio_)
    n_par = sum(p.numel() for p in mlp.parameters())
    report["pca_mlp"] = dict(val_vfair=best, epochs=ep, params=n_par, basis_params=int(pca.components_.size),
                             var_explained_first8=float(pca.explained_variance_ratio_[:8].sum()), train_s=time.time() - t0)
    print(f"pca_mlp : val V_fair {best:.5f} at {ep} epochs, {n_par} trainable params + {pca.components_.size} basis values "
          f"({time.time()-t0:.0f}s)")

    # ---- residual GRU ------------------------------------------------------
    t0 = time.time()
    gru = GRUSeq()
    Wt = torch.tensor(W[tr])
    best, ep = fit(gru, Xt, Wt, Xv, Wv, lambda m, x: m(x), lambda x, y: gru(x, teacher=y))
    torch.save(gru.state_dict(), OUT / "gru_seq.pth")
    n_par = sum(p.numel() for p in gru.parameters())
    with torch.no_grad():
        P = gru(torch.tensor(X[te])).numpy()
    jump = float(np.abs(P[:, -1] - P[:, 0]).mean())
    gt_jump = float(np.abs(W[te][:, -1] - W[te][:, 0]).mean())
    report["gru_seq"] = dict(val_vfair=best, epochs=ep, params=n_par, first_last_jump=jump, gt_first_last_jump=gt_jump,
                             train_s=time.time() - t0)
    print(f"gru_seq : val V_fair {best:.5f} at {ep} epochs, {n_par} params, free-running first-to-last jump "
          f"{jump:.3f} rad (ground-truth window {gt_jump:.3f}) ({time.time()-t0:.0f}s)")
    fft_params = sum(p.numel() for p in SimpleFCNN(3, 136, 512).parameters())
    report["fft_mlp_params_4joint"] = fft_params
    print(f"reference: the 4-joint FFT-MLP trunk has {fft_params} params")
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
