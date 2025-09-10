
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- SB3 imports
from stable_baselines3 import PPO
try:
    from sb3_contrib import RecurrentPPO
    HAS_RPPO = True
except Exception:
    RecurrentPPO = None
    HAS_RPPO = False

# --- Your environment
from ppoenv_guide import BipedEnv

# -------------------------------
# Plotting utilities (no seaborn)
# -------------------------------

def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.25,
        "lines.linewidth": 1.6,
    })

def edges_from_centers(vals):
    vals = np.asarray(vals)
    if len(vals) < 2:
        step = 1.0
    else:
        step = np.diff(vals)
        step = np.r_[step, step[-1]]
    left = vals[0] - step[0]/2
    right = vals[-1] + step[-1]/2
    return np.r_[left, (vals[:-1] + vals[1:]) / 2, right]

def pcolor_from_grid(ax, Xc, Yc, Z, cmap, vmin=None, vmax=None, centered=False):
    Xe = edges_from_centers(Xc); Ye = edges_from_centers(Yc)
    if centered:
        lim = np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 1.0
        vmin, vmax = -lim, +lim
    h = ax.pcolormesh(Xe, Ye, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    return h

# ---------------
# Env integration
# ---------------

def get_forward_pos(env):
    """Return forward (Y) position if available; fallback to PyBullet link-2 Y."""
    try:
        es = getattr(env, "external_states", None)
        if es is not None and len(es) >= 2 and es[1] is not None:
            return float(es[1])
    except Exception:
        pass
    try:
        import pybullet as p
        for cand in ["robot", "robot_id", "biped_id"]:
            if hasattr(env, cand):
                rid = getattr(env, cand)
                if rid is not None:
                    link_state = p.getLinkState(rid, 2)
                    return float(link_state[0][1])
        if hasattr(env, "robot") and hasattr(env.robot, "id"):
            link_state = p.getLinkState(env.robot.id, 2)
            return float(link_state[0][1])
    except Exception:
        pass
    return float("nan")

def extract_success(info):
    for k in ["success", "succeeded", "no_fall", "not_fallen"]:
        if k in info:
            return bool(info[k])
    for k in ["fell", "is_fallen", "fallen"]:
        if k in info:
            return not bool(info[k])
    return None

# ----------------
# Eval primitives
# ----------------

def run_episode(model, env, horizon_steps, deterministic=True):
    obs, info = env.reset()
    done = False
    total_r = 0.0
    xs, vs = [], []
    success_flag = None
    lstm_state = None
    episode_starts = True

    for t in range(horizon_steps):
        if hasattr(model.policy, "lstm_hidden_state_shape") or hasattr(model.policy, "rnn"):
            action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_starts, deterministic=deterministic)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += float(reward)
        done = terminated or truncated
        episode_starts = done

        y = get_forward_pos(env)
        xs.append(y)
        v = None
        for k in ["vx", "vy", "forward_speed", "speed"]:
            if k in info:
                v = float(info[k]); break
        if v is None:
            if len(xs) >= 2 and np.isfinite(xs[-1]) and np.isfinite(xs[-2]):
                v = xs[-1] - xs[-2]
            else:
                v = np.nan
        vs.append(v)

        suc = extract_success(info)
        if suc is not None:
            success_flag = suc

        if done:
            break

    xs = np.array(xs, dtype=float)
    vs = np.array(vs, dtype=float)
    displacement = np.nan
    if xs.size and np.isfinite(xs[0]) and np.isfinite(xs[-1]):
        displacement = xs[-1] - xs[0]
    mean_speed = np.nanmean(vs) if vs.size else np.nan

    if success_flag is None:
        success_flag = not (info.get("fell", False) or info.get("is_fallen", False))

    return {"R": total_r, "success": bool(success_flag),
            "mean_speed": float(mean_speed) if mean_speed==mean_speed else np.nan,
            "displacement": float(displacement) if displacement==displacement else np.nan,
            "steps": t+1}

def eval_grid(model, env, speeds, angles=None, noises=None, trials=5, horizon_steps=3200):
    assert (angles is None) ^ (noises is None), "Provide angles XOR noises"
    A = angles if angles is not None else noises
    ns, na = len(speeds), len(A)
    success_acc = np.full((na, ns), np.nan, dtype=float)
    mean_speed_acc = np.full((na, ns), np.nan, dtype=float)
    range_acc = np.full((na, ns), np.nan, dtype=float)

    for i, a in enumerate(A):
        for j, s in enumerate(speeds):
            successes, speeds_trial, ranges = [], [], []
            for _ in range(trials):
                # Directly pass grid params to reset; matches your BipedEnv.reset signature.
                obs, info = env.reset(test_speed=float(s),
                                      test_angle=float(a) if angles is not None else None,
                                      ground_noise=float(a) if noises is not None else None)
                metrics = run_episode(model, env, horizon_steps=horizon_steps, deterministic=True)
                successes.append(1.0 if metrics["success"] else 0.0)
                speeds_trial.append(metrics["mean_speed"])
                ranges.append(metrics["displacement"])
            success_acc[i, j] = np.nanmean(successes) if len(successes) else np.nan
            mean_speed_acc[i, j] = np.nanmean(speeds_trial) if len(speeds_trial) else np.nan
            try:
                range_acc[i, j] = np.nanmax(ranges)
            except Exception:
                range_acc[i, j] = np.nan

    return {"success_prob": success_acc, "mean_speed": mean_speed_acc, "max_range": range_acc}

# ------------------
# Publication plots
# ------------------

def plot_vel_tracking(commanded, samples_per_speed, outfile):
    set_pub_style()
    means = np.array([np.nanmean(s) for s in samples_per_speed])
    stds  = np.array([np.nanstd(s, ddof=1) for s in samples_per_speed])
    ns    = np.array([len(s) for s in samples_per_speed])
    ci = 1.96 * stds / np.maximum(1, np.sqrt(ns))

    A = np.vstack([commanded, np.ones_like(commanded)]).T
    if np.isfinite(means).any():
        a, b = np.linalg.lstsq(A, np.nan_to_num(means, nan=0.0), rcond=None)[0]
    else:
        a, b = np.nan, np.nan
    pred = a*commanded + b if np.isfinite(a) else np.full_like(commanded, np.nan)
    residuals = means - commanded
    rmse = np.sqrt(np.nanmean((means - commanded)**2))
    mae  = np.nanmean(np.abs(means - commanded))

    fig = plt.figure(figsize=(3.35, 2.6))
    ax = fig.add_subplot(111)
    ax.plot(commanded, commanded, color="0.6", linestyle="--", label="Unity")
    ax.plot(commanded, means, label="Measured")
    if np.isfinite(ci).any():
        ax.fill_between(commanded, means-ci, means+ci, alpha=0.25, linewidth=0, label="95% CI")
    if np.isfinite(a):
        ax.plot(commanded, pred, color="k", linewidth=1.0, label=f"Fit: y={a:.2f}x+{b:.2f}")
    ax.set_xlabel("Commanded speed $u_d$ (m/s)")
    ax.set_ylabel("Actual speed (m/s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", frameon=False)

    axins = ax.inset_axes([0.58, 0.12, 0.38, 0.38])
    axins.plot(commanded, residuals, marker="o", linewidth=0.8)
    axins.axhline(0, color="0.5", lw=0.8)
    axins.set_title(f"Residuals (RMSE={rmse:.2f}, MAE={mae:.2f})", pad=2)
    axins.set_xlabel("$u_d$", fontsize=7); axins.set_ylabel("err", fontsize=7)
    fig.tight_layout()
    fig.savefig(outfile.replace(".png", ".pdf"), bbox_inches="tight")
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

def plot_ramp_speed_map(speeds, angles, mean_speed_mat, success_prob_mat, outfile):
    set_pub_style()
    err = mean_speed_mat - speeds[np.newaxis, :]
    fig, ax = plt.subplots(figsize=(3.35, 2.8))
    h = pcolor_from_grid(ax, speeds, angles, err, cmap="coolwarm", centered=True)
    cbar = fig.colorbar(h, ax=ax, pad=0.02)
    cbar.set_label("Speed error (m/s)")
    try:
        CS = ax.contour(speeds, angles, success_prob_mat, levels=[0.2, 0.5, 0.8],
                        colors=["0.2","0.2","0.2"], linestyles=["dotted","dashdot","solid"], linewidths=1.0)
        ax.clabel(CS, inline=True, fmt={0.2:"20%",0.5:"50%",0.8:"80%"}, fontsize=7)
    except Exception:
        pass
    ax.set_xlabel("Desired speed $u_d$ (m/s)")
    ax.set_ylabel("Ramp angle $\\alpha$ (deg)")
    ax.set_title("Tracking error with success contours")
    fig.tight_layout()
    fig.savefig(outfile.replace(".png", ".pdf"), bbox_inches="tight")
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

def plot_noise_speed_triptych(speeds, noises, mean_speed_mat, success_prob_mat, max_range_mat, outfile):
    set_pub_style()
    fig = plt.figure(figsize=(6.9, 2.6))
    gs = fig.add_gridspec(1, 3, wspace=0.25)

    axA = fig.add_subplot(gs[0, 0])
    hA = pcolor_from_grid(axA, speeds, noises, success_prob_mat, cmap="Greys", vmin=0, vmax=1)
    fig.colorbar(hA, ax=axA, pad=0.02, ticks=[0,0.5,1]).set_label("Success")
    axA.set_title("(A) Success probability")
    axA.set_xlabel("$u_d$ (m/s)"); axA.set_ylabel("Noise level")

    axB = fig.add_subplot(gs[0, 1], sharex=axA, sharey=axA)
    finite_vals = mean_speed_mat[np.isfinite(mean_speed_mat)]
    if finite_vals.size:
        vmin, vmax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
        levels = np.linspace(vmin, vmax, 6)
        CS = axB.contour(speeds, noises, mean_speed_mat, levels=levels, colors="k", linewidths=0.9)
        axB.clabel(CS, inline=True, fontsize=7, fmt="%.1f")
    axB.set_title("(B) Mean actual speed (m/s)")
    axB.set_xlabel("$u_d$ (m/s)"); axB.tick_params(labelleft=False)

    axC = fig.add_subplot(gs[0, 2], sharex=axA, sharey=axA)
    Xe, Ye = np.meshgrid(speeds, noises)
    norm = np.nanmax(max_range_mat) if np.isfinite(max_range_mat).any() else 1.0
    sizes = 20 + 180 * (max_range_mat / (norm if norm != 0 else 1.0))
    axC.scatter(Xe.ravel(), Ye.ravel(), s=sizes.ravel(), facecolors="none", edgecolors="k", linewidths=0.8)
    axC.set_title("(C) Max distance (m)")
    axC.set_xlabel("$u_d$ (m/s)"); axC.tick_params(labelleft=False)

    fig.suptitle("Robustness on noisy terrain (shared axes)", y=1.05)
    fig.tight_layout()
    fig.savefig(outfile.replace(".png", ".pdf"), bbox_inches="tight")
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

# -----------------
# Command-line app
# -----------------

def main():
    parser = argparse.ArgumentParser(description="Demo: publication-grade plots + MLP vs LSTM comparison")
    parser.add_argument("--mlp", type=str, default=None, help="Path to PPO (MLP) model zip")
    parser.add_argument("--lstm", type=str, default=None, help="Path to RecurrentPPO (LSTM) model zip")
    parser.add_argument("--out", type=str, default="figs_demo", help="Output directory")
    parser.add_argument("--trials", type=int, default=5, help="Trials per grid point")
    parser.add_argument("--headless", action="store_true", help="Run env without GUI")
    parser.add_argument("--horizon", type=float, default=3.2, help="Episode length in seconds")
    parser.add_argument("--dt", type=float, default=None, help="Env dt override (s); default: env.dt or 1e-3")
    parser.add_argument("--speeds", type=str, default="0.1,2.2,12", help="start,end,count for speed grid")
    parser.add_argument("--angles", type=str, default="-15,15,21", help="start,end,count for ramp-angle grid")
    parser.add_argument("--noises", type=str, default="0.00,0.20,10", help="start,end,count for noise grid (height amp)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    def parse_triplet(s):
        a,b,c = [float(x) for x in s.split(",")]
        c = int(c)
        return np.linspace(a,b,c)
    speeds = parse_triplet(args.speeds)
    angles = parse_triplet(args.angles)
    noises = parse_triplet(args.noises)

    env = BipedEnv(render=not args.headless, render_mode=None, demo_mode=True)
    dt = args.dt if args.dt is not None else getattr(env, "dt", 1e-3)
    horizon_steps = int(round(args.horizon / dt))

    models = {}
    if args.mlp is not None and os.path.exists(args.mlp):
        models["MLP"] = PPO.load(args.mlp, device="cpu")
    if args.lstm is not None and os.path.exists(args.lstm):
        if HAS_RPPO:
            models["LSTM"] = RecurrentPPO.load(args.lstm, device="cpu")
        else:
            print("[WARN] sb3-contrib not available; skipping LSTM model.")
    if not models:
        raise FileNotFoundError("No models provided. Use --mlp and/or --lstm with valid paths.")

    # Test 1
    for name, mdl in models.items():
        samples = []
        for s in speeds:
            per = []
            for _ in range(args.trials):
                obs, info = env.reset(test_speed=float(s))
                m = run_episode(mdl, env, horizon_steps=horizon_steps, deterministic=True)
                per.append(m["mean_speed"])
            samples.append(np.array(per, dtype=float))
        plot_vel_tracking(speeds, samples, outfile=os.path.join(args.out, f"{name.lower()}_vel_tracking.png"))

    # Test 2: ramp × speed
    results_ramp = {}
    for name, mdl in models.items():
        res = eval_grid(mdl, env, speeds=speeds, angles=angles, trials=args.trials, horizon_steps=horizon_steps)
        results_ramp[name] = res
        plot_ramp_speed_map(speeds, angles, res["mean_speed"], res["success_prob"],
                            outfile=os.path.join(args.out, f"{name.lower()}_ramp_speed_map.png"))

    if set(models.keys()) >= {"MLP", "LSTM"}:
        diff_succ = results_ramp["LSTM"]["success_prob"] - results_ramp["MLP"]["success_prob"]
        fig, ax = plt.subplots(figsize=(3.35, 2.8)); set_pub_style()
        h = pcolor_from_grid(ax, speeds, angles, diff_succ, cmap="coolwarm", centered=True)
        fig.colorbar(h, ax=ax, pad=0.02).set_label("Δ success (LSTM–MLP)")
        ax.set_xlabel("Desired speed $u_d$ (m/s)"); ax.set_ylabel("Ramp angle (deg)")
        ax.set_title("Ramp: success advantage (LSTM − MLP)")
        fig.tight_layout(); path = os.path.join(args.out, "compare_ramp_success_diff.png")
        fig.savefig(path.replace(".png",".pdf"), bbox_inches="tight"); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

        diff_err = (results_ramp["LSTM"]["mean_speed"] - speeds[np.newaxis,:]) - (results_ramp["MLP"]["mean_speed"] - speeds[np.newaxis,:])
        fig, ax = plt.subplots(figsize=(3.35, 2.8)); set_pub_style()
        h = pcolor_from_grid(ax, speeds, angles, diff_err, cmap="coolwarm", centered=True)
        fig.colorbar(h, ax=ax, pad=0.02).set_label("Δ tracking error (m/s)")
        ax.set_xlabel("Desired speed $u_d$ (m/s)"); ax.set_ylabel("Ramp angle (deg)")
        ax.set_title("Ramp: tracking error advantage (LSTM − MLP)")
        fig.tight_layout(); path = os.path.join(args.out, "compare_ramp_err_diff.png")
        fig.savefig(path.replace(".png",".pdf"), bbox_inches="tight"); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

    # Test 3: noise × speed
    results_noise = {}
    for name, mdl in models.items():
        res = eval_grid(mdl, env, speeds=speeds, noises=noises, trials=args.trials, horizon_steps=horizon_steps)
        results_noise[name] = res
        plot_noise_speed_triptych(speeds, noises, res["mean_speed"], res["success_prob"], res["max_range"],
                                  outfile=os.path.join(args.out, f"{name.lower()}_noise_speed_triptych.png"))

    if set(models.keys()) >= {"MLP", "LSTM"}:
        diff_succ = results_noise["LSTM"]["success_prob"] - results_noise["MLP"]["success_prob"]
        fig, ax = plt.subplots(figsize=(3.35, 2.8)); set_pub_style()
        h = pcolor_from_grid(ax, speeds, noises, diff_succ, cmap="coolwarm", centered=True)
        fig.colorbar(h, ax=ax, pad=0.02).set_label("Δ success (LSTM–MLP)")
        ax.set_xlabel("Desired speed $u_d$ (m/s)"); ax.set_ylabel("Noise level")
        ax.set_title("Noise: success advantage (LSTM − MLP)")
        fig.tight_layout(); path = os.path.join(args.out, "compare_noise_success_diff.png")
        fig.savefig(path.replace(".png",".pdf"), bbox_inches="tight"); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

        diff_range = results_noise["LSTM"]["max_range"] - results_noise["MLP"]["max_range"]
        fig, ax = plt.subplots(figsize=(3.35, 2.8)); set_pub_style()
        h = pcolor_from_grid(ax, speeds, noises, diff_range, cmap="coolwarm", centered=True)
        fig.colorbar(h, ax=ax, pad=0.02).set_label("Δ max distance (m)")
        ax.set_xlabel("Desired speed $u_d$ (m/s)"); ax.set_ylabel("Noise level")
        ax.set_title("Noise: distance advantage (LSTM − MLP)")
        fig.tight_layout(); path = os.path.join(args.out, "compare_noise_range_diff.png")
        fig.savefig(path.replace(".png",".pdf"), bbox_inches="tight"); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

    print(f"[OK] Figures saved to: {args.out}")

if __name__ == "__main__":
    main()