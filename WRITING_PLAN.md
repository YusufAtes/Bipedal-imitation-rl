# Bipedal Paper Resubmission — Writing & Presentation Plan

This plan captures all non-code changes for the resubmission of *Human Imitated Bipedal Locomotion with Frequency Based Gait Generator Network* (arXiv 2511.17387). Code-level changes live in the companion plan at `.cursor/plans/bipedal_paper_resubmission_plan_*.plan.md` and are applied in the `review-suggestions` branch.

**Scope separation rule:** nothing in this file should be executed against the source tree. All items here are manuscript edits, figure regenerations driven by finalized experimental results, and reference additions.

## Reviewer-to-action map

| Item | Reviewer & point | Type |
|---|---|---|
| D1 | Review 2 #5 | Soften claims |
| D2 | Review 2 #1 | Abstract scope |
| D3 | Review 2 #3 | Missing references |
| D4 | Review 2 #2 | Figure legibility |
| D5 | Review 1 #4 | 3D roadmap in Conclusion |
| D6 | Review 2 #4 | Justify three-config choice |

## D1 — Soften sharp claims (Review 2 #5)

Review 2 flagged this sentence explicitly: *"imitation based policy achieves a precise alignment with the natural kinematic structure of the motion capture gait..."*

Action: sweep the whole manuscript for absolute language. Every surviving phrase must point to a specific quantitative result from the new Track B / Track C analyses.

Checklist of phrases to revise:
- "precise alignment with the natural kinematic structure" → "matches the mocap mean within X° RMSE across all six joints (Fig. B4)".
- "demonstrating the necessity of both imitation rewards and reference-state conditioning" → "ablation in Table C3 shows removing either increases vel-MSE by a factor of Y".
- "This approach therefore offers a niche yet straightforward and implementable solution" — drop "niche" (self-deprecating and vague) and replace with a precise statement of scope.
- "general-purpose gait-generation module" → "a drop-in gait generator for 2D planar bipeds with 7 DoF".
- "The PPO baseline performs well at lower noise levels but experiences a sharp performance drop beyond 10 cm for Noisy Plane 0. In contrast, the CMA–ES SIMBICON baseline collapses early" — keep but anchor every claim to the new 95% CI bars.

Do this pass **after** Track B3 and Track C1 are finalized, so that each replacement has a number to cite.

## D2 — Abstract: state simulation scope (Review 2 #1)

Current abstract (line 9 of the paper):
> "Results suggest that pairing spectral motion priors with Deep Reinforcement Learning (DRL) offers a practical path toward natural and robust bipedal locomotion with modest training cost."

Add, ideally before that sentence:
> "All experiments are conducted in the PyBullet physics simulator; sim-to-real transfer is left for future work."

Also add "in simulation" to the second sentence: *"We propose a lightweight framework that combines a gait generator network learned from human motion with a Proximal Policy Optimization (PPO) controller for torque control **in simulation**."*

## D3 — Introduction: add missing references (Review 2 #3)

Add the four references Reviewer 2 asked for, each in a single sentence of context. Rough placements:

1. **Bioinspired walking / passive dynamics** — in the first paragraph of Sec. 1, after the LIPM / SLIP sentence:
   - Collins S, Ruina A. *A bipedal walking robot with efficient and human-like gait.* Proc. IEEE ICRA 2005, 1983–8.
   - Suggested phrasing: "Passive-dynamic walkers [Collins & Ruina 2005] demonstrated that near-human gaits can emerge with minimal actuation, motivating morphology-aware controllers."

2. **Templates-and-anchors / SLIP context** — extend the existing SLIP citation:
   - Full RJ, Koditschek DE. *Templates and anchors: neuromechanical hypotheses of legged locomotion on land.* J Exp Biol 1999; 202: 3325–32.

3. **Inverted pendulum energetics** — in the paragraph discussing LIPM:
   - Kuo AD. *Energetics of actively powered locomotion using the simplest walking model.* J Biomech Eng 2002; 124: 113–20.

4. **Compliant-ankle / bioinspired control** — in the paragraph bridging classical and learned control:
   - Kerimoglu D, Karkoub M, Ismail U, Morgul O, Saranli U. *Efficient bipedal locomotion on rough terrain via compliant ankle actuation with energy regulation.* Bioinspiration & Biomimetics 2021; 16(5): 056011.

Each addition should be one sentence of genuine context, not a citation dump. Tie Kerimoglu et al. in particular to the paper's noisy-plane results (shared theme of rough-terrain robustness) — this is a citation the senior author is likely to value.

## D4 — Figure quality (Review 2 #2)

All figures were regenerated in the repository with `plot_style.py` (Track D4 code-side is the style module; the paper-side work is *deciding which figures to include and redrawing Fig. 1*).

Paper-side action items:
- Replace Fig. 1 (Controller Loop) with a clearer schematic. Use `env.get_image()` in [`ppoenv_guide.py`](ppoenv_guide.py) to render the 2D biped, then overlay labelled joint markers (torso, hip L/R, knee L/R, ankle L/R). Current figure is cluttered and reviewers couldn't parse the morphology (explicit Review 2 complaint).
- Replace Fig. 2 (Policy Network Architectures) with a single panel showing MLP / small-LSTM / large-LSTM side-by-side, axis-aligned. Currently the subfigures are different scales.
- Fig. 3 (joint comparison) — increase font to ≥14 pt, thicken lines, move legend out of plot area. Add a clarifying caption: what speeds are averaged, what seeds.
- Fig. 4 (Velocity Comparison) — currently has every configuration overlaid; split into 2 panels (imitation-based vs baselines) to reduce clutter.
- Figs. 7 & 8 (Noisy Plane) — use the same y-axis range across both so the "NP0 is harsher" point is visually immediate.
- Fig. 9 (Velocity Tracking) — add grid, mark the requested speed profile in a distinct colour, note phase lag numerically in the caption.

## D5 — Concrete 3D roadmap in the Conclusion (Review 1 #4)

Review 1 #4 called the Conclusion "hasty" regarding 3D extension. Replace the current single-sentence future-work line with a dedicated subsection (~0.5 column):

> ### 4.1 Extension to 3D Humanoids
> We outline a concrete migration path for the present framework.
> **(i) Morphology.** Swap the 7-DoF planar URDF in [`assets/biped2d.urdf`](assets/biped2d.urdf) for a full 3D humanoid (Cassie, MIT Humanoid or Unitree H1), adding frontal-plane hips and ankles, yaw at the torso, and upper-body DoFs with arms held compliant.
> **(ii) Gait generator extension.** Extend the 6-joint × 17-harmonic Fourier encoding to include frontal-plane joints and yaw as two additional Fourier channels, conditioned on pelvic width and shoulder width in addition to leg length. Retraining on the same [Schreiber & Moissenet 2019] dataset is feasible because their capture protocol records full-body markers.
> **(iii) Observation.** Add pelvic orientation (quaternion), 6-DoF CoM twist, and per-foot wrench measurements to the 58-D observation.
> **(iv) Rewards.** Keep the imitation reward structure, add a frontal-plane CoM tracking term and a heading-stability term; remove the explicit foot-clearance sigmoid now that yaw can deviate.
> **(v) Curriculum.** Begin with a horizontal pelvis constraint (sagittal-only motion) to verify the transfer, then release lateral DoFs. This mirrors the 2D→3D progression that worked for DeepMimic.

If time permits, include one preliminary gif / screenshot of a 3D humanoid standing or starting to walk under this framework, even without full quantitative results. A visual makes the promise concrete to reviewers.

## D6 — Justify the three-configuration choice (Review 2 #4)

Add a paragraph in Sec. 2.5 (Agent Training) or at the start of Sec. 3 explaining **why** three policy networks were evaluated, rather than treating it as arbitrary. Proposed text:

> "We designed the three policy architectures as a memory-capacity ablation rather than a hyperparameter sweep. Configuration 1 (MLP 256×256) has no explicit memory and relies solely on the history-and-preview observation to recover the gait phase. Configuration 2 (LSTM 64) introduces minimal recurrent capacity and tests whether a short-range hidden state adds value on top of the observation. Configuration 3 (LSTM 256) expands recurrent capacity by 16× to test whether longer-range memory pays off given the periodic nature of the task. We hypothesized, and the results confirm (Table X), that the periodicity is already carried by the gait-generator preview, so the feed-forward policy is sufficient; recurrent variants trade training speed for no improvement in robustness."

Also fold in the bonus `nostate_mlp/PPO_47` result: *"As a further sanity check, we ran a configuration where both the imitation reward and the reference state were removed (`nostate_mlp`). This policy fails to learn a forward gait (Table X), confirming that the gait-generator output is load-bearing even when used only as an observation, not as a reward target."*

Also answer Reviewer 2's specific numeric question ("what is the maximum value for the last 2 cols in Table 3 (Range)"): add a table-caption clarifier *"Range (NP0/NP1) is the maximum forward distance before early termination, clipped at the floor length of 10.0 m."*

## Execution order and dependencies

D2 and D3 are essentially free and can be done immediately.

D6 can also be drafted immediately; the `nostate_mlp` paragraph needs the existing Table 3 numbers.

D1 and D4 depend on new numbers / figures produced by the code tracks (Tracks B and C). Schedule these **last**, after experiments settle.

D5 is a self-contained writing task.

## Draft timeline (writing only)

1. Week 1: D2 (abstract), D3 (references), D6 (three-config justification), D5 (3D roadmap) — all writable without new experiments.
2. Weeks 2–6: wait for code tracks to produce final numbers.
3. Weeks 7–8: D1 (soften claims with new numbers), D4 (redraw figures with new numbers and `plot_style.py`).
4. Week 9: full manuscript pass, camera-ready.

## Out of scope

- Any code change. Writing-only plan.
- Any change to experimental hypotheses. This plan presents existing results more honestly; it does not invent new ones.
- Sim2real. Explicitly deferred in D2.
