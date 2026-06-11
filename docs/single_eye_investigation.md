# Single-eye failure investigation — handoff

**Repo:** ACI / cambrian (What if Eye…? paper, arXiv 2501.15001). **Branch:** saccade-and-fixate-agents. **Date:** 2026-06-10.

## The big-picture goal

The branch's purpose is the **saccade-and-fixate eye**: can ONE *movable* eye recover the performance that the paper gets from THREE *fixed* eyes? Before building that, we spent this investigation answering a prerequisite question — **why does a single fixed eye fail at the tracking/detection task while the 3-eye baseline succeeds?** That answer defines what the movable eye actually has to fix.

## Anchors (eval mean reward, full task)

- **1 frozen eye, 105°, 35×35** → caps **+5**, then *collapses to −6* over a 2M run.
- **3 fixed eyes (baseline, `example=tracking`, multi-eye, 45° ea, 20×20, lon ±30)** → **+37**.

## What we RULED OUT (overnight + fair-test work)

- **Not obs format / body code / blindness / sensors / pixel count / action-dim.**
- **Not early-stopping.** A fair 2M run with early-stop OFF (`exp_m2_frozen_2m_full`, `max_no_improvement_evals=100000`) still failed. The earlier "ROOT CAUSE = early-stopping" claim OVERSTATES it — early-stop is a real artifact (it does cut noisy runs ~480k) but is NOT why the single eye fails.
- **Not undertraining.** More samples made it *worse*: peak +5.17 @320k, then declined to −6 by 1.84M. That decline is an **optimization collapse**, not a sample-count problem.
- **Not resolution.** Exp 3 (`exp_frozen_hires_2m`, 64×64, 3.3× the pixels, full task) peaked **+5.2** and ended **0.0** — identical cap to 35×35. More pixels on one welded eye buys nothing.

## What we LEARNED — the real mechanism (TWO stacked problems)

The task is **fine-grained discrimination under asymmetric risk**:

1. **Goal and adversary are the SAME striped sphere**, adversary just rotated 90° (`task/detection.yaml`, `object_sphere_textured_adversary.yaml`, euler `0 90 0`). Telling approach(+) from avoid(−) requires resolving **vertical vs horizontal stripes** on a small sphere.
2. **Reward is risk-asymmetric**: eval goal +10, adversary **−20**, do-nothing 0 (train +1/−1/0). Approaching the wrong sphere costs MORE than the right one gains → rational policy = approach only when confident.

These produce two distinct failures, cleanly separated by experiments:

- **Risk-asymmetry → PARKING / do-nothing basin.** Confirmed by Exp 1 (read-only): single eye confuses goal/adversary **42.9% vs 6.4%** baseline, and **parks** (path length 3.8 vs 86.4; 57% do-nothing episodes). Confirmed by Exp 2 (goal-only, adversary removed): the agent stops parking and starts moving (negative → +6.8, peak +21).
- **Welded gaze + weak monocular signal → CAN'T TRACK even when free to move.** Exp 2's goal-only video shows it WANDERS instead of chasing: a frozen forward eye welds gaze to heading, so the target leaves the cone whenever the body turns or closes distance (POV goes black). When in view it's a few faint pixels. Reward is contact-only (no approach shaping), so it can't bootstrap homing → random walk that occasionally bumps a goal (per-rollout 0→33 spread).

**Why 3 eyes win:** NOT coverage (matched ~105°). It's **discrimination quality** — overlapping views give higher central angular resolution + **parallax** (multiple simultaneous vantage points) → reliable stripe discrimination AND persistent target visibility → confident approach → robust +37.

### Correction logged honestly

The earlier claim that the policy is "memoryless" was WRONG: **frame-stacking is active** (`frame_stack_wrapper`, `stack_size: 10`) — the agent does see short-term motion. Doesn't change the diagnosis.

## Experiments run (all early-stop OFF, 2M unless noted)

| Exp | Config | Result |
|---|---|---|
| Exp 1 read-only | confusion + body-motion on existing policies | baseline +37.1 / 30-of-30; single eye 42.9% confusion, parks (57% do-nothing) |
| Exp 2 goal-only | 1 eye 35×35, adversary neutralized | peak +21 @1.16M, final +6.8 — moves but can't track |
| Exp 3 hi-res | 1 eye 64×64, full task | peak +5.2, final 0.0 — resolution doesn't help |
| Exp 4 two-eye | 2 eyes, full task, parallax test | RUNNING (see below) |

## Exp 4 — IN PROGRESS (the parallax test)

Question: does 1→2 eyes (binocular) recover the gap toward +37? **Correct config** = the +37 baseline (`example=tracking`, multi-eye) with ONLY `num_eyes [1,3]→[1,2]`.

- **GOTCHA #1:** `example=tracking_eye` uses the SINGLE-eye class (`MjCambrianEye`) which has **no `num_eyes` field** → Hydra error `Key 'num_eyes' not in 'MjCambrianEyeConfig'`. `num_eyes`/`lon_range`/`lat_range` only exist on **`MjCambrianMultiEye`** (`example=tracking`/`detection`). This silently killed 3 launch attempts.
- **GOTCHA #2 (geometry):** with `num_eyes [1,2]`, eyes are placed by `np.linspace` across `lon_range`. At the baseline's `lon_range [−30,30]` the two eyes land at the ENDPOINTS ±30° → 15° BLIND GAP dead ahead, zero binocular overlap (worst case for a parallax test). **Fix:** `lon_range [−15,15]` → eyes at ±15°, 45° FOV each → 15° frontal binocular overlap, no gap.
- **Current run:** `exp_frozen_2eye_2m`, binocular geometry, `num_eyes [1,2]`, `lon_range [−15,15]`, early-stop off. Monitor to 2M; record final eval vs anchors 1-eye +5 / 3-eye +37.
- **GOTCHA #3 (zsh):** list args must be single-quoted in the shell (`'env...num_eyes=[1,2]'`) or zsh glob-expands them.

## Reframing that matters

The single-eye "brittleness" is largely an **ablation artifact**, not a flaw in the paper. The task is calibrated for a compound multi-eye agent; the canonical `example=detection` ships **`num_eyes [1,3]` = three 20×20 eyes over 60°**. We stripped it to one welded eye to probe failure. "1 eye fails, 3 succeed" is closer to the paper's *thesis* (eye number/placement/morphology change capability) than to a bug. Our single-eye runs also pushed FOV wider (105°) than canonical (60°), amplifying the cliff.

## Operational notes for future agents

- **Launch runs DETACHED** (`nohup … & disown`, log to `/tmp/m2dbg/`) and **verify healthy** before ending: `ps` alive, compiled `logs/<date>/<expname>/config.yaml` shows expected geometry + `max_no_improvement_evals: 100000`, log shows stepping + first eval (~−470..−508 random-init band).
- **Delegation to implementor agents was UNRELIABLE** this session (repeated silent stalls at launch). Running launches directly via shell worked. Read the actual error before theorizing.
- Python for eval curves: `/Users/jake/opt/anaconda3/envs/aci/bin/python`, load `evaluations.npz` → `timesteps` + `results.mean(axis=1)`.
- **`logs/` is gitignored**; don't commit training output (checkpoints, videos, eval npz).

## Likely next steps (not yet started)

- Finish Exp 4, read the 1 / 2 / 3-eye trend. If two binocular eyes recover most of the gap → parallax/redundancy is the lever, and the saccade-and-fixate (one movable eye) is well-motivated. If only three clear it → a single eye (even moving) may not suffice; important to know first.
- Candidate follow-ups: the **moving/actuated eye** on goal-only or full task (does decoupled gaze let it fixate + chase?); **approach-shaping reward** to break contact-only sparsity; a wide-baseline 2-eye variant to separate stereo overlap from raw parallax baseline.
