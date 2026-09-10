# Saccade-and-fixate agents — ACI research fork

This project studies the minimum requirements for agents to learn **saccades and
fixations**: rapid gaze shifts interspersed with periods of stable visual sampling.
It builds on [Artificial Cambrian Intelligence](https://github.com/cambrian-org/ACI),
using embodied agents with independently movable binocular eyes in MuJoCo.

**Work from `main`.** It is the maintained starting point for this research fork.
Start with the [student quickstart](docs/student_quickstart.md), then create a
short-lived branch for each change and open a pull request back to `main`.

## Start here

```bash
git clone https://github.com/jcbyts/ACI.git
cd ACI
git switch main
```

Use Python 3.12 in an isolated environment. On the existing workstation, activate
`aci312`; for a new installation, follow the setup in the
[quickstart](docs/student_quickstart.md#environment).

The first milestone is to run a small training smoke test, load its checkpoint,
and produce an evaluation video and behavior report. The quickstart supplies the
commands, expected files, and the full baseline training recipe.

## Current research status

| Status | What to use / what it establishes |
|---|---|
| **Recommended starting baseline** | `actuated-2eye-rppo` with `overlay=[tracking_documented_reward,textured_ground]`: matched binocular cameras, yaw-rate body control, shared spatiotemporal retina, motor-conditioned binocular fusion, and LSTM PPO. Use `fixed-2eye-rppo` with the same settings as the control. |
| **Working infrastructure** | Training/checkpoint/evaluation paths, corrected camera geometry and image layout, recurrent gradients/reset tests, and behavior CSV/JSON/plot generation. |
| **Still to validate scientifically** | Robust held-out performance, fixation/saccade segmentation, and which components are necessary. Two recorded actuated recurrent seeds improve but lose performance after their best checkpoints. |
| **Experimental** | Motor-command costs, heavy-body physics, sensor noise/integration, and alternative retinal codes. The historical metabolism sweep failed before producing checkpoints. |
| **Historical** | Earlier PPO and R2-Dreamer approaches are documented in the [experiment review](docs/review_2026-09-10/README.md). Some used simplified tasks or incorrect geometry; compare resolved configurations before comparing scores. |

The recommended baseline retains a moving target and a moving adversary. Training
adds a small progress reward; evaluation disables that shaping. Both a ten-frame
visual history and persistent LSTM state are enabled. These choices differ from
the older [June 22 no-shaping contract](docs/experiment_contract.md).

## Where to look

- [Student quickstart](docs/student_quickstart.md): setup, smoke test, training, evaluation, analysis, and first research tasks.
- [Experiment review and handoff](docs/review_2026-09-10/README.md): branch map, evidence, results, limitations, and controlled-ablation plan.
- [Contributing](docs/contributing.md): branch and pull-request workflow.
- `cambrian/configs/example/tracking_2eye_recurrent_actuated.yaml`: current actuated experiment.
- `cambrian/ml/features_extractors.py` and `cambrian/ml/policies.py`: visual architecture and recurrent policy.
- `cambrian/analysis/tracking_behavior.py`: current behavior measurements.

## Branches and artifacts

`main` is the student entry point. Older feature branches are historical snapshots;
there is no separate permanent development branch named `saccade-and-fixate`.
The previous `main` is preserved at annotated tag **`legacy-main-v0.0.0`**
(commit `69525c7`). This is an archival tag, not a validated research release.

Training outputs under `logs/` are local artifacts and are not included in a fresh
clone. Train a smoke checkpoint using the quickstart, or obtain a complete recorded
run from the project maintainer. Keep its saved configurations and source revision
with the checkpoint. Use a fresh output directory for each evaluation.

## Upstream ACI

ACI supplies the simulator, eye models, configuration framework, and original
navigation/detection/tracking tasks. Its [documentation](https://eyes.mit.edu/ACI/)
provides background on those components. The student workflow for this fork is
maintained here. The upstream papers and citation are retained below.

## Project Papers

<div style="display: flex; align-items: center; gap: 1rem;">

<img src="https://eyes.mit.edu/ACI/_static/whatifeye.png" alt="What if Eye...?" width="100">

***What if Eye...?* Computationally Recreating Vision Evolution** \
[Kushagra Tiwary\*](https://kushagratiwary.com/), [Aaron Young\*](https://AaronYoung5.github.io/), [Zaid Tasneem](https://zaidtas.github.io/), [Tzofi Klinghoffer](https://tzofi.github.io/), [Akshat Dave](https://akshatdave.github.io/), [Tomaso Poggio](https://mcgovern.mit.edu/profile/tomaso-poggio/), [Dan-Eric Nilsson](https://portal.research.lu.se/en/persons/dan-eric-nilsson), [Brian Cheung<sup>†</sup>](https://briancheung.github.io/), [Ramesh Raskar<sup>†</sup>](https://www.media.mit.edu/people/raskar/overview/)
<br>
\[[Paper](https://arxiv.org/pdf/2501.15001) | [Website](https://eyes.mit.edu) | [Code](https://github.com/cambrian-org/ACI) | [Documentation](https://eyes.mit.edu/ACI/)\]




</div>

<div style="display: flex; align-items: center; gap: 1rem;">

<img src="https://eyes.mit.edu/ACI/_static/genvi.png" alt="GenVI" width="100">

**A Roadmap for Generative Design of Visual Intelligence** \
[Kushagra Tiwary](https://kushagratiwary.com/), [Tzofi Klinghoffer\*](https://tzofi.github.io/), [Aaron Young\*](https://AaronYoung5.github.io/), [Siddharth Somasundaram](https://sidsoma.github.io/), [Nikhil Behari](https://nikhilbehari.github.io/), [Akshat Dave](https://akshatdave.github.io/), [Brian Cheung](https://briancheung.github.io/), [Dan-Eric Nilsson](https://portal.research.lu.se/en/persons/dan-eric-nilsson), [Tomaso Poggio](https://mcgovern.mit.edu/profile/tomaso-poggio/), [Ramesh Raskar](https://www.media.mit.edu/people/raskar/overview/)
<br>
\[[Paper](https://mit-genai.pubpub.org/pub/bcfcb6lu/release/3) | [Code](https://github.com/cambrian-org/ACI) | [Documentation](https://eyes.mit.edu/ACI/)\]



</div>

## Citation

If you use ACI in your research, please consider citing:

```bibtex
@software{aci,
    author = {Aaron Young and Kushagra Tiwary and Zaid Tasneem and Tzofi Klinghoffer and Bhavya Agrawalla and Sanjana Duttagupta and Akshat Dave and Brian Cheung},
    title = {{Artificial Cambrian Intelligence}},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/cambrian-org/ACI}},
}
```
