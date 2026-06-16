"""Check that a composed training config exposes only vision plus action history.

This intentionally fails if the learner receives world position, target position,
contact bits, or observations from non-learning agents.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hydra import compose, initialize_config_dir
from hydra_config import HydraContainerConfig

from cambrian import MjCambrianConfig  # noqa: F401 - registers config/resolvers
import cambrian.agents  # noqa: F401 - make package re-exports locatable
import cambrian.agents.point  # noqa: F401 - make custom point agents locatable
import cambrian.ml.features_extractors  # noqa: F401 - make custom extractors locatable
from cambrian.envs.env import MjCambrianEnvConfig


def _main(config):
    env_config = MjCambrianEnvConfig.instantiate(config.env)
    env = env_config.instance(env_config)
    wrappers = HydraContainerConfig.instantiate(config.trainer.wrappers)
    for wrapper in wrappers.values():
        if wrapper:
            env = wrapper(env)
    obs, _ = env.reset(seed=int(config.seed))

    obs_space = env.observation_space
    action_space = env.action_space
    keys = sorted(obs_space.spaces.keys())

    forbidden = {
        "qpos",
        "qvel",
        "pos",
        "position",
        "target",
        "goal",
        "contacts",
        "agent_qpos",
        "goal0",
        "adversary0",
    }
    bad = [key for key in keys if key in forbidden or key.startswith("goal")]

    print("OBS_KEYS:", keys)
    print("OBS_SPACE:", obs_space)
    print("ACTION_SPACE:", action_space)
    print("RESET_OBS_KEYS:", sorted(obs.keys()) if isinstance(obs, dict) else type(obs))
    assert not bad, f"forbidden learner observation keys present: {bad}"
    assert "action" in keys, "previous-action/proprioceptive observation missing"
    assert any(key.startswith("eye") or "eye" in key for key in keys), (
        "vision observation missing"
    )
    print("PREFLIGHT_OK: learner obs is vision plus action history only")


if __name__ == "__main__":
    config_path = REPO_ROOT / "cambrian" / "configs"
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="base", overrides=sys.argv[1:])
    _main(cfg)
