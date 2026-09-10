"""Regression checks for diagnostics crossing the training process boundary."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from cambrian.ml.callbacks import MjCambrianEvalCallback
from cambrian.utils.wrappers import MjCambrianSingleAgentEnvWrapper


class _StatefulInfoEnv(gym.Env):
    def __init__(self):
        space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.agents = {
            "agent": SimpleNamespace(
                name="agent", action_space=space, observation_space=space
            )
        }
        self.info = {"agent": {"has_contacts": False}, "goal0": {"respawned": True}}

    def step(self, action):
        return (
            {"agent": np.zeros(1, dtype=np.float32)},
            {"agent": 0.0},
            {"agent": False},
            {"agent": False},
            self.info,
        )


def test_diagnostics_do_not_mutate_persistent_info_or_grow_between_steps():
    env = _StatefulInfoEnv()
    wrapped = MjCambrianSingleAgentEnvWrapper(env)
    sizes = []
    for _ in range(100):
        *_, info = wrapped.step(np.zeros(1, dtype=np.float32))
        data = pickle.dumps(info)
        sizes.append(len(data))
        restored = pickle.loads(data)
        assert restored["_all_agents"]["goal0"]["respawned"] is True
        assert "_all_agents" not in env.info["agent"]
        assert "_all_agents" not in info["_all_agents"]["agent"]
    assert len(set(sizes)) == 1


@pytest.mark.parametrize("save_checkpoint", [True, False])
def test_best_metadata_uses_checkpoint_directory(tmp_path: Path, save_checkpoint):
    # Rendering and SB3 checkpoints are deliberately stored in different directories.
    callback = object.__new__(MjCambrianEvalCallback)
    callback.log_path = tmp_path / "evaluations"
    callback.log_path.mkdir()
    callback.best_model_save_path = str(tmp_path) if save_checkpoint else None
    callback.num_timesteps = 100
    callback._write_best_metadata(2, 3.5)
    metadata = json.loads((callback.log_path / "best_model_metadata.json").read_text())
    expected = str(tmp_path / "best_model.zip") if save_checkpoint else None
    assert metadata["best_model_zip"] == expected
    assert metadata["best_eval_timesteps"] == 100
