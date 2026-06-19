#!/usr/bin/env python3
"""Train NM512/r2dreamer on ACI's frozen binocular pursuit task.

This script intentionally keeps r2dreamer outside this repository. Pass
``--r2dreamer-root`` to a checkout of https://github.com/NM512/r2dreamer and this
runner will import its Dreamer, Buffer, and OnlineTrainer classes while supplying a
Dreamer-compatible ACI environment.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra_config import HydraContainerConfig
from omegaconf import OmegaConf
from tensordict import TensorDict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cambrian import MjCambrianConfig  # noqa: F401 - registers config/resolvers
import cambrian.agents  # noqa: F401 - make package re-exports locatable
import cambrian.agents.point  # noqa: F401 - make custom point agents locatable
from cambrian.envs.env import MjCambrianEnvConfig
from cambrian.ml.behavior_audit import BehaviorGateThresholds, evaluate_behavior_gate
from cambrian.utils.wrappers import (
    MjCambrianSingleAgentEnvWrapper,
    MjCambrianTorchToNumpyWrapper,
)
from scripts.preflight_binocular_geometry import CameraSummary
from scripts.preflight_binocular_geometry import (
    _check_summaries as check_binocular_geometry,
)
from scripts.preflight_binocular_geometry import (
    _summarize_compiled_xml as summarize_compiled_xml,
)


ACI_FROZEN2_OVERRIDES = [
    "example=tracking",
    "env/agents@env.agents.agent=point_relative",
    "env/agents/eyes@env.agents.agent.eyes.eye=multi_eye",
    "~env.agents.adversary0",
    "~eval_env.reward_fn.penalize_if_adversary_respawned",
    "+env.truncation_fn.truncate_if_close_to_adversary.disable=true",
    "eval_env.truncation_fn.truncate_if_close_to_adversary.disable=true",
    "+env.termination_fn.terminate_if_close_to_goal.disable=true",
    "eval_env.termination_fn.terminate_if_close_to_goal.disable=true",
    "trainer.max_episode_steps=512",
    "env.agents.agent.instance.max_relative_heading=0.6",
    "env.agents.agent.use_action_obs=true",
    "env.agents.agent.use_contact_obs=false",
    "env.agents.agent.check_contacts=true",
    "env.agents.goal0.instance.speed=-0.9",
    "+env.agents.goal0.instance.random_target_locations=reset",
    "++env.mazes.maze.floor_texture=checker",
    "++env.mazes.maze.floor_texrepeat=[10,10]",
    "env.agents.agent.eyes.eye.num_eyes=[1,2]",
    "env.agents.agent.eyes.eye.lat_range=[0,0]",
    "env.agents.agent.eyes.eye.lon_range=[-18.75,18.75]",
    "env.agents.agent.eyes.eye.fov=[45,67.5]",
    "env.agents.agent.eyes.eye.sensorsize=[0.008284271247461903,0.008909048505590654]",
    "env.agents.agent.eyes.eye.resolution=[20,30]",
    "env.agents.agent.eyes.eye.noise_std=0",
    "env.agents.agent.eyes.eye.integration_factor=0",
    "+env.step_fn.respawn_objects_if_agent_close._target_=cambrian.envs.step_fns.step_respawn_agents_if_close_to_agents",
    "+env.step_fn.respawn_objects_if_agent_close._partial_=true",
    "+env.step_fn.respawn_objects_if_agent_close.for_agents=[goal0]",
    "+env.step_fn.respawn_objects_if_agent_close.to_agents=[agent]",
    "+env.step_fn.respawn_objects_if_agent_close.distance_threshold=1.0",
    "+env.reward_fn.time_penalty._target_=cambrian.envs.reward_fns.reward_fn_constant",
    "+env.reward_fn.time_penalty._partial_=true",
    "+env.reward_fn.time_penalty.for_agents=[agent]",
    "+env.reward_fn.time_penalty.reward=-0.01",
    "+env.reward_fn.approach_goal._target_=cambrian.envs.reward_fns.reward_fn_euclidean_delta_to_agent",
    "+env.reward_fn.approach_goal._partial_=true",
    "+env.reward_fn.approach_goal.for_agents=[agent]",
    "+env.reward_fn.approach_goal.to_agents=[goal0]",
    "+env.reward_fn.approach_goal.reward=8.0",
    "+env.reward_fn.reward_if_goal_respawned._target_=cambrian.envs.reward_fns.reward_fn_agent_respawned",
    "+env.reward_fn.reward_if_goal_respawned._partial_=true",
    "+env.reward_fn.reward_if_goal_respawned.for_agents=[goal0]",
    "+env.reward_fn.reward_if_goal_respawned.reward=10.0",
    "+env.reward_fn.reward_if_goal_respawned.scale_by_quickness=false",
    "env.reward_fn.penalize_if_has_contacts.reward=-1.0",
    "+eval_env.reward_fn.approach_goal.disable=true",
    "+eval_env.reward_fn.time_penalty.disable=true",
    "eval_env.reward_fn.reward_if_goal_respawned.scale_by_quickness=false",
    "eval_env.reward_fn.reward_if_goal_respawned.reward=10.0",
    "eval_env.reward_fn.penalize_if_has_contacts.reward=0.0",
]


@dataclass
class EpisodeSummary:
    steps: int
    captures: int
    total_reward: float
    mean_reward: float
    initial_distance: float
    final_distance: float
    median_distance: float
    min_distance: float
    near_goal_fraction: float
    total_path: float
    net_displacement: float
    stationary_fraction: float
    contact_fraction: float
    had_contact: bool
    terminal_contact: bool
    near_border_fraction: float
    action_mean: list[float]
    action_min: list[float]
    action_max: list[float]
    physical_speed_command_mean: float
    near_zero_speed_command_fraction: float
    contact_pairs: dict[str, int]


def clear_hydra() -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def compose_aci_config(extra_overrides: list[str]) -> Any:
    clear_hydra()
    config_path = REPO_ROOT / "cambrian" / "configs"
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        return compose(
            config_name="base",
            overrides=[*ACI_FROZEN2_OVERRIDES, *extra_overrides],
        )


def compose_r2dreamer_config(args: argparse.Namespace) -> Any:
    clear_hydra()
    config_path = Path(args.r2dreamer_root) / "configs"
    overrides = [
        f"logdir={args.logdir}",
        f"seed={args.seed}",
        f"device={args.device}",
        "env=dmc_vision",
        f"env.steps={args.steps}",
        f"env.env_num={args.env_num}",
        f"env.eval_episode_num={args.eval_episode_num}",
        "env.action_repeat=1",
        "env.time_limit=512",
        f"env.train_ratio={args.train_ratio}",
        "env.encoder.cnn_keys=image",
        "env.encoder.mlp_keys=proprio",
        "env.decoder.cnn_keys=image",
        "env.decoder.mlp_keys=proprio",
        f"batch_size={args.batch_size}",
        f"batch_length={args.batch_length}",
        f"buffer.max_size={args.buffer_size}",
        "buffer.storage_device=cpu",
        f"trainer.eval_every={args.eval_every}",
        f"trainer.pretrain={args.pretrain}",
        f"trainer.update_log_every={args.update_log_every}",
        "trainer.action_repeat=1",
        "trainer.video_pred_log=false",
        "trainer.params_hist_log=false",
        f"model.rep_loss={args.rep_loss}",
        "model.compile=false",
        f"model.deter={args.deter}",
        f"model.hidden={args.hidden}",
        f"model.discrete={args.discrete}",
        f"model.depth={args.depth}",
        f"model.units={args.units}",
        f"model.lr={args.learning_rate}",
        f"model.act_entropy={args.act_entropy}",
    ]
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        return compose(config_name="configs", overrides=overrides)


def instantiate_single_agent_env(config: Any, *, eval_env: bool, seed: int, name: str):
    env_config_source = config.eval_env if eval_env else config.env
    env_config = MjCambrianEnvConfig.instantiate(env_config_source)
    env = env_config.instance(env_config, name=name)
    env = MjCambrianSingleAgentEnvWrapper(env, agent_name="agent")
    env = MjCambrianTorchToNumpyWrapper(env, convert_action=False)
    env.unwrapped.set_random_seed(seed)
    return env


def strip_optional_stack(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def maze_border_distance(env: Any, xy: np.ndarray) -> float | None:
    maze = getattr(env, "maze", None)
    if maze is None:
        return None

    half_w = maze.map_width_scaled / 2.0
    half_h = maze.map_length_scaled / 2.0
    center_x = -float(getattr(maze, "_starting_x", 0.0))
    left = center_x - half_w
    right = center_x + half_w
    bottom = -half_h
    top = half_h
    x, y = float(xy[0]), float(xy[1])
    return min(x - left, right - x, y - bottom, top - y)


def agent_contact_pairs(env: Any, agent_name: str = "agent") -> list[str]:
    import mujoco as mj

    agent = env.agents[agent_name]
    pairs: list[str] = []
    for contact in env.data.contact:
        if contact.exclude:
            continue
        geom1 = int(contact.geom[0])
        geom2 = int(contact.geom[1])
        body1 = env.model.geom_bodyid[geom1]
        body2 = env.model.geom_bodyid[geom2]
        rootbody1 = env.model.body_rootid[body1]
        rootbody2 = env.model.body_rootid[body2]
        if rootbody1 != agent._body_id and rootbody2 != agent._body_id:
            continue
        name1 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, geom1) or f"geom_{geom1}"
        name2 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, geom2) or f"geom_{geom2}"
        pairs.append("::".join(sorted((name1, name2))))
    return sorted(set(pairs))


class ACIDreamerEnv(gym.Env):
    """Old-style Gym API wrapper expected by r2dreamer."""

    metadata: dict[str, Any] = {}

    def __init__(self, config: Any, *, eval_env: bool, seed: int, name: str):
        super().__init__()
        self._env = instantiate_single_agent_env(
            config,
            eval_env=eval_env,
            seed=seed,
            name=name,
        )
        self._cambrian_env = self._env.unwrapped
        self._seed = seed
        self._reset_count = 0
        self._last_samples: list[dict[str, Any]] = []
        self.action_space = self._env.action_space
        self._action_obs_size = int(np.prod(self._env.observation_space["action"].shape))
        agent = self._cambrian_env.agents["agent"]
        self._physical_eye_action_size = int(getattr(agent, "_n_eye_actuators", 0))
        self._proprio_size = self._action_obs_size + 2 * self._physical_eye_action_size

        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(20, 60, 3),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=-5.0,
                    high=5.0,
                    shape=(self._proprio_size,),
                    dtype=np.float32,
                ),
                "is_first": gym.spaces.Box(0, 1, shape=(), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, shape=(), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, shape=(), dtype=bool),
                "log_capture": gym.spaces.Box(0.0, np.inf, shape=(), dtype=np.float32),
                "log_contact": gym.spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
                "log_distance": gym.spaces.Box(0.0, np.inf, shape=(), dtype=np.float32),
            }
        )

    @property
    def cambrian_env(self) -> Any:
        return self._cambrian_env

    @property
    def last_samples(self) -> list[dict[str, Any]]:
        return self._last_samples

    def close(self) -> None:
        self._env.close()

    def reset(self) -> dict[str, np.ndarray]:
        reset_seed = self._seed if self._reset_count == 0 else None
        obs, _ = self._env.reset(seed=reset_seed)
        self._reset_count += 1
        self._last_samples = []
        return self._convert_obs(
            obs,
            is_first=True,
            is_last=False,
            is_terminal=False,
            info={},
        )

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        self._record_sample(action, float(reward), done, info)
        converted = self._convert_obs(
            obs,
            is_first=False,
            is_last=done,
            is_terminal=bool(terminated),
            info=info,
        )
        return converted, float(reward), done, info

    def _convert_obs(
        self,
        obs: dict[str, Any],
        *,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
        info: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        eye_keys = sorted(key for key in obs if key.startswith("agent_eye_"))
        if len(eye_keys) != 2:
            raise RuntimeError(f"Expected exactly two eye observations, got {eye_keys}")
        eyes = [strip_optional_stack(obs[key]) for key in eye_keys]
        image = np.concatenate(eyes, axis=1)
        image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)

        proprio = strip_optional_stack(obs["action"]).astype(np.float32).reshape(-1)
        gaze_state = self._eye_joint_state().astype(np.float32)
        if gaze_state.size:
            proprio = np.concatenate([proprio, gaze_state]).astype(np.float32)
        agent = self._cambrian_env.agents["agent"]
        goal = self._cambrian_env.agents["goal0"]
        distance = float(np.linalg.norm(np.asarray(agent.pos[:2]) - np.asarray(goal.pos[:2])))
        goal_info = self._cambrian_env._info.get("goal0", {})
        has_contacts = bool(info.get("has_contacts", False))

        return {
            "image": image,
            "proprio": proprio,
            "is_first": np.asarray(is_first, dtype=bool),
            "is_last": np.asarray(is_last, dtype=bool),
            "is_terminal": np.asarray(is_terminal, dtype=bool),
            "log_capture": np.asarray(
                float(bool(goal_info.get("respawned", False))),
                dtype=np.float32,
            ),
            "log_contact": np.asarray(float(has_contacts), dtype=np.float32),
            "log_distance": np.asarray(distance, dtype=np.float32),
        }

    def _eye_joint_state(self) -> np.ndarray:
        if self._physical_eye_action_size == 0:
            return np.zeros(0, dtype=np.float32)

        agent = self._cambrian_env.agents["agent"]
        eye_actuators = getattr(agent, "_eye_actuators", [])
        if not eye_actuators:
            return np.zeros(2 * self._physical_eye_action_size, dtype=np.float32)

        qpos: list[float] = []
        qvel: list[float] = []
        for actuator in eye_actuators:
            joint_id = int(actuator.trnadr)
            qpos_adr = int(self._cambrian_env.model.jnt_qposadr[joint_id])
            qvel_adr = int(self._cambrian_env.model.jnt_dofadr[joint_id])
            value = float(self._cambrian_env.data.qpos[qpos_adr])
            velocity = float(self._cambrian_env.data.qvel[qvel_adr])
            if actuator.ctrllimited:
                value = float(np.interp(value, actuator.ctrlrange, [-1.0, 1.0]))
                scale = float(np.max(np.abs(actuator.ctrlrange)))
                if scale > 1e-8:
                    velocity = velocity / scale
            qpos.append(float(np.clip(value, -5.0, 5.0)))
            qvel.append(float(np.clip(velocity, -5.0, 5.0)))
        return np.asarray([*qpos, *qvel], dtype=np.float32)

    def _record_sample(
        self,
        action: np.ndarray,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        agent = self._cambrian_env.agents["agent"]
        goal = self._cambrian_env.agents["goal0"]
        agent_xy = np.asarray(agent.pos[:2], dtype=np.float64)
        goal_xy = np.asarray(goal.pos[:2], dtype=np.float64)
        goal_info = self._cambrian_env._info.get("goal0", {})
        has_contacts = bool(info.get("has_contacts", False))
        self._last_samples.append(
            {
                "agent_xy": agent_xy.tolist(),
                "goal_xy": goal_xy.tolist(),
                "action": np.asarray(action, dtype=np.float64).reshape(-1).tolist(),
                "reward": reward,
                "goal_respawned": bool(goal_info.get("respawned", False)),
                "has_contacts": has_contacts,
                "contact_pairs": (
                    agent_contact_pairs(self._cambrian_env) if has_contacts else []
                ),
                "border_distance": maze_border_distance(self._cambrian_env, agent_xy),
                "done": done,
            }
        )


class ACIDreamerVec:
    """Minimal r2dreamer-compatible vector env without subprocesses."""

    def __init__(self, constructors: list[Callable[[], ACIDreamerEnv]], device: str):
        self.envs = [constructor() for constructor in constructors]
        self.device = torch.device(device)

    @property
    def observation_space(self) -> gym.Space:
        return self.envs[0].observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.envs[0].action_space

    @property
    def env_num(self) -> int:
        return len(self.envs)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    @staticmethod
    def lift_dim(td: TensorDict) -> TensorDict:
        for key in list(td.keys()):
            if td[key].ndim == 1:
                td[key] = td[key].unsqueeze(-1)
        return td

    def step(self, action: torch.Tensor, done: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
        action_np = action.detach().cpu().numpy()
        done_np = done.detach().cpu().numpy().astype(bool)
        observations: list[dict[str, np.ndarray]] = []
        rewards: list[float] = []
        dones: list[bool] = []

        for env, env_action, env_done in zip(self.envs, action_np, done_np):
            if env_done:
                observations.append(env.reset())
                rewards.append(0.0)
                dones.append(False)
            else:
                obs, reward, step_done, _ = env.step(env_action)
                observations.append(obs)
                rewards.append(float(reward))
                dones.append(bool(step_done))

        obs_stacked = {
            key: np.stack([obs[key] for obs in observations])
            for key in observations[0].keys()
        }
        tensors = {
            key: torch.as_tensor(value, device="cpu")
            for key, value in obs_stacked.items()
        }
        td = TensorDict(
            {
                **tensors,
                "reward": torch.as_tensor(rewards, dtype=torch.float32, device="cpu"),
            },
            batch_size=(self.env_num,),
            device="cpu",
        )
        if self.device.type == "cuda":
            td = td.pin_memory()
        return self.lift_dim(td), torch.as_tensor(dones, device="cpu")


def summarize_episode(samples: list[dict[str, Any]]) -> EpisodeSummary:
    agent = np.asarray([s["agent_xy"] for s in samples], dtype=np.float64)
    goal = np.asarray([s["goal_xy"] for s in samples], dtype=np.float64)
    actions = np.asarray([s["action"] for s in samples], dtype=np.float64)
    rewards = np.asarray([s["reward"] for s in samples], dtype=np.float64)
    distances = np.linalg.norm(agent - goal, axis=1)
    deltas = np.linalg.norm(np.diff(agent, axis=0), axis=1)
    border_values = [
        s["border_distance"] for s in samples if s["border_distance"] is not None
    ]
    border = np.asarray(border_values, dtype=np.float64)
    contact_pairs = Counter(pair for sample in samples for pair in sample["contact_pairs"])
    physical_speed = (np.clip(actions[:, 0], -1.0, 1.0) + 1.0) / 2.0

    return EpisodeSummary(
        steps=len(samples),
        captures=int(sum(1 for s in samples if s["goal_respawned"])),
        total_reward=float(np.sum(rewards)),
        mean_reward=float(np.mean(rewards)),
        initial_distance=float(distances[0]),
        final_distance=float(distances[-1]),
        median_distance=float(np.median(distances)),
        min_distance=float(np.min(distances)),
        near_goal_fraction=float(np.mean(distances < 1.25)),
        total_path=float(np.sum(deltas)) if len(deltas) else 0.0,
        net_displacement=float(np.linalg.norm(agent[-1] - agent[0])),
        stationary_fraction=float(np.mean(deltas < 0.02)) if len(deltas) else 0.0,
        contact_fraction=float(np.mean([s["has_contacts"] for s in samples])),
        had_contact=bool(any(s["has_contacts"] for s in samples)),
        terminal_contact=bool(samples[-1]["done"] and samples[-1]["has_contacts"]),
        near_border_fraction=float(np.mean(border < 1.25)) if len(border) else 0.0,
        action_mean=actions.mean(axis=0).round(4).tolist(),
        action_min=actions.min(axis=0).round(4).tolist(),
        action_max=actions.max(axis=0).round(4).tolist(),
        physical_speed_command_mean=float(np.mean(physical_speed)),
        near_zero_speed_command_fraction=float(np.mean(physical_speed < 0.02)),
        contact_pairs=dict(contact_pairs.most_common()),
    )


def stack_obs_for_agent(obs: dict[str, np.ndarray], device: torch.device) -> TensorDict:
    tensors = {
        key: torch.as_tensor(np.stack([value]), device="cpu")
        for key, value in obs.items()
    }
    td = TensorDict(tensors, batch_size=(1,), device="cpu")
    td = ACIDreamerVec.lift_dim(td)
    return td.to(device, non_blocking=True)


def evaluate_final_policy(
    agent: Any,
    config: Any,
    *,
    seed: int,
    n_episodes: int,
    outdir: Path,
    device: torch.device,
) -> dict[str, Any]:
    env = ACIDreamerEnv(config, eval_env=True, seed=seed, name="r2dreamer_eval_final")
    summaries: list[EpisodeSummary] = []
    first_episode_frames: list[np.ndarray] = []
    agent.eval()
    try:
        for episode in range(n_episodes):
            obs = env.reset()
            state = agent.get_initial_state(1)
            done = False
            if episode == 0:
                first_episode_frames.append(obs["image"])
            while not done:
                obs_td = stack_obs_for_agent(obs, device)
                action, state = agent.act(obs_td, state, eval=True)
                obs, _, done, _ = env.step(action.detach().cpu().numpy()[0])
                if episode == 0:
                    first_episode_frames.append(obs["image"])
            summaries.append(summarize_episode(env.last_samples))
    finally:
        env.close()
        agent.train()

    episodes = [asdict(summary) for summary in summaries]
    audit = {
        "n_episodes": len(episodes),
        "mean_captures": float(np.mean([e["captures"] for e in episodes])),
        "all_episodes_captured": bool(all(e["captures"] > 0 for e in episodes)),
        "mean_stationary_fraction": float(
            np.mean([e["stationary_fraction"] for e in episodes])
        ),
        "mean_median_distance": float(np.mean([e["median_distance"] for e in episodes])),
        "max_contact_fraction": float(np.max([e["contact_fraction"] for e in episodes])),
        "episode_contact_fraction": float(np.mean([e["had_contact"] for e in episodes])),
        "terminal_contact_fraction": float(
            np.mean([e["terminal_contact"] for e in episodes])
        ),
        "max_near_border_fraction": float(
            np.max([e["near_border_fraction"] for e in episodes])
        ),
        "episodes": episodes,
    }
    thresholds = BehaviorGateThresholds(
        min_mean_captures=1.0,
        require_all_episodes_capture=True,
        max_stationary_fraction=0.5,
        max_median_distance=5.0,
        max_contact_fraction=0.0,
        max_near_border_fraction=0.25,
    )
    passed, failures = evaluate_behavior_gate(audit, thresholds)
    audit["behavior_gate"] = {
        "passed": passed,
        "failures": failures,
        "thresholds": asdict(thresholds),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "dreamer_behavior_eval.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    if first_episode_frames:
        imageio.mimsave(
            outdir / "dreamer_eval_retina_episode0.mp4",
            first_episode_frames,
            fps=16,
            macro_block_size=1,
        )
    return audit


def summarize_live_cameras(env: Any) -> list[CameraSummary]:
    import mujoco as mj

    cambrian_env = env.unwrapped
    model = cambrian_env.model
    data = cambrian_env.data
    agent = cambrian_env.agents["agent"]
    agent_heading = float(agent.qpos[2])
    c, s = np.cos(-agent_heading), np.sin(-agent_heading)
    world_to_agent_xy = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    summaries: list[CameraSummary] = []
    for cam_id in range(model.ncam):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, cam_id) or ""
        if not name.startswith("agent_eye"):
            continue
        xmat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
        forward = -xmat[:, 2]
        world_yaw = float((np.rad2deg(np.arctan2(forward[1], forward[0])) + 180.0) % 360.0 - 180.0)
        yaw = float((world_yaw - np.rad2deg(agent_heading) + 180.0) % 360.0 - 180.0)
        pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
        local_xy = world_to_agent_xy @ (pos[:2] - np.asarray(agent.pos[:2], dtype=np.float64))
        position_yaw = float((np.rad2deg(np.arctan2(local_xy[1], local_xy[0])) + 180.0) % 360.0 - 180.0)
        for target_yaw in (-18.75, 18.75):
            if abs(yaw - target_yaw) < 0.05:
                yaw = target_yaw
            if abs(position_yaw - target_yaw) < 0.05:
                position_yaw = target_yaw
        resolution = [int(v) for v in model.cam_resolution[cam_id].tolist()]
        sensorsize = [float(v) for v in model.cam_sensorsize[cam_id].tolist()]
        horizontal_fov = 67.5
        summaries.append(
            CameraSummary(
                name=name,
                yaw_deg=yaw,
                position_yaw_deg=position_yaw,
                resolution=resolution,
                sensorsize=sensorsize,
                coverage_deg=[yaw - horizontal_fov / 2.0, yaw + horizontal_fov / 2.0],
            )
        )
    return sorted(summaries, key=lambda c: c.yaw_deg)


def write_run_artifacts(config: Any, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=outdir / "aci_config.yaml", resolve=True)
    env = instantiate_single_agent_env(
        config,
        eval_env=True,
        seed=int(config.seed),
        name="r2dreamer_preflight",
    )
    try:
        env.unwrapped.xml.write(outdir / "env.xml")
        compiled_xml = outdir / "compiled_env.xml"
        compiled_xml.write_text(env.unwrapped.spec.to_xml())
        obs, _ = env.reset(seed=int(config.seed))
        try:
            camera_summaries = summarize_compiled_xml(compiled_xml)
        except (KeyError, AssertionError):
            camera_summaries = summarize_live_cameras(env)
        geometry = check_binocular_geometry(camera_summaries)
        (outdir / "binocular_geometry_preflight.json").write_text(
            json.dumps(geometry, indent=2) + "\n"
        )
        summary = {
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
            "reset_obs_keys": sorted(obs.keys()),
        }
        (outdir / "aci_env_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2dreamer-root", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--env-num", type=int, default=1)
    parser.add_argument("--eval-episode-num", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save Dreamer checkpoints every N environment steps after eval; defaults to eval_every. Set 0 to disable periodic checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-keep",
        type=int,
        default=0,
        help="Keep only the newest N checkpoint_step_*.pt files; 0 keeps all periodic checkpoints.",
    )
    parser.add_argument("--final-eval-episodes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-length", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--train-ratio", type=int, default=32)
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--update-log-every", type=int, default=2_000)
    parser.add_argument("--rep-loss", choices=["r2dreamer", "dreamer"], default="r2dreamer")
    parser.add_argument("--deter", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--discrete", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--units", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--act-entropy", type=float, default=3e-4)
    parser.add_argument(
        "aci_overrides",
        nargs="*",
        help="Extra Hydra overrides applied to the ACI environment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    r2dreamer_root = Path(args.r2dreamer_root).expanduser().resolve()
    if not (r2dreamer_root / "dreamer.py").exists():
        raise FileNotFoundError(f"r2dreamer checkout not found at {r2dreamer_root}")
    sys.path.insert(0, str(r2dreamer_root))

    import tools as r2tools
    from buffer import Buffer
    from dreamer import Dreamer
    from trainer import OnlineTrainer

    torch.set_float32_matmul_precision("high")
    r2tools.set_seed_everywhere(args.seed)

    aci_config = compose_aci_config(args.aci_overrides)
    args.logdir.mkdir(parents=True, exist_ok=True)
    (args.logdir / "aci_overrides.txt").write_text(
        "\n".join(args.aci_overrides) + ("\n" if args.aci_overrides else "")
    )
    write_run_artifacts(aci_config, args.logdir)

    r2_config = compose_r2dreamer_config(args)
    OmegaConf.save(config=r2_config, f=args.logdir / "r2dreamer_config.yaml", resolve=True)

    console_f = r2tools.setup_console_log(args.logdir, filename="console.log")
    atexit.register(lambda: console_f.close())
    logger = r2tools.Logger(args.logdir)
    logger.log_hydra_config(r2_config)
    replay_buffer = Buffer(r2_config.buffer)

    def train_constructor(idx: int) -> Callable[[], ACIDreamerEnv]:
        return lambda: ACIDreamerEnv(
            aci_config,
            eval_env=False,
            seed=args.seed + idx,
            name=f"r2dreamer_train_{idx}",
        )

    def eval_constructor(idx: int) -> Callable[[], ACIDreamerEnv]:
        return lambda: ACIDreamerEnv(
            aci_config,
            eval_env=True,
            seed=args.seed + 10_000 + idx,
            name=f"r2dreamer_eval_{idx}",
        )

    train_envs = ACIDreamerVec(
        [train_constructor(i) for i in range(args.env_num)],
        args.device,
    )
    eval_envs = ACIDreamerVec(
        [eval_constructor(i) for i in range(args.eval_episode_num)],
        args.device,
    )
    try:
        agent = Dreamer(
            r2_config.model,
            train_envs.observation_space,
            train_envs.action_space,
        ).to(r2_config.device)
        policy_trainer = OnlineTrainer(
            r2_config.trainer,
            replay_buffer,
            logger,
            args.logdir,
            train_envs,
            eval_envs,
        )

        checkpoint_every = (
            args.eval_every if args.checkpoint_every is None else args.checkpoint_every
        )
        checkpoint_steps: list[Path] = []

        def write_checkpoint(path: Path) -> None:
            payload = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": r2tools.recursively_collect_optim_state_dict(agent),
            }
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(path)

        def save_periodic_checkpoint(train_step: int) -> None:
            if checkpoint_every <= 0:
                return
            if train_step % checkpoint_every != 0:
                return
            checkpoint_path = args.logdir / f"checkpoint_step_{train_step:06d}.pt"
            write_checkpoint(checkpoint_path)
            write_checkpoint(args.logdir / "latest.pt")
            checkpoint_steps.append(checkpoint_path)
            if args.checkpoint_keep > 0:
                while len(checkpoint_steps) > args.checkpoint_keep:
                    old_checkpoint = checkpoint_steps.pop(0)
                    old_checkpoint.unlink(missing_ok=True)
            print(f"Saved checkpoint: {checkpoint_path}")

        original_eval = policy_trainer.eval

        def eval_and_checkpoint(eval_agent: Any, train_step: int) -> None:
            original_eval(eval_agent, train_step)
            save_periodic_checkpoint(train_step)

        policy_trainer.eval = eval_and_checkpoint
        policy_trainer.begin(agent)
        write_checkpoint(args.logdir / "latest.pt")
        audit = evaluate_final_policy(
            agent,
            aci_config,
            seed=args.seed + 20_000,
            n_episodes=args.final_eval_episodes,
            outdir=args.logdir,
            device=torch.device(r2_config.device),
        )
        print(
            "Final behavior gate:",
            "passed" if audit["behavior_gate"]["passed"] else "failed",
            audit["behavior_gate"]["failures"],
        )
    finally:
        train_envs.close()
        eval_envs.close()


if __name__ == "__main__":
    main()
