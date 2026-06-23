"""Callbacks used during training and/or evaluation."""

import csv
import glob
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.experimental.callbacks import Callback as HydraCallback
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from cambrian.envs import MjCambrianEnv
from cambrian.ml.behavior_audit import (
    BehaviorGateThresholds,
    audit_policy_behavior,
    evaluate_behavior_gate,
    save_behavior_audit,
    thresholds_to_dict,
)
from cambrian.ml.model import MjCambrianModel, MjCambrianRecurrentModel
from cambrian.utils.logger import get_logger


class MjCambrianPlotMonitorCallback(BaseCallback):
    """Should be used with an EvalCallback to plot the evaluation results.

    This callback will take the monitor.csv file produced by the VecMonitor and
    plot the results and save it as an image. Should be passed as the
    `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory where the evaluation results are stored. The
            evaluations.npz file is expected to be at `<logdir>/<filename>.csv`. The
            resulting plot is going to be stored at
            `<logdir>/evaluations/<filename>.png`.
        filename (Path | str): The filename of the monitor file. The saved file will be
            `<logdir>/<filename>.csv`. And the resulting plot will be saved as
            `<logdir>/evaluations/<filename>.png`.
    """

    parent: EvalCallback

    def __init__(self, logdir: Path | str, filename: Path | str, n_episodes: int = 1):
        self.logdir = Path(logdir)
        self.filename = Path(filename)
        self.filename_csv = self.filename.with_suffix(".csv")
        self.filename_png = self.filename.with_suffix(".png")
        self.evaldir = self.logdir / "evaluations"
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.n_episodes = n_episodes
        self.n_calls = 0

    def _on_step(self) -> bool:
        if not (self.logdir / self.filename_csv).exists():
            get_logger().warning(f"No {self.filename_csv} file found.")
            return

        # Temporarily set the monitor ext so that the right file is read
        old_ext = Monitor.EXT
        Monitor.EXT = str(self.filename_csv)
        x, y = ts2xy(load_results(self.logdir), "timesteps")
        Monitor.EXT = old_ext
        if len(x) <= 20 or len(y) <= 20:
            get_logger().warning(f"Not enough {self.filename} data to plot.")
            return True
        original_x, original_y = x.copy(), y.copy()

        get_logger().info(f"Plotting {self.filename} results at {self.evaldir}")

        def moving_average(data, window=1):
            return np.convolve(data, np.ones(window), "valid") / window

        n = min(len(y) // 10, 1000)
        y = y.astype(float)

        if self.n_episodes > 1:
            assert len(y) % self.n_episodes == 0, (
                "n_episodes must be a common factor of the"
                f" number of episodes in the {self.filename} data."
            )
            y = y.reshape(-1, self.n_episodes).mean(axis=1)
        else:
            y = moving_average(y, window=n)

        x = moving_average(x, window=n).astype(int)

        # Make sure the x, y are of the same length
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        plt.plot(x, y)
        plt.plot(original_x, original_y, color="grey", alpha=0.3)

        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.savefig(self.evaldir / self.filename.with_suffix(".png"))
        plt.cla()

        return True


class MjCambrianEvalCallback(EvalCallback):
    """Overwrites the default EvalCallback to support saving visualizations at the same
    time as the evaluation.

    Note:
        Only the first environment is visualized
    """

    def _init_callback(self):
        self.log_path = Path(self.log_path)
        self.n_evals = 0

        # Delete all the existing renders
        for f in glob.glob(str(self.log_path / "vis_*")):
            get_logger().info(f"Deleting {f}")
            Path(f).unlink()

        super()._init_callback()

    def _on_step(self) -> bool:
        # Early exit
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        env: MjCambrianEnv = self.eval_env.envs[0].unwrapped

        # Add some overlays
        # env.overlays["Exp"] = env.config.expname # TODO
        env.overlays["Best Mean Reward"] = f"{self.best_mean_reward:.2f}"
        env.overlays["Total Timesteps"] = f"{self.num_timesteps}"

        # Run the evaluation
        get_logger().info(f"Starting {self.n_eval_episodes} evaluation run(s)...")
        env.record(self.render)
        continue_training = super()._on_step()

        if self.render:
            # Save the visualization
            filename = Path(f"vis_{self.n_evals}")
            env.save(self.log_path / filename)
            env.record(False)

        if self.render:
            # Copy the most recent gif to latest.gif so that we can just watch this file
            for f in self.log_path.glob(str(filename.with_suffix(".*"))):
                shutil.copy(f, f.with_stem("latest"))

        self.n_evals += 1
        return continue_training

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        env: MjCambrianEnv = self.eval_env.envs[0].unwrapped

        # If done, do some logging
        if locals_["done"]:
            run = locals_["episode_counts"][locals_["i"]]
            cumulative_reward = env.stashed_cumulative_reward
            get_logger().info(f"Run {run} done. Cumulative reward: {cumulative_reward}")

        super()._log_success_callback(locals_, globals_)


class MjCambrianGPUUsageCallback(BaseCallback):
    """This callback will log the GPU usage at the end of each evaluation.
    We'll log to a csv."""

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        logfile: Path | str = "gpu_usage.csv",
        *,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.logfile = self.logdir / logfile
        with open(self.logfile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timesteps",
                    "memory_reserved",
                    "max_memory_reserved",
                    "memory_available",
                ]
            )

    def _on_step(self) -> bool:
        if torch.cuda.is_available():
            # Get the GPU usage, log it and save it to the file
            device = torch.cuda.current_device()
            memory_reserved = torch.cuda.memory_reserved(device)
            max_memory_reserved = torch.cuda.max_memory_reserved(device)
            memory_available = torch.cuda.get_device_properties(device).total_memory

            # Log to the output file
            with open(self.logfile, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.num_timesteps,
                        memory_reserved,
                        max_memory_reserved,
                        memory_available,
                    ]
                )

            # Log to stdout
            if self.verbose > 0:
                get_logger().debug(subprocess.getoutput("nvidia-smi"))
                get_logger().debug(torch.cuda.memory_summary())

        return True


class MjCambrianActionPriorCallback(BaseCallback):
    """Set an initial Gaussian action prior before PPO collects rollouts.

    The point-agent action semantics make the default SB3 zero-mean Gaussian a poor
    prior: action 0 maps to half forward speed. This callback lets experiments set a
    safer prior without changing rewards, termination, or environment dynamics.
    """

    def __init__(
        self,
        *,
        action_bias: Sequence[float] | None = None,
        log_std: float | Sequence[float] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.action_bias = list(action_bias) if action_bias is not None else None
        self.log_std = log_std
        self._applied = False

    def _init_callback(self) -> None:
        policy = self.model.policy
        with torch.no_grad():
            if self.action_bias is not None:
                if not hasattr(policy, "action_net") or policy.action_net.bias is None:
                    raise RuntimeError("Policy action_net bias is not available.")
                bias = torch.as_tensor(
                    self.action_bias,
                    dtype=policy.action_net.bias.dtype,
                    device=policy.action_net.bias.device,
                )
                if bias.numel() != policy.action_net.bias.numel():
                    raise ValueError(
                        "action_bias length must match action dimension: "
                        f"{bias.numel()} != {policy.action_net.bias.numel()}"
                    )
                policy.action_net.bias.copy_(bias)

            if self.log_std is not None:
                if not hasattr(policy, "log_std"):
                    raise RuntimeError("Policy log_std is not available.")
                log_std = torch.as_tensor(
                    self.log_std,
                    dtype=policy.log_std.dtype,
                    device=policy.log_std.device,
                )
                if log_std.numel() == 1:
                    policy.log_std.fill_(float(log_std.item()))
                elif log_std.numel() == policy.log_std.numel():
                    policy.log_std.copy_(log_std.reshape_as(policy.log_std))
                else:
                    raise ValueError(
                        "log_std must be scalar or match action dimension: "
                        f"{log_std.numel()} not in (1, {policy.log_std.numel()})"
                    )

        self._applied = True
        if self.verbose > 0:
            get_logger().info(
                "Applied action prior: "
                f"action_bias={self.action_bias}, log_std={self.log_std}"
            )

    def _on_step(self) -> bool:
        return True


class MjCambrianPPODiagnosticsCallback(BaseCallback):
    """Log PPO rollout, action-distribution, and gradient diagnostics.

    This callback is passive: it runs one extra PPO-style backward pass on a sampled
    rollout minibatch after advantages have been computed, logs the resulting
    gradients, then clears the gradients before SB3 performs the real optimizer step.
    """

    def __init__(
        self,
        logdir: Path | str,
        logfile: Path | str = "ppo_diagnostics.csv",
        *,
        batch_size: int = 256,
        max_action_dims: int = 4,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.logfile = self.logdir / logfile
        self.batch_size = batch_size
        self.max_action_dims = max_action_dims
        self.fieldnames: list[str] | None = None
        self.previous_params: dict[str, torch.Tensor] | None = None
        self.rows: list[dict[str, float]] = []
        self.step_records: list[dict[str, float]] = []

        if self.logfile.exists():
            self.logfile.unlink()

    def _init_callback(self) -> None:
        if isinstance(self.model, MjCambrianRecurrentModel):
            raise RuntimeError(
                "MjCambrianPPODiagnosticsCallback is not recurrent-policy compatible."
            )

    @staticmethod
    def _swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    @staticmethod
    def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return {}
        flat = values.reshape(-1)
        return {
            f"{prefix}_mean": float(np.mean(flat)),
            f"{prefix}_std": float(np.std(flat)),
            f"{prefix}_min": float(np.min(flat)),
            f"{prefix}_max": float(np.max(flat)),
        }

    @staticmethod
    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def _mean_where(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(values[mask]))

    @staticmethod
    def _tensor_norm(tensor: torch.Tensor | None) -> float:
        if tensor is None:
            return 0.0
        return float(torch.linalg.vector_norm(tensor.detach().float()).cpu())

    @staticmethod
    def _sanitize(name: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in name)

    def _rollout_arrays(self) -> dict[str, Any]:
        buffer = self.model.rollout_buffer
        if isinstance(buffer.observations, dict):
            observations = {
                key: self._swap_and_flatten(value)
                for key, value in buffer.observations.items()
            }
        else:
            observations = self._swap_and_flatten(buffer.observations)

        return {
            "observations": observations,
            "actions": self._swap_and_flatten(buffer.actions),
            "rewards": self._swap_and_flatten(buffer.rewards).reshape(-1),
            "values": self._swap_and_flatten(buffer.values).reshape(-1),
            "returns": self._swap_and_flatten(buffer.returns).reshape(-1),
            "advantages": self._swap_and_flatten(buffer.advantages).reshape(-1),
            "log_probs": self._swap_and_flatten(buffer.log_probs).reshape(-1),
        }

    def _sample_batch(self, arrays: dict[str, Any]) -> dict[str, Any]:
        actions = arrays["actions"]
        batch_size = min(self.batch_size, actions.shape[0])
        # Deterministic sampling keeps this callback from perturbing PPO's RNG state.
        indices = np.linspace(0, actions.shape[0] - 1, batch_size, dtype=np.int64)
        device = self.model.rollout_buffer.device

        observations = arrays["observations"]
        if isinstance(observations, dict):
            obs_batch = {
                key: torch.as_tensor(value[indices], device=device)
                for key, value in observations.items()
            }
        else:
            obs_batch = torch.as_tensor(observations[indices], device=device)

        return {
            "observations": obs_batch,
            "actions": torch.as_tensor(actions[indices], device=device),
            "old_values": torch.as_tensor(arrays["values"][indices], device=device),
            "old_log_prob": torch.as_tensor(
                arrays["log_probs"][indices], device=device
            ),
            "advantages": torch.as_tensor(
                arrays["advantages"][indices], device=device
            ),
            "returns": torch.as_tensor(arrays["returns"][indices], device=device),
        }

    def _add_observation_stats(self, row: dict[str, float], observations: Any) -> None:
        if not isinstance(observations, dict):
            row.update(self._stats("obs", observations))
            row["obs_nonzero_fraction"] = float(np.mean(observations != 0.0))
            return

        for key, value in observations.items():
            prefix = f"obs_{self._sanitize(key)}"
            row.update(self._stats(prefix, value))
            row[f"{prefix}_nonzero_fraction"] = float(np.mean(value != 0.0))

    def _add_rollout_stats(self, row: dict[str, float], arrays: dict[str, Any]) -> None:
        row.update(self._stats("rollout_reward", arrays["rewards"]))
        row.update(self._stats("rollout_value", arrays["values"]))
        row.update(self._stats("rollout_return", arrays["returns"]))
        row.update(self._stats("rollout_advantage", arrays["advantages"]))
        row.update(self._stats("rollout_old_log_prob", arrays["log_probs"]))
        row["rollout_advantage_positive_fraction"] = float(
            np.mean(arrays["advantages"] > 0.0)
        )

        actions = arrays["actions"]
        clipped_actions = np.clip(actions, -1.0, 1.0)
        speed = clipped_actions[:, 0] if clipped_actions.shape[1] > 0 else np.array([])
        physical_speed = (speed + 1.0) / 2.0 if speed.size else np.array([])
        if speed.size:
            row.update(self._stats("rollout_speed_action", speed))
            row.update(self._stats("rollout_physical_speed", physical_speed))
            row["rollout_near_zero_speed_fraction"] = float(
                np.mean(physical_speed < 0.02)
            )
            row["rollout_advantage_speed_corr"] = self._safe_corr(
                physical_speed, arrays["advantages"]
            )
            row["rollout_advantage_near_zero_speed_mean"] = self._mean_where(
                arrays["advantages"], physical_speed < 0.02
            )
            row["rollout_advantage_slow_speed_mean"] = self._mean_where(
                arrays["advantages"], physical_speed < 0.1
            )
            row["rollout_advantage_fast_speed_mean"] = self._mean_where(
                arrays["advantages"], physical_speed > 0.2
            )
        if clipped_actions.shape[1] > 1:
            row.update(
                self._stats(
                    "rollout_turn_action_abs",
                    np.abs(clipped_actions[:, 1]),
                )
            )

        for i in range(min(self.max_action_dims, clipped_actions.shape[1])):
            row.update(self._stats(f"rollout_action_{i}", clipped_actions[:, i]))

        self._add_observation_stats(row, arrays["observations"])

    def _add_step_stats(self, row: dict[str, float]) -> None:
        if not self.step_records:
            return

        rewards = np.asarray([r["reward"] for r in self.step_records], dtype=np.float64)
        dones = np.asarray([r["done"] for r in self.step_records], dtype=bool)
        contacts = np.asarray([r["has_contacts"] for r in self.step_records], dtype=bool)
        physical_speed = np.asarray(
            [r["physical_speed"] for r in self.step_records],
            dtype=np.float64,
        )

        row.update(self._stats("step_reward", rewards))
        row.update(self._stats("step_physical_speed", physical_speed))
        row["step_done_fraction"] = float(np.mean(dones))
        row["step_contact_fraction"] = float(np.mean(contacts))
        row["step_done_contact_fraction"] = (
            float(np.mean(contacts[dones])) if np.any(dones) else 0.0
        )
        row["step_contact_physical_speed_mean"] = self._mean_where(
            physical_speed, contacts
        )
        row["step_noncontact_physical_speed_mean"] = self._mean_where(
            physical_speed, ~contacts
        )
        row["step_done_physical_speed_mean"] = self._mean_where(physical_speed, dones)
        row["step_reward_near_zero_speed_mean"] = self._mean_where(
            rewards, physical_speed < 0.02
        )
        row["step_reward_slow_speed_mean"] = self._mean_where(
            rewards, physical_speed < 0.1
        )
        row["step_reward_fast_speed_mean"] = self._mean_where(
            rewards, physical_speed > 0.2
        )
        row["step_contact_slow_fraction"] = (
            float(np.mean(contacts[physical_speed < 0.1]))
            if np.any(physical_speed < 0.1)
            else float("nan")
        )
        row["step_contact_fast_fraction"] = (
            float(np.mean(contacts[physical_speed > 0.2]))
            if np.any(physical_speed > 0.2)
            else float("nan")
        )

        episode_lengths = [
            r["episode_length"]
            for r in self.step_records
            if not np.isnan(r["episode_length"])
        ]
        if episode_lengths:
            row.update(self._stats("step_episode_length", np.asarray(episode_lengths)))

    def _add_distribution_stats(self, row: dict[str, float], observations: Any) -> None:
        policy = self.model.policy
        with torch.no_grad():
            distribution = policy.get_distribution(observations)
            torch_action = policy._predict(observations, deterministic=True)

        action = torch_action.detach().float().cpu().numpy()
        action = np.clip(action, -1.0, 1.0)
        if action.ndim == 1:
            action = action.reshape(-1, 1)
        if action.shape[1] > 0:
            physical_speed = (action[:, 0] + 1.0) / 2.0
            row.update(self._stats("policy_det_speed_action", action[:, 0]))
            row.update(self._stats("policy_det_physical_speed", physical_speed))
            row["policy_det_near_zero_speed_fraction"] = float(
                np.mean(physical_speed < 0.02)
            )
        if action.shape[1] > 1:
            row.update(self._stats("policy_det_turn_action_abs", np.abs(action[:, 1])))

        dist = getattr(distribution, "distribution", None)
        for attr in ("mean", "stddev"):
            tensor = getattr(dist, attr, None)
            if tensor is not None:
                values = tensor.detach().float().cpu().numpy()
                clipped = np.clip(values, -1.0, 1.0) if attr == "mean" else values
                row.update(self._stats(f"policy_dist_{attr}", clipped))
                for i in range(min(self.max_action_dims, values.shape[-1])):
                    row.update(self._stats(f"policy_dist_{attr}_action_{i}", values[:, i]))

        if hasattr(policy, "log_std"):
            log_std = policy.log_std.detach().float().cpu().numpy().reshape(-1)
            row.update(self._stats("policy_log_std", log_std))
            for i in range(min(self.max_action_dims, log_std.shape[0])):
                row[f"policy_log_std_action_{i}"] = float(log_std[i])
                row[f"policy_std_action_{i}"] = float(np.exp(log_std[i]))

        action_net = getattr(policy, "action_net", None)
        if action_net is not None:
            bias = getattr(action_net, "bias", None)
            weight = getattr(action_net, "weight", None)
            if bias is not None:
                bias_np = bias.detach().float().cpu().numpy().reshape(-1)
                for i in range(min(self.max_action_dims, bias_np.shape[0])):
                    row[f"action_net_bias_{i}"] = float(bias_np[i])
            if weight is not None:
                weight_np = weight.detach().float().cpu().numpy()
                for i in range(min(self.max_action_dims, weight_np.shape[0])):
                    row[f"action_net_weight_row_norm_{i}"] = float(
                        np.linalg.norm(weight_np[i])
                    )

    def _loss_and_grads(self, batch: dict[str, Any]) -> dict[str, float]:
        policy = self.model.policy
        policy.set_training_mode(True)
        policy.optimizer.zero_grad(set_to_none=True)

        actions = batch["actions"]
        if isinstance(self.model.action_space, gym.spaces.Discrete):
            actions = actions.long().flatten()

        values, log_prob, entropy = policy.evaluate_actions(
            batch["observations"], actions
        )
        values = values.flatten()

        advantages = batch["advantages"]
        if self.model.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clip_range = float(self.model.clip_range(self.model._current_progress_remaining))
        ratio = torch.exp(log_prob - batch["old_log_prob"])
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        if self.model.clip_range_vf is None:
            values_pred = values
            clip_range_vf = float("nan")
        else:
            clip_range_vf = float(
                self.model.clip_range_vf(self.model._current_progress_remaining)
            )
            values_pred = batch["old_values"] + torch.clamp(
                values - batch["old_values"], -clip_range_vf, clip_range_vf
            )
        value_loss = torch.nn.functional.mse_loss(batch["returns"], values_pred)

        if entropy is None:
            entropy_loss = -torch.mean(-log_prob)
            entropy_mean = float("nan")
        else:
            entropy_loss = -torch.mean(entropy)
            entropy_mean = float(entropy.detach().mean().cpu())

        loss = (
            policy_loss
            + self.model.ent_coef * entropy_loss
            + self.model.vf_coef * value_loss
        )

        with torch.no_grad():
            log_ratio = log_prob - batch["old_log_prob"]
            approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())

        loss.backward()

        row = {
            "diag_loss": float(loss.detach().cpu()),
            "diag_policy_loss": float(policy_loss.detach().cpu()),
            "diag_value_loss": float(value_loss.detach().cpu()),
            "diag_entropy_loss": float(entropy_loss.detach().cpu()),
            "diag_entropy_mean": entropy_mean,
            "diag_approx_kl": float(approx_kl.detach().cpu()),
            "diag_clip_fraction": float(clip_fraction.detach().cpu()),
            "diag_clip_range": clip_range,
            "diag_clip_range_vf": clip_range_vf,
            "diag_ratio_mean": float(ratio.detach().mean().cpu()),
            "diag_ratio_std": float(ratio.detach().std().cpu()),
        }
        row.update(self._grad_and_param_stats())
        policy.optimizer.zero_grad(set_to_none=True)
        return row

    def _group_for_param(self, name: str) -> list[str]:
        groups = ["all"]
        if "features_extractor" in name:
            groups.append("features_extractor")
        if ".cnn." in name or name.endswith(".cnn"):
            groups.append("image_cnn")
        if ".linear." in name or name.endswith(".linear"):
            groups.append("image_linear")
        if "temporal_linear" in name:
            groups.append("temporal_linear")
        if "mlp_extractor" in name:
            groups.append("mlp_extractor")
        if "action_net" in name:
            groups.append("action_net")
        if "value_net" in name:
            groups.append("value_net")
        if "log_std" in name:
            groups.append("log_std")
        return groups

    def _grad_and_param_stats(self) -> dict[str, float]:
        grad_squares: dict[str, float] = {}
        param_squares: dict[str, float] = {}
        delta_squares: dict[str, float] = {}
        current_params: dict[str, torch.Tensor] = {}
        action_head_rows: dict[str, float] = {}

        for name, param in self.model.policy.named_parameters():
            detached = param.detach().float().cpu()
            current_params[name] = detached.clone()
            groups = self._group_for_param(name)
            param_norm_sq = float(torch.sum(detached * detached))
            if param.grad is None:
                grad_norm_sq = 0.0
            else:
                grad = param.grad.detach().float().cpu()
                grad_norm_sq = float(torch.sum(grad * grad))
                if name == "action_net.bias":
                    for i in range(min(self.max_action_dims, grad.numel())):
                        action_head_rows[f"action_net_bias_grad_{i}"] = float(
                            grad.reshape(-1)[i]
                        )
                if name == "action_net.weight":
                    for i in range(min(self.max_action_dims, grad.shape[0])):
                        action_head_rows[f"action_net_weight_row_grad_norm_{i}"] = (
                            float(torch.linalg.vector_norm(grad[i]))
                        )

            if self.previous_params is None or name not in self.previous_params:
                delta_norm_sq = float("nan")
            else:
                delta = detached - self.previous_params[name]
                delta_norm_sq = float(torch.sum(delta * delta))

            for group in groups:
                grad_squares[group] = grad_squares.get(group, 0.0) + grad_norm_sq
                param_squares[group] = param_squares.get(group, 0.0) + param_norm_sq
                if not np.isnan(delta_norm_sq):
                    delta_squares[group] = (
                        delta_squares.get(group, 0.0) + delta_norm_sq
                    )

        self.previous_params = current_params

        row: dict[str, float] = {}
        for group, value in grad_squares.items():
            row[f"grad_norm_{group}"] = float(np.sqrt(value))
        for group, value in param_squares.items():
            row[f"param_norm_{group}"] = float(np.sqrt(value))
        for group, value in delta_squares.items():
            row[f"update_delta_norm_{group}"] = float(np.sqrt(value))
        row.update(action_head_rows)
        return row

    def _write_row(self, row: dict[str, float]) -> None:
        self.rows.append(row)
        fieldnames = list(dict.fromkeys(key for saved in self.rows for key in saved))
        if self.fieldnames != fieldnames:
            self.fieldnames = fieldnames
            with open(self.logfile, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
            return

        with open(self.logfile, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.fieldnames, extrasaction="ignore"
            )
            writer.writerow(row)

    def _on_step(self) -> bool:
        actions = self.locals.get("clipped_actions", self.locals.get("actions"))
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if actions is None or rewards is None or dones is None or infos is None:
            return True

        actions = np.asarray(actions)
        rewards = np.asarray(rewards).reshape(-1)
        dones = np.asarray(dones).reshape(-1)
        for i, info in enumerate(infos):
            recorded_action = (
                info.get("rescaled_action", actions[i])
                if isinstance(info, dict)
                else actions[i]
            )
            action = np.asarray(recorded_action).reshape(-1)
            physical_speed = float((np.clip(action[0], -1.0, 1.0) + 1.0) / 2.0)
            episode_info = info.get("episode", {}) if isinstance(info, dict) else {}
            self.step_records.append(
                {
                    "reward": float(rewards[i]),
                    "done": float(bool(dones[i])),
                    "has_contacts": float(
                        bool(info.get("has_contacts", False))
                        if isinstance(info, dict)
                        else False
                    ),
                    "physical_speed": physical_speed,
                    "episode_length": float(episode_info.get("l", np.nan)),
                }
            )
        return True

    def _on_rollout_start(self) -> None:
        self.step_records = []

    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer
        if not buffer.full:
            return

        arrays = self._rollout_arrays()
        batch = self._sample_batch(arrays)
        row: dict[str, float] = {
            "timesteps": float(self.num_timesteps),
            "n_updates": float(getattr(self.model, "_n_updates", 0)),
            "progress_remaining": float(self.model._current_progress_remaining),
            "learning_rate": float(self.model.lr_schedule(self.model._current_progress_remaining)),
            "ent_coef": float(self.model.ent_coef),
            "vf_coef": float(self.model.vf_coef),
        }
        self._add_rollout_stats(row, arrays)
        self._add_step_stats(row)
        self._add_distribution_stats(row, batch["observations"])
        row.update(self._loss_and_grads(batch))
        self._write_row(row)

        self.logger.record(
            "diagnostics/policy_det_physical_speed_mean",
            row.get("policy_det_physical_speed_mean", float("nan")),
        )
        self.logger.record(
            "diagnostics/rollout_advantage_speed_corr",
            row.get("rollout_advantage_speed_corr", float("nan")),
        )
        self.logger.record(
            "diagnostics/grad_norm_features_extractor",
            row.get("grad_norm_features_extractor", float("nan")),
        )
        self.logger.record(
            "diagnostics/grad_norm_action_net",
            row.get("grad_norm_action_net", float("nan")),
        )

        if self.verbose > 0:
            get_logger().info(f"Wrote PPO diagnostics to {self.logfile}")


class MjCambrianBehaviorGateCallback(BaseCallback):
    """Stop training only when behavior-level pursuit metrics pass a gate.

    This is intended to be used as an ``EvalCallback.callback_after_eval`` callback.
    It runs a fresh deterministic audit on the eval environment and writes a JSON
    summary next to the rendered evaluation artifacts.
    """

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        n_eval_episodes: int,
        *,
        stochastic_audit: bool = True,
        min_mean_captures: float = 1.0,
        require_all_episodes_capture: bool = True,
        max_stationary_fraction: float = 0.5,
        max_median_distance: float = 5.0,
        max_contact_fraction: float = 0.0,
        max_near_border_fraction: float = 0.25,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.logdir = Path(logdir)
        self.evaldir = self.logdir / "evaluations"
        self.evaldir.mkdir(parents=True, exist_ok=True)
        self.n_eval_episodes = n_eval_episodes
        self.stochastic_audit = stochastic_audit
        self.thresholds = BehaviorGateThresholds(
            min_mean_captures=min_mean_captures,
            require_all_episodes_capture=require_all_episodes_capture,
            max_stationary_fraction=max_stationary_fraction,
            max_median_distance=max_median_distance,
            max_contact_fraction=max_contact_fraction,
            max_near_border_fraction=max_near_border_fraction,
        )
        self.n_audits = 0

    def _on_step(self) -> bool:
        if not hasattr(self, "parent") or self.parent is None:
            raise RuntimeError("Behavior gate callback must be attached to EvalCallback")

        audit = audit_policy_behavior(
            self.parent.eval_env,
            self.model,
            n_episodes=self.n_eval_episodes,
            deterministic=True,
        )
        passed, failures = evaluate_behavior_gate(audit, self.thresholds)
        audit["behavior_gate"] = {
            "passed": passed,
            "failures": failures,
            "thresholds": thresholds_to_dict(self.thresholds),
        }
        save_behavior_audit(
            self.evaldir / f"behavior_gate_{self.n_audits}.json",
            audit,
        )
        if self.stochastic_audit:
            stochastic_audit = audit_policy_behavior(
                self.parent.eval_env,
                self.model,
                n_episodes=self.n_eval_episodes,
                deterministic=False,
            )
            stochastic_passed, stochastic_failures = evaluate_behavior_gate(
                stochastic_audit,
                self.thresholds,
            )
            stochastic_audit["behavior_gate"] = {
                "passed": stochastic_passed,
                "failures": stochastic_failures,
                "thresholds": thresholds_to_dict(self.thresholds),
            }
            save_behavior_audit(
                self.evaldir / f"behavior_gate_stochastic_{self.n_audits}.json",
                stochastic_audit,
            )
        self.n_audits += 1

        if passed:
            get_logger().info("Behavior gate passed; stopping training.")
            self.model.save(self.logdir / "behavior_gate_model")
            if hasattr(self.model, "save_policy"):
                self.model.save_policy(self.logdir / "behavior_gate_policy")
            return False

        get_logger().info(f"Behavior gate failed: {failures}")
        return True


class MjCambrianSavePolicyCallback(BaseCallback):
    """Should be used with an EvalCallback to save the policy.

    This callback will save the policy at the end of each evaluation. Should be passed
    as the `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory to store the generated visualizations. The
            resulting visualizations are going to be stored at
            `<logdir>/evaluations/visualization.gif`.
    """

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        *,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.model: MjCambrianModel = None

    def _on_step(self) -> bool:
        self.model.save_policy(self.logdir)

        return True


class MjCambrianProgressBarCallback(ProgressBarCallback):
    """Overwrite the default progress bar callback to flush the pbar on deconstruct."""

    def __del__(self):
        """This string will restore the terminal back to its original state."""
        if hasattr(self, "pbar"):
            print("\x1b[?25h")


class MjCambrianCallbackListWithSharedParent(CallbackList):
    def __init__(self, callbacks: Iterable[BaseCallback] | Dict[str, BaseCallback]):
        if isinstance(callbacks, dict):
            callbacks = callbacks.values()

        self.callbacks = []
        super().__init__(list(callbacks))

    @property
    def parent(self):
        return getattr(self.callbacks[0], "parent", None)

    @parent.setter
    def parent(self, parent):
        for cb in self.callbacks:
            cb.parent = parent


# ==================


class MjCambrianSaveConfigCallback(HydraCallback):
    """This callback will save the resolved hydra config to the logdir."""

    def on_run_start(self, config: DictConfig, **kwargs):
        self._save_config(config)

    def on_multirun_start(self, config: DictConfig, **kwargs):
        self._save_config(config)

    def _save_config(self, config: DictConfig):
        from omegaconf import OmegaConf

        config.logdir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, config.logdir / "full.yaml")
