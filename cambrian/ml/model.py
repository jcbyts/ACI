"""Custom model classes for Cambrian PPO baselines.

The feed-forward and recurrent PPO implementations share policy-only save/load and
rollout replay helpers, while leaving rollout collection, training, buffers, and
recurrent-state handling to Stable Baselines / sb3-contrib.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from cambrian.ml.policies import MjCambrianCyclopeanLstmPolicy
from cambrian.utils.logger import get_logger


class _MjCambrianModelMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rollout: List[Any] | None = None

    def save_policy(self, path: Path | str):
        """Save only the policy weights."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path / "policy.pt")

    def load_policy(self, path: Path | str, *, allow_partial: bool = False):
        """Load policy weights, failing closed on architecture mismatches.

        Scientific evaluation must not silently run a partly random policy.  Set
        ``allow_partial=True`` only for an explicitly documented transfer-learning
        operation; all ordinary replay/evaluation code should retain the strict default.
        """

        policy_path = Path(path) / "policy.pt"
        if not policy_path.exists():
            raise FileNotFoundError(f"Could not find policy.pt file at {policy_path}.")

        # Loop through the loaded state_dict and remove any layers that don't match in
        # shape with the current policy
        saved_state_dict = torch.load(
            policy_path, map_location=self.device, weights_only=True
        )
        policy_state_dict = self.policy.state_dict()
        missing = sorted(set(policy_state_dict) - set(saved_state_dict))
        unexpected = sorted(set(saved_state_dict) - set(policy_state_dict))
        shape_mismatches = []
        for saved_state_dict_key in list(saved_state_dict.keys()):
            if saved_state_dict_key not in policy_state_dict:
                if allow_partial:
                    get_logger().warning(
                        f"Key '{saved_state_dict_key}' not found in policy "
                        "state_dict. Skipping it because allow_partial=True."
                    )
                del saved_state_dict[saved_state_dict_key]
                continue

            saved_state_dict_var = saved_state_dict[saved_state_dict_key]
            policy_state_dict_var = policy_state_dict[saved_state_dict_key]

            if saved_state_dict_var.shape != policy_state_dict_var.shape:
                shape_mismatches.append(
                    (
                        saved_state_dict_key,
                        tuple(saved_state_dict_var.shape),
                        tuple(policy_state_dict_var.shape),
                    )
                )
                del saved_state_dict[saved_state_dict_key]

        if (missing or unexpected or shape_mismatches) and not allow_partial:
            lines = [
                "Policy checkpoint is incompatible with the current architecture."
            ]
            if missing:
                lines.append(f"Missing keys ({len(missing)}): {missing}")
            if unexpected:
                lines.append(f"Unexpected keys ({len(unexpected)}): {unexpected}")
            if shape_mismatches:
                details = [
                    f"{name}: saved={saved}, current={current}"
                    for name, saved, current in shape_mismatches
                ]
                lines.append(
                    f"Shape mismatches ({len(shape_mismatches)}): {details}"
                )
            raise RuntimeError("\n".join(lines))

        if allow_partial:
            for name, saved, current in shape_mismatches:
                get_logger().warning(
                    f"Skipping shape-mismatched key '{name}': "
                    f"saved={saved}, current={current}"
                )
        self.policy.load_state_dict(saved_state_dict, strict=not allow_partial)

    def load_rollout(self, path: Path | str):
        """Load the rollout data from a previous training run. The rollout is a list
        of actions based on a current step. The model.predict call will then be
        overwritten to return the next action. This loader is "dumb" in the sense that
        it doesn't actually process the observations when it's using rollout, it will
        simply keep track of the current step and return the next action in the
        rollout.
        """
        with open(path, "rb") as f:
            self._rollout = pickle.load(f)["actions"]

    @classmethod
    def load_weights(cls, weights: Dict[str, List[float]], **kwargs):
        """Load the weights for the policy. This is useful for testing the
        evolutionary loop without having to train the agent each time."""
        model = cls(**kwargs)

        # Iteratively load the weights into the model
        state_dict = model.policy.state_dict()
        for name, weight in weights.items():
            name = name.replace("__", ".")

            weight = torch.tensor(weight)
            assert name in state_dict, f"Layer {name} not found in model"
            assert state_dict[name].shape == weight.shape, (
                f"Shape mismatch for layer {name}: {state_dict[name].shape} != "
                f"{weight.shape}"
            )

            state_dict[name] = weight

        return model

    def predict(self, *args, **kwargs):
        if self._rollout is not None:
            state = kwargs.get("state")
            if state is None and len(args) > 1:
                state = args[1]
            return self._rollout.pop(0), state

        return super().predict(*args, **kwargs)


class MjCambrianModel(_MjCambrianModelMixin, PPO):
    pass


class MjCambrianRecurrentModel(_MjCambrianModelMixin, RecurrentPPO):
    policy_aliases = {
        **RecurrentPPO.policy_aliases,
        "CyclopeanMultiInputLstmPolicy": MjCambrianCyclopeanLstmPolicy,
    }
