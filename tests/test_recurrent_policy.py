from __future__ import annotations

from functools import partial

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from cambrian.ml.features_extractors import (
    MjCambrianCyclopeanR2Plus1DExtractor,
)
from cambrian.ml.policies import MjCambrianCyclopeanLstmPolicy


def _observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "eye_0_0": spaces.Box(
                0.0, 1.0, shape=(3, 8, 12, 3), dtype=np.float32
            ),
            "eye_0_1": spaces.Box(
                0.0, 1.0, shape=(3, 8, 12, 3), dtype=np.float32
            ),
            "action": spaces.Box(-1.0, 1.0, shape=(3, 6), dtype=np.float32),
            "eye_state": spaces.Box(-5.0, 5.0, shape=(3, 8), dtype=np.float32),
            "contacts": spaces.Box(0, 1, shape=(3, 1), dtype=np.int32),
        }
    )


def _observations(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "eye_0_0": torch.rand(batch_size, 3, 8, 12, 3),
        "eye_0_1": torch.rand(batch_size, 3, 8, 12, 3),
        "action": 2.0 * torch.rand(batch_size, 3, 6) - 1.0,
        "eye_state": 2.0 * torch.rand(batch_size, 3, 8) - 1.0,
        "contacts": torch.zeros(batch_size, 3, 1),
    }


def _extractor_partial(cyclopean_dim: int = 32):
    return partial(
        MjCambrianCyclopeanR2Plus1DExtractor,
        normalized_image=True,
        stem_channels=4,
        block_channels=[4, 8, 8],
        temporal_channels=8,
        latent_channels=4,
        norm_groups=2,
        code_mode="signed_silu",
        motor_hidden_dim=32,
        motor_context_dim=16,
        fusion_hidden_dim=48,
        cyclopean_dim=cyclopean_dim,
        film_conditioning=True,
    )


def test_cyclopean_extractor_fuses_two_eyes_and_motor_history() -> None:
    extractor = _extractor_partial()(_observation_space())
    observations = _observations(batch_size=3)

    eye_maps, motor_context, cyclopean = extractor.encode_components(observations)

    assert extractor.image_keys == ("eye_0_0", "eye_0_1")
    assert extractor.retinal_code_shape == (4, 1, 1)
    assert eye_maps.shape == (3, 2, 4, 1, 1)
    assert motor_context.shape == (3, 16)
    assert cyclopean.shape == (3, 32)
    assert extractor(observations).shape == (3, 32)

    changed_motor = {key: value.clone() for key, value in observations.items()}
    changed_motor["action"] = torch.ones_like(changed_motor["action"])
    assert not torch.allclose(cyclopean, extractor(changed_motor))


def _policy(hidden_size: int = 16) -> MjCambrianCyclopeanLstmPolicy:
    return MjCambrianCyclopeanLstmPolicy(
        _observation_space(),
        spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=_extractor_partial(cyclopean_dim=32),
        normalize_images=True,
        net_arch={"pi": [32, 16], "vf": [32, 16]},
        activation_fn=torch.nn.SiLU,
        ortho_init=False,
        lstm_hidden_size=hidden_size,
        n_lstm_layers=1,
        use_current_features_skip=True,
    )


def _states(batch_size: int, hidden_size: int, fill: float = 0.0) -> RNNStates:
    hidden = torch.full((1, batch_size, hidden_size), fill)
    cell = torch.full((1, batch_size, hidden_size), fill)
    return RNNStates((hidden, cell), (hidden.clone(), cell.clone()))


def test_recurrent_policy_has_finite_nonzero_lstm_gradients() -> None:
    torch.manual_seed(0)
    policy = _policy(hidden_size=16)
    observations = _observations(batch_size=4)
    states = _states(batch_size=2, hidden_size=16)
    episode_starts = torch.zeros(4)
    actions = torch.zeros(4, 6)

    before = policy.lstm_actor.weight_hh_l0.detach().clone()
    values, log_prob, entropy = policy.evaluate_actions(
        observations,
        actions,
        states,
        episode_starts,
    )
    loss = -log_prob.mean() + 0.5 * values.square().mean()
    if entropy is not None:
        loss -= 0.01 * entropy.mean()

    policy.optimizer.zero_grad()
    loss.backward()
    gradient = policy.lstm_actor.weight_hh_l0.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0
    policy.optimizer.step()
    assert not torch.equal(before, policy.lstm_actor.weight_hh_l0.detach())


def test_episode_start_resets_shared_recurrent_state() -> None:
    torch.manual_seed(1)
    policy = _policy(hidden_size=16)
    policy.eval()
    observations = _observations(batch_size=2)

    with torch.no_grad():
        _, values_reset, _, states_reset = policy(
            observations,
            _states(batch_size=2, hidden_size=16, fill=3.0),
            torch.ones(2),
            deterministic=True,
        )
        _, values_zero, _, states_zero = policy(
            observations,
            _states(batch_size=2, hidden_size=16),
            torch.zeros(2),
            deterministic=True,
        )

    torch.testing.assert_close(values_reset, values_zero)
    torch.testing.assert_close(states_reset.pi[0], states_zero.pi[0])
    torch.testing.assert_close(states_reset.pi[1], states_zero.pi[1])
    torch.testing.assert_close(states_reset.pi[0], states_reset.vf[0])
    torch.testing.assert_close(states_reset.pi[1], states_reset.vf[1])
