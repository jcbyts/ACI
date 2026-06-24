"""Recurrent policies for binocular tracking experiments."""

from typing import Dict, Tuple

import torch
from sb3_contrib.common.recurrent.policies import (
    RecurrentMultiInputActorCriticPolicy,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.torch_layers import MlpExtractor


class MjCambrianCyclopeanLstmPolicy(RecurrentMultiInputActorCriticPolicy):
    """One shared LSTM state plus a direct current-vision skip connection.

    The feature extractor constructs the current cyclopean visual-motor code ``c_t``.
    A single LSTM updates persistent state ``h_t``. Actor and critic read
    ``[c_t, h_t]``, so recurrence adds memory without forcing immediate visual control
    through an initially untrained recurrent bottleneck. Both PPO policy and value
    losses train the shared recurrent state.
    """

    def __init__(self, *args, use_current_features_skip: bool = True, **kwargs) -> None:
        self.use_current_features_skip = use_current_features_skip
        # sb3-contrib's buffer stores actor and critic state slots. We fill both with
        # the same state while using exactly one recurrent module.
        kwargs["shared_lstm"] = True
        kwargs["enable_critic_lstm"] = False
        kwargs["share_features_extractor"] = True
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        policy_input_dim = self.lstm_output_dim
        if self.use_current_features_skip:
            policy_input_dim += self.features_dim
        self.mlp_extractor = MlpExtractor(
            policy_input_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _policy_input(
        self,
        current_features: torch.Tensor,
        recurrent_features: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_current_features_skip:
            return recurrent_features
        return torch.cat((current_features, recurrent_features), dim=1)

    @staticmethod
    def _duplicate_state(
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> RNNStates:
        return RNNStates(state, state)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        features = self.extract_features(obs)
        recurrent, next_state = self._process_sequence(
            features,
            lstm_states.pi,
            episode_starts,
            self.lstm_actor,
        )
        policy_input = self._policy_input(features, recurrent)
        latent_pi, latent_vf = self.mlp_extractor(policy_input)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, self._duplicate_state(next_state)

    def get_distribution(
        self,
        obs: Dict[str, torch.Tensor],
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Tuple[Distribution, Tuple[torch.Tensor, torch.Tensor]]:
        features = self.extract_features(obs)
        recurrent, next_state = self._process_sequence(
            features,
            lstm_states,
            episode_starts,
            self.lstm_actor,
        )
        latent_pi = self.mlp_extractor.forward_actor(
            self._policy_input(features, recurrent)
        )
        return self._get_action_dist_from_latent(latent_pi), next_state

    def predict_values(
        self,
        obs: Dict[str, torch.Tensor],
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        features = self.extract_features(obs)
        recurrent, _ = self._process_sequence(
            features,
            lstm_states,
            episode_starts,
            self.lstm_actor,
        )
        latent_vf = self.mlp_extractor.forward_critic(
            self._policy_input(features, recurrent)
        )
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        features = self.extract_features(obs)
        recurrent, _ = self._process_sequence(
            features,
            lstm_states.pi,
            episode_starts,
            self.lstm_actor,
        )
        policy_input = self._policy_input(features, recurrent)
        latent_pi, latent_vf = self.mlp_extractor(policy_input)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
