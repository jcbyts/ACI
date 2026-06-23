"""Custom feature extractors used by the PPO policies.

Cambrian eye observations are emitted in channels-last form (``H, W, C``), while
``torch.nn.Conv2d`` requires channels-first tensors (``C, H, W``).  Stable
Baselines normally inserts a transpose wrapper for uint8 images, but Cambrian's
retinal observations are normalized float tensors.  Consequently, layout handling
must be explicit here rather than delegated to SB3.
"""

from typing import Dict, List, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import TensorDict

# ==================
# Utils


def is_image_space(
    observation_space: gym.Space,
    check_channels: bool = False,
    normalized_image: bool = False,
) -> bool:
    """This is an extension of the sb3 is_image_space to support both regular images
    (HxWxC) and images with an additional dimension (NxHxWxC)."""
    from stable_baselines3.common.preprocessing import (
        is_image_space as sb3_is_image_space,
    )

    return len(observation_space.shape) == 4 or sb3_is_image_space(
        observation_space, normalized_image=normalized_image
    )


def _image_channels_last(shape: Sequence[int]) -> bool:
    """Infer whether a 3D/4D image shape is channels-last.

    Supported forms are ``HWC``, ``CHW``, ``NHWC``, and ``NCHW``.  The inference is
    deliberately strict: a silent layout guess is worse than a hard failure because a
    mistaken guess still produces a trainable network with the wrong semantic axes.
    """

    if len(shape) == 3:
        first_channel, first_spatial = shape[0], shape[1:]
        last_channel, last_spatial = shape[-1], shape[:-1]
    elif len(shape) == 4:
        # The first axis is the temporal/frame-stack axis.
        first_channel, first_spatial = shape[1], shape[2:]
        last_channel, last_spatial = shape[-1], shape[1:-1]
    else:
        raise ValueError(f"Expected a 3D or 4D image space, got shape {tuple(shape)}")

    first_plausible = first_channel < min(first_spatial)
    last_plausible = last_channel < min(last_spatial)

    if first_plausible == last_plausible:
        raise ValueError(
            "Could not unambiguously infer image layout for shape "
            f"{tuple(shape)}. Expected HWC/CHW or NHWC/NCHW with a channel axis "
            "smaller than both spatial axes."
        )
    return last_plausible


def maybe_transpose_space(observation_space: spaces.Box) -> spaces.Box:
    """Return an equivalent channels-first image space.

    ``numpy.transpose`` is required here.  Reshaping an HWC bounds array into CHW does
    not preserve the mapping between pixels and channels when bounds are non-uniform.
    """

    if not _image_channels_last(observation_space.shape):
        return observation_space

    if len(observation_space.shape) == 3:
        axes = (2, 0, 1)  # HWC -> CHW
    else:
        axes = (0, 3, 1, 2)  # NHWC -> NCHW

    return spaces.Box(
        low=np.transpose(observation_space.low, axes),
        high=np.transpose(observation_space.high, axes),
        dtype=observation_space.dtype,
    )


def maybe_transpose_obs(
    observation: torch.Tensor, *, channels_last: bool
) -> torch.Tensor:
    """Convert a batched image tensor to channels-first form when required."""

    if not channels_last:
        return observation

    if observation.dim() == 4:
        return observation.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW
    if observation.dim() == 5:
        return observation.permute(0, 1, 4, 2, 3).contiguous()  # BNHWC -> BNCHW
    raise ValueError(
        "Expected a batched HWC or NHWC image tensor, got "
        f"shape {tuple(observation.shape)}"
    )


# ==================
# Feature Extractors


class MjCambrianCombinedExtractor(BaseFeaturesExtractor):
    """Overwrite of the default feature extractor of Stable Baselines 3."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        normalized_image: bool,
        image_extractor: BaseFeaturesExtractor,
        share_image_extractor: bool = False,
    ) -> None:
        # We do not know features-dim here before going over all the items, so put
        # something there.
        super().__init__(observation_space, features_dim=1)

        self._image_extractor = None
        self._image_channels_last: Dict[str, bool] = {}
        if share_image_extractor:
            # Verify all the image spaces have the same shape
            image_space = None
            for subspace in observation_space.values():
                if is_image_space(subspace, normalized_image=normalized_image):
                    subspace = maybe_transpose_space(subspace)
                    if image_space is None:
                        image_space = subspace
                    assert image_space.shape == subspace.shape, (
                        "All the image spaces must have the same shape if "
                        + "using shared image extractor"
                    )
            assert image_space is not None, "There must be at least one image space"
            self._image_extractor = image_extractor(image_space)

        extractors: Dict[str, BaseFeaturesExtractor] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                self._image_channels_last[key] = _image_channels_last(subspace.shape)
                subspace = maybe_transpose_space(subspace)
                if share_image_extractor:
                    extractors[key] = self._image_extractor
                else:
                    extractors[key] = image_extractor(subspace)
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = FlattenExtractor(subspace)
            total_concat_size += extractors[key].features_dim

        self.extractors = torch.nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key in self._image_channels_last:
                obs = maybe_transpose_obs(
                    obs, channels_last=self._image_channels_last[key]
                )
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


class PermutedFlattenExtractor(FlattenExtractor):
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        flattened = super().forward(observations)
        perm = torch.randperm(flattened.size(-1))
        return flattened[:, perm]


class MjCambrianImageFeaturesExtractor(BaseFeaturesExtractor):
    """Base class for image feature extractors."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: torch.nn.Module,
    ):
        self._leading_dim = 1
        if len(observation_space.shape) == 4:
            self._leading_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim)

        if len(observation_space.shape) == 4:
            _, n_channels, height, width = observation_space.shape
        elif len(observation_space.shape) == 3:
            n_channels, height, width = observation_space.shape
        else:
            raise ValueError(
                "Expected a CHW or NCHW image space, got "
                f"shape {observation_space.shape}"
            )
        if n_channels <= 0 or height <= 0 or width <= 0:
            raise ValueError(
                f"Invalid image shape {observation_space.shape}: all axes must be > 0"
            )
        self._num_pixels = n_channels * height * width
        if self._leading_dim > 1:
            self.temporal_linear = torch.nn.Sequential(
                torch.nn.Linear(features_dim * self._leading_dim, features_dim),
                activation(),
            )
        else:
            self.temporal_linear = torch.nn.Identity()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.temporal_linear(observations)


class MjCambrianMLPExtractor(MjCambrianImageFeaturesExtractor):
    """MLP feature extractor for small images. Essentially NatureCNN but with MLPs."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: torch.nn.Module,
        architecture: List[int],
    ) -> None:
        super().__init__(observation_space, features_dim, activation)

        layers = []
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(self._num_pixels, architecture[0]))
        layers.append(activation())
        for i in range(1, len(architecture)):
            layers.append(torch.nn.Linear(architecture[i - 1], architecture[i]))
            layers.append(activation())
        layers.append(torch.nn.Linear(architecture[-1], features_dim))
        layers.append(activation())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        observations = observations.reshape(-1, self._num_pixels)  # [B, C * H * W]
        encodings = self.mlp(observations)
        encodings = encodings.reshape(B, -1)

        return super().forward(encodings)


class MjCambrianSmallCNNExtractor(MjCambrianImageFeaturesExtractor):
    """Small CNN with global average pooling.

    This extractor is retained for historical compatibility.  Its terminal
    ``AdaptiveAvgPool2d(1, 1)`` discards explicit within-eye spatial layout, so it
    should not be used as the localization encoder for the binocular tracking
    baselines.  Use :class:`MjCambrianSpatialCNNExtractor` instead.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: torch.nn.Module,
        channels: List[int] = [16, 32],
        kernel_sizes: List[int] = [5, 3],
        strides: List[int] = [2, 1],
        **_,
    ) -> None:
        super().__init__(observation_space, features_dim, activation)

        assert len(channels) > 0, "SmallCNN requires at least one channel size."
        assert len(channels) == len(kernel_sizes) == len(strides), (
            "channels, kernel_sizes, and strides must have the same length."
        )

        if len(observation_space.shape) == 4:
            _, n_channels, height, width = observation_space.shape
        else:
            n_channels, height, width = observation_space.shape

        layers = []
        in_channels = n_channels
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
            kernel_size = min(kernel_size, height, width)
            layers.append(
                torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(activation())
            in_channels = out_channels
        layers.extend([torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten()])
        self.cnn = torch.nn.Sequential(*layers)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(channels[-1], features_dim),
            activation(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        if observations.dim() == 5:
            observations = observations.reshape(-1, *observations.shape[2:])
        elif observations.dim() != 4:
            raise ValueError(
                "Expected image observations with shape [B,C,H,W] or [B,T,C,H,W], "
                f"got {tuple(observations.shape)}"
            )
        encodings = self.linear(self.cnn(observations))
        encodings = encodings.reshape(B, -1)
        return super().forward(encodings)


class MjCambrianSpatialCNNExtractor(MjCambrianImageFeaturesExtractor):
    """CNN extractor that preserves spatial retinal layout before projection."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: torch.nn.Module,
        channels: List[int] = [16, 32, 64],
        kernel_sizes: List[int] = [5, 3, 3],
        strides: List[int] = [2, 1, 1],
        hidden_dim: int = 128,
        **_,
    ) -> None:
        super().__init__(observation_space, features_dim, activation)

        assert len(channels) > 0, "SpatialCNN requires at least one channel size."
        assert len(channels) == len(kernel_sizes) == len(strides), (
            "channels, kernel_sizes, and strides must have the same length."
        )

        if len(observation_space.shape) == 4:
            _, n_channels, height, width = observation_space.shape
        else:
            n_channels, height, width = observation_space.shape

        layers = []
        in_channels = n_channels
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
            kernel_size = min(kernel_size, height, width)
            layers.append(
                torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(activation())
            in_channels = out_channels
        layers.append(torch.nn.Flatten())
        self.cnn = torch.nn.Sequential(*layers)

        with torch.no_grad():
            sample = torch.zeros(1, n_channels, height, width)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, features_dim),
            activation(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        if observations.dim() == 5:
            observations = observations.reshape(-1, *observations.shape[2:])
        elif observations.dim() != 4:
            raise ValueError(
                "Expected image observations with shape [B,C,H,W] or [B,T,C,H,W], "
                f"got {tuple(observations.shape)}"
            )
        encodings = self.linear(self.cnn(observations))
        encodings = encodings.reshape(B, -1)
        return super().forward(encodings)


class MjCambrianCoordCNNExtractor(MjCambrianImageFeaturesExtractor):
    """CNN extractor that appends normalized x/y coordinate channels.

    Unlike ``MjCambrianSmallCNNExtractor``, this preserves spatial layout until the
    final projection. It is a lightweight first step toward a binocular encoder that
    can reason about target position within each retina.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: torch.nn.Module,
        channels: List[int] = [16, 32, 64],
        kernel_sizes: List[int] = [5, 3, 3],
        strides: List[int] = [2, 1, 1],
        hidden_dim: int = 128,
        **_,
    ) -> None:
        super().__init__(observation_space, features_dim, activation)

        assert len(channels) > 0, "CoordCNN requires at least one channel size."
        assert len(channels) == len(kernel_sizes) == len(strides), (
            "channels, kernel_sizes, and strides must have the same length."
        )

        if len(observation_space.shape) == 4:
            _, n_channels, height, width = observation_space.shape
        else:
            n_channels, height, width = observation_space.shape

        layers = []
        in_channels = n_channels + 2
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
            kernel_size = min(kernel_size, height, width)
            layers.append(
                torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(activation())
            in_channels = out_channels
        layers.append(torch.nn.Flatten())
        self.cnn = torch.nn.Sequential(*layers)

        with torch.no_grad():
            sample = torch.zeros(1, n_channels + 2, height, width)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, features_dim),
            activation(),
        )

    @staticmethod
    def _coord_channels(
        batch_size: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=0).unsqueeze(0)
        return coords.expand(batch_size, -1, -1, -1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        if observations.dim() == 5:
            observations = observations.reshape(-1, *observations.shape[2:])
        elif observations.dim() != 4:
            raise ValueError(
                "Expected image observations with shape [B,C,H,W] or [B,T,C,H,W], "
                f"got {tuple(observations.shape)}"
            )

        _, _, height, width = observations.shape
        coords = self._coord_channels(
            observations.shape[0],
            height,
            width,
            device=observations.device,
            dtype=observations.dtype,
        )
        observations = torch.cat((observations, coords), dim=1)
        encodings = self.linear(self.cnn(observations))
        encodings = encodings.reshape(B, -1)
        return super().forward(encodings)


class MjCambrianNatureCNNExtractor(MjCambrianImageFeaturesExtractor):
    """Nature CNN feature extractor for images. This is the default feature extractor
    for stable baseline3 images. The main differences between this and the original
    is that this supports temporal features (i.e. image stacks) and dynamically
    calculates the kernel sizes and strides. In sb3, the fixed kernel sizes and strides
    restricted the image size to be > 36x36, which is a bd assumption here."""

    def __init__(
        self, observation_space: gym.Space, features_dim, activation: torch.nn.Module
    ):
        super().__init__(observation_space, features_dim, activation)

        if len(observation_space.shape) == 4:
            _, n_channels, height, width = observation_space.shape
        elif len(observation_space.shape) == 3:
            n_channels, height, width = observation_space.shape
        else:
            raise ValueError(
                "Expected a CHW or NCHW image space, got "
                f"shape {observation_space.shape}"
            )

        # Dynamically calculate kernel sizes and strides
        k_sizes, strides = self.calculate_dynamic_params(width, height)

        # Create CNN layers
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, 32, kernel_size=k_sizes[0], stride=strides[0]),
            activation(),
            torch.nn.Conv2d(32, 64, kernel_size=k_sizes[1], stride=strides[1]),
            activation(),
            torch.nn.Conv2d(64, 64, kernel_size=k_sizes[2], stride=strides[2]),
            activation(),
            torch.nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, height, width)
            n_flatten = self.cnn(sample).shape[1]

        # Encode each frame independently, then let ``temporal_linear`` combine
        # the resulting T feature vectors. Multiplying the output width by T here
        # creates T^2 features and breaks the frame-stacked path.
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim), activation()
        )

    def calculate_dynamic_params(self, width, height):
        # Define max sizes and strides (from sb3, i.e. if width x height > 36x36, it's
        # the same).
        max_kernel_sizes = [8, 4, 2]
        max_strides = [4, 2, 1]

        # Adjust kernel sizes and strides based on input dimensions
        kernel_sizes = [min(k, height, width) for k in max_kernel_sizes]
        strides = [
            min(s, height // k, width // k) for s, k in zip(max_strides, kernel_sizes)
        ]

        return kernel_sizes, strides

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        if observations.dim() == 5:
            observations = observations.reshape(-1, *observations.shape[2:])
        elif observations.dim() != 4:
            raise ValueError(
                "Expected image observations with shape [B,C,H,W] or [B,T,C,H,W], "
                f"got {tuple(observations.shape)}"
            )
        observations = self.linear(self.cnn(observations))
        observations = observations.reshape(B, -1)
        return super().forward(observations)
