"""Custom feature extractors used by the PPO policies.

Cambrian eye observations are emitted in channels-last form (``H, W, C``), while
``torch.nn.Conv2d`` and ``torch.nn.Conv3d`` require channels-first tensors.
Stable Baselines normally inserts a transpose wrapper for uint8 images, but Cambrian's
retinal observations are normalized float tensors.  Consequently, layout handling
must be explicit here rather than delegated to SB3.
"""

from typing import Dict, List, Sequence, Tuple

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


class _R2Plus1DResidualBlock(torch.nn.Module):
    """Lightweight residual video block with factorized space/time filtering.

    The spatial and temporal operations are separated so that an activation lies
    between them.  This is cheaper than a monolithic 3-D convolution and makes the
    temporal operation explicit.  GroupNorm is used instead of BatchNorm because PPO
    minibatches are correlated and non-stationary, and because train/eval behavior
    should not depend on running batch statistics.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm_groups: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
    ) -> None:
        super().__init__()

        if spatial_stride < 1 or temporal_stride < 1:
            raise ValueError("Residual-block strides must be positive integers.")
        for channels in (in_channels, out_channels):
            if channels % norm_groups != 0:
                raise ValueError(
                    f"norm_groups={norm_groups} must divide channels={channels}."
                )

        self.norm1 = torch.nn.GroupNorm(norm_groups, in_channels)
        self.spatial = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, spatial_stride, spatial_stride),
            padding=(0, 1, 1),
            bias=False,
        )
        self.norm2 = torch.nn.GroupNorm(norm_groups, out_channels)
        self.temporal = torch.nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            stride=(temporal_stride, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        )

        if (
            in_channels != out_channels
            or spatial_stride != 1
            or temporal_stride != 1
        ):
            self.projection = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
            )
        else:
            self.projection = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        x = torch.nn.functional.silu(self.norm1(x))
        x = self.spatial(x)
        x = torch.nn.functional.silu(self.norm2(x))
        x = self.temporal(x)
        return x + residual


class MjCambrianR2Plus1DExtractor(BaseFeaturesExtractor):
    """Shared-eye spatiotemporal retinal encoder for frame-stacked PPO.

    Input observations must be frame-stacked in ``[T, C, H, W]`` form after the
    combined extractor performs the channels-last conversion.  The forward pass
    receives ``[B, T, C, H, W]`` and explicitly permutes it to the Conv3d layout
    ``[B, C, T, H, W]``.  Temporal adjacency is therefore preserved rather than
    flattening the frame axis into the batch axis.

    The encoder uses residual R(2+1)D blocks, GroupNorm, and SiLU.  It performs a
    learned temporal collapse and a learned spatial projection, then flattens the
    remaining retinotopic map without global pooling.  When used with
    ``MjCambrianCombinedExtractor(..., share_image_extractor=True)``, the same module
    and weights encode both eyes, while the two flattened outputs remain separate in
    the combined feature vector.

    ``code_mode`` controls only the final retinal code:

    - ``signed_silu``: signed, smooth code used for the initial architecture test.
    - ``identity``: unconstrained signed linear code.
    - ``on_off_relu``: non-negative ON/OFF split with a fixed total channel count.
    """

    _VALID_CODE_MODES = {"signed_silu", "identity", "on_off_relu"}

    def __init__(
        self,
        observation_space: gym.Space,
        *,
        stem_channels: int = 24,
        block_channels: Sequence[int] = (32, 48, 48),
        temporal_channels: int = 32,
        latent_channels: int = 24,
        norm_groups: int = 8,
        code_mode: str = "signed_silu",
    ) -> None:
        if len(observation_space.shape) != 4:
            raise ValueError(
                "MjCambrianR2Plus1DExtractor requires a frame-stacked image space "
                "with shape [T,C,H,W]. Keep the frame-stack wrapper enabled. Got "
                f"shape {observation_space.shape}."
            )

        n_frames, n_channels, height, width = observation_space.shape
        if min(n_frames, n_channels, height, width) <= 0:
            raise ValueError(
                f"Invalid stacked image shape {observation_space.shape}: all axes "
                "must be positive."
            )
        if n_frames < 3:
            raise ValueError(
                "The spatiotemporal encoder requires at least three stacked frames; "
                f"got {n_frames}."
            )
        if len(block_channels) != 3:
            raise ValueError(
                "block_channels must contain exactly three channel sizes: one "
                "spatial-downsampling block, one temporal-downsampling block, and "
                "one refinement block."
            )
        if code_mode not in self._VALID_CODE_MODES:
            raise ValueError(
                f"Unknown code_mode={code_mode!r}; expected one of "
                f"{sorted(self._VALID_CODE_MODES)}."
            )
        if code_mode == "on_off_relu" and latent_channels % 2 != 0:
            raise ValueError(
                "latent_channels must be even when code_mode='on_off_relu' so the "
                "ON and OFF populations have equal size."
            )

        if norm_groups <= 0:
            raise ValueError("norm_groups must be positive.")

        normalized_channels = [
            stem_channels,
            *block_channels,
            temporal_channels,
        ]
        for channels in normalized_channels:
            if channels <= 0:
                raise ValueError("All encoder channel counts must be positive.")
            if channels % norm_groups != 0:
                raise ValueError(
                    f"norm_groups={norm_groups} must divide channels={channels}."
                )
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive.")

        # The precise flattened size is inferred from the actual network below.
        super().__init__(observation_space, features_dim=1)

        self._input_shape = tuple(int(v) for v in observation_space.shape)
        self.code_mode = code_mode

        self.spatial_stem = torch.nn.Sequential(
            torch.nn.Conv3d(
                n_channels,
                stem_channels,
                kernel_size=(1, 5, 5),
                stride=(1, 2, 2),
                padding=(0, 2, 2),
                bias=False,
            ),
            torch.nn.GroupNorm(norm_groups, stem_channels),
            torch.nn.SiLU(),
        )
        self.temporal_stem = torch.nn.Sequential(
            torch.nn.Conv3d(
                stem_channels,
                stem_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=(1, 0, 0),
                bias=False,
            ),
            torch.nn.GroupNorm(norm_groups, stem_channels),
            torch.nn.SiLU(),
        )

        c0, c1, c2 = (int(v) for v in block_channels)
        self.residual_blocks = torch.nn.Sequential(
            _R2Plus1DResidualBlock(
                stem_channels,
                c0,
                norm_groups=norm_groups,
                spatial_stride=2,
            ),
            _R2Plus1DResidualBlock(
                c0,
                c1,
                norm_groups=norm_groups,
                temporal_stride=2,
            ),
            _R2Plus1DResidualBlock(
                c1,
                c2,
                norm_groups=norm_groups,
            ),
        )
        self.pre_collapse = torch.nn.Sequential(
            torch.nn.GroupNorm(norm_groups, c2),
            torch.nn.SiLU(),
        )

        # Infer the remaining temporal extent and collapse it with a learned kernel.
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, n_frames, height, width)
            sample = self._forward_trunk(sample)
            remaining_frames = int(sample.shape[2])
        if remaining_frames <= 0:
            raise ValueError(
                "The configured encoder collapsed the temporal axis before the "
                "learned temporal projection."
            )

        self.temporal_collapse = torch.nn.Sequential(
            torch.nn.Conv3d(
                c2,
                temporal_channels,
                kernel_size=(remaining_frames, 1, 1),
                bias=False,
            ),
            torch.nn.GroupNorm(norm_groups, temporal_channels),
            torch.nn.SiLU(),
        )

        signed_channels = (
            latent_channels // 2 if code_mode == "on_off_relu" else latent_channels
        )
        self.spatial_projection = torch.nn.Conv2d(
            temporal_channels,
            signed_channels,
            kernel_size=(2, 3),
            stride=1,
            padding=0,
        )

        with torch.no_grad():
            sample = torch.zeros(1, n_frames, n_channels, height, width)
            retinal_code = self.encode_retinal_map(sample)
        if retinal_code.shape[-2] <= 0 or retinal_code.shape[-1] <= 0:
            raise ValueError(
                "The configured encoder produced a non-positive spatial output size."
            )

        self.retinal_code_shape = tuple(int(v) for v in retinal_code.shape[1:])
        self._features_dim = int(retinal_code[0].numel())

    def _forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_stem(x)
        x = self.temporal_stem(x)
        x = self.residual_blocks(x)
        return self.pre_collapse(x)

    def encode_retinal_map(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the unflattened retinal code as ``[B,C,H,W]``."""
        if observations.dim() != 5:
            raise ValueError(
                "Expected stacked retinal observations [B,T,C,H,W], got "
                f"shape {tuple(observations.shape)}."
            )
        if tuple(observations.shape[1:]) != self._input_shape:
            raise ValueError(
                "Stacked retinal observation shape changed after initialization: "
                f"expected [B,{','.join(map(str, self._input_shape))}], got "
                f"{tuple(observations.shape)}."
            )

        # CombinedExtractor yields [B,T,C,H,W]; Conv3d requires [B,C,T,H,W].
        x = observations.permute(0, 2, 1, 3, 4).contiguous()
        x = self._forward_trunk(x)
        x = self.temporal_collapse(x)
        if x.shape[2] != 1:
            raise RuntimeError(
                "Learned temporal collapse must produce exactly one time slice; "
                f"got shape {tuple(x.shape)}."
            )
        x = self.spatial_projection(x.squeeze(2))

        if self.code_mode == "signed_silu":
            return torch.nn.functional.silu(x)
        if self.code_mode == "identity":
            return x
        # Split every signed feature into separate non-negative ON and OFF channels.
        return torch.cat((torch.relu(x), torch.relu(-x)), dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.encode_retinal_map(observations), start_dim=1)


class MjCambrianCyclopeanR2Plus1DExtractor(BaseFeaturesExtractor):
    """Shared-eye retinal encoding followed by motor-conditioned binocular fusion.

    This extractor consumes the complete frame-stacked ``Dict`` observation.  One
    shared R(2+1)D encoder maps the two eye clips to separate retinotopic feature maps.
    The stacked action/efference-copy, eye-state, contact, and other vector observations
    are encoded into a motor context.  That context FiLM-modulates each eye map before
    a learned MLP constructs one cyclopean visual-motor representation.

    The default tracking geometry yields one ``[24, 4, 6]`` map per eye.  Eye identity
    and retinal position remain explicit until the fusion layer; the two maps are never
    averaged or globally pooled.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        normalized_image: bool = True,
        stem_channels: int = 24,
        block_channels: Sequence[int] = (32, 48, 48),
        temporal_channels: int = 32,
        latent_channels: int = 24,
        norm_groups: int = 8,
        code_mode: str = "signed_silu",
        motor_hidden_dim: int = 128,
        motor_context_dim: int = 64,
        fusion_hidden_dim: int = 256,
        cyclopean_dim: int = 192,
        film_conditioning: bool = True,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                "MjCambrianCyclopeanR2Plus1DExtractor requires a Dict observation "
                f"space, got {type(observation_space).__name__}."
            )
        for name, value in {
            "motor_hidden_dim": motor_hidden_dim,
            "motor_context_dim": motor_context_dim,
            "fusion_hidden_dim": fusion_hidden_dim,
            "cyclopean_dim": cyclopean_dim,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        image_keys = [
            key
            for key, subspace in observation_space.spaces.items()
            if is_image_space(subspace, normalized_image=normalized_image)
        ]
        if len(image_keys) != 2:
            raise ValueError(
                "Cyclopean fusion requires exactly two retinal image keys; found "
                f"{len(image_keys)}: {image_keys}."
            )

        # Sorting makes left/right assignment deterministic while preserving identity.
        self.image_keys: Tuple[str, str] = tuple(  # type: ignore[assignment]
            sorted(image_keys)
        )
        self.motor_keys: Tuple[str, ...] = tuple(
            sorted(key for key in observation_space.spaces if key not in image_keys)
        )
        if len(self.motor_keys) == 0:
            raise ValueError(
                "Cyclopean fusion requires vector observations carrying efference "
                "copy and/or proprioception."
            )

        self._image_channels_last = {
            key: _image_channels_last(observation_space[key].shape)
            for key in self.image_keys
        }
        retinal_space = maybe_transpose_space(observation_space[self.image_keys[0]])
        for key in self.image_keys[1:]:
            candidate = maybe_transpose_space(observation_space[key])
            if candidate.shape != retinal_space.shape:
                raise ValueError(
                    "Both eyes must have identical stacked image shapes for a shared "
                    f"encoder; got {retinal_space.shape} and {candidate.shape}."
                )

        super().__init__(observation_space, features_dim=cyclopean_dim)

        self.retinal_encoder = MjCambrianR2Plus1DExtractor(
            retinal_space,
            stem_channels=stem_channels,
            block_channels=block_channels,
            temporal_channels=temporal_channels,
            latent_channels=latent_channels,
            norm_groups=norm_groups,
            code_mode=code_mode,
        )
        self.retinal_code_shape = self.retinal_encoder.retinal_code_shape
        retinal_channels = self.retinal_code_shape[0]
        per_eye_features = int(np.prod(self.retinal_code_shape))

        self.motor_input_dim = sum(
            int(np.prod(observation_space[key].shape)) for key in self.motor_keys
        )
        self.motor_context_dim = motor_context_dim
        self.motor_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.motor_input_dim, motor_hidden_dim),
            torch.nn.LayerNorm(motor_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(motor_hidden_dim, motor_context_dim),
            torch.nn.LayerNorm(motor_context_dim),
            torch.nn.SiLU(),
        )

        self.film_conditioning = film_conditioning
        if film_conditioning:
            # Two eyes x (gain, bias) x C channels. Zero initialization makes the
            # initial transformation exactly identity while gradients can learn how
            # eye/body motion changes retinal coordinates.
            self.eye_film: torch.nn.Module | None = torch.nn.Linear(
                motor_context_dim, 4 * retinal_channels
            )
            torch.nn.init.zeros_(self.eye_film.weight)
            torch.nn.init.zeros_(self.eye_film.bias)
        else:
            self.eye_film = None

        fusion_input_dim = 2 * per_eye_features + motor_context_dim
        self.cyclopean_fusion = torch.nn.Sequential(
            torch.nn.Linear(fusion_input_dim, fusion_hidden_dim),
            torch.nn.LayerNorm(fusion_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(fusion_hidden_dim, cyclopean_dim),
            torch.nn.LayerNorm(cyclopean_dim),
            torch.nn.SiLU(),
        )

    def _encode_motor_context(
        self,
        observations: TensorDict,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        motor_parts = []
        for key in self.motor_keys:
            value = observations[key].to(device=device, dtype=dtype)
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"Observation key {key!r} has batch size {value.shape[0]}, "
                    f"expected {batch_size}."
                )
            motor_parts.append(torch.flatten(value, start_dim=1))
        motor_input = torch.cat(motor_parts, dim=1)
        if motor_input.shape[1] != self.motor_input_dim:
            raise ValueError(
                "Non-image observation shape changed after initialization: expected "
                f"{self.motor_input_dim} flattened values, got {motor_input.shape[1]}."
            )
        return self.motor_encoder(motor_input)

    def encode_components(
        self, observations: TensorDict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return eye maps, motor context, and fused cyclopean code.

        Returns:
            conditioned_maps: ``[B, 2, C, H, W]`` in deterministic eye-key order.
            motor_context: ``[B, M]``.
            cyclopean_code: ``[B, D]``.
        """

        eye_batches = []
        batch_size: int | None = None
        for key in self.image_keys:
            eye = observations[key]
            if batch_size is None:
                batch_size = int(eye.shape[0])
            elif eye.shape[0] != batch_size:
                raise ValueError("Both retinal observations must share a batch size.")
            eye_batches.append(
                maybe_transpose_obs(
                    eye,
                    channels_last=self._image_channels_last[key],
                )
            )
        assert batch_size is not None

        # One encoder call guarantees exact weight sharing and efficient GPU use.
        retinal_maps = self.retinal_encoder.encode_retinal_map(
            torch.cat(eye_batches, dim=0)
        )
        left_map, right_map = retinal_maps.chunk(2, dim=0)
        conditioned_maps = torch.stack((left_map, right_map), dim=1)

        motor_context = self._encode_motor_context(
            observations,
            batch_size=batch_size,
            device=conditioned_maps.device,
            dtype=conditioned_maps.dtype,
        )

        if self.eye_film is not None:
            channels = conditioned_maps.shape[2]
            film = self.eye_film(motor_context).reshape(batch_size, 2, 2, channels)
            # Bounded modulation avoids an unstable scale explosion. At initialization
            # gain=1 and bias=0 exactly.
            gain = 1.0 + torch.tanh(film[:, :, 0]).unsqueeze(-1).unsqueeze(-1)
            bias = torch.tanh(film[:, :, 1]).unsqueeze(-1).unsqueeze(-1)
            conditioned_maps = gain * conditioned_maps + bias

        fusion_input = torch.cat(
            (torch.flatten(conditioned_maps, start_dim=1), motor_context), dim=1
        )
        cyclopean_code = self.cyclopean_fusion(fusion_input)
        return conditioned_maps, motor_context, cyclopean_code

    def forward(self, observations: TensorDict) -> torch.Tensor:
        return self.encode_components(observations)[2]


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
