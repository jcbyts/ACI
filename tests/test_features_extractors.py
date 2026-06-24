from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
from gymnasium import spaces

from cambrian.ml.features_extractors import (
    MjCambrianCombinedExtractor,
    MjCambrianR2Plus1DExtractor,
    MjCambrianSpatialCNNExtractor,
    _image_channels_last,
    maybe_transpose_space,
)


def _spatial_extractor(features_dim: int = 16):
    return partial(
        MjCambrianSpatialCNNExtractor,
        features_dim=features_dim,
        activation=torch.nn.ReLU,
        channels=[8, 8],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        hidden_dim=32,
    )


def test_hwc_retina_is_transposed_to_rgb_channels_first() -> None:
    observation_space = spaces.Dict(
        {
            "eye0": spaces.Box(0.0, 1.0, shape=(20, 30, 3), dtype=np.float32),
            "eye1": spaces.Box(0.0, 1.0, shape=(20, 30, 3), dtype=np.float32),
            "action": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
        }
    )
    extractor = MjCambrianCombinedExtractor(
        observation_space,
        normalized_image=True,
        image_extractor=_spatial_extractor(features_dim=16),
        share_image_extractor=True,
    )

    first_conv = extractor._image_extractor.cnn[0]
    assert isinstance(first_conv, torch.nn.Conv2d)
    assert first_conv.in_channels == 3
    assert maybe_transpose_space(observation_space["eye0"]).shape == (3, 20, 30)

    batch_size = 4
    output = extractor(
        {
            "eye0": torch.rand(batch_size, 20, 30, 3),
            "eye1": torch.rand(batch_size, 20, 30, 3),
            "action": torch.rand(batch_size, 6),
        }
    )
    assert output.shape == (batch_size, 16 + 16 + 6)


def test_nhwc_temporal_retina_encodes_each_frame_then_combines_time() -> None:
    observation_space = spaces.Dict(
        {
            "eye0": spaces.Box(
                0.0, 1.0, shape=(4, 20, 30, 3), dtype=np.float32
            )
        }
    )
    extractor = MjCambrianCombinedExtractor(
        observation_space,
        normalized_image=True,
        image_extractor=_spatial_extractor(features_dim=12),
        share_image_extractor=False,
    )
    image_extractor = extractor.extractors["eye0"]
    first_conv = image_extractor.cnn[0]
    assert first_conv.in_channels == 3
    assert maybe_transpose_space(observation_space["eye0"]).shape == (4, 3, 20, 30)
    assert image_extractor.temporal_linear[0].in_features == 4 * 12
    assert image_extractor.temporal_linear[0].out_features == 12

    output = extractor({"eye0": torch.rand(2, 4, 20, 30, 3)})
    assert output.shape == (2, 12)


def test_space_transpose_preserves_nonuniform_bounds() -> None:
    low = np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2)
    high = low + 100.0
    space = spaces.Box(low=low, high=high, dtype=np.float32)
    transposed = maybe_transpose_space(space)

    assert transposed.shape == (2, 4, 5)
    np.testing.assert_array_equal(transposed.low, np.transpose(low, (2, 0, 1)))
    np.testing.assert_array_equal(transposed.high, np.transpose(high, (2, 0, 1)))


def test_ambiguous_layout_fails_instead_of_guessing() -> None:
    with pytest.raises(ValueError, match="unambiguously infer image layout"):
        _image_channels_last((3, 3, 3))


def _spatiotemporal_extractor(code_mode: str = "signed_silu"):
    return partial(
        MjCambrianR2Plus1DExtractor,
        stem_channels=8,
        block_channels=[8, 16, 16],
        temporal_channels=16,
        latent_channels=8,
        norm_groups=4,
        code_mode=code_mode,
    )


def test_shared_r2plus1d_encoder_preserves_temporal_and_eye_structure() -> None:
    observation_space = spaces.Dict(
        {
            "eye0": spaces.Box(
                0.0, 1.0, shape=(10, 20, 30, 3), dtype=np.float32
            ),
            "eye1": spaces.Box(
                0.0, 1.0, shape=(10, 20, 30, 3), dtype=np.float32
            ),
            "action": spaces.Box(-1.0, 1.0, shape=(10, 6), dtype=np.float32),
        }
    )
    extractor = MjCambrianCombinedExtractor(
        observation_space,
        normalized_image=True,
        image_extractor=_spatiotemporal_extractor(),
        share_image_extractor=True,
    )

    image_extractor = extractor._image_extractor
    assert extractor.extractors["eye0"] is extractor.extractors["eye1"]
    assert image_extractor.spatial_stem[0].in_channels == 3
    assert image_extractor.temporal_stem[0].kernel_size == (3, 1, 1)
    assert image_extractor.retinal_code_shape == (8, 4, 6)
    assert maybe_transpose_space(observation_space["eye0"]).shape == (
        10,
        3,
        20,
        30,
    )

    batch_size = 2
    output = extractor(
        {
            "eye0": torch.rand(batch_size, 10, 20, 30, 3),
            "eye1": torch.rand(batch_size, 10, 20, 30, 3),
            "action": torch.rand(batch_size, 10, 6),
        }
    )
    per_eye_features = 8 * 4 * 6
    assert output.shape == (batch_size, 2 * per_eye_features + 10 * 6)


def test_r2plus1d_on_off_code_is_nonnegative() -> None:
    observation_space = spaces.Box(
        0.0, 1.0, shape=(10, 3, 20, 30), dtype=np.float32
    )
    extractor = _spatiotemporal_extractor(code_mode="on_off_relu")(
        observation_space
    )
    output = extractor(torch.rand(3, 10, 3, 20, 30))

    assert extractor.retinal_code_shape == (8, 4, 6)
    assert output.shape == (3, 8 * 4 * 6)
    assert torch.all(output >= 0)


def test_r2plus1d_rejects_unstacked_images() -> None:
    with pytest.raises(ValueError, match="frame-stacked"):
        _spatiotemporal_extractor()(
            spaces.Box(0.0, 1.0, shape=(3, 20, 30), dtype=np.float32)
        )
