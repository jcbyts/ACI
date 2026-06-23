from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
from gymnasium import spaces

from cambrian.ml.features_extractors import (
    MjCambrianCombinedExtractor,
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
