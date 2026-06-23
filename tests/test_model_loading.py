from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cambrian.ml.model import _MjCambrianModelMixin


class _DummyModel:
    load_policy = _MjCambrianModelMixin.load_policy

    def __init__(self, policy: torch.nn.Module):
        self.policy = policy
        self.device = torch.device("cpu")


def _save_policy(path: Path, policy: torch.nn.Module) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), path / "policy.pt")


def test_load_policy_rejects_shape_mismatch_by_default(tmp_path: Path) -> None:
    _save_policy(tmp_path, torch.nn.Linear(3, 2))
    model = _DummyModel(torch.nn.Linear(4, 2))

    with pytest.raises(RuntimeError, match="Shape mismatches"):
        model.load_policy(tmp_path)


def test_load_policy_allows_explicit_partial_transfer(tmp_path: Path) -> None:
    source = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    target = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.Linear(2, 1))
    _save_policy(tmp_path, source)

    target_second_before = target[1].weight.detach().clone()
    model = _DummyModel(target)
    model.load_policy(tmp_path, allow_partial=True)

    # The incompatible first layer is retained, while the compatible second layer is
    # transferred. Partial loading is therefore explicit and testable rather than
    # silently producing a partly random evaluation policy.
    assert target[0].weight.shape == (2, 4)
    assert not torch.equal(target[1].weight, target_second_before)
    torch.testing.assert_close(target[1].weight, source[1].weight)
