"""Exact boundary-contract checks for synthesized replacements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ContractValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Boundary:
    shape: tuple[int, ...]
    dtype: str
    stride: tuple[float, ...]


def _boundary(value: Any, input_size: tuple[int, int] = (640, 640)) -> Boundary:
    shape = tuple(int(item) for item in value.shape)
    dtype = str(value.dtype)
    if len(shape) >= 4:
        stride = (input_size[0] / shape[-2], input_size[1] / shape[-1])
    else:
        stride = ()
    return Boundary(shape, dtype, stride)


def validate_boundary(source_output: Any, replacement_output: Any,
                      input_size: tuple[int, int] = (640, 640)) -> Boundary:
    source = _boundary(source_output, input_size)
    replacement = _boundary(replacement_output, input_size)
    if source != replacement:
        raise ContractValidationError(f"boundary mismatch: source={source}, replacement={replacement}")
    return replacement
