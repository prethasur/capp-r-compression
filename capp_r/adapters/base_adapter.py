"""Base interface for architecture-specific protected semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SemanticDecision:
    protected: bool
    reason: str = ""
    region_kind: str = "generic"


class ArchitectureAdapter:
    family = "generic"
    native_postprocessing = "architecture_native"

    def unwrap(self, model: Any) -> Any:
        try:
            import torch.nn as nn
            if isinstance(model, nn.Module):
                return model
        except ImportError:
            pass
        return getattr(model, "model", model)

    def protected_semantics(self, name: str, module: Any) -> SemanticDecision:
        del name, module
        return SemanticDecision(False)

    def output_semantics(self, model: Any) -> dict[str, Any]:
        del model
        return {"family": self.family, "native_postprocessing": self.native_postprocessing}

    @staticmethod
    def module_type(module: Any) -> str:
        return f"{module.__class__.__module__}.{module.__class__.__name__}"
