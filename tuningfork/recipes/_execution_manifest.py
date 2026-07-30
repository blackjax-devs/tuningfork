"""Immutable manifest for a resolved execution plan.

The manifest is the small, serialisable contract shared by code generation and
the later launcher.  It records plan identity only; execution receipts and
emitted-source digests belong to later lifecycle stages.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._execution_plan import (
    ExecutionPlan,
    _freeze,
    _thaw,
    canonical_json,
    execution_config_hash,
    execution_plan_hash,
)

MANIFEST_VERSION = "tuningfork.execution-manifest.v1"
DEFAULT_GENERATOR_CONTRACT = "tuningfork.execution-plan.v1"


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ExecutionManifest:
    """Versioned, immutable identity for one resolved :class:`ExecutionPlan`."""

    manifest_version: str
    generator_contract: str
    generator_version: str
    recipe_ref: str
    executable_config: Mapping[str, Any]
    normalized_plan: Mapping[str, Any]
    executable_config_hash: str
    plan_hash: str

    @classmethod
    def from_plan(
        cls,
        plan: ExecutionPlan,
        *,
        generator_version: str,
        generator_contract: str = DEFAULT_GENERATOR_CONTRACT,
    ) -> ExecutionManifest:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be an ExecutionPlan")
        generator_version = _require_text("generator_version", generator_version)
        generator_contract = _require_text("generator_contract", generator_contract)
        if generator_contract != DEFAULT_GENERATOR_CONTRACT:
            raise ValueError(f"unsupported generator_contract: {generator_contract!r}")
        executable_config = plan.config.as_dict()
        normalized_plan = plan.as_dict()
        # Recompute rather than trusting fields on the plan object, so the
        # manifest always describes the values it serialises.
        config_hash = execution_config_hash(executable_config)
        plan_hash = execution_plan_hash(executable_config, plan.artifact_filename)
        if config_hash != plan.executable_config_hash or plan_hash != plan.plan_hash:
            raise ValueError("execution plan hashes do not match its contents")
        return cls(
            MANIFEST_VERSION,
            generator_contract,
            generator_version,
            plan.recipe_ref,
            _freeze(executable_config),
            _freeze(normalized_plan),
            config_hash,
            plan_hash,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "generator_contract": self.generator_contract,
            "generator_version": self.generator_version,
            "recipe_ref": self.recipe_ref,
            "executable_config": _thaw(self.executable_config),
            "normalized_plan": _thaw(self.normalized_plan),
            "executable_config_hash": self.executable_config_hash,
            "plan_hash": self.plan_hash,
        }

    def to_json(self) -> str:
        """Return strict, deterministic JSON for embedding or persistence."""
        return canonical_json(self.as_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionManifest:
        """Load a v1 manifest and reject shape, identity, or hash tampering."""
        if not isinstance(data, Mapping):
            raise TypeError("manifest data must be a mapping")
        required = {
            "manifest_version",
            "generator_contract",
            "generator_version",
            "recipe_ref",
            "executable_config",
            "normalized_plan",
            "executable_config_hash",
            "plan_hash",
        }
        unknown = set(data).difference(required)
        if unknown:
            raise ValueError(f"manifest has unsupported fields: {sorted(unknown)!r}")
        missing = required.difference(data)
        if missing:
            raise ValueError(f"manifest is missing fields: {sorted(missing)!r}")
        canonical_json(data)
        version = _require_text("manifest_version", data["manifest_version"])
        if version != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest_version: {version!r}")
        contract = _require_text("generator_contract", data["generator_contract"])
        if contract != DEFAULT_GENERATOR_CONTRACT:
            raise ValueError(f"unsupported generator_contract: {contract!r}")
        generator = _require_text("generator_version", data["generator_version"])
        recipe_ref = _require_text("recipe_ref", data["recipe_ref"])
        config = data["executable_config"]
        normalized = data["normalized_plan"]
        if not isinstance(config, Mapping) or not isinstance(normalized, Mapping):
            raise ValueError(
                "manifest executable_config and normalized_plan must be mappings"
            )
        if set(normalized) != {"config", "recipe_ref", "artifact_filename"}:
            raise ValueError("normalized_plan has unsupported or missing fields")
        if normalized.get("config") != dict(config):
            raise ValueError("normalized_plan config does not match executable_config")
        if normalized.get("recipe_ref") != recipe_ref:
            raise ValueError("normalized_plan recipe_ref does not match recipe_ref")
        artifact = normalized.get("artifact_filename")
        if not isinstance(artifact, str):
            raise ValueError("normalized_plan must contain artifact_filename")
        config_hash = execution_config_hash(config)
        plan_hash = execution_plan_hash(config, artifact)
        if data["executable_config_hash"] != config_hash:
            raise ValueError("executable_config_hash does not match executable_config")
        if data["plan_hash"] != plan_hash:
            raise ValueError("plan_hash does not match normalized_plan")
        return cls(
            version,
            contract,
            generator,
            recipe_ref,
            _freeze(dict(config)),
            _freeze(dict(normalized)),
            config_hash,
            plan_hash,
        )


__all__ = [
    "DEFAULT_GENERATOR_CONTRACT",
    "MANIFEST_VERSION",
    "ExecutionManifest",
]
