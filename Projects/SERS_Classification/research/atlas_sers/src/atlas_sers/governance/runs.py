"""Deterministic identities for provisional and definitive experiment runs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace

from atlas_sers.governance.canonical import sha256_value

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RunIdentity:
    protocol_version: str
    experiment_id: str
    task_id: str
    information_regime: str
    outer_repeat: int | str
    outer_fold: int | str
    held_domain: str
    population_id: str
    representation_id: str
    model_id: str
    hyperparameter_sha256: str
    seed: int | str
    code_sha256: str
    config_sha256: str
    input_sha256: str

    def validate(self) -> None:
        state = asdict(self)
        empty = [key for key, value in state.items() if value == "" or value is None]
        if empty:
            raise ValueError(f"Run identity contains empty fields: {empty}")
        for field in (
            "hyperparameter_sha256",
            "code_sha256",
            "config_sha256",
            "input_sha256",
        ):
            if not SHA256_PATTERN.fullmatch(str(state[field])):
                raise ValueError(f"{field} is not a lowercase SHA-256 digest")

    def changed(self, field: str, value: object) -> RunIdentity:
        if field not in self.__dataclass_fields__:
            raise KeyError(field)
        return replace(self, **{field: value})


def protected_state_sha256(identity: RunIdentity) -> str:
    identity.validate()
    return sha256_value(asdict(identity))


def deterministic_run_id(identity: RunIdentity, *, prefix: str = "RUN") -> str:
    digest = protected_state_sha256(identity)
    return f"{prefix}-{digest[:24]}"
