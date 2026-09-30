"""P08-T021: metadata-only QC plan catalog assembly.

This module assembles a machine-readable, execution-disabled catalog of future
QC jobs.  It performs no numerical preprocessing, fitting, routing, scoring or
permission granting, and it never touches the filesystem.

Boundary: ``_parse_contexts`` normalises compact universal-context metadata and
checks basic differences between hashes, but it does NOT establish concrete
calibration exclusions.  A caller must still obtain an authenticated universal
calibration-role audit before any runtime use.  The supplied
``gate_library_sha256`` binds claimed gate-library bytes; it does not
authenticate an external file.
"""

from __future__ import annotations

from .p08_plan import (
    _parse_actions,
    _parse_candidates,
    _parse_contexts,
    _parse_model_spec,
)
from .p08_qc_blocks import (
    FALLBACK_TARGET,
    canonical_sha256,
    make_alias,
    seal_catalog,
)
from .p08_qc_final import _build_final_blocks
from .p08_qc_policy import build_policy_blocks
from .p08_qc_roles import build_nested_qc_roles

_GENERIC_CODE = "invalid_qc_catalog_input"

_KNOWN_CODES = frozenset(
    (
        "gate_library_sha256_invalid",
        "context_id_set_mismatch",
        "selection_mode_mismatch",
        "outer_fit_hash_mismatch",
        "outer_test_hash_mismatch",
        "selection_unit_id_mismatch",
        "selection_unit_fit_hash_mismatch",
        "selection_unit_validation_hash_mismatch",
        "invalid_qc_catalog_input",
        "scientific_execution_not_authorized",
    )
)

_HEX_CHARS = frozenset("0123456789abcdef")

_ALIAS_PAIRS = (
    ("C-RBF-SVM", "C-RBF-SVM"),
    ("C-RANDOM-FOREST", "C-RANDOM-FOREST"),
    ("D0-M", "D0-M"),
    ("P05-SELECTED", None),
)


class QCPlanError(ValueError):
    """Sanitised catalog error carrying a finite, known code."""

    def __init__(self, code):
        safe = code if isinstance(code, str) and code in _KNOWN_CODES else _GENERIC_CODE
        self.code = safe
        super().__init__(safe)


def require_scientific_execution(*args, **kwargs):
    """Reserved hook: scientific execution is never authorised here."""
    raise QCPlanError("scientific_execution_not_authorized")


def _is_lower_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in _HEX_CHARS for ch in value)


def _hash_uids(uids):
    return canonical_sha256(sorted(uids))


def build_qc_catalog(
    contexts,
    roles,
    universal_contexts,
    candidates,
    actions,
    model_spec_sha256,
    gate_library_sha256,
):
    """Build a sealed, metadata-only catalog of future QC jobs."""
    try:
        if not _is_lower_sha256(gate_library_sha256):
            raise QCPlanError("gate_library_sha256_invalid")

        nested = build_nested_qc_roles(contexts, roles)
        compact = _parse_contexts(universal_contexts)
        parsed_candidates = _parse_candidates(candidates)
        parsed_actions = _parse_actions(actions)
        spec = _parse_model_spec(model_spec_sha256)

        bindings = {
            "protocol_namespace": "nato-sers-p08-qc-catalog-v1",
            "nested_registry_sha256": nested["registry_sha256"],
            "universal_contexts": compact,
            "candidates": parsed_candidates,
            "actions": parsed_actions,
            "model_spec_sha256": spec,
            "gate_library_sha256": gate_library_sha256,
        }
        binding_sha = canonical_sha256(bindings)

        raw_by_id = {c["context_id"]: c for c in contexts}
        compact_by_id = {c["context_id"]: c for c in compact}
        nested_by_id = {e["context_id"]: e for e in nested["contexts"]}
        if set(raw_by_id) != set(compact_by_id) or set(raw_by_id) != set(nested_by_id):
            raise QCPlanError("context_id_set_mismatch")

        candidates_by_model = {}
        for cand in parsed_candidates:
            candidates_by_model.setdefault(cand["model_id"], []).append(
                {
                    "candidate_id": cand["candidate_id"],
                    "hyperparameter_sha256": cand["hyperparameter_sha256"],
                }
            )
        svm_pairs = candidates_by_model.get("C-RBF-SVM", [])
        final_candidates = {
            "C-RBF-SVM": svm_pairs,
            "C-RANDOM-FOREST": candidates_by_model.get("C-RANDOM-FOREST", []),
        }

        all_blocks = []
        all_aliases = []
        for context_id, compact_ctx in compact_by_id.items():
            raw_ctx = raw_by_id[context_id]
            entry = nested_by_id[context_id]

            if raw_ctx["selection_mode"] != compact_ctx["selection_mode"]:
                raise QCPlanError("selection_mode_mismatch")
            if entry["outer_fit_uid_sha256"] != compact_ctx["outer_fit_uid_sha256"]:
                raise QCPlanError("outer_fit_hash_mismatch")
            if entry["outer_test_uid_sha256"] != compact_ctx["outer_test_uid_sha256"]:
                raise QCPlanError("outer_test_hash_mismatch")

            raw_units = {u["unit_id"]: u for u in raw_ctx["selection_units"]}
            compact_units = {u["unit_id"]: u for u in compact_ctx["selection_units"]}
            if set(raw_units) != set(compact_units):
                raise QCPlanError("selection_unit_id_mismatch")
            for unit_id, compact_unit in compact_units.items():
                raw_unit = raw_units[unit_id]
                if _hash_uids(raw_unit["fit_uids"]) != compact_unit["fit_uid_sha256"]:
                    raise QCPlanError("selection_unit_fit_hash_mismatch")
                if _hash_uids(raw_unit["validation_uids"]) != compact_unit["validation_uid_sha256"]:
                    raise QCPlanError("selection_unit_validation_hash_mismatch")

            if entry["eligible"]:
                policy_result = build_policy_blocks(binding_sha, entry, svm_pairs)
                final_result = _build_final_blocks(
                    binding_sha, compact_ctx, final_candidates, policy_result
                )
                all_blocks.extend(policy_result["blocks"])
                all_blocks.extend(final_result["blocks"])
                endpoints = final_result["endpoint_blocks_by_model"]
                evidence_status = "unapproved_future_job"
                reason_code = "eligible"
            else:
                endpoints = None
                evidence_status = "requires_authenticated_minimal_endpoint"
                reason_code = entry["reason_code"]

            for strategy, fixed_recipe in _ALIAS_PAIRS:
                recipe_id = (
                    fixed_recipe if fixed_recipe is not None else compact_ctx["selected_recipe_id"]
                )
                target = FALLBACK_TARGET if endpoints is None else endpoints[recipe_id]
                metadata = {
                    "binding_sha256": binding_sha,
                    "outer_fit_uid_sha256": entry["outer_fit_uid_sha256"],
                    "outer_test_uid_sha256": entry["outer_test_uid_sha256"],
                    "minimal_array_sha256": parsed_actions["R_MIN_400_1800"],
                    "model_spec_sha256": spec[recipe_id],
                }
                all_aliases.append(
                    make_alias(
                        context_id=context_id,
                        strategy=strategy,
                        recipe_id=recipe_id,
                        target_block_id=target,
                        evidence_status=evidence_status,
                        reason_code=reason_code,
                        metadata=metadata,
                    )
                )

        return seal_catalog(bindings, all_blocks, all_aliases)
    except QCPlanError:
        raise
    except Exception:
        raise QCPlanError(_GENERIC_CODE) from None
