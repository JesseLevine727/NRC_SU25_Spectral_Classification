"""Bounded QC policy-selection subgraph for P08-T015.

The builder consumes an eligible context produced by
``build_nested_qc_roles`` together with the pre-filtered SVM candidate
registry and emits the bounded policy-selection block graph.  It reads only
already-audited role hashes and identifiers from the upstream role graph.

Scope and non-guarantees
------------------------
* This module does NOT authenticate upstream memberships, master/class or
  instrument disjointness, or upstream canonical hashes; the upstream
  registry builder owns those checks.
* No raw membership, master or class field is copied to any output.
* No QC quantile, numerical routing, model fit, prediction, score or gate
  selection is computed here; placeholder axes stay unresolved.
* No block in this graph carries a test (T) role hash.

The returned blocks are content-addressed by the shared ``p08_qc_blocks``
primitives and form the policy-selection subgraph that a separate assembler
combines with the remaining subgraphs and seals.
"""

from __future__ import annotations

from .p08_plan import SEEDS, SVM_SEED
from .p08_qc_blocks import BlockError, make_block

_NOT_APPLICABLE = "not_applicable"
_FIT_ONLY_F = "fit_only_F"
_SVM_MODEL = "C-RBF-SVM"
_D0_MODEL = "D0-M"

REASON_BINDING_SHA256_INVALID = "binding_sha256_invalid"
REASON_CONTEXT_NOT_MAPPING = "context_not_mapping"
REASON_UNSUPPORTED_CONTEXT = "unsupported_context"
REASON_CONTEXT_ID_INVALID = "context_id_invalid"
REASON_OUTER_FIT_UID_INVALID = "outer_fit_uid_invalid"
REASON_OUTER_TEST_UID_INVALID = "outer_test_uid_invalid"
REASON_OUTER_FIT_TEST_SAME = "outer_fit_test_same"
REASON_CONTEXT_QUANTILE_UID_INVALID = "context_quantile_uid_invalid"
REASON_CONTEXT_QUANTILE_BINDING_MISMATCH = "context_quantile_binding_mismatch"
REASON_POLICY_UNITS_INVALID = "policy_units_invalid"
REASON_POLICY_UNITS_COUNT_INVALID = "policy_units_count_invalid"
REASON_UNIT_NOT_MAPPING = "unit_not_mapping"
REASON_UNIT_ID_INVALID = "unit_id_invalid"
REASON_UNIT_IDS_DUPLICATE = "unit_ids_duplicate"
REASON_UNIT_FIT_UID_INVALID = "unit_fit_uid_invalid"
REASON_UNIT_VALIDATION_UID_INVALID = "unit_validation_uid_invalid"
REASON_UNIT_FIT_VALIDATION_SAME = "unit_fit_validation_same"
REASON_UNIT_HASH_EQUALS_OUTER_TEST = "unit_hash_equals_outer_test"
REASON_UNIT_QUANTILE_UID_INVALID = "unit_quantile_uid_invalid"
REASON_UNIT_QUANTILE_BINDING_MISMATCH = "unit_quantile_binding_mismatch"
REASON_INNER_FOLDS_INVALID = "inner_folds_invalid"
REASON_FOLD_NOT_MAPPING = "fold_not_mapping"
REASON_FOLD_INDEX_INVALID = "fold_index_invalid"
REASON_ROLE_PAIR_ID_INVALID = "role_pair_id_invalid"
REASON_ROLE_PAIR_IDS_DUPLICATE = "role_pair_ids_duplicate"
REASON_FOLD_FIT_UID_INVALID = "fold_fit_uid_invalid"
REASON_FOLD_VALIDATION_UID_INVALID = "fold_validation_uid_invalid"
REASON_FOLD_FIT_VALIDATION_SAME = "fold_fit_validation_same"
REASON_FOLD_HASH_EQUALS_OUTER_TEST = "fold_hash_equals_outer_test"
REASON_FOLD_QUANTILE_UID_INVALID = "fold_quantile_uid_invalid"
REASON_FOLD_QUANTILE_BINDING_MISMATCH = "fold_quantile_binding_mismatch"
REASON_SVM_CANDIDATES_COUNT_INVALID = "svm_candidates_count_invalid"
REASON_SVM_CANDIDATE_NOT_MAPPING = "svm_candidate_not_mapping"
REASON_SVM_CANDIDATE_KEYS_INVALID = "svm_candidate_keys_invalid"
REASON_SVM_CANDIDATE_ID_INVALID = "svm_candidate_id_invalid"
REASON_SVM_CANDIDATE_HASH_INVALID = "svm_candidate_hash_invalid"
REASON_SVM_CANDIDATE_IDS_DUPLICATE = "svm_candidate_ids_duplicate"
REASON_SCIENTIFIC_EXECUTION_NOT_AUTHORIZED = "scientific_execution_not_authorized"

__all__ = (
    "PolicyBlockError",
    "SVM_SEED",
    "build_policy_blocks",
    "require_scientific_execution",
)


class PolicyBlockError(ValueError):
    """Static-reason error raised while building the policy subgraph."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def require_scientific_execution(*args, **kwargs):
    """Always refuse: this subgraph never authorizes scientific execution."""

    raise PolicyBlockError(REASON_SCIENTIFIC_EXECUTION_NOT_AUTHORIZED)


def _require(condition: bool, reason_code: str) -> None:
    if not condition:
        raise PolicyBlockError(reason_code)


def _is_hex64(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _is_clean_str(value) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _gate_ids():
    singles = tuple(f"QC-{index:03d}-SINGLE" for index in range(1, 16))
    duals = tuple(f"QC-{index:03d}-DUAL" for index in range(16, 124))
    return ("QC-000-MIN",) + singles + duals


_GATE_IDS = _gate_ids()
_D0_SEEDS = tuple(sorted(int(seed) for seed in SEEDS))


def _candidate(candidate_id: str, hyperparameter_sha256: str):
    return {
        "candidate_id": candidate_id,
        "hyperparameter_sha256": hyperparameter_sha256,
    }


_NA_CANDIDATE = _candidate(_NOT_APPLICABLE, _NOT_APPLICABLE)
_SELECTED_CANDIDATE = _candidate("source_selected_candidate", _NOT_APPLICABLE)
_FIXED_SPEC = _candidate("fixed_spec", _NOT_APPLICABLE)


def _axes(gate_ids, candidates, seeds):
    return {
        "gate_id": list(gate_ids),
        "candidate": [dict(candidate) for candidate in candidates],
        "seed": list(seeds),
    }


def _block_id(block):
    return block["block_id"]


def _validate_context(context):
    if not isinstance(context, dict):
        raise PolicyBlockError(REASON_CONTEXT_NOT_MAPPING)
    if context.get("eligible") is not True:
        raise PolicyBlockError(REASON_UNSUPPORTED_CONTEXT)
    context_id = context.get("context_id")
    _require(_is_clean_str(context_id), REASON_CONTEXT_ID_INVALID)
    outer_fit = context.get("outer_fit_uid_sha256")
    outer_test = context.get("outer_test_uid_sha256")
    _require(_is_hex64(outer_fit), REASON_OUTER_FIT_UID_INVALID)
    _require(_is_hex64(outer_test), REASON_OUTER_TEST_UID_INVALID)
    _require(outer_fit != outer_test, REASON_OUTER_FIT_TEST_SAME)
    context_quantile = context.get("final_refit_quantile_fit_uid_sha256")
    _require(_is_hex64(context_quantile), REASON_CONTEXT_QUANTILE_UID_INVALID)
    _require(
        context_quantile == outer_fit,
        REASON_CONTEXT_QUANTILE_BINDING_MISMATCH,
    )
    units = context.get("policy_units")
    _require(isinstance(units, list), REASON_POLICY_UNITS_INVALID)
    _require(len(units) >= 2, REASON_POLICY_UNITS_COUNT_INVALID)
    seen_unit_ids = set()
    seen_role_ids = set()
    normalized_units = []
    for unit in units:
        _require(isinstance(unit, dict), REASON_UNIT_NOT_MAPPING)
        unit_id = unit.get("parent_unit_id")
        _require(_is_clean_str(unit_id), REASON_UNIT_ID_INVALID)
        _require(unit_id not in seen_unit_ids, REASON_UNIT_IDS_DUPLICATE)
        seen_unit_ids.add(unit_id)
        unit_fit = unit.get("policy_fit_uid_sha256")
        unit_validation = unit.get("policy_validation_uid_sha256")
        _require(_is_hex64(unit_fit), REASON_UNIT_FIT_UID_INVALID)
        _require(_is_hex64(unit_validation), REASON_UNIT_VALIDATION_UID_INVALID)
        _require(unit_fit != unit_validation, REASON_UNIT_FIT_VALIDATION_SAME)
        _require(
            unit_fit != outer_test and unit_validation != outer_test,
            REASON_UNIT_HASH_EQUALS_OUTER_TEST,
        )
        unit_quantile = unit.get("policy_refit_quantile_fit_uid_sha256")
        _require(_is_hex64(unit_quantile), REASON_UNIT_QUANTILE_UID_INVALID)
        _require(
            unit_quantile == unit_fit,
            REASON_UNIT_QUANTILE_BINDING_MISMATCH,
        )
        folds = unit.get("inner_folds")
        _require(
            isinstance(folds, list) and len(folds) == 3,
            REASON_INNER_FOLDS_INVALID,
        )
        normalized_folds = []
        fold_indices = set()
        for fold in folds:
            _require(isinstance(fold, dict), REASON_FOLD_NOT_MAPPING)
            fold_index = fold.get("fold_index")
            _require(type(fold_index) is int, REASON_FOLD_INDEX_INVALID)
            fold_indices.add(fold_index)
            role_pair_id = fold.get("role_pair_id")
            _require(_is_clean_str(role_pair_id), REASON_ROLE_PAIR_ID_INVALID)
            _require(
                role_pair_id not in seen_role_ids,
                REASON_ROLE_PAIR_IDS_DUPLICATE,
            )
            seen_role_ids.add(role_pair_id)
            fold_fit = fold.get("fit_uid_sha256")
            fold_validation = fold.get("validation_uid_sha256")
            _require(_is_hex64(fold_fit), REASON_FOLD_FIT_UID_INVALID)
            _require(
                _is_hex64(fold_validation),
                REASON_FOLD_VALIDATION_UID_INVALID,
            )
            _require(
                fold_fit != fold_validation,
                REASON_FOLD_FIT_VALIDATION_SAME,
            )
            _require(
                fold_fit != outer_test and fold_validation != outer_test,
                REASON_FOLD_HASH_EQUALS_OUTER_TEST,
            )
            fold_quantile = fold.get("quantile_fit_uid_sha256")
            _require(_is_hex64(fold_quantile), REASON_FOLD_QUANTILE_UID_INVALID)
            _require(
                fold_quantile == fold_fit,
                REASON_FOLD_QUANTILE_BINDING_MISMATCH,
            )
            normalized_folds.append(
                {
                    "fold_index": fold_index,
                    "role_pair_id": role_pair_id,
                    "fit_uid_sha256": fold_fit,
                    "validation_uid_sha256": fold_validation,
                }
            )
        _require(fold_indices == {0, 1, 2}, REASON_FOLD_INDEX_INVALID)
        normalized_folds.sort(key=lambda fold: fold["fold_index"])
        normalized_units.append(
            {
                "unit_id": unit_id,
                "fit_uid_sha256": unit_fit,
                "validation_uid_sha256": unit_validation,
                "folds": normalized_folds,
            }
        )
    normalized_units.sort(key=lambda unit: unit["unit_id"])
    return {
        "context_id": context_id,
        "outer_fit_uid_sha256": outer_fit,
        "units": normalized_units,
    }


def _validate_svm_candidates(svm_candidates):
    if not isinstance(svm_candidates, list) or len(svm_candidates) != 36:
        raise PolicyBlockError(REASON_SVM_CANDIDATES_COUNT_INVALID)
    seen_ids = set()
    normalized = []
    for candidate in svm_candidates:
        _require(isinstance(candidate, dict), REASON_SVM_CANDIDATE_NOT_MAPPING)
        _require(
            set(candidate.keys()) == {"candidate_id", "hyperparameter_sha256"},
            REASON_SVM_CANDIDATE_KEYS_INVALID,
        )
        candidate_id = candidate["candidate_id"]
        _require(_is_clean_str(candidate_id), REASON_SVM_CANDIDATE_ID_INVALID)
        hyperparameter_sha256 = candidate["hyperparameter_sha256"]
        _require(
            _is_hex64(hyperparameter_sha256),
            REASON_SVM_CANDIDATE_HASH_INVALID,
        )
        _require(
            candidate_id not in seen_ids,
            REASON_SVM_CANDIDATE_IDS_DUPLICATE,
        )
        seen_ids.add(candidate_id)
        normalized.append(_candidate(candidate_id, hyperparameter_sha256))
    normalized.sort(key=lambda candidate: candidate["candidate_id"])
    return normalized


class _Builder:
    def __init__(self, binding_sha256: str, context_id: str) -> None:
        self._binding_sha256 = binding_sha256
        self._context_id = context_id
        self.blocks = []

    def add(
        self,
        *,
        stage,
        model_id,
        role_id,
        fit_uid_sha256,
        validation_uid_sha256,
        test_uid_sha256,
        axes,
        depends_on_blocks,
        resolution,
    ):
        try:
            block = make_block(
                binding_sha256=self._binding_sha256,
                context_id=self._context_id,
                stage=stage,
                model_id=model_id,
                role_id=role_id,
                fit_uid_sha256=fit_uid_sha256,
                validation_uid_sha256=validation_uid_sha256,
                test_uid_sha256=test_uid_sha256,
                axes=axes,
                depends_on_blocks=list(depends_on_blocks),
                resolution=resolution,
            )
        except BlockError as error:
            raise PolicyBlockError(error.reason_code) from None
        self.blocks.append(block)
        return _block_id(block)


def _build_inner_fold(builder, fold, candidates):
    role_id = fold["role_pair_id"]
    fit_hash = fold["fit_uid_sha256"]
    validation_hash = fold["validation_uid_sha256"]

    quantile = builder.add(
        stage="inner_quantile_fit",
        model_id=_NOT_APPLICABLE,
        role_id=role_id,
        fit_uid_sha256=fit_hash,
        validation_uid_sha256=_NOT_APPLICABLE,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes((_NOT_APPLICABLE,), (_NA_CANDIDATE,), (_NOT_APPLICABLE,)),
        depends_on_blocks=(),
        resolution="source_fit_only_shared_all_gates",
    )
    route = builder.add(
        stage="inner_route_pair",
        model_id=_NOT_APPLICABLE,
        role_id=role_id,
        fit_uid_sha256=fit_hash,
        validation_uid_sha256=validation_hash,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_NA_CANDIDATE,), (_NOT_APPLICABLE,)),
        depends_on_blocks=(quantile,),
        resolution="inner_route_pair_same_gate_candidate_seed_parent_quantile",
    )
    svm_fit = builder.add(
        stage="inner_source_fit",
        model_id=_SVM_MODEL,
        role_id=role_id,
        fit_uid_sha256=fit_hash,
        validation_uid_sha256=_NOT_APPLICABLE,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, candidates, (SVM_SEED,)),
        depends_on_blocks=(route,),
        resolution="inner_source_fit_same_gate_candidate_seed",
    )
    d0_fit = builder.add(
        stage="inner_source_fit",
        model_id=_D0_MODEL,
        role_id=role_id,
        fit_uid_sha256=fit_hash,
        validation_uid_sha256=validation_hash,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_FIXED_SPEC,), _D0_SEEDS),
        depends_on_blocks=(route,),
        resolution=(
            "inner_validation_stopping_same_gate_same_seed_"
            "source_selected_epochs"
        ),
    )
    svm_prediction = builder.add(
        stage="inner_source_prediction",
        model_id=_SVM_MODEL,
        role_id=role_id,
        fit_uid_sha256=fit_hash,
        validation_uid_sha256=validation_hash,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, candidates, (SVM_SEED,)),
        depends_on_blocks=(svm_fit,),
        resolution="inner_source_prediction_same_gate_candidate_seed_parent_fit",
    )
    d0_prediction = builder.add(
        stage="inner_source_prediction",
        model_id=_D0_MODEL,
        role_id=role_id,
        fit_uid_sha256=fit_hash,
        validation_uid_sha256=validation_hash,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_FIXED_SPEC,), _D0_SEEDS),
        depends_on_blocks=(d0_fit,),
        resolution="inner_source_prediction_same_gate_candidate_seed_parent_fit",
    )
    return {
        "quantile": quantile,
        "route": route,
        "svm_fit": svm_fit,
        "d0_fit": d0_fit,
        "svm_prediction": svm_prediction,
        "d0_prediction": d0_prediction,
    }


def _build_unit(builder, unit, candidates):
    unit_id = unit["unit_id"]
    unit_fit = unit["fit_uid_sha256"]
    unit_validation = unit["validation_uid_sha256"]
    inner = [_build_inner_fold(builder, fold, candidates) for fold in unit["folds"]]

    policy_quantile = builder.add(
        stage="policy_refit_quantile_fit",
        model_id=_NOT_APPLICABLE,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=_NOT_APPLICABLE,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes((_NOT_APPLICABLE,), (_NA_CANDIDATE,), (_NOT_APPLICABLE,)),
        depends_on_blocks=(),
        resolution="source_fit_only_shared_all_gates",
    )
    policy_route = builder.add(
        stage="policy_route_pair",
        model_id=_NOT_APPLICABLE,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=unit_validation,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_NA_CANDIDATE,), (_NOT_APPLICABLE,)),
        depends_on_blocks=(policy_quantile,),
        resolution="policy_route_pair_same_gate_candidate_seed_parent_quantile",
    )
    select_hyperparameters = builder.add(
        stage="inner_select_hyperparameters",
        model_id=_SVM_MODEL,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=_NOT_APPLICABLE,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_SELECTED_CANDIDATE,), (SVM_SEED,)),
        depends_on_blocks=tuple(item["svm_prediction"] for item in inner),
        resolution="selectedcandidate-all3folds",
    )
    select_epochs = builder.add(
        stage="inner_select_refit_epochs",
        model_id=_D0_MODEL,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=_NOT_APPLICABLE,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_FIXED_SPEC,), _D0_SEEDS),
        depends_on_blocks=tuple(item["d0_fit"] for item in inner),
        resolution="samegate-sameseed-median3bestepochs-pythonround-clip30_200",
    )
    scalar_svm = builder.add(
        stage="policy_scalar_calibration",
        model_id=_SVM_MODEL,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=_NOT_APPLICABLE,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_SELECTED_CANDIDATE,), (SVM_SEED,)),
        depends_on_blocks=(select_hyperparameters,)
        + tuple(item["svm_prediction"] for item in inner),
        resolution="svm_single_temperature_same_gate_same_seed_by_selector",
    )
    scalar_d0 = builder.add(
        stage="policy_scalar_calibration",
        model_id=_D0_MODEL,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=_NOT_APPLICABLE,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_FIXED_SPEC,), _D0_SEEDS),
        depends_on_blocks=tuple(item["d0_prediction"] for item in inner)
        + (select_epochs,),
        resolution="d0_per_seed_temperature_same_gate_same_seed_inner_predictions",
    )
    refit_svm = builder.add(
        stage="policy_refit",
        model_id=_SVM_MODEL,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=_NOT_APPLICABLE,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_SELECTED_CANDIDATE,), (SVM_SEED,)),
        depends_on_blocks=(policy_route, select_hyperparameters),
        resolution=f"{_FIT_ONLY_F}_same_gate_selected_candidate",
    )
    refit_d0 = builder.add(
        stage="policy_refit",
        model_id=_D0_MODEL,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=_NOT_APPLICABLE,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_FIXED_SPEC,), _D0_SEEDS),
        depends_on_blocks=(policy_route, select_epochs),
        resolution=f"{_FIT_ONLY_F}_same_gate_same_seed_source_selected_epochs",
    )
    validation_svm = builder.add(
        stage="policy_validation_prediction",
        model_id=_SVM_MODEL,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=unit_validation,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_SELECTED_CANDIDATE,), (SVM_SEED,)),
        depends_on_blocks=(refit_svm, scalar_svm),
        resolution="parent_FV_prediction_from_refit_and_calibration",
    )
    validation_d0 = builder.add(
        stage="policy_validation_prediction",
        model_id=_D0_MODEL,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=unit_validation,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_FIXED_SPEC,), _D0_SEEDS),
        depends_on_blocks=(refit_d0, scalar_d0),
        resolution="parent_FV_prediction_from_refit_and_calibration",
    )
    ensemble_svm = builder.add(
        stage="policy_seed_ensemble_prediction",
        model_id=_SVM_MODEL,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=unit_validation,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_SELECTED_CANDIDATE,), (_NOT_APPLICABLE,)),
        depends_on_blocks=(validation_svm,),
        resolution="SVM_single_calibrated",
    )
    ensemble_d0 = builder.add(
        stage="policy_seed_ensemble_prediction",
        model_id=_D0_MODEL,
        role_id=unit_id,
        fit_uid_sha256=unit_fit,
        validation_uid_sha256=unit_validation,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_FIXED_SPEC,), (_NOT_APPLICABLE,)),
        depends_on_blocks=(validation_d0,),
        resolution="D0_average_of_3_calibrated_probabilities",
    )
    panel = builder.add(
        stage="policy_panel_score",
        model_id=_NOT_APPLICABLE,
        role_id=unit_id,
        fit_uid_sha256=_NOT_APPLICABLE,
        validation_uid_sha256=unit_validation,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_NA_CANDIDATE,), (_NOT_APPLICABLE,)),
        depends_on_blocks=(ensemble_svm, ensemble_d0),
        resolution=(
            "M01_same_gate_equal_svm_d0_weights_balanced_accuracy_"
            "validation_V"
        ),
    )
    return panel, policy_route


def build_policy_blocks(binding_sha256, context, svm_candidates):
    """Build the bounded QC policy-selection subgraph.

    Consumes an eligible context and the filtered 36-candidate SVM registry.
    Returns the sorted blocks, the gate-selection block id and per-unit
    policy-route block ids.  Never authorizes scientific execution and never
    resolves a real gate, candidate or epoch.
    """

    _require(_is_hex64(binding_sha256), REASON_BINDING_SHA256_INVALID)
    normalized_context = _validate_context(context)
    candidates = _validate_svm_candidates(svm_candidates)
    context_id = normalized_context["context_id"]
    outer_fit = normalized_context["outer_fit_uid_sha256"]
    builder = _Builder(binding_sha256, context_id)

    unit_panels = []
    unit_routes = []
    routes_by_unit = {}
    for unit in normalized_context["units"]:
        panel, policy_route = _build_unit(builder, unit, candidates)
        unit_panels.append(panel)
        unit_routes.append(policy_route)
        routes_by_unit[unit["unit_id"]] = policy_route

    objective = builder.add(
        stage="gate_objective",
        model_id=_NOT_APPLICABLE,
        role_id=context_id,
        fit_uid_sha256=outer_fit,
        validation_uid_sha256=_NOT_APPLICABLE,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(_GATE_IDS, (_NA_CANDIDATE,), (_NOT_APPLICABLE,)),
        depends_on_blocks=tuple(unit_panels) + tuple(unit_routes),
        resolution=(
            "equal_unit_weights_mean_then_worst_then_fraction_strict_improvements_"
            "vs_NESTED_minimal_gate_then_mean_V_nonMIN_fraction_ascending_"
            "then_declared_order"
        ),
    )
    selection = builder.add(
        stage="gate_selection",
        model_id=_NOT_APPLICABLE,
        role_id=context_id,
        fit_uid_sha256=outer_fit,
        validation_uid_sha256=_NOT_APPLICABLE,
        test_uid_sha256=_NOT_APPLICABLE,
        axes=_axes(("source_selected_gate",), (_NA_CANDIDATE,), (_NOT_APPLICABLE,)),
        depends_on_blocks=(objective,),
        resolution=(
            "complete_units_seeds_valid_nested_MIN_required_source_only_no_held_"
            "no_winner_computed"
        ),
    )

    blocks = sorted(builder.blocks, key=_block_id)
    ordered_routes = {
        unit_id: routes_by_unit[unit_id] for unit_id in sorted(routes_by_unit)
    }
    return {
        "execution_authorized": False,
        "blocks": blocks,
        "gate_selection_block_id": selection,
        "policy_route_blocks_by_unit": ordered_routes,
    }
