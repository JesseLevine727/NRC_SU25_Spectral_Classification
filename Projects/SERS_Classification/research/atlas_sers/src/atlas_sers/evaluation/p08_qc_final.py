"""Internal final-QC descriptor constructor (T018).

This module is intentionally INTERNAL.  ``_build_final_blocks`` is a pure
descriptor constructor: it consumes already validated and normalized metadata
and emits block descriptors.  It computes no numerical results, reads no data
and performs no scientific work.

The caller is responsible for authenticating and validating roles, input
schemas, the policy graph and every dependency before sealing the joint graph.
This constructor never authorizes scientific execution.
"""

from .p08_plan import SEEDS, SVM_SEED
from .p08_qc_blocks import make_block

_NOT_APPLICABLE = 'not_applicable'

_CLASSICAL_MODELS = ('C-RBF-SVM', 'C-RANDOM-FOREST')
_DEFAULT_NEURAL_MODEL = 'D0-M'

# Resolution tags bind a stage to the exact inputs it may resolve against.
# Nothing here is ever resolved numerically.
_SOURCE_ROUTE_ALIAS_RESOLUTION = (
    'selected_gate_from_same_unit_source_fitted_thresholds'
)
_SOURCE_FIT_RESOLUTION = 'same_gate_candidate_seed_neural_stopping_on_unit_validation'
_SOURCE_PREDICTION_RESOLUTION = 'same_gate_candidate_seed'
_SELECT_HYPERPARAMETERS_RESOLUTION = 'all_original_source_units_inherited_objective'
_EPOCH_RESOLUTION = 'same_seed_median_best_round_clip_30_200'
_QUANTILE_FIT_RESOLUTION = 'calibration_quantile_fit_role_only'
_CALIBRATION_ROUTE_RESOLUTION = 'same_source_thresholds'
_CALIBRATION_MODEL_RESOLUTION = 'selected_candidate_same_gate_same_seed'
_SCALAR_CLASSICAL_RESOLUTION = 'seed_average_then_single_temperature_master_equal'
_SCALAR_NEURAL_RESOLUTION = 'same_gate_same_seed_source_logits_master_equal'
_SOURCE_ROUTE_RESOLUTION = 'source_frozen_thresholds_selection_only'
_REFIT_CLASSICAL_RESOLUTION = 'source_frozen_selected_candidate_selection_only'
_REFIT_NEURAL_RESOLUTION = 'same_seed_selected_epochs_selection_only'
_FINAL_ROUTE_RESOLUTION = (
    'source_frozen_thresholds_row_local_after_all_final_models_frozen'
)
_HELD_CLASSICAL_RESOLUTION = 'raw_scores'
_HELD_NEURAL_RESOLUTION = 'same_seed_calibrated_scores'
_ENSEMBLE_CLASSICAL_RESOLUTION = 'seed_average_then_single_temperature_logclip1e_7'
_ENSEMBLE_NEURAL_RESOLUTION = 'average_calibrated_seed_probabilities'


def _is_classical(model_id):
    return model_id in _CLASSICAL_MODELS


def _seeds_for(model_id):
    if model_id == 'C-RBF-SVM':
        return [SVM_SEED]
    return list(SEEDS)


def _selected_candidate(model_id):
    if _is_classical(model_id):
        return {
            'candidate_id': 'source_selected_candidate',
            'hyperparameter_sha256': _NOT_APPLICABLE,
        }
    return {'candidate_id': 'fixed_spec', 'hyperparameter_sha256': _NOT_APPLICABLE}


def _source_candidate_axis(model_id, candidates_by_model):
    if not _is_classical(model_id):
        return [
            {'candidate_id': 'fixed_spec', 'hyperparameter_sha256': _NOT_APPLICABLE}
        ]
    pairs = (dict(pair) for pair in candidates_by_model[model_id])
    return sorted(
        pairs,
        key=lambda pair: (pair['candidate_id'], pair['hyperparameter_sha256']),
    )


def _emit(binding_sha256, context_id, stage, *, model_id=_NOT_APPLICABLE,
          role_id=_NOT_APPLICABLE, fit_uid_sha256=_NOT_APPLICABLE,
          validation_uid_sha256=_NOT_APPLICABLE, test_uid_sha256=_NOT_APPLICABLE,
          candidate=None, seeds=None, depends_on_blocks=(), resolution):
    if candidate is None:
        candidate = [
            {'candidate_id': _NOT_APPLICABLE, 'hyperparameter_sha256': _NOT_APPLICABLE}
        ]
    if seeds is None:
        seeds = [_NOT_APPLICABLE]
    return make_block(
        binding_sha256=binding_sha256,
        context_id=context_id,
        stage=stage,
        model_id=model_id,
        role_id=role_id,
        fit_uid_sha256=fit_uid_sha256,
        validation_uid_sha256=validation_uid_sha256,
        test_uid_sha256=test_uid_sha256,
        axes={
            'gate_id': ['source_selected_gate'],
            'candidate': candidate,
            'seed': list(seeds),
        },
        depends_on_blocks=list(depends_on_blocks),
        resolution=resolution,
    )


def _build_final_blocks(binding_sha256, context, candidates_by_model, policy_result):
    """Build the internal final-QC graph for already validated metadata.

    The caller must authenticate and validate roles, input schemas, the policy
    graph and every dependency, then seal the joint graph.  This function only
    emits descriptors and cannot authorize execution.  No numerical result is
    computed and no scientific or IO work is performed.
    """
    context_id = context['context_id']
    outer_fit = context['outer_fit_uid_sha256']
    outer_test = context['outer_test_uid_sha256']
    selection_units = context['selection_units']
    calibration_units = context['calibration_units']
    selected_recipe_id = context['selected_recipe_id']

    gate_choice = policy_result['gate_selection_block_id']
    policy_routes = policy_result['policy_route_blocks_by_unit']

    neural_models = sorted({_DEFAULT_NEURAL_MODEL, selected_recipe_id})
    models = list(_CLASSICAL_MODELS) + neural_models

    new_blocks = []

    def add(block):
        new_blocks.append(block)
        return block

    # Stage 1: per source unit route alias, then per model fit and prediction.
    source_route_alias = {}
    source_fit = {}
    source_pred = {}
    for unit in selection_units:
        unit_id = unit['unit_id']
        unit_fit = unit['fit_uid_sha256']
        unit_val = unit['validation_uid_sha256']
        route = add(_emit(
            binding_sha256, context_id, 'final_source_route_alias', role_id=unit_id,
            fit_uid_sha256=unit_fit, validation_uid_sha256=unit_val,
            depends_on_blocks=[policy_routes[unit_id], gate_choice],
            resolution=_SOURCE_ROUTE_ALIAS_RESOLUTION))
        source_route_alias[unit_id] = route
        for model_id in models:
            seeds = _seeds_for(model_id)
            fit_validation = _NOT_APPLICABLE if _is_classical(model_id) else unit_val
            fit = add(_emit(
                binding_sha256, context_id, 'final_source_fit', model_id=model_id,
                role_id=unit_id, fit_uid_sha256=unit_fit,
                validation_uid_sha256=fit_validation,
                candidate=_source_candidate_axis(model_id, candidates_by_model),
                seeds=seeds, depends_on_blocks=[route['block_id']],
                resolution=_SOURCE_FIT_RESOLUTION))
            source_fit[(unit_id, model_id)] = fit
            pred = add(_emit(
                binding_sha256, context_id, 'final_source_prediction', model_id=model_id,
                role_id=unit_id, fit_uid_sha256=unit_fit,
                validation_uid_sha256=unit_val,
                candidate=_source_candidate_axis(model_id, candidates_by_model),
                seeds=seeds, depends_on_blocks=[fit['block_id']],
                resolution=_SOURCE_PREDICTION_RESOLUTION))
            source_pred[(unit_id, model_id)] = pred

    # Stage 2: classical hyperparameter selection and neural epoch selection.
    selection = {}
    for model_id in _CLASSICAL_MODELS:
        parents = [source_pred[(unit['unit_id'], model_id)]['block_id']
                   for unit in selection_units]
        parents.append(gate_choice)
        selection[model_id] = add(_emit(
            binding_sha256, context_id, 'final_select_hyperparameters',
            model_id=model_id, role_id=context_id, fit_uid_sha256=outer_fit,
            candidate=[_selected_candidate(model_id)], depends_on_blocks=parents,
            resolution=_SELECT_HYPERPARAMETERS_RESOLUTION))
    for model_id in neural_models:
        parents = [source_fit[(unit['unit_id'], model_id)]['block_id']
                   for unit in selection_units]
        parents.append(gate_choice)
        selection[model_id] = add(_emit(
            binding_sha256, context_id, 'final_select_refit_epochs', model_id=model_id,
            role_id=context_id, fit_uid_sha256=outer_fit,
            candidate=[_selected_candidate(model_id)], seeds=_seeds_for(model_id),
            depends_on_blocks=parents, resolution=_EPOCH_RESOLUTION))

    # Stage 3: per calibration unit quantile fit, route pair and classical model.
    cal_quantile = {}
    cal_route = {}
    cal_model_pred = {}
    for unit in calibration_units:
        unit_id = unit['unit_id']
        unit_fit = unit['fit_uid_sha256']
        unit_val = unit['validation_uid_sha256']
        quantile = add(_emit(
            binding_sha256, context_id, 'final_calibration_quantile_fit',
            role_id=unit_id, fit_uid_sha256=unit_fit,
            depends_on_blocks=[gate_choice],
            resolution=_QUANTILE_FIT_RESOLUTION))
        cal_quantile[unit_id] = quantile
        route_pair = add(_emit(
            binding_sha256, context_id, 'final_calibration_route_pair',
            role_id=unit_id, fit_uid_sha256=unit_fit,
            validation_uid_sha256=unit_val,
            depends_on_blocks=[quantile['block_id'], gate_choice],
            resolution=_CALIBRATION_ROUTE_RESOLUTION))
        cal_route[unit_id] = route_pair
        for model_id in _CLASSICAL_MODELS:
            model_fit = add(_emit(
                binding_sha256, context_id, 'final_calibration_model_fit',
                model_id=model_id, role_id=unit_id, fit_uid_sha256=unit_fit,
                candidate=[_selected_candidate(model_id)], seeds=_seeds_for(model_id),
                depends_on_blocks=[
                    route_pair['block_id'], selection[model_id]['block_id']
                ],
                resolution=_CALIBRATION_MODEL_RESOLUTION))
            model_pred = add(_emit(
                binding_sha256, context_id, 'final_calibration_model_prediction',
                model_id=model_id, role_id=unit_id, fit_uid_sha256=unit_fit,
                validation_uid_sha256=unit_val,
                candidate=[_selected_candidate(model_id)], seeds=_seeds_for(model_id),
                depends_on_blocks=[model_fit['block_id']],
                resolution=_CALIBRATION_MODEL_RESOLUTION))
            cal_model_pred[(unit_id, model_id)] = model_pred

    # Stage 4: scalar calibration per model.
    scalar = {}
    for model_id in _CLASSICAL_MODELS:
        parents = [selection[model_id]['block_id']]
        parents.extend(cal_model_pred[(unit['unit_id'], model_id)]['block_id']
                       for unit in calibration_units)
        scalar[model_id] = add(_emit(
            binding_sha256, context_id, 'final_scalar_calibration', model_id=model_id,
            role_id=context_id, fit_uid_sha256=outer_fit, seeds=[_NOT_APPLICABLE],
            candidate=[_selected_candidate(model_id)], depends_on_blocks=parents,
            resolution=_SCALAR_CLASSICAL_RESOLUTION))
    for model_id in neural_models:
        parents = [source_pred[(unit['unit_id'], model_id)]['block_id']
                   for unit in selection_units]
        parents.append(selection[model_id]['block_id'])
        scalar[model_id] = add(_emit(
            binding_sha256, context_id, 'final_scalar_calibration', model_id=model_id,
            role_id=context_id, fit_uid_sha256=outer_fit, seeds=_seeds_for(model_id),
            candidate=[_selected_candidate(model_id)], depends_on_blocks=parents,
            resolution=_SCALAR_NEURAL_RESOLUTION))

    # Stage 5: refit quantile, source route and per model final refit.
    refit_quantile = add(_emit(
        binding_sha256, context_id, 'final_refit_quantile_fit', role_id=context_id,
        fit_uid_sha256=outer_fit, depends_on_blocks=[gate_choice],
        resolution=_QUANTILE_FIT_RESOLUTION))
    source_route = add(_emit(
        binding_sha256, context_id, 'final_source_route', role_id=context_id,
        fit_uid_sha256=outer_fit,
        depends_on_blocks=[refit_quantile['block_id'], gate_choice],
        resolution=_SOURCE_ROUTE_RESOLUTION))
    refit = {}
    for model_id in models:
        if _is_classical(model_id):
            resolution = _REFIT_CLASSICAL_RESOLUTION
        else:
            resolution = _REFIT_NEURAL_RESOLUTION
        refit[model_id] = add(_emit(
            binding_sha256, context_id, 'final_refit', model_id=model_id,
            role_id=context_id, fit_uid_sha256=outer_fit,
            candidate=[_selected_candidate(model_id)], seeds=_seeds_for(model_id),
            depends_on_blocks=[source_route['block_id'], selection[model_id]['block_id']],
            resolution=resolution))

    # Stage 6: first test-bearing route, after every final refit exists.
    test_route = add(_emit(
        binding_sha256, context_id, 'final_test_route', role_id=context_id,
        test_uid_sha256=outer_test,
        depends_on_blocks=[refit_quantile['block_id'], gate_choice]
        + [refit[model_id]['block_id'] for model_id in models],
        resolution=_FINAL_ROUTE_RESOLUTION))

    # Stage 7: per model held prediction and seed ensemble endpoint.
    held = {}
    ensemble = {}
    for model_id in models:
        parents = [refit[model_id]['block_id'], test_route['block_id']]
        if _is_classical(model_id):
            held_resolution = _HELD_CLASSICAL_RESOLUTION
        else:
            parents.append(scalar[model_id]['block_id'])
            held_resolution = _HELD_NEURAL_RESOLUTION
        held[model_id] = add(_emit(
            binding_sha256, context_id, 'final_held_prediction', model_id=model_id,
            role_id=context_id, fit_uid_sha256=outer_fit, test_uid_sha256=outer_test,
            candidate=[_selected_candidate(model_id)], seeds=_seeds_for(model_id),
            depends_on_blocks=parents, resolution=held_resolution))
        if _is_classical(model_id):
            ensemble_parents = [held[model_id]['block_id'], scalar[model_id]['block_id']]
            resolution = _ENSEMBLE_CLASSICAL_RESOLUTION
        else:
            ensemble_parents = [held[model_id]['block_id']]
            resolution = _ENSEMBLE_NEURAL_RESOLUTION
        ensemble[model_id] = add(_emit(
            binding_sha256, context_id, 'final_seed_ensemble_prediction',
            model_id=model_id, role_id=context_id, fit_uid_sha256=outer_fit,
            test_uid_sha256=outer_test,
            candidate=[_selected_candidate(model_id)], seeds=[_NOT_APPLICABLE],
            depends_on_blocks=ensemble_parents, resolution=resolution))

    endpoint_blocks_by_model = {
        model_id: ensemble[model_id]['block_id'] for model_id in sorted(ensemble)
    }
    return {
        'blocks': sorted(new_blocks, key=lambda block: block['block_id']),
        'endpoint_blocks_by_model': endpoint_blocks_by_model,
    }


def require_scientific_execution(*args, **kwargs):
    """Refuse to authorize scientific execution from this descriptor module."""
    raise ValueError('scientific_execution_not_authorized')
