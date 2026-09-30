"""Focused tests for the internal final-QC block constructor (T018)."""

import copy
import hashlib

import pytest

from atlas_sers.evaluation.p08_qc_blocks import seal_catalog
from atlas_sers.evaluation.p08_qc_final import (
    _build_final_blocks,
    require_scientific_execution,
)
from atlas_sers.evaluation.p08_qc_policy import build_policy_blocks
from tests.test_p08_qc_policy import (
    _binding_obj,
    _binding_sha,
    _make_candidates,
    _make_context,
)

_NOT_APPLICABLE = 'not_applicable'
_CLASSICAL = ('C-RBF-SVM', 'C-RANDOM-FOREST')
_NEURAL = ('D0-M', 'D1', 'D2', 'D3')


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _classical_candidates():
    return {
        'C-RBF-SVM': [dict(pair) for pair in _make_candidates()],
        'C-RANDOM-FOREST': [
            {
                'candidate_id': f'rf-{index:02d}',
                'hyperparameter_sha256': _sha(f'rf-{index}'),
            }
            for index in range(16)
        ],
    }


def _compact_units(raw):
    return [
        {
            'unit_id': unit['parent_unit_id'],
            'fit_uid_sha256': unit['policy_fit_uid_sha256'],
            'validation_uid_sha256': unit['policy_validation_uid_sha256'],
        }
        for unit in raw['policy_units']
    ]


def _calibration_units():
    return [
        {
            'unit_id': f'calibration-{index}',
            'fit_uid_sha256': _sha(f'calibration-fit-{index}'),
            'validation_uid_sha256': _sha(f'calibration-validation-{index}'),
        }
        for index in range(3)
    ]


def _context(raw, selected_recipe_id):
    return {
        'context_id': raw['context_id'],
        'selected_recipe_id': selected_recipe_id,
        'outer_fit_uid_sha256': raw['outer_fit_uid_sha256'],
        'outer_test_uid_sha256': raw['outer_test_uid_sha256'],
        'selection_units': _compact_units(raw),
        'calibration_units': _calibration_units(),
        'selection_mode': 'pseudo_domain',
    }


def _run(selected_recipe_id, unit_count=2):
    raw = _make_context(unit_count=unit_count)
    candidates = _classical_candidates()
    policy = build_policy_blocks(_binding_sha(), raw, _make_candidates())
    context = _context(raw, selected_recipe_id)
    new = _build_final_blocks(_binding_sha(), context, candidates, policy)
    return raw, candidates, policy, context, new


def _slots(blocks, stage):
    return sum(block['slot_count'] for block in blocks if block['stage'] == stage)


def _model_fit_slots(blocks, model_id):
    stages = ('final_source_fit', 'final_calibration_model_fit', 'final_refit')
    return sum(block['slot_count'] for block in blocks
               if block['stage'] in stages and block['model_id'] == model_id)


def _by_id(blocks):
    return {block['block_id']: block for block in blocks}


def _find(blocks, stage, model_id):
    return [block for block in blocks
            if block['stage'] == stage and block['model_id'] == model_id]


def _ancestors(blocks, ids):
    index = _by_id(blocks)
    seen = set()
    stack = list(ids)
    while stack:
        block_id = stack.pop()
        if block_id in seen:
            continue
        seen.add(block_id)
        assert block_id in index, f'missing ancestor: {block_id}'
        stack.extend(index[block_id]['depends_on_blocks'])
    return seen


def test_new_fit_counts_d0_selection():
    _, _, _, _, new = _run('D0-M')
    blocks = new['blocks']
    assert _slots(blocks, 'final_source_fit') == 174
    assert _slots(blocks, 'final_calibration_model_fit') == 12
    assert _slots(blocks, 'final_refit') == 7
    assert _slots(blocks, 'final_scalar_calibration') == 5
    assert _slots(blocks, 'final_held_prediction') == 7
    assert sum(1 for block in blocks
               if block['stage'] == 'final_seed_ensemble_prediction') == 3
    assert 174 + 12 + 7 == 193
    assert _model_fit_slots(blocks, 'C-RBF-SVM') == 76
    assert _model_fit_slots(blocks, 'C-RANDOM-FOREST') == 108
    assert _model_fit_slots(blocks, 'D0-M') == 9
    assert len(new['endpoint_blocks_by_model']) == 3


def test_new_fit_counts_d3_selection():
    _, _, _, _, new = _run('D3')
    blocks = new['blocks']
    assert _slots(blocks, 'final_source_fit') == 180
    assert _slots(blocks, 'final_refit') == 10
    assert _slots(blocks, 'final_scalar_calibration') == 8
    assert _slots(blocks, 'final_held_prediction') == 10
    assert sum(1 for block in blocks
               if block['stage'] == 'final_seed_ensemble_prediction') == 4
    assert 180 + 12 + 10 == 202
    assert _model_fit_slots(blocks, 'D3') == 9
    assert len(new['endpoint_blocks_by_model']) == 4


def test_source_blocks_bind_exact_unit_hashes_and_role_ids():
    _, _, _, context, new = _run('D0-M')
    by_unit = {unit['unit_id']: unit for unit in context['selection_units']}
    stages = (
        'final_source_route_alias',
        'final_source_fit',
        'final_source_prediction',
    )
    for block in new['blocks']:
        if block['stage'] not in stages:
            continue
        unit = by_unit[block['role_id']]
        assert block['role_id'] == unit['unit_id']
        assert block['fit_uid_sha256'] == unit['fit_uid_sha256']
        if block['stage'] == 'final_source_fit' and block['model_id'] in _CLASSICAL:
            assert block['validation_uid_sha256'] == _NOT_APPLICABLE
        else:
            assert block['validation_uid_sha256'] == unit['validation_uid_sha256']


def test_calibration_blocks_bind_exact_unit_hashes_and_role_ids():
    _, _, _, context, new = _run('D0-M')
    by_unit = {unit['unit_id']: unit for unit in context['calibration_units']}
    stages = (
        'final_calibration_quantile_fit',
        'final_calibration_route_pair',
        'final_calibration_model_fit',
        'final_calibration_model_prediction',
    )
    for block in new['blocks']:
        if block['stage'] not in stages:
            continue
        unit = by_unit[block['role_id']]
        assert block['role_id'] == unit['unit_id']
        assert block['fit_uid_sha256'] == unit['fit_uid_sha256']
        if block['stage'] in (
            'final_calibration_quantile_fit',
            'final_calibration_model_fit',
        ):
            assert block['validation_uid_sha256'] == _NOT_APPLICABLE
        else:
            assert block['validation_uid_sha256'] == unit['validation_uid_sha256']


def test_context_blocks_use_context_role_id():
    _, _, _, context, new = _run('D0-M')
    context_stages = {
        'final_select_hyperparameters',
        'final_select_refit_epochs',
        'final_scalar_calibration',
        'final_refit_quantile_fit',
        'final_source_route',
        'final_refit',
        'final_test_route',
        'final_held_prediction',
        'final_seed_ensemble_prediction',
    }
    for block in new['blocks']:
        if block['stage'] in context_stages:
            assert block['role_id'] == context['context_id']


def test_all_stages_have_non_static_resolution():
    _, _, _, _, new = _run('D0-M')
    for block in new['blocks']:
        assert block['resolution'] != 'static_tag'
        assert block['resolution'] != _NOT_APPLICABLE


def test_fit_selection_scalar_ancestors_avoid_outer_test():
    _, _, policy, context, new = _run('D0-M')
    joint = policy['blocks'] + new['blocks']
    seal_catalog(_binding_obj(), joint, [])
    outer_test = context['outer_test_uid_sha256']
    protected_stages = (
        'final_source_fit',
        'final_select_hyperparameters',
        'final_select_refit_epochs',
        'final_calibration_quantile_fit',
        'final_calibration_model_fit',
        'final_scalar_calibration',
        'final_refit',
        'final_refit_quantile_fit',
    )
    protected = [block['block_id'] for block in new['blocks']
                 if block['stage'] in protected_stages]
    index = _by_id(joint)
    for block_id in _ancestors(joint, protected):
        block = index[block_id]
        assert block['test_uid_sha256'] == _NOT_APPLICABLE
        assert block['fit_uid_sha256'] != outer_test
        assert block['validation_uid_sha256'] != outer_test


def test_test_route_parents_every_final_refit():
    _, _, _, context, new = _run('D3')
    blocks = new['blocks']
    refits = {block['block_id'] for block in blocks if block['stage'] == 'final_refit'}
    route = [block for block in blocks if block['stage'] == 'final_test_route'][0]
    assert refits <= set(route['depends_on_blocks'])
    assert route['test_uid_sha256'] == context['outer_test_uid_sha256']
    assert route['fit_uid_sha256'] == _NOT_APPLICABLE
    assert route['validation_uid_sha256'] == _NOT_APPLICABLE


def test_neural_stopping_uses_validation_split():
    _, _, _, context, new = _run('D3')
    blocks = new['blocks']
    validation = {unit['validation_uid_sha256'] for unit in context['selection_units']}
    neural_values = {block['validation_uid_sha256'] for block in blocks
                     if block['stage'] == 'final_source_fit'
                     and block['model_id'] in _NEURAL}
    classical_values = {block['validation_uid_sha256'] for block in blocks
                        if block['stage'] == 'final_source_fit'
                        and block['model_id'] in _CLASSICAL}
    assert neural_values == validation
    assert classical_values == {_NOT_APPLICABLE}


def test_calibration_selector_and_scalar_parent_identity():
    _, _, _, _, new = _run('D0-M')
    blocks = new['blocks']
    route_ids = {block['block_id'] for block in blocks
                 if block['stage'] == 'final_calibration_route_pair'}
    selector_by_model = {block['model_id']: block['block_id'] for block in blocks
                         if block['stage'] == 'final_select_hyperparameters'}
    epoch_ids = {block['block_id'] for block in blocks
                 if block['stage'] == 'final_select_refit_epochs'}
    for block in blocks:
        parents = set(block['depends_on_blocks'])
        if block['stage'] == 'final_calibration_model_fit':
            assert parents & route_ids
            assert selector_by_model[block['model_id']] in parents
        if (block['stage'] == 'final_scalar_calibration'
                and block['model_id'] in _NEURAL):
            assert parents & epoch_ids


def test_neural_scalar_precedes_seed_ensemble():
    _, _, policy, _, new = _run('D0-M')
    joint = policy['blocks'] + new['blocks']
    models = {block['model_id'] for block in new['blocks']
              if block['model_id'] in _NEURAL}
    for model_id in models:
        scalar = _find(new['blocks'], 'final_scalar_calibration', model_id)[0]
        held = _find(new['blocks'], 'final_held_prediction', model_id)[0]
        ensemble = _find(new['blocks'], 'final_seed_ensemble_prediction', model_id)[0]
        assert scalar['block_id'] in held['depends_on_blocks']
        assert held['block_id'] in ensemble['depends_on_blocks']
        assert scalar['block_id'] in _ancestors(joint, [ensemble['block_id']])
        assert ensemble['block_id'] not in _ancestors(joint, [scalar['block_id']])


def test_endpoint_map_is_deduplicated_without_extra_trees():
    _, _, _, _, new = _run('D0-M')
    blocks = new['blocks']
    endpoints = new['endpoint_blocks_by_model']
    assert 'D0-M' in endpoints
    assert list(endpoints).count('D0-M') == 1
    assert 'C-EXTRA-TREES' not in endpoints
    assert 'C-EXTRA-TREES' not in {block['model_id'] for block in blocks}
    block_ids = {block['block_id'] for block in blocks}
    assert all(endpoint in block_ids for endpoint in endpoints.values())


def test_three_source_units_scale():
    _, _, _, _, new = _run('D0-M', unit_count=3)
    blocks = new['blocks']
    assert _slots(blocks, 'final_source_fit') == 261
    assert _slots(blocks, 'final_calibration_model_fit') == 12
    assert _slots(blocks, 'final_refit') == 7
    assert 261 + 12 + 7 == 280
    assert len(new['endpoint_blocks_by_model']) == 3
    block_ids = [block['block_id'] for block in blocks]
    assert len(block_ids) == len(set(block_ids))


def test_blocks_sorted_and_inputs_unmutated():
    raw = _make_context(unit_count=2)
    candidates = _classical_candidates()
    policy = build_policy_blocks(_binding_sha(), raw, _make_candidates())
    context = _context(raw, 'D0-M')
    raw_before = copy.deepcopy(raw)
    candidates_before = copy.deepcopy(candidates)
    policy_before = copy.deepcopy(policy)
    context_before = copy.deepcopy(context)
    new = _build_final_blocks(_binding_sha(), context, candidates, policy)
    assert raw == raw_before
    assert candidates == candidates_before
    assert policy == policy_before
    assert context == context_before
    blocks = new['blocks']
    assert blocks == sorted(blocks, key=lambda block: block['block_id'])
    assert all(block['block_id'].startswith('P08QCBLOCK-') for block in blocks)
    assert all(block['stage'].startswith('final_') for block in blocks)


def test_permutation_preserves_block_identity():
    raw = _make_context(unit_count=2)
    candidates = _classical_candidates()
    policy = build_policy_blocks(_binding_sha(), raw, _make_candidates())
    context = _context(raw, 'D0-M')
    base = _build_final_blocks(_binding_sha(), context, candidates, policy)

    permuted_context = copy.deepcopy(context)
    permuted_context['selection_units'].reverse()
    permuted_context['calibration_units'].reverse()
    permuted_candidates = {
        model_id: list(reversed(pairs)) for model_id, pairs in candidates.items()
    }
    permuted = _build_final_blocks(
        _binding_sha(), permuted_context, permuted_candidates, policy
    )

    base_ids = [block['block_id'] for block in base['blocks']]
    permuted_ids = [block['block_id'] for block in permuted['blocks']]
    assert base_ids == permuted_ids
    assert base['endpoint_blocks_by_model'] == permuted['endpoint_blocks_by_model']


def test_require_scientific_execution_denied():
    with pytest.raises(ValueError, match='scientific_execution_not_authorized'):
        require_scientific_execution()
    with pytest.raises(ValueError, match='scientific_execution_not_authorized'):
        require_scientific_execution(1, 2, binding_sha256='x')


def test_seal_catalog_accepts_policy_and_final_blocks():
    _, _, policy, _, new = _run('D3')
    sealed = seal_catalog(_binding_obj(), policy['blocks'] + new['blocks'], [])
    assert sealed['summary']['stage_counts']
    assert sealed['summary']['stage_block_counts']
