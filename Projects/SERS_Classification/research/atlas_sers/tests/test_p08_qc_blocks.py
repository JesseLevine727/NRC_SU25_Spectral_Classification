"""Compact tests for the P08-T012 no-fit block primitives."""
import hashlib
import json

import pytest

from atlas_sers.evaluation import p08_qc_blocks as qc

HEX = 'a' * 64
N = 'not_applicable'
BIND = {'split': 'demo', 'note': 'ø'}


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def sha(obj):
    return hashlib.sha256(canon(obj).encode('utf-8')).hexdigest()


BHASH = sha(BIND)


def ax(**over):
    base = {
        'gate_id': ['g'],
        'candidate': [
            {'candidate_id': 'c', 'hyperparameter_sha256': N}],
        'seed': [0],
    }
    base.update(over)
    return base


def mkblock(**over):
    kw = dict(
        binding_sha256=BHASH, context_id='ctx', stage='fit',
        model_id='model', role_id='role', fit_uid_sha256=N,
        validation_uid_sha256=N, test_uid_sha256=N,
        axes={
            'gate_id': ['g1', 'g2'],
            'candidate': [
                {'candidate_id': 'c1', 'hyperparameter_sha256': N}],
            'seed': [0, 1, 'deterministic'],
        },
        depends_on_blocks=[], resolution='res')
    kw.update(over)
    return qc.make_block(**kw)


def mkalias(**over):
    kw = dict(context_id='ctx', strategy='strat', recipe_id='recipe',
              target_block_id=qc.FALLBACK_TARGET, evidence_status='ok',
              reason_code='none', metadata={})
    kw.update(over)
    return qc.make_alias(**kw)


def mkcatalog(blocks=(), aliases=(), bindings=BIND):
    return qc.seal_catalog(bindings, list(blocks), list(aliases))


def test_manual_block_and_slot_hashes():
    b = mkblock(axes=ax(), resolution='r')
    payload = {k: v for k, v in b.items() if k != 'block_id'}
    assert b['block_id'] == 'P08QCBLOCK-' + sha(payload)
    assert b['slot_count'] == 1
    slot = next(qc.iter_slots(mkcatalog([b])))
    expected = 'P08QCSLOT-' + sha({
        'block_id': b['block_id'], 'gate_id': 'g',
        'candidate': {'candidate_id': 'c', 'hyperparameter_sha256': N},
        'seed': 0})
    assert slot['slot_id'] == expected


def test_six_slot_product_seed_candidate_pairing():
    cat = mkcatalog([mkblock()])
    slots = list(qc.iter_slots(cat))
    assert len(slots) == 6
    assert len({s['slot_id'] for s in slots}) == 6
    triples = {
        (s['gate_id'], s['candidate']['candidate_id'], s['seed'])
        for s in slots}
    assert len(triples) == 6
    for slot in slots:
        assert slot['candidate'] == {
            'candidate_id': 'c1', 'hyperparameter_sha256': N}


def test_candidate_hash_pairing_preserved():
    axes = ax(candidate=[
        {'candidate_id': 'b', 'hyperparameter_sha256': HEX},
        {'candidate_id': 'a', 'hyperparameter_sha256': N}], seed=[0])
    slots = qc.iter_slots(mkcatalog([mkblock(axes=axes)]))
    got = {
        s['candidate']['candidate_id']:
            s['candidate']['hyperparameter_sha256']
        for s in slots}
    assert got == {'a': N, 'b': HEX}


def test_axes_and_dependency_canonical_order():
    b = mkblock(
        axes=ax(gate_id=['b', 'a'], seed=['deterministic', 2, 0]),
        depends_on_blocks=[
            'P08QCBLOCK-' + 'f' * 64,
            'P08QCBLOCK-' + '0' * 64])
    assert b['axes']['gate_id'] == ['a', 'b']
    assert b['axes']['seed'] == ['deterministic', 0, 2]
    assert b['depends_on_blocks'] == [
        'P08QCBLOCK-' + '0' * 64, 'P08QCBLOCK-' + 'f' * 64]


def test_catalog_permutation_identity():
    a = mkblock(stage='fit')
    b = mkblock(stage='eval', model_id='m2')
    c1, c2 = mkcatalog([a, b]), mkcatalog([b, a])
    assert c1 == c2
    assert c1['catalog_sha256'] == c2['catalog_sha256']


def test_axes_permutation_identity():
    a1 = mkblock(axes={
        'gate_id': ['b', 'a'],
        'candidate': [
            {'candidate_id': 'c2', 'hyperparameter_sha256': N},
            {'candidate_id': 'c1', 'hyperparameter_sha256': N}],
        'seed': [2, 0, 1]})
    a2 = mkblock(axes={
        'gate_id': ['a', 'b'],
        'candidate': [
            {'candidate_id': 'c1', 'hyperparameter_sha256': N},
            {'candidate_id': 'c2', 'hyperparameter_sha256': N}],
        'seed': [0, 1, 2]})
    assert a1['axes'] == a2['axes']
    assert a1['block_id'] == a2['block_id']
    assert a1['slot_count'] == a2['slot_count'] == 12


def test_alias_permutation_identity():
    m1 = {'a': 1, 'b': [2, {'c': 3}]}
    m2 = {'b': [2, {'c': 3}], 'a': 1}
    assert (mkalias(metadata=m1)['alias_id']
            == mkalias(metadata=m2)['alias_id'])


def test_unicode_canonicalization():
    assert 'é' in qc.canonical_json({'x': 'é'})
    escaped = hashlib.sha256(
        json.dumps({'x': 'é'}, sort_keys=True, ensure_ascii=True,
                   separators=(',', ':')).encode('utf-8')).hexdigest()
    assert qc.canonical_sha256({'x': 'é'}) != escaped
    b = mkblock(context_id='cøntext', stage='fé', resolution='rés')
    assert next(
        qc.iter_slots(mkcatalog([b])))['context_id'] == 'cøntext'


def test_non_mutation_of_inputs_and_catalog():
    axes = {
        'gate_id': ['b', 'a'],
        'candidate': [
            {'candidate_id': 'c', 'hyperparameter_sha256': N}],
        'seed': [1, 0]}
    axes_snap = json.loads(json.dumps(axes))
    b = mkblock(axes=axes)
    assert axes == axes_snap
    meta = {'k': ['x', {'y': 1}]}
    meta_snap = json.loads(json.dumps(meta))
    mkalias(metadata=meta)
    assert meta == meta_snap
    cat = mkcatalog([b])
    cat_snap = json.loads(json.dumps(cat))
    list(qc.iter_slots(cat))
    assert cat == cat_snap


def test_catalog_copy_isolated_from_source_mutation():
    bindings = {'x': [1, {'y': 2}]}
    metadata = {'m': [1, {'z': 3}]}
    block = mkblock(binding_sha256=sha(bindings))
    cat = qc.seal_catalog(
        bindings, [block], [mkalias(metadata=metadata)])
    frozen = qc.canonical_json(cat)
    bindings['x'][1]['y'] = 99
    metadata['m'][1]['z'] = 99
    block['slot_count'] = 999
    block['axes']['gate_id'].append('extra')
    assert qc.canonical_json(cat) == frozen
    assert cat['bindings']['x'][1]['y'] == 2
    assert cat['blocks'][0]['slot_count'] == 6


def test_slots_are_fresh_objects():
    cat = mkcatalog([mkblock()])
    first = list(qc.iter_slots(cat))
    first[0]['candidate']['candidate_id'] = 'mutated'
    first[0]['depends_on_blocks'].append('P08QCBLOCK-' + '0' * 64)
    again = list(qc.iter_slots(cat))
    assert again[0]['candidate']['candidate_id'] == 'c1'
    assert again[0]['depends_on_blocks'] == []


def test_empty_catalog_with_fallback_aliases():
    cat = mkcatalog([], [mkalias(), mkalias(strategy='s2')])
    assert cat['execution_authorized'] is False
    assert cat['summary'] == {
        'block_count': 0,
        'expanded_operation_slots': 0,
        'stage_counts': {},
        'stage_block_counts': {},
        'alias_count': 2,
    }
    assert list(qc.iter_slots(cat)) == []


def test_valid_chain_counts_and_nonfallback_alias():
    a = mkblock(stage='fit', model_id='ma')
    b = mkblock(stage='eval', model_id='mb',
                depends_on_blocks=[a['block_id']])
    cat = mkcatalog([a, b], [mkalias(target_block_id=b['block_id'])])
    assert cat['summary'] == {
        'block_count': 2,
        'expanded_operation_slots': 12,
        'stage_counts': {'eval': 6, 'fit': 6},
        'stage_block_counts': {'eval': 1, 'fit': 1},
        'alias_count': 1,
    }
    deps = {s['block_id']: s['depends_on_blocks']
            for s in qc.iter_slots(cat)}
    assert deps[b['block_id']] == [a['block_id']]
    assert deps[a['block_id']] == []


def test_stage_counts_sum_expanded_slots():
    a = mkblock(stage='fit', model_id='ma')
    b = mkblock(stage='eval', model_id='mb')
    cat = mkcatalog([a, b])
    assert cat['summary']['stage_counts'] == {'eval': 6, 'fit': 6}
    assert cat['summary']['stage_block_counts'] == {'eval': 1, 'fit': 1}


def test_stream_first_slot_of_large_product_only():
    gates = [f'g{i:06d}' for i in range(20000)]
    b = mkblock(axes=ax(gate_id=gates))
    assert b['slot_count'] == 20000
    first = next(qc.iter_slots(mkcatalog([b])))
    assert first['gate_id'] == 'g000000'


BLOCK_BAD = [
    ({'context_id': ''}, 'invalid_context_id'),
    ({'context_id': ' x'}, 'invalid_context_id'),
    ({'context_id': 7}, 'invalid_context_id'),
    ({'stage': ''}, 'invalid_stage'),
    ({'stage': 'e '}, 'invalid_stage'),
    ({'model_id': None}, 'invalid_model_id'),
    ({'role_id': 'r\t'}, 'invalid_role_id'),
    ({'resolution': ''}, 'invalid_resolution'),
    ({'resolution': 3}, 'invalid_resolution'),
    ({'binding_sha256': 'A' * 64}, 'invalid_binding_sha256'),
    ({'binding_sha256': 'a' * 63}, 'invalid_binding_sha256'),
    ({'binding_sha256': N}, 'invalid_binding_sha256'),
    ({'binding_sha256': None}, 'invalid_binding_sha256'),
    ({'fit_uid_sha256': 'zz'}, 'invalid_fit_uid_sha256'),
    ({'validation_uid_sha256': 5}, 'invalid_validation_uid_sha256'),
    ({'test_uid_sha256': 'A' * 64}, 'invalid_test_uid_sha256'),
    ({'axes': []}, 'invalid_axes'),
    ({'axes': {'gate_id': ['g'], 'candidate': []}}, 'invalid_axes'),
    ({'axes': ax(gate_id=[])}, 'invalid_gate_id'),
    ({'axes': ax(gate_id=[1])}, 'invalid_gate_id'),
    ({'axes': ax(gate_id=['g', 'g'])}, 'duplicate_gate_id'),
    ({'axes': ax(candidate={})}, 'invalid_candidate'),
    ({'axes': ax(candidate=[])}, 'invalid_candidate'),
    ({'axes': ax(candidate=[{'candidate_id': 'c'}])},
     'invalid_candidate'),
    ({'axes': ax(candidate=[
        {'candidate_id': 'c', 'hyperparameter_sha256': N,
         'x': 1}])}, 'invalid_candidate'),
    ({'axes': ax(candidate=[
        {'candidate_id': '', 'hyperparameter_sha256': N}])},
     'invalid_candidate_id'),
    ({'axes': ax(candidate=[
        {'candidate_id': 'c', 'hyperparameter_sha256': 'Z' * 64}])},
     'invalid_hyperparameter_sha256'),
    ({'axes': ax(candidate=[
        {'candidate_id': 'c', 'hyperparameter_sha256': N},
        {'candidate_id': 'c', 'hyperparameter_sha256': HEX}])},
     'duplicate_candidate_id'),
    ({'axes': ax(seed=[])}, 'invalid_seed'),
    ({'axes': ax(seed=[-1])}, 'invalid_seed'),
    ({'axes': ax(seed=[True])}, 'invalid_seed'),
    ({'axes': ax(seed=['bogus'])}, 'invalid_seed'),
    ({'axes': ax(seed=[1, 1])}, 'duplicate_seed'),
    ({'depends_on_blocks': 'x'}, 'invalid_depends_on_blocks'),
    ({'depends_on_blocks': [qc.BLOCK_PREFIX + 'z' * 64]},
     'invalid_depends_on_blocks'),
    ({'depends_on_blocks': [qc.BLOCK_PREFIX]},
     'invalid_depends_on_blocks'),
    ({'depends_on_blocks': ['P08QCBLOCK-' + HEX,
                            'P08QCBLOCK-' + HEX]},
     'duplicate_dependency'),
]


@pytest.mark.parametrize('over,code', BLOCK_BAD)
def test_make_block_static_errors(over, code):
    with pytest.raises(qc.BlockError) as ei:
        mkblock(**over)
    assert str(ei.value) == code
    assert ei.value.reason_code == code


@pytest.mark.parametrize('over,code', [
    ({'context_id': ''}, 'invalid_context_id'),
    ({'strategy': ' s'}, 'invalid_strategy'),
    ({'recipe_id': 1}, 'invalid_recipe_id'),
    ({'evidence_status': ''}, 'invalid_evidence_status'),
    ({'reason_code': ' '}, 'invalid_reason_code'),
    ({'target_block_id': 'nope'}, 'invalid_target_block_id'),
    ({'target_block_id': 'P08QCBLOCK-' + 'Z' * 64},
     'invalid_target_block_id'),
    ({'target_block_id': 5}, 'invalid_target_block_id'),
    ({'metadata': []}, 'invalid_metadata'),
])
def test_make_alias_static_errors(over, code):
    with pytest.raises(qc.BlockError) as ei:
        mkalias(**over)
    assert str(ei.value) == code


def test_seal_block_schema_and_hash_errors():
    b = mkblock()
    extra = dict(b, nope=1)
    missing = {k: v for k, v in b.items() if k != 'resolution'}
    cases = [
        (extra, 'invalid_block_schema'),
        (missing, 'invalid_block_schema'),
        (dict(b, schema_version='x'), 'invalid_block_schema'),
        (dict(b, block_id='P08QCBLOCK-' + '0' * 64),
         'block_hash_mismatch'),
        (dict(b, slot_count=b['slot_count'] + 1),
         'block_hash_mismatch'),
        (dict(b, slot_count=True), 'block_hash_mismatch'),
        (dict(b, slot_count=float(b['slot_count'])),
         'block_hash_mismatch'),
        (dict(b, slot_count=0), 'block_hash_mismatch'),
    ]
    for bad, code in cases:
        with pytest.raises(qc.BlockError) as ei:
            qc.seal_catalog(BIND, [bad], [])
        assert str(ei.value) == code
    with pytest.raises(qc.BlockError) as ei:
        qc.seal_catalog(BIND, [b, b], [])
    assert str(ei.value) == 'duplicate_block'
    other = mkblock(binding_sha256=HEX)
    with pytest.raises(qc.BlockError) as ei:
        qc.seal_catalog(BIND, [other], [])
    assert str(ei.value) == 'binding_hash_mismatch'


def test_seal_dependency_errors():
    missing = mkblock(depends_on_blocks=['P08QCBLOCK-' + HEX])
    with pytest.raises(qc.BlockError) as ei:
        qc.seal_catalog(BIND, [missing], [])
    assert str(ei.value) == 'dependency_missing'
    a = mkblock(context_id='ctx-a')
    b = mkblock(context_id='ctx-b',
                depends_on_blocks=[a['block_id']])
    with pytest.raises(qc.BlockError) as ei:
        qc.seal_catalog(BIND, [a, b], [])
    assert str(ei.value) == 'dependency_missing'


def test_cycle_helper_direct_and_long_chain():
    a = {'block_id': 'P08QCBLOCK-' + '0' * 64,
         'depends_on_blocks': ['P08QCBLOCK-' + '1' * 64]}
    b = {'block_id': 'P08QCBLOCK-' + '1' * 64,
         'depends_on_blocks': ['P08QCBLOCK-' + '0' * 64]}
    with pytest.raises(qc.BlockError) as ei:
        qc.assert_acyclic([a, b])
    assert str(ei.value) == 'catalog_dependency_cycle'
    self_dep = {'block_id': 'P08QCBLOCK-' + '2' * 64,
                'depends_on_blocks': ['P08QCBLOCK-' + '2' * 64]}
    with pytest.raises(qc.BlockError):
        qc.assert_acyclic([self_dep])
    chain = [
        {'block_id': 'P08QCBLOCK-' + format(i, '064x'),
         'depends_on_blocks':
             ([] if i == 0
              else ['P08QCBLOCK-' + format(i - 1, '064x')])}
        for i in range(2000)]
    assert qc.assert_acyclic(chain) == 2000


def test_seal_alias_errors_and_target_scope():
    al = mkalias()
    cases = [
        (dict(al, nope=1), 'invalid_alias_schema'),
        (dict(al, alias_id='P08QCALIAS-' + '0' * 64),
         'alias_hash_mismatch'),
    ]
    for bad, code in cases:
        with pytest.raises(qc.BlockError) as ei:
            qc.seal_catalog(BIND, [], [bad])
        assert str(ei.value) == code
    with pytest.raises(qc.BlockError) as ei:
        qc.seal_catalog(BIND, [], [al, al])
    assert str(ei.value) == 'duplicate_alias'
    with pytest.raises(qc.BlockError) as ei:
        qc.seal_catalog(BIND, [], [al, mkalias(reason_code='other')])
    assert str(ei.value) == 'duplicate_alias_scope'
    with pytest.raises(qc.BlockError) as ei:
        qc.seal_catalog(
            BIND, [], [mkalias(target_block_id='P08QCBLOCK-' + HEX)])
    assert str(ei.value) == 'alias_target_missing'
    a = mkblock(context_id='ctx-a')
    cross = mkalias(context_id='ctx-b', target_block_id=a['block_id'])
    with pytest.raises(qc.BlockError) as ei:
        qc.seal_catalog(BIND, [a], [cross])
    assert str(ei.value) == 'alias_target_missing'


@pytest.mark.parametrize('bindings', [
    {'x': float('inf')},
    {'x': float('nan')},
    {'x': {1, 2}},
    {'x': (1, 2)},
    {1: 'x'},
    {'x': {'y': {1, 2}}},
    {'x': '\ud800'},
    [],
])
def test_seal_bindings_strict_json_errors(bindings):
    with pytest.raises(qc.BlockError) as ei:
        qc.seal_catalog(bindings, [], [])
    assert ei.value.reason_code == 'invalid_bindings'


@pytest.mark.parametrize('metadata', [
    float('inf'),
    {'x': {1}},
    {'x': float('nan')},
    {'x': (1, 2)},
    {1: 'x'},
    {'x': {'y': {1, 2}}},
    {'x': '\ud800'},
])
def test_alias_metadata_strict_json_errors(metadata):
    with pytest.raises(qc.BlockError) as ei:
        mkalias(metadata=metadata)
    assert ei.value.reason_code == 'invalid_metadata'


def test_iter_slots_catalog_schema_and_hash_forgery():
    cat = mkcatalog([mkblock()])
    with pytest.raises(qc.BlockError) as ei:
        list(qc.iter_slots(dict(cat, catalog_sha256='0' * 64)))
    assert str(ei.value) == 'catalog_hash_mismatch'
    with pytest.raises(qc.BlockError) as ei:
        list(qc.iter_slots(dict(cat, extra=1)))
    assert str(ei.value) == 'invalid_catalog_schema'
    with pytest.raises(qc.BlockError) as ei:
        list(qc.iter_slots(dict(cat, schema_version='other')))
    assert str(ei.value) == 'invalid_catalog_schema'


def test_catalog_digest_must_be_lower_hex64():
    cat = mkcatalog([mkblock()])
    for bad_digest in ('A' * 64, 'a' * 63, 'z' * 64, 123, None, True):
        broken = dict(cat, catalog_sha256=bad_digest)
        with pytest.raises(qc.BlockError) as ei:
            list(qc.iter_slots(broken))
        assert ei.value.reason_code == 'catalog_hash_mismatch'


def test_slot_count_type_forgery_rejected():
    cat = mkcatalog([mkblock()])

    old_hash = json.loads(json.dumps(cat))
    old_hash['blocks'][0]['slot_count'] = True
    with pytest.raises(qc.BlockError) as ei:
        list(qc.iter_slots(old_hash))
    assert ei.value.reason_code == 'catalog_hash_mismatch'

    for bad_count in (True, 1.0, 0, -1):
        forged = json.loads(json.dumps(cat))
        forged['blocks'][0]['slot_count'] = bad_count
        body = {k: v for k, v in forged.items() if k != 'catalog_sha256'}
        forged['catalog_sha256'] = sha(body)
        with pytest.raises(qc.BlockError) as ei:
            list(qc.iter_slots(forged))
        assert ei.value.reason_code == 'block_hash_mismatch'


def test_forged_authorization_rejected_even_with_recomputed_hash():
    cat = mkcatalog([mkblock()])
    forged = dict(cat, execution_authorized=True)
    body = {k: v for k, v in forged.items() if k != 'catalog_sha256'}
    forged['catalog_sha256'] = sha(body)
    with pytest.raises(qc.BlockError) as ei:
        list(qc.iter_slots(forged))
    assert str(ei.value) == 'execution_not_authorized'


def test_require_scientific_execution_always_denies():
    with pytest.raises(qc.BlockError) as ei:
        qc.require_scientific_execution(True, authorized=True, token='x')
    assert str(ei.value) == 'scientific_execution_not_authorized'


def test_block_error_exposes_reason_code():
    err = qc.BlockError('some_code')
    assert isinstance(err, ValueError)
    assert str(err) == 'some_code'
    assert err.reason_code == 'some_code'


def test_error_codes_do_not_leak_private_values():
    with pytest.raises(qc.BlockError) as ei:
        mkblock(context_id='  secret-id  ')
    assert str(ei.value) == 'invalid_context_id'
    assert ei.value.reason_code == 'invalid_context_id'
    assert 'secret-id' not in str(ei.value)


def test_canonical_json_strict_primitives():
    text = qc.canonical_json({'a': [1, 1.5, True, None, 'x']})
    assert text == '{"a":[1,1.5,true,null,"x"]}'
    assert qc.canonical_sha256({'a': 1}) == sha({'a': 1})


@pytest.mark.parametrize('value', [
    {'x': float('inf')},
    {'x': float('nan')},
    {'x': {1, 2}},
    {'x': (1, 2)},
    {1: 'x'},
    {'x': {'y': {1, 2}}},
    {'x': '\ud800'},
    {'x': object()},
])
def test_canonical_json_rejects_non_json(value):
    with pytest.raises(qc.BlockError) as ei:
        qc.canonical_json(value)
    assert ei.value.reason_code == 'invalid_json_value'
    with pytest.raises(qc.BlockError) as ei:
        qc.canonical_sha256(value)
    assert ei.value.reason_code == 'invalid_json_value'


def test_canonical_json_rejects_mixed_key_collision():
    with pytest.raises(qc.BlockError) as ei:
        qc.canonical_json({1: 'a', '1': 'b'})
    assert ei.value.reason_code == 'invalid_json_value'


def test_canonical_json_rejects_cycle():
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    with pytest.raises(qc.BlockError) as ei:
        qc.canonical_json(cyclic_list)
    assert ei.value.reason_code == 'invalid_json_value'
    cyclic_dict = {}
    cyclic_dict['self'] = cyclic_dict
    with pytest.raises(qc.BlockError) as ei:
        qc.canonical_json(cyclic_dict)
    assert ei.value.reason_code == 'invalid_json_value'


def test_canonical_json_rejects_deep_recursion():
    root = []
    cursor = root
    for _ in range(5000):
        child = []
        cursor.append(child)
        cursor = child
    with pytest.raises(qc.BlockError) as ei:
        qc.canonical_json(root)
    assert ei.value.reason_code == 'invalid_json_value'


def test_no_fit_boundary_and_metadata_scope_warning():
    b = mkblock(stage='fit', fit_uid_sha256=N)
    assert b['fit_uid_sha256'] == N
    cat = mkcatalog([b], [mkalias(metadata={'model_hash': HEX})])
    assert cat['aliases'][0]['metadata']['model_hash'] == HEX
    assert cat['execution_authorized'] is False
