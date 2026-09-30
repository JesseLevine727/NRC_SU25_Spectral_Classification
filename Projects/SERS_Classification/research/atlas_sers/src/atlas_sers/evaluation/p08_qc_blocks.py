"""Bounded, no-fit Cartesian QC block primitives for P08-T012.

Scope is deliberately limited to generic bookkeeping.  This module
canonicalises declarative block/alias metadata, computes content hashes,
checks structural consistency (uniqueness, same-context
targets/dependencies, acyclicity) and streams Cartesian slot descriptors.

It computes nothing scientific.  Stage strings are shape-checked only and
are interpreted later by a workflow factory.  Metadata hashes
(role/input/spec) are copied verbatim and do NOT prove that any data,
inputs or splits are disjoint.  An alias on a block is not evidence that
the data are disjoint.  Aliases create no jobs and authorize no
execution.  Callers may bind full role/spec metadata in ``bindings``; no
actual model input is required or evaluated here.
"""
import hashlib
import json
import math

SCHEMA_VERSION = 'nato-sers-p08-qc-block-catalog-v1'
BLOCK_PREFIX = 'P08QCBLOCK-'
SLOT_PREFIX = 'P08QCSLOT-'
ALIAS_PREFIX = 'P08QCALIAS-'
FALLBACK_TARGET = 'PP-U-MIN-COMPLETE-PIPELINE'
NOT_APPLICABLE = 'not_applicable'
SEED_DETERMINISTIC = 'deterministic'

_HEX_CHARS = frozenset('0123456789abcdef')
_BLOCK_FIELDS = frozenset({
    'schema_version', 'binding_sha256', 'context_id', 'stage', 'model_id',
    'role_id', 'fit_uid_sha256', 'validation_uid_sha256', 'test_uid_sha256',
    'axes', 'depends_on_blocks', 'resolution', 'slot_count', 'block_id'})
_ALIAS_FIELDS = frozenset({
    'schema_version', 'context_id', 'strategy', 'recipe_id',
    'target_block_id', 'evidence_status', 'reason_code', 'metadata',
    'alias_id'})
_CATALOG_FIELDS = frozenset({
    'schema_version', 'execution_authorized', 'bindings', 'blocks',
    'aliases', 'summary', 'catalog_sha256'})
_AXES_FIELDS = frozenset({'gate_id', 'candidate', 'seed'})
_CANDIDATE_FIELDS = frozenset({'candidate_id', 'hyperparameter_sha256'})

__all__ = [
    'SCHEMA_VERSION', 'BLOCK_PREFIX', 'SLOT_PREFIX', 'ALIAS_PREFIX',
    'FALLBACK_TARGET', 'NOT_APPLICABLE', 'SEED_DETERMINISTIC', 'BlockError',
    'canonical_json', 'canonical_sha256', 'make_block', 'make_alias',
    'seal_catalog', 'iter_slots', 'assert_acyclic',
    'require_scientific_execution']


class BlockError(ValueError):
    """ValueError carrying one static reason code (never private values)."""

    def __init__(self, reason_code):
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(code):
    raise BlockError(code)


def _check_text(text):
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        _fail('invalid_json_value')


def _validate_json(value, _seen=None):
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        _check_text(value)
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail('invalid_json_value')
        return
    if isinstance(value, (dict, list)):
        marker = id(value)
        if marker in _seen:
            _fail('invalid_json_value')
        _seen.add(marker)
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    _fail('invalid_json_value')
                _check_text(key)
                _validate_json(item, _seen)
        else:
            for item in value:
                _validate_json(item, _seen)
        _seen.discard(marker)
        return
    _fail('invalid_json_value')


def _canonical_json(value):
    _validate_json(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def canonical_json(value):
    """Canonical UTF-8 JSON text; static ``invalid_json_value`` on bad input."""
    try:
        text = _canonical_json(value)
        text.encode('utf-8')
    except BlockError:
        raise
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise BlockError('invalid_json_value') from None
    return text


def canonical_sha256(value):
    """SHA-256 hex digest of the canonical UTF-8 JSON encoding."""
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def _sha(value):
    text = _canonical_json(value)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _is_hex64(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in _HEX_CHARS for ch in value)


def _require_id(value, code):
    if not isinstance(value, str) or value == '' or value != value.strip():
        _fail(code)
    return value


def _require_hash(value, code, allow_not_applicable=False):
    if allow_not_applicable and value == NOT_APPLICABLE:
        return value
    if not _is_hex64(value):
        _fail(code)
    return value


def _copy_json_safe(value, code):
    try:
        _validate_json(value)
        text = json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(',', ':'), allow_nan=False)
        return json.loads(text)
    except BlockError:
        raise BlockError(code) from None
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise BlockError(code) from None


def _valid_seed(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, str):
        return value in (SEED_DETERMINISTIC, NOT_APPLICABLE)
    return False


def _normalize_axes(axes):
    if (not isinstance(axes, dict)
            or set(axes.keys()) != _AXES_FIELDS):
        _fail('invalid_axes')
    gates = axes['gate_id']
    if not isinstance(gates, list) or not gates:
        _fail('invalid_gate_id')
    gate_ids, seen_gates = [], set()
    for gate in gates:
        gate = _require_id(gate, 'invalid_gate_id')
        if gate in seen_gates:
            _fail('duplicate_gate_id')
        seen_gates.add(gate)
        gate_ids.append(gate)
    cands = axes['candidate']
    if not isinstance(cands, list) or not cands:
        _fail('invalid_candidate')
    norm_cands, seen_cands = [], set()
    for cand in cands:
        if (not isinstance(cand, dict)
                or set(cand.keys()) != _CANDIDATE_FIELDS):
            _fail('invalid_candidate')
        cid = _require_id(cand['candidate_id'], 'invalid_candidate_id')
        hp = _require_hash(
            cand['hyperparameter_sha256'],
            'invalid_hyperparameter_sha256', allow_not_applicable=True)
        if cid in seen_cands:
            _fail('duplicate_candidate_id')
        seen_cands.add(cid)
        norm_cands.append(
            {'candidate_id': cid, 'hyperparameter_sha256': hp})
    seeds = axes['seed']
    if not isinstance(seeds, list) or not seeds:
        _fail('invalid_seed')
    norm_seeds, seen_seeds = [], set()
    for seed in seeds:
        if not _valid_seed(seed):
            _fail('invalid_seed')
        if seed in seen_seeds:
            _fail('duplicate_seed')
        seen_seeds.add(seed)
        norm_seeds.append(seed)
    norm_cands.sort(key=lambda c: c['candidate_id'])
    norm_seeds.sort(key=_canonical_json)
    return {
        'gate_id': sorted(gate_ids),
        'candidate': norm_cands,
        'seed': norm_seeds,
    }


def _normalize_deps(deps):
    if not isinstance(deps, list):
        _fail('invalid_depends_on_blocks')
    out, seen = [], set()
    for dep in deps:
        if not isinstance(dep, str) or not dep.startswith(BLOCK_PREFIX):
            _fail('invalid_depends_on_blocks')
        if not _is_hex64(dep[len(BLOCK_PREFIX):]):
            _fail('invalid_depends_on_blocks')
        if dep in seen:
            _fail('duplicate_dependency')
        seen.add(dep)
        out.append(dep)
    out.sort()
    return out


def make_block(*, binding_sha256, context_id, stage, model_id, role_id,
               fit_uid_sha256, validation_uid_sha256, test_uid_sha256,
               axes, depends_on_blocks, resolution):
    """Build a fresh content-addressed block; no caller id/count inputs."""
    try:
        binding = _require_hash(binding_sha256, 'invalid_binding_sha256')
        ctx = _require_id(context_id, 'invalid_context_id')
        stage_id = _require_id(stage, 'invalid_stage')
        model = _require_id(model_id, 'invalid_model_id')
        role = _require_id(role_id, 'invalid_role_id')
        fit = _require_hash(fit_uid_sha256, 'invalid_fit_uid_sha256', True)
        val = _require_hash(
            validation_uid_sha256, 'invalid_validation_uid_sha256', True)
        tst = _require_hash(
            test_uid_sha256, 'invalid_test_uid_sha256', True)
        norm_axes = _normalize_axes(axes)
        deps = _normalize_deps(depends_on_blocks)
        res = _require_id(resolution, 'invalid_resolution')
        slot_count = (
            len(norm_axes['gate_id'])
            * len(norm_axes['candidate'])
            * len(norm_axes['seed']))
        payload = {
            'schema_version': SCHEMA_VERSION,
            'binding_sha256': binding,
            'context_id': ctx,
            'stage': stage_id,
            'model_id': model,
            'role_id': role,
            'fit_uid_sha256': fit,
            'validation_uid_sha256': val,
            'test_uid_sha256': tst,
            'axes': norm_axes,
            'depends_on_blocks': deps,
            'resolution': res,
            'slot_count': slot_count,
        }
        block = dict(payload)
        block['block_id'] = BLOCK_PREFIX + _sha(payload)
        return block
    except BlockError:
        raise
    except Exception:
        raise BlockError('invalid_block_input') from None


def make_alias(*, context_id, strategy, recipe_id, target_block_id,
               evidence_status, reason_code, metadata):
    """Build a fresh alias descriptor; aliases create no jobs."""
    try:
        ctx = _require_id(context_id, 'invalid_context_id')
        strat = _require_id(strategy, 'invalid_strategy')
        recipe = _require_id(recipe_id, 'invalid_recipe_id')
        ev = _require_id(evidence_status, 'invalid_evidence_status')
        rc = _require_id(reason_code, 'invalid_reason_code')
        if (not isinstance(target_block_id, str)
                or target_block_id == ''
                or target_block_id != target_block_id.strip()):
            _fail('invalid_target_block_id')
        if target_block_id != FALLBACK_TARGET:
            suffix = target_block_id[len(BLOCK_PREFIX):]
            if (not target_block_id.startswith(BLOCK_PREFIX)
                    or not _is_hex64(suffix)):
                _fail('invalid_target_block_id')
        if not isinstance(metadata, dict):
            _fail('invalid_metadata')
        meta = _copy_json_safe(metadata, 'invalid_metadata')
        payload = {
            'schema_version': SCHEMA_VERSION,
            'context_id': ctx,
            'strategy': strat,
            'recipe_id': recipe,
            'target_block_id': target_block_id,
            'evidence_status': ev,
            'reason_code': rc,
            'metadata': meta,
        }
        alias = dict(payload)
        alias['alias_id'] = ALIAS_PREFIX + _sha(payload)
        return alias
    except BlockError:
        raise
    except Exception:
        raise BlockError('invalid_alias_input') from None


def assert_acyclic(blocks):
    """Iterative topological check over ``{block_id, depends_on_blocks}``."""
    try:
        graph = {}
        for block in blocks:
            node = graph.setdefault(block['block_id'], set())
            for dep in block['depends_on_blocks']:
                graph.setdefault(dep, set())
                node.add(dep)
        dependents = {node: set() for node in graph}
        indegree = {node: len(graph[node]) for node in graph}
        for node, deps in graph.items():
            for dep in deps:
                dependents[dep].add(node)
        queue = [node for node, deg in indegree.items() if deg == 0]
        processed = 0
        while queue:
            node = queue.pop()
            processed += 1
            for follower in dependents[node]:
                indegree[follower] -= 1
                if indegree[follower] == 0:
                    queue.append(follower)
        if processed != len(graph):
            _fail('catalog_dependency_cycle')
        return processed
    except BlockError:
        raise
    except Exception:
        raise BlockError('catalog_dependency_cycle') from None


def seal_catalog(bindings, blocks, aliases):
    """Validate and rebuild a catalog; returns a fresh hash-stamped structure."""
    try:
        if not isinstance(bindings, dict):
            _fail('invalid_bindings')
        bindings_copy = _copy_json_safe(bindings, 'invalid_bindings')
        binding_sha = _sha(bindings_copy)
        if not isinstance(blocks, list):
            _fail('invalid_blocks')
        if not isinstance(aliases, list):
            _fail('invalid_aliases')

        normalized_blocks, seen_block_ids = [], set()
        for raw in blocks:
            if not isinstance(raw, dict) or set(raw.keys()) != _BLOCK_FIELDS:
                _fail('invalid_block_schema')
            if raw.get('schema_version') != SCHEMA_VERSION:
                _fail('invalid_block_schema')
            raw_count = raw.get('slot_count')
            if type(raw_count) is not int or raw_count <= 0:
                _fail('block_hash_mismatch')
            rebuilt = make_block(
                binding_sha256=raw['binding_sha256'],
                context_id=raw['context_id'],
                stage=raw['stage'], model_id=raw['model_id'],
                role_id=raw['role_id'],
                fit_uid_sha256=raw['fit_uid_sha256'],
                validation_uid_sha256=raw['validation_uid_sha256'],
                test_uid_sha256=raw['test_uid_sha256'],
                axes=raw['axes'],
                depends_on_blocks=raw['depends_on_blocks'],
                resolution=raw['resolution'])
            if (rebuilt['block_id'] != raw['block_id']
                    or rebuilt['slot_count'] != raw_count):
                _fail('block_hash_mismatch')
            if rebuilt['block_id'] in seen_block_ids:
                _fail('duplicate_block')
            if rebuilt['binding_sha256'] != binding_sha:
                _fail('binding_hash_mismatch')
            seen_block_ids.add(rebuilt['block_id'])
            normalized_blocks.append(rebuilt)

        normalized_aliases = []
        seen_alias_ids, seen_alias_scopes = set(), set()
        for raw in aliases:
            if not isinstance(raw, dict) or set(raw.keys()) != _ALIAS_FIELDS:
                _fail('invalid_alias_schema')
            if raw.get('schema_version') != SCHEMA_VERSION:
                _fail('invalid_alias_schema')
            rebuilt = make_alias(
                context_id=raw['context_id'], strategy=raw['strategy'],
                recipe_id=raw['recipe_id'],
                target_block_id=raw['target_block_id'],
                evidence_status=raw['evidence_status'],
                reason_code=raw['reason_code'], metadata=raw['metadata'])
            if rebuilt['alias_id'] != raw['alias_id']:
                _fail('alias_hash_mismatch')
            if rebuilt['alias_id'] in seen_alias_ids:
                _fail('duplicate_alias')
            scope = (rebuilt['context_id'], rebuilt['strategy'])
            if scope in seen_alias_scopes:
                _fail('duplicate_alias_scope')
            seen_alias_ids.add(rebuilt['alias_id'])
            seen_alias_scopes.add(scope)
            normalized_aliases.append(rebuilt)

        by_block = {}
        for block in normalized_blocks:
            by_block.setdefault(block['block_id'], block)

        for alias in normalized_aliases:
            target = alias['target_block_id']
            if target == FALLBACK_TARGET:
                continue
            block = by_block.get(target)
            if block is None or block['context_id'] != alias['context_id']:
                _fail('alias_target_missing')

        for block in normalized_blocks:
            for dep in block['depends_on_blocks']:
                target = by_block.get(dep)
                if target is None or target['context_id'] != block['context_id']:
                    _fail('dependency_missing')

        assert_acyclic(normalized_blocks)

        stage_counts, stage_block_counts = {}, {}
        for block in normalized_blocks:
            stage = block['stage']
            stage_counts[stage] = stage_counts.get(stage, 0) + block['slot_count']
            stage_block_counts[stage] = stage_block_counts.get(stage, 0) + 1
        summary = {
            'block_count': len(normalized_blocks),
            'expanded_operation_slots': sum(
                b['slot_count'] for b in normalized_blocks),
            'stage_counts': dict(sorted(stage_counts.items())),
            'stage_block_counts': dict(sorted(stage_block_counts.items())),
            'alias_count': len(normalized_aliases),
        }
        payload = {
            'schema_version': SCHEMA_VERSION,
            'execution_authorized': False,
            'bindings': bindings_copy,
            'blocks': sorted(normalized_blocks, key=lambda b: b['block_id']),
            'aliases': sorted(normalized_aliases, key=lambda a: a['alias_id']),
            'summary': summary,
        }
        catalog = dict(payload)
        catalog['catalog_sha256'] = _sha(payload)
        return catalog
    except BlockError:
        raise
    except Exception:
        raise BlockError('invalid_catalog_input') from None


def iter_slots(catalog):
    """Eagerly validate a canonical catalog, then stream fresh slot dicts."""
    if not isinstance(catalog, dict) or set(catalog.keys()) != _CATALOG_FIELDS:
        _fail('invalid_catalog_schema')
    if catalog.get('schema_version') != SCHEMA_VERSION:
        _fail('invalid_catalog_schema')
    if catalog.get('execution_authorized') is not False:
        _fail('execution_not_authorized')

    raw_digest = catalog.get('catalog_sha256')
    if not _is_hex64(raw_digest):
        _fail('catalog_hash_mismatch')

    try:
        body = {k: v for k, v in catalog.items() if k != 'catalog_sha256'}
        expected_digest = _sha(body)
    except BlockError:
        _fail('catalog_hash_mismatch')
    except (ValueError, TypeError, RecursionError, UnicodeError):
        _fail('catalog_hash_mismatch')
    if expected_digest != raw_digest:
        _fail('catalog_hash_mismatch')

    sealed = seal_catalog(
        catalog['bindings'], catalog['blocks'], catalog['aliases'])
    if sealed != catalog:
        _fail('catalog_hash_mismatch')

    def _generate():
        for block in sealed['blocks']:
            axes = block['axes']
            for gate_id in axes['gate_id']:
                for candidate in axes['candidate']:
                    for seed in axes['seed']:
                        slot_id = SLOT_PREFIX + _sha({
                            'block_id': block['block_id'],
                            'gate_id': gate_id,
                            'candidate': candidate,
                            'seed': seed})
                        yield {
                            'slot_id': slot_id,
                            'block_id': block['block_id'],
                            'context_id': block['context_id'],
                            'stage': block['stage'],
                            'model_id': block['model_id'],
                            'role_id': block['role_id'],
                            'fit_uid_sha256': block['fit_uid_sha256'],
                            'validation_uid_sha256':
                                block['validation_uid_sha256'],
                            'test_uid_sha256': block['test_uid_sha256'],
                            'gate_id': gate_id,
                            'candidate': dict(candidate),
                            'seed': seed,
                            'resolution': block['resolution'],
                            'depends_on_blocks':
                                list(block['depends_on_blocks']),
                        }

    return _generate()


def require_scientific_execution(*args, **kwargs):
    """Always deny execution, regardless of forged flags or arguments."""
    raise BlockError('scientific_execution_not_authorized')
