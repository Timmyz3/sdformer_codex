"""Small CPU reference DP for equal REAL slow-service work across group policies.

This is an offline/reference planner, not a synthesizable runtime scheduler.
All four policies must use this same optimizer. A group contributes its existing
max-member score ONCE; maximize the sum of selected group scores at exact cost.
Negative finite scores are legal. Float inputs are compared as exact rational
values to avoid order-dependent accumulated floating-point score ties.
Costs are real spatial points; T10/channels/fast paths/scorers are not priced here.
"""
import math
from fractions import Fraction

PROPOSAL_SCHEMA = 'flow_motion_group_proposal_v1'
POLICIES = ('independent', 'motion', 'rotated90', 'horizontal_local')
MAX_TILES = 32
MAX_COST = 2048


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bound(value, name):
    require(type(value) is int and 0 <= value <= MAX_COST,
            name+' must be an integer in [0,2048]')


def _validate(proposal):
    require(isinstance(proposal, dict) and proposal.get('schema') == PROPOSAL_SCHEMA,
            'Foreign proposal schema')
    require(proposal.get('policy') in POLICIES, 'Unknown policy')
    shape = proposal.get('shape')
    require(isinstance(shape, (list, tuple)) and len(shape) == 2
            and all(type(v) is int and v > 0 for v in shape), 'Invalid grid shape')
    h, w = shape
    require(h*w <= MAX_TILES, 'CPU reference limit is32 grid tiles including empty tiles')
    tiles, groups = proposal.get('tiles'), proposal.get('groups')
    require(isinstance(tiles, list) and len(tiles) == h*w and isinstance(groups, list), 'Invalid proposal tiles/groups')
    active = set()
    for i, tile in enumerate(tiles):
        require(tile.get('tile_id') == i and type(tile['tile_id']) is int
                and tile.get('row') == i//w and tile.get('col') == i%w, 'Tile IDs/coordinates must be raster ordered')
        n, cost = tile.get('valid_points'), tile.get('slow_service_cost')
        require(type(n) is int and type(cost) is int and 0 <= n == cost,
                'This planner costs actual spatial points: slow_service_cost must equal valid_points')
        require(n == 0 or finite(tile.get('score')), 'Active tile score must be finite, negative allowed')
        if n:
            active.add(i)
    total = sum(t['slow_service_cost'] for t in tiles)
    require(total <= MAX_COST, 'CPU reference total spatial-point limit is2048')
    seen, group_ids = set(), set()
    for group in groups:
        group_id, ids = group.get('group_id'), group.get('members')
        require(type(group_id) is int and group_id >= 0 and group_id not in group_ids, 'Invalid/duplicate group ID')
        require(isinstance(ids, list) and len(ids) in (1, 2)
                and all(type(i) is int for i in ids) and ids == sorted(set(ids))
                and all(i in active and i not in seen for i in ids), 'Groups overlap, split, or contain invalid members')
        cost = sum(tiles[i]['slow_service_cost'] for i in ids)
        require(type(group.get('slow_service_cost')) is int and group['slow_service_cost'] == cost
                and type(group.get('valid_points')) is int and group['valid_points'] == cost, 'Group actual-point cost differs')
        require(finite(group.get('score')) and group['score'] == max(tiles[i]['score'] for i in ids), 'Group score must be existing max-member score')
        seen.update(ids); group_ids.add(group_id)
    require(seen == active, 'Every active tile must appear in exactly one whole group')
    require(proposal['policy'] != 'independent' or all(len(g['members']) == 1 for g in groups), 'Independent policy must use singleton groups')
    require(proposal.get('association_mode') in ('direction_one_hop', 'endpoint'), 'Unknown association mode')
    return tiles, sorted(groups, key=lambda g: tuple(g['members'])), total


def _counters(groups, total, cap):
    return {'group_count': len(groups), 'available_spatial_points': total,
        'cost_limit': cap, 'state_visits': 0, 'exclude_transitions': 0,
        'include_transitions_attempted': 0, 'include_transitions_within_cap': 0,
        'include_transitions_over_cap': 0, 'peak_reachable_states': 1,
        'reachable_states_after_each_group': [1], 'initial_states': 1,
        'score_additions': 0, 'score_comparisons': 0, 'equal_score_tie_comparisons': 0,
        'member_tuple_materializations': 0, 'selected_member_tuple_elements_materialized': 0,
        'score_rational_max_bits': 0, 'transition_count': 0,
        'representation': 'Python sparse cost map/set; exact Fraction scores for selection',
        'complexity_bound': 'O(G*(cap+1)) state transitions; tuple/rational arithmetic has additional variable cost',
        'runtime_ASIC_ready': False, 'measured_cycles': False,
        'excluded_costs': ['pair/score production', 'proposal construction and validation',
                           'Python allocation/hash/sort/control timing', 'PE/PSN/BN/fast path', 'data transfers and queues']}


def _visit(counters, number):
    counters['state_visits'] += number
    counters['exclude_transitions'] += number
    counters['include_transitions_attempted'] += number
    counters['transition_count'] += 2*number


def reachable_costs(proposal, cap):
    """Return {'costs': sorted reachable exact costs, 'counters': ...}; includes0."""
    _bound(cap, 'cap')
    _, groups, total = _validate(proposal)
    states = {0}; counters = _counters(groups, total, cap)
    for group in groups:
        next_states = set(states)
        _visit(counters, len(states))
        for cost in states:
            new_cost = cost + group['slow_service_cost']
            if new_cost <= cap:
                next_states.add(new_cost); counters['include_transitions_within_cap'] += 1
            else:
                counters['include_transitions_over_cap'] += 1
        states = next_states
        counters['peak_reachable_states'] = max(counters['peak_reachable_states'], len(states))
        counters['reachable_states_after_each_group'].append(len(states))
    return {'schema': 'flow_exact_budget_reachability_v1', 'policy': proposal['policy'],
            'cap': cap, 'costs': sorted(states), 'counters': counters,
            'cost_kind': 'real_spatial_points_not_cycles', 'scores_used_for_reachability': False}


def _score_number(score):
    try:
        result = float(score)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def select_exact(proposal, target_cost):
    """Maximize group-score sum at EXACT target_cost; infeasible is explicit.

    Equal objective -> lexicographically smallest sorted selected tile IDs.
    All groups remain indivisible. Positive costs mean tied-cost selections
    cannot be strict subsets, so this tie order survives common extensions.
    Returns old-consumer-compatible slow_mask/selected_* plus audit metadata.
    """
    _bound(target_cost, 'target_cost')
    tiles, groups, total = _validate(proposal)
    # cost -> (exact score, sorted selected tile IDs, canonical group IDs)
    states = {0: (Fraction(0), (), ())}
    counters = _counters(groups, total, target_cost)
    for group in groups:
        next_states = dict(states)
        _visit(counters, len(states))
        score = Fraction(group['score'])
        for cost, previous in states.items():
            new_cost = cost + group['slow_service_cost']
            if new_cost > target_cost:
                counters['include_transitions_over_cap'] += 1
                continue
            counters['include_transitions_within_cap'] += 1
            value = previous[0]+score
            members = tuple(sorted(previous[1]+tuple(group['members'])))
            candidate = (value, members, previous[2]+(group['group_id'],))
            counters['score_additions'] += 1
            counters['member_tuple_materializations'] += 1
            counters['selected_member_tuple_elements_materialized'] += len(members)
            counters['score_rational_max_bits'] = max(counters['score_rational_max_bits'],
                value.numerator.bit_length(), value.denominator.bit_length())
            old = next_states.get(new_cost)
            replace = old is None
            if old is not None:
                counters['score_comparisons'] += 1
                if value == old[0]:
                    counters['equal_score_tie_comparisons'] += 1
                    replace = members < old[1]
                else:
                    replace = value > old[0]
            if replace:
                next_states[new_cost] = candidate
        states = next_states
        counters['peak_reachable_states'] = max(counters['peak_reachable_states'], len(states))
        counters['reachable_states_after_each_group'].append(len(states))
    best = states.get(target_cost)
    result = {'schema': 'flow_exact_budget_selection_v1', 'policy': proposal['policy'],
        'association_mode': proposal['association_mode'], 'feasible': best is not None,
        'target_cost': target_cost, 'slow_cost_cap': target_cost, 'achieved_cost': None,
        'slow_service_cost': None, 'unused_slow_cost': None, 'objective_score': None,
        'objective_score_exact': None, 'slow_mask': None, 'selected_tile_ids': None,
        'selected_group_ids': None, 'unselected_group_ids': None, 'selected_tile_count': None,
        'selected_group_count': None, 'selected_valid_points': None,
        'available_slow_service_cost': total, 'counters': counters,
        'objective': 'sum of selected existing max-member group scores; each group counted once',
        'score_semantics': 'finite source int/float values represented as exact rationals; no score calibration claim',
        'tie_break': 'lexicographically smallest sorted selected tile IDs',
        'budget_semantics': 'EXACT real spatial-point slow budget; complete T10/channels required by consumer',
        'optimality_claim': 'only declared finite grouped exact-cost objective',
        'runtime_ASIC_ready': False, 'cycle_or_energy_measurement': False}
    if best is None:
        result['status'] = 'INFEASIBLE_EXACT_COST'
        return result
    score, members, selected_groups = best
    h, w = proposal['shape']; selected_set = set(members)
    actual = sum(tiles[i]['slow_service_cost'] for i in members)
    require(actual == target_cost, 'Internal exact-cost error')
    result.update(status='EXACT_COST_SELECTED', achieved_cost=actual, slow_service_cost=actual,
        unused_slow_cost=0, objective_score=_score_number(score),
        objective_score_exact={'numerator': str(score.numerator), 'denominator': str(score.denominator)},
        slow_mask=[[y*w+x in selected_set for x in range(w)] for y in range(h)],
        selected_tile_ids=list(members), selected_group_ids=list(selected_groups),
        unselected_group_ids=[g['group_id'] for g in groups if g['group_id'] not in selected_groups],
        selected_tile_count=len(members), selected_group_count=len(selected_groups), selected_valid_points=actual)
    return result


def common_target(proposals, cap):
    """Maximum common reachable cost <=cap, independently of scores/GT/loss.

    Accept exactly four proposals (list/tuple or policy->proposal dict).
    Validate equal raster tiles, actual costs, scores, shape and association mode;
    different groupings are the experimental intervention. Zero is always legal.
    Includes all four reachability planners' control counts; selection DP costs
    must ALSO be charged by the caller, rather than hidden in this reference.
    """
    _bound(cap, 'cap')
    if isinstance(proposals, dict):
        require(set(proposals) == set(POLICIES)
                and all(p.get('policy') == k for k,p in proposals.items()), 'Need exactly four correctly named proposals')
        mapping = proposals
    else:
        require(isinstance(proposals, (list, tuple)) and len(proposals) == 4, 'Need exactly four proposals')
        mapping = {p.get('policy'): p for p in proposals}
        require(set(mapping) == set(POLICIES), 'Duplicate/missing policy')
    signature = None
    for name in POLICIES:
        p = mapping[name]; tiles, _, _ = _validate(p)
        current = (tuple(p['shape']), p['association_mode'],
                   tuple((t['tile_id'], t['valid_points'], t['slow_service_cost'], t['score']) for t in tiles))
        require(signature is None or current == signature, 'Four policies must share tiles/scores/real costs')
        signature = current
    results = {name: reachable_costs(mapping[name], cap) for name in POLICIES}
    shared = set.intersection(*(set(results[name]['costs']) for name in POLICIES))
    require(0 in shared, 'Internal error: zero must be reachable')
    target = max(shared)
    return {'schema': 'flow_common_exact_budget_v1', 'target_cost': target,
        'cap': cap, 'unused_cap': cap-target, 'common_reachable_costs': sorted(shared),
        'per_policy_reachability': results, 'target_selection_uses_only_cost_reachability': True,
        'GT_or_loss_used': False, 'cost_kind': 'real_spatial_points_not_cycles',
        'reachability_transition_count': sum(r['counters']['transition_count'] for r in results.values()),
        'selection_DP_still_required_per_policy': True, 'runtime_ASIC_ready': False,
        'warning': 'All four arms must call select_exact at target_cost; same cap alone is insufficient'}
