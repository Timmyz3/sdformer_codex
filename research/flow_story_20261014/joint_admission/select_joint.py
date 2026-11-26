"""Bounded CPU reference for joint admission under an EXACT real-point budget.

Ordinary candidate controls, not a new pruning theorem or a hardware scheduler.
The input is an existing complete pair+singleton proposal. No features, GT, flow
inference, grouping changes, fallback budgets, or dataset selection occur here.
Three rules use SUM OF SELECTED TILE SCORES; legacy_group_max alone preserves
the earlier all-or-none max-member-group objective as a provenance control.
"""
from fractions import Fraction
import math

RULES = ('free', 'at_least_one', 'all_or_none', 'legacy_group_max')
POLICIES = ('independent', 'motion', 'rotated90', 'horizontal_local')
MAX_TILES = 32
MAX_COST = 2048


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate(proposal, target_cost):
    require(type(target_cost) is int and 0 <= target_cost <= MAX_COST,
            'Exact target must be an integer in [0,2048]')
    require(isinstance(proposal, dict) and proposal.get('schema') == 'flow_motion_group_proposal_v1',
            'Expected frozen pair+singleton proposal schema')
    require(proposal.get('policy') in POLICIES and
            proposal.get('association_mode') in ('direction_one_hop', 'endpoint'), 'Unknown topology/association mode')
    shape = proposal.get('shape')
    require(isinstance(shape, (list, tuple)) and len(shape) == 2 and
            all(type(v) is int and v > 0 for v in shape) and shape[0]*shape[1] <= MAX_TILES,
            'Invalid/beyond32 tile grid')
    h, w = shape
    tiles, groups = proposal.get('tiles'), proposal.get('groups')
    require(isinstance(tiles, list) and len(tiles) == h*w and isinstance(groups, list), 'Invalid tiles/groups')
    active = set(); total = 0
    for index, tile in enumerate(tiles):
        require(type(tile.get('tile_id')) is int and tile['tile_id'] == index and
                tile.get('row') == index//w and tile.get('col') == index%w, 'Raster tile identities differ')
        n, cost = tile.get('valid_points'), tile.get('slow_service_cost')
        require(type(n) is int and type(cost) is int and 0 <= n == cost,
                'This reference costs actual valid spatial points, not padded tiles/cycles')
        require(n == 0 or finite(tile.get('score')), 'Active tile score must be finite; missing is not imputed')
        total += cost
        if n:
            active.add(index)
    require(total <= MAX_COST, 'Total point count beyond2048 CPU reference limit')
    seen = set(); group_ids = set()
    for group in groups:
        gid, ids = group.get('group_id'), group.get('members')
        require(type(gid) is int and gid >= 0 and gid not in group_ids, 'Duplicate/invalid group ID')
        require(isinstance(ids, list) and len(ids) in (1, 2) and
                all(type(i) is int for i in ids) and ids == sorted(set(ids)) and
                all(i in active and i not in seen for i in ids), 'Groups overlap or contain invalid members')
        cost = sum(tiles[i]['slow_service_cost'] for i in ids)
        require(type(group.get('slow_service_cost')) is int and group['slow_service_cost'] == cost and
                type(group.get('valid_points')) is int and group['valid_points'] == cost, 'Group cost closure differs')
        require(finite(group.get('score')) and group['score'] == max(tiles[i]['score'] for i in ids),
                'Input proposal no longer matches inherited max-member-score provenance')
        seen.update(ids); group_ids.add(gid)
    require(seen == active, 'Every active tile needs exactly one pair or singleton')
    require(proposal['policy'] != 'independent' or all(len(g['members']) == 1 for g in groups),
            'Independent source may contain only singleton groups')
    return tiles, sorted(groups, key=lambda g: tuple(g['members'])), total


def _options(group, tiles, rule):
    ids = tuple(group['members'])
    if len(ids) == 1:
        choices = ((), ids)  # singletons always optional, even under at_least_one
    elif rule == 'free':
        choices = ((), ids[:1], ids[1:], ids)
    elif rule == 'at_least_one':
        choices = (ids[:1], ids[1:], ids)
    else:
        choices = ((), ids)
    out = []
    for chosen in choices:
        cost = sum(tiles[i]['slow_service_cost'] for i in chosen)
        if rule == 'legacy_group_max':
            score = Fraction(group['score']) if chosen else Fraction(0)
        else:
            score = sum((Fraction(tiles[i]['score']) for i in chosen), Fraction(0))
        out.append((cost, score, chosen))
    return out


def _float_or_none(value):
    try:
        answer = float(value)
    except OverflowError:
        return None
    return answer if math.isfinite(answer) else None


def select_exact(proposal, target_cost, rule='free'):
    """Return full spatial bool mask or explicit infeasible; never relax target.

    Pair states: free={00,10,01,11}; at_least_one={10,01,11};
    all_or_none={00,11}. Singleton states are {0,1} under every rule.
    Sum selected tile scores, exact Fraction comparisons, then lexicographically
    smallest sorted selected tile IDs. Free has exactly the old independent
    feasible masks/objective/tie rule for any fixed complete pair topology.
    Caller applies each bool to complete T10/all channels and charges fast/head/
    control/BN service separately. target_cost is spatial points, not PE cycles.
    """
    require(rule in RULES, 'Unknown admission rule')
    tiles, groups, total = _validate(proposal, target_cost)
    # cost -> (exact objective, sorted chosen tile IDs). All groups are disjoint.
    states = {0: (Fraction(0), ())}
    counters = {'group_count': len(groups), 'tile_count': len(tiles), 'state_visits': 0,
        'option_transitions_attempted': 0, 'option_transitions_within_cap': 0,
        'option_transitions_over_cap': 0, 'score_additions': 0, 'score_comparisons': 0,
        'equal_score_tie_comparisons': 0, 'selected_tuple_materializations': 0,
        'peak_reachable_states': 1, 'reachable_states_after_group': [], 'max_score_rational_bits': 0,
        'cycle_measurement': False, 'note': 'Reference DP counts; Fraction/tuple/memory/control cost is additional'}
    for group in groups:
        options = _options(group, tiles, rule)
        updated = {}
        counters['state_visits'] += len(states)
        for cost, (value, ids) in states.items():
            for extra_cost, extra_score, extra_ids in options:
                counters['option_transitions_attempted'] += 1
                new_cost = cost+extra_cost
                if new_cost > target_cost:
                    counters['option_transitions_over_cap'] += 1
                    continue
                counters['option_transitions_within_cap'] += 1
                new_score = value+extra_score
                members = tuple(sorted(ids+extra_ids))
                counters['score_additions'] += 1
                counters['selected_tuple_materializations'] += 1
                counters['max_score_rational_bits'] = max(counters['max_score_rational_bits'],
                    new_score.numerator.bit_length(), new_score.denominator.bit_length())
                incumbent = updated.get(new_cost)
                replace = incumbent is None
                if incumbent is not None:
                    counters['score_comparisons'] += 1
                    if new_score == incumbent[0]:
                        counters['equal_score_tie_comparisons'] += 1
                        replace = members < incumbent[1]
                    else:
                        replace = new_score > incumbent[0]
                if replace:
                    updated[new_cost] = (new_score, members)
        states = updated
        counters['peak_reachable_states'] = max(counters['peak_reachable_states'], len(states))
        counters['reachable_states_after_group'].append(len(states))
    best = states.get(target_cost)
    result = {'schema': 'flow_joint_admission_exact_v1', 'policy': proposal['policy'], 'rule': rule,
        'association_mode': proposal['association_mode'], 'feasible': best is not None,
        'status': 'EXACT_COST_SELECTED' if best is not None else 'INFEASIBLE_EXACT_COST',
        'target_cost': target_cost, 'slow_cost_cap': target_cost, 'slow_service_cost': None,
        'achieved_cost': None, 'unused_slow_cost': None, 'slow_mask': None,
        'selected_tile_ids': None, 'selected_tile_count': None, 'selected_valid_points': None,
        'objective_score': None, 'objective_score_exact': None, 'selected_tile_score_exact': None,
        'selected_group_ids': None, 'fully_selected_group_ids': None, 'unselected_group_ids': None,
        'group_states': None, 'pair_state_counts': None, 'available_slow_service_cost': total,
        'reachable_costs_up_to_target': sorted(states),
        'minimum_required_cost': sum(min(tiles[i]['slow_service_cost'] for i in g['members'])
            for g in groups if len(g['members']) == 2) if rule == 'at_least_one' else 0,
        'objective': 'sum selected inherited max-member group scores, once per full group' if rule == 'legacy_group_max'
                     else 'sum selected TILE scores, once per selected tile',
        'tie_break': 'lexicographically smallest sorted selected tile IDs',
        'score_semantics': 'exact Fraction of supplied finite int/float; magnitude is an ordinary proxy, not calibrated risk',
        'budget_semantics': 'EXACT real valid spatial points; infeasible never falls back or drops a sample',
        'optimality_scope': 'only this finite proxy objective/constraint set, not task AEE',
        'GT_or_model_used': False, 'runtime_ASIC_ready': False, 'cycle_or_energy_measurement': False,
        'counters': counters}
    if best is None:
        return result
    value, ids = best; chosen = set(ids); h, w = proposal['shape']
    actual = sum(tiles[i]['slow_service_cost'] for i in ids)
    require(actual == target_cost, 'Internal exact-cost violation')
    pair_counts = {'00': 0, '10': 0, '01': 0, '11': 0}
    group_states = []
    for group in groups:
        state = ''.join('1' if i in chosen else '0' for i in group['members'])
        if len(group['members']) == 2:
            pair_counts[state] += 1
        group_states.append({'group_id': group['group_id'], 'members': list(group['members']), 'state': state})
    tile_value = sum((Fraction(tiles[i]['score']) for i in ids), Fraction(0))
    result.update(slow_service_cost=actual, achieved_cost=actual, unused_slow_cost=0,
        slow_mask=[[y*w+x in chosen for x in range(w)] for y in range(h)], selected_tile_ids=list(ids),
        selected_tile_count=len(ids), selected_valid_points=actual, objective_score=_float_or_none(value),
        objective_score_exact={'numerator': str(value.numerator), 'denominator': str(value.denominator)},
        selected_tile_score_exact={'numerator': str(tile_value.numerator), 'denominator': str(tile_value.denominator)},
        selected_group_ids=[g['group_id'] for g in groups if any(i in chosen for i in g['members'])],
        fully_selected_group_ids=[g['group_id'] for g in groups if all(i in chosen for i in g['members'])],
        unselected_group_ids=[g['group_id'] for g in groups if not any(i in chosen for i in g['members'])],
        group_states=group_states, pair_state_counts=pair_counts)
    return result
