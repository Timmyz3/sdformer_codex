"""Frozen H0/H1 arithmetic on already independently audited in-memory records.

No file reader, runtime imports, GT, model execution, fitting, or queue side effects.
Real-data admission and raw-prediction recomputation must be implemented separately.
"""
import itertools
import math

REGIONS = ('all_valid', 'B', 'L', 'B_and_L', 'B_only', 'L_only_B_known',
           'neither_B_known', 'B_unknown', 'voxel_nonzero', 'voxel_zero',
           'B_voxel_nonzero', 'B_voxel_zero', 'L_voxel_nonzero', 'L_voxel_zero',
           'B_unknown_voxel_nonzero', 'B_unknown_voxel_zero')
ANCHORS = ('native', 'FF', 'CC', 'P', 'F12', 'T8', 'matched_N', 'matched_F')
REFERENCES = ('native', 'FF', 'matched_N', 'matched_F')
SCORES = ('amplitude_abs_mean', 'feature_rms', 'pooling_residual_mse',
          'spatial_neighbor_abs_difference', 'temporal_neighbor_abs_difference',
          'psn1_activity', 'psn1_spatial_neighbor_hamming')
PRIMARY_SCORES = ('amplitude_abs_mean', 'pooling_residual_mse',
                  'psn1_activity', 'psn1_spatial_neighbor_hamming')
TILES = tuple(itertools.product(range(3), range(4)))
PAIRS = tuple(itertools.combinations(range(12), 2))
LIMITS = {'all_valid': .03, 'B': .05, 'L': .05}
UNIFORM_ORDER = (3, 8, 1, 10, 5, 6, 0, 11, 2, 9, 4, 7)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def action(indices):
    indices = tuple(indices)
    require(len(indices) in (1, 2) and tuple(sorted(set(indices))) == indices
            and all(i in range(12) for i in indices), 'Canonical singleton or pair required')
    return 'k%d_' % len(indices) + '_'.join('r%dc%d' % TILES[i] for i in indices)


def validate_metrics(value):
    require(set(value) == set(REGIONS), 'Exactly all16 regions required')
    for region, metric in value.items():
        n, e, a = (metric[k] for k in ('valid_pixels', 'error_sum', 'AEE'))
        require(isinstance(n, int) and not isinstance(n, bool) and n >= 0, 'Nonnegative integer support required')
        require(isinstance(e, (int, float)) and math.isfinite(e) and e >= 0, 'Finite nonnegative error sum required')
        if n:
            require(isinstance(a, (int, float)) and math.isfinite(a) and a >= 0
                    and math.isclose(a, e/n, rel_tol=1e-12, abs_tol=1e-10), 'AEE/error/support mismatch')
        else:
            require(a is None and e == 0, 'Empty region is null, never zero AEE')
    require(value['all_valid']['valid_pixels'] > 0, 'All-valid support required')


def validate_frames(frames, members, h1=False):
    require(len(frames) == len(members) == 8, 'Complete H8 only')
    require(len({m['sequence'] for m in members}) == len({m['target_id'] for m in members}) == 8,
            'Exactly one identity per distinct sequence')
    names = {action((i,)) for i in range(12)} | {action(p) for p in PAIRS} if h1 else set(ANCHORS)
    for frame, member in zip(frames, members):
        require(frame['target_id'] == member['target_id'] and frame['sequence'] == member['sequence'], 'H8 identity/order differs')
        require(set(frame['arms']) == names, 'Complete endpoint set required')
        supports = []
        for metrics in frame['arms'].values():
            validate_metrics(metrics)
            supports.append(tuple(metrics[r]['valid_pixels'] for r in REGIONS))
        require(len(set(supports)) == 1, 'Different endpoint region support')
        if h1:
            rows = frame['ordinary_scores']
            require(len(rows) == 12 and [tuple(r['tile']) for r in rows] == list(TILES), 'Canonical twelve scores required')
            require(all(set(r) == {'tile', *SCORES} and all(isinstance(r[s], (int, float))
                        and math.isfinite(r[s]) for s in SCORES) for r in rows), 'Finite seven ordinary scores required')


def aggregate_metrics(values):
    out = {}
    for region in REGIONS:
        rows = [v[region] for v in values]
        present = [v for v in rows if v['valid_pixels']]
        n, e = sum(v['valid_pixels'] for v in rows), sum(v['error_sum'] for v in rows)
        out[region] = {'frames_with_region': len(present), 'empty_region_frames': len(rows)-len(present),
                       'valid_pixels': n, 'error_sum': e,
                       'frame_mean_AEE': sum(v['AEE'] for v in present)/len(present) if present else None,
                       'pixel_weighted_AEE': e/n if n else None}
    return out


def summarize(values, members):
    require(len(values) == len(members) == 8, 'Summary requires complete H8')
    return {'overall': aggregate_metrics(values), **{m['sequence']: aggregate_metrics([v]) for m, v in zip(members, values)}}


def compare(candidate, reference):
    require(set(candidate) == set(reference), 'Scope mismatch')
    passed, observed, records = True, True, {}
    for scope in candidate:
        records[scope] = {}
        for region, limit in LIMITS.items():
            c, r = candidate[scope][region], reference[scope][region]
            require(c['valid_pixels'] == r['valid_pixels'] and c['frames_with_region'] == r['frames_with_region'], 'Comparison support mismatch')
            x, y = c['frame_mean_AEE'], r['frame_mean_AEE']
            require((x is None) == (y is None), 'Missing-region mismatch')
            evidence = x is not None
            ok = not evidence or x <= y * (1 + limit)
            records[scope][region] = {'candidate_AEE': x, 'reference_AEE': y, 'relative_limit': limit,
                                      'relative_change': x/y-1 if y not in (None, 0) else None,
                                      'evidence': evidence, 'pass_supported_constraint': ok}
            passed, observed = passed and ok, observed and evidence
    return {'pass_supported_constraints': passed, 'all_gate_regions_observed': observed, 'scopes': records}


def gate_against_references(summary, references):
    comparisons = {name: compare(summary, ref) for name, ref in references.items()}
    return {'comparisons': comparisons,
            'pass_supported_constraints': all(c['pass_supported_constraints'] for c in comparisons.values()),
            'all_gate_regions_observed': all(c['all_gate_regions_observed'] for c in comparisons.values())}


def delta_summary(candidate, reference):
    return {scope: {r: (candidate[scope][r]['frame_mean_AEE'] - reference[scope][r]['frame_mean_AEE']
                       if candidate[scope][r]['frame_mean_AEE'] is not None else None)
                    for r in REGIONS} for scope in candidate}


def analyze_h0(frames, members):
    validate_frames(frames, members)
    summaries = {a: summarize([f['arms'][a] for f in frames], members) for a in ANCHORS}
    protection = {'native': compare(summaries['native'], summaries['matched_N']),
                  'FF': compare(summaries['FF'], summaries['matched_F'])}
    references = {a: summaries[a] for a in REFERENCES}
    return {'schema': 'flow_h0_pure_metric_analysis_v1', 'actual_source_audit_performed_here': False,
            'summaries': summaries, 'same_W_fine_protection': protection,
            'same_W_fine_pass': all(g['pass_supported_constraints'] for g in protection.values()),
            'same_W_fine_all_regions_observed': all(g['all_gate_regions_observed'] for g in protection.values()),
            'action_gates': {a: gate_against_references(summaries[a], references) for a in ('CC','P','F12','T8')},
            'deltas': {a: {b: delta_summary(summaries[a], summaries[b]) for b in ('native','FF','CC')}
                       for a in ('CC','P','F12','T8')},
            'complete_tail_interaction': {scope: {r: (summaries['F12'][scope][r]['frame_mean_AEE']
                + summaries['T8'][scope][r]['frame_mean_AEE'] - summaries['CC'][scope][r]['frame_mean_AEE']
                - summaries['P'][scope][r]['frame_mean_AEE'] if summaries['CC'][scope][r]['frame_mean_AEE'] is not None else None)
                for r in REGIONS} for scope in summaries['CC']},
            'normality_W0_gate': None,
            'scope': 'Eight training-side one-frame sequences. No W0 normality, deployment, or hardware inference.'}


def frame_feasible(metrics, h0frame):
    for ref in REFERENCES:
        for region, limit in LIMITS.items():
            x, y = metrics[region], h0frame['arms'][ref][region]
            require(x['valid_pixels'] == y['valid_pixels'], 'H0/H1 region support differs')
            if x['AEE'] is not None and x['AEE'] > y['AEE'] * (1 + limit):
                return False
    return True


def policy_record(selected, frames, members, references):
    require(len(selected) == 8, 'One fixed-policy action per member')
    values = [f['arms'][action(pair)] for f, pair in zip(frames, selected)]
    summary = summarize(values, members)
    return {'selected_pairs': [list(p) for p in selected], 'summary': summary,
            'gate': gate_against_references(summary, references)}


def relative_margin(candidate, reference):
    return 1 - candidate/reference if reference > 0 else None


def analyze_h1(h0frames, frames, members):
    h0 = analyze_h0(h0frames, members)
    validate_frames(frames, members, h1=True)
    for before, after in zip(h0frames, frames):
        for region in REGIONS:
            require(before['arms']['CC'][region]['valid_pixels'] == after['arms'][action((0,))][region]['valid_pixels'],
                    'H0/H1 support differs')
    references = {r: h0['summaries'][r] for r in REFERENCES}
    selections = {'fixed_uniform': [tuple(sorted(UNIFORM_ORDER[:2]))]*8, 'fixed_position_01': [(0,1)]*8}
    for score in SCORES:
        selections[score] = [tuple(sorted(sorted(range(12), key=lambda i: (-f['ordinary_scores'][i][score], TILES[i]))[:2])) for f in frames]
    ordinary = {name: policy_record(selected, frames, members, references) for name, selected in selections.items()}
    fixed = {p: policy_record([p]*8, frames, members, references) for p in PAIRS}
    cost = lambda rec: rec['summary']['overall']['all_valid']['frame_mean_AEE']
    best_fixed = min(PAIRS, key=lambda p: (cost(fixed[p]), p))
    feasible_fixed = [p for p in PAIRS if fixed[p]['gate']['pass_supported_constraints']]
    best_feasible_fixed = min(feasible_fixed, key=lambda p: (cost(fixed[p]), p)) if feasible_fixed else None
    if best_feasible_fixed is not None:
        ordinary['H8_fitted_feasible_fixed_pair'] = fixed[best_feasible_fixed]
    allgt, feasible, singleton_top2, interaction = [], [], [], []
    feasible_counts = []
    for f, base in zip(frames, h0frames):
        energy = lambda p: f['arms'][action(p)]['all_valid']['AEE']
        allgt.append(min(PAIRS, key=lambda p: (energy(p), p)))
        allowed = [p for p in PAIRS if frame_feasible(f['arms'][action(p)], base)]
        feasible_counts.append(len(allowed))
        feasible.append(min(allowed, key=lambda p: (energy(p), p)) if allowed else None)
        e0 = base['arms']['CC']['all_valid']['AEE']
        singleton = [energy((i,)) for i in range(12)]
        singleton_top2.append(tuple(sorted(sorted(range(12), key=lambda i: (singleton[i], i))[:2])))
        pairs, conditional = [], []
        for i, j in PAIRS:
            e = energy((i,j))
            pairs.append({'pair': [i,j], 'interaction': singleton[i]+singleton[j]-e0-e,
                          'actual_joint_gain': e0-e, 'sum_singleton_gains': 2*e0-singleton[i]-singleton[j]})
            for bg, tile in ((i,j),(j,i)):
                g0, g = e0-singleton[tile], singleton[bg]-e
                conditional.append({'background': bg, 'tile': tile, 'empty_gain': g0, 'conditional_gain': g,
                                    'strict_sign_flip': g0*g < 0})
        rankings = []
        for bg in range(12):
            available = [j for j in range(12) if j != bg]
            a = sorted(available, key=lambda j: (singleton[j],j))
            b = sorted(available, key=lambda j: (energy(tuple(sorted((bg,j)))),j))
            rankings.append({'background':bg, 'empty_order':a, 'conditional_order':b, 'changed':a!=b})
        interaction.append({'target_id':f['target_id'], 'pairs':pairs, 'ordered_conditional':conditional, 'rankings':rankings})
    allgt_result = policy_record(allgt, frames, members, references)
    feasible_result = policy_record(feasible, frames, members, references) if all(p is not None for p in feasible) else None
    additive_result = policy_record(singleton_top2, frames, members, references)
    names = [n for n in ordinary if ordinary[n]['gate']['pass_supported_constraints']]
    strongest_name = min(names, key=lambda n: (cost(ordinary[n]), list(ordinary).index(n))) if names else None
    screen = {'strongest_single_ordinary_name':strongest_name, 'allGT_oracle_relative_margin':None,
              'feasible_oracle_relative_margin':None, 'threshold':.01, 'automatic_execution':False}
    if feasible_result is None:
        screen['decision'] = 'NO_REGION_FEASIBLE_ORACLE'
    elif strongest_name is None:
        screen['decision'] = 'NO_FEASIBLE_ORDINARY_COMPARATOR'
    else:
        strongest = ordinary[strongest_name]
        screen['allGT_oracle_relative_margin'] = relative_margin(cost(allgt_result),cost(strongest))
        screen['feasible_oracle_relative_margin'] = relative_margin(cost(feasible_result),cost(strongest)) if feasible_result else None
        if not h0['same_W_fine_pass']:
            screen['decision'] = 'SAME_W_FINE_PROTECTION_FAILED_NO_INFORMATION_ATTRIBUTION'
        elif not h0['same_W_fine_all_regions_observed'] or not strongest['gate']['all_gate_regions_observed']:
            screen['decision'] = 'MISSING_REGION_EVIDENCE'
        elif screen['feasible_oracle_relative_margin'] is None:
            screen['decision'] = 'RELATIVE_MARGIN_UNDEFINED_ZERO_REFERENCE'
        elif screen['feasible_oracle_relative_margin'] < .01:
            screen['decision'] = 'STOP_THIS_W_Q_K2_HEAD_INSUFFICIENT_MARGIN'
        else:
            screen['decision'] = 'MARGIN_FOR_FURTHER_REVIEW_ONLY'
        if feasible_result is not None:
            gains = []
            refs, opts = [], []
            for m in members:
                seq = m['sequence']
                refs.append(strongest['summary'][seq]['all_valid']['frame_mean_AEE'])
                opts.append(feasible_result['summary'][seq]['all_valid']['frame_mean_AEE'])
                gains.append(refs[-1]-opts[-1])
            positive = sum(max(x,0) for x in gains)
            loo = [relative_margin(sum(opts)-opts[i],sum(refs)-refs[i]) for i in range(8)]
            best = max(range(8),key=lambda i:(gains[i],-i))
            screen['concentration'] = {'per_frame_absolute_gain':gains,
                'largest_positive_share':max(gains)/positive if positive else None,
                'leave_one_out_relative_margin_fixed_method':loo,
                'positive_frame_count':sum(x>0 for x in gains), 'largest_gain_frame':members[best]['target_id'],
                'single_frame_sensitive':loo[best] is None or loo[best]<=0}
    return {'schema':'flow_h1_pure_metric_analysis_v1', 'actual_source_audit_performed_here':False,
            'all_actual_endpoint_summaries': {name:summarize([f['arms'][name] for f in frames],members)
                for name in [action((i,)) for i in range(12)] + [action(p) for p in PAIRS]},
            'ordinary_controls':ordinary, 'ordinary_primary_score_names':list(PRIMARY_SCORES),
            'ordinary_selection_scope':'One score rule across H8; best method and fitted pair are H8 development selections.',
            'allGT_best_fixed_pair':dict(pair=list(best_fixed),result=fixed[best_fixed]),
            'feasible_best_fixed_pair':dict(pair=list(best_feasible_fixed),result=fixed[best_feasible_fixed]) if best_feasible_fixed is not None else None,
            'feasible_fixed_pair_count':len(feasible_fixed), 'per_frame_feasible_pair_count':feasible_counts,
            'allGT_oracle':allgt_result, 'region_feasible_oracle':feasible_result,
            'singleton_top2_actual_endpoint':additive_result,
            'singleton_top2_minus_allGT_oracle':delta_summary(additive_result['summary'],allgt_result['summary']),
            'interaction':interaction, 'investment_screen':screen,
            'label_counts':{'empty_background':96,'ordered_single_background':1056,'independent_frames':8},
            'scope':'Finite actual k2 collection conditional on fixed W/Q. No deployed selector or hardware gain.'}
