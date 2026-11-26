"""CPU-only motion-associated two-tile grouping and exact slow-cost caps.

This is a research policy adapter, not a flow estimator, warp, optimizer or
cycle model. x points right and y down. Flow values supplied to quantize_offset
are displacement in original-image pixels over the caller's declared time
interval. tile_size_pixels=(width,height) is the physical image footprint of a
feature tile, NOT necessarily (9,9). This module cannot infer DSEC direction,
time interval, shift coordinates or feature stride. It does not use GT.

All masks are spatial; the caller applies each decision to complete T10 and
all channels. A pair jointly chooses its slow/fast path; it does not merge
features. The slow-cost cap excludes fast paths, predictors and bookkeeping.
Those costs must be added by the consumer; no count here is a measured cycle.
"""
import math

SCHEMA = 'flow_motion_group_proposal_v1'
POLICIES = ('independent', 'horizontal_local', 'motion', 'rotated90')
MODES = ('direction_one_hop', 'endpoint')
# Exact ordering is the deterministic angular tie rule. Diagonal directions
# have unit length when taking dot products below.
DIRECTIONS = ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))


def require(ok, message):
    if not ok: raise ValueError(message)


def real(value):
    return type(value) in (int,float) and math.isfinite(value)


def _config(tile_size_pixels, association_mode, min_motion_pixels, max_distance_tiles):
    require(association_mode in MODES, 'Unknown association mode')
    require(isinstance(tile_size_pixels,(tuple,list)) and len(tile_size_pixels)==2 and
            all(real(v) and v>0 for v in tile_size_pixels), 'Positive finite tile pixel width/height required')
    require(real(min_motion_pixels) and min_motion_pixels>=0, 'Nonnegative finite minimum motion required')
    require(type(max_distance_tiles) is int and 1<=max_distance_tiles<=2, 'Only a finite one/two-hop table is supported')


def _offset_and_reason(dx,dy,tile_size_pixels,association_mode,min_motion_pixels,max_distance_tiles):
    if dx is None or dy is None:return None,'missing_flow'
    if not real(dx) or not real(dy):return None,'nonfinite_or_invalid_flow'
    if dx==0 and dy==0:return None,'zero_motion'
    if math.hypot(dx,dy)<min_motion_pixels:return None,'below_min_motion_pixels'
    nx,ny=dx/tile_size_pixels[0],dy/tile_size_pixels[1]
    if not math.isfinite(nx) or not math.isfinite(ny):return None,'nonfinite_scaled_flow'
    magnitude=max(abs(nx),abs(ny))
    if magnitude==0:return None,'underflow_scaled_flow'
    distance=1
    if association_mode=='endpoint':
        # Explicit half-up Chebyshev-distance quantization, not Python bankers rounding.
        distance=math.floor(magnitude+.5)
        if distance<1:return None,'sub_half_tile_endpoint'
        if distance>max_distance_tiles:return None,'endpoint_outside_distance_table'
    # Normalize by max component to avoid overflow; division does not change
    # the best direction. Equal computed dot products keep the first direction.
    x,y=nx/magnitude,ny/magnitude
    best=None;best_dot=None
    for ox,oy in DIRECTIONS:
        dot=(x*ox+y*oy)/(math.sqrt(2.) if ox and oy else 1.)
        if best_dot is None or dot>best_dot:
            best,best_dot=(ox*distance,oy*distance),dot
    return best,('directional_neighbor_not_correspondence' if association_mode=='direction_one_hop'
                 else 'quantized_endpoint_proxy_not_verified_correspondence')


def quantize_offset(dx_pixels,dy_pixels,*,tile_size_pixels=(9.,9.),
                    association_mode='direction_one_hop',min_motion_pixels=1.,max_distance_tiles=2):
    """Return (offset_x,offset_y) or None; no clipping/wrapping or GT imputation.

    direction_one_hop: any eligible nonzero motion chooses one neighboring
    tile by direction. A 1-pixel flow may protect a neighbor 144 pixels away;
    that is a direction prior, explicitly NOT endpoint correspondence.
    endpoint: the scaled displacement selects distance one or two. Sub-half
    tile or farther-than-table endpoints remain unpaired, never clamped.
    rotated90 control should call this with (-dy_pixels, dx_pixels), i.e.
    clockwise in image coordinates, before anisotropic tile normalization.
    """
    _config(tile_size_pixels,association_mode,min_motion_pixels,max_distance_tiles)
    return _offset_and_reason(dx_pixels,dy_pixels,tile_size_pixels,association_mode,
                              min_motion_pixels,max_distance_tiles)[0]


def _shape(grid,name,shape=None):
    require(isinstance(grid,(tuple,list)) and len(grid)>0 and
            all(isinstance(row,(tuple,list)) for row in grid),'Nonempty rectangular '+name+' required')
    width=len(grid[0])
    require(width>0 and all(len(row)==width for row in grid),'Ragged '+name)
    result=(len(grid),width)
    require(shape is None or result==shape,'Grid shape mismatch: '+name)
    return result


def propose_pairs(scores,flow_dxdy,valid_points,slow_costs,*,policy='motion',
                  association_mode='direction_one_hop',tile_size_pixels=(9.,9.),
                  flow_scale_to_pixels=(1.,1.),min_motion_pixels=1.,max_distance_tiles=2):
    """Build deterministic, disjoint pairs plus all active singleton tiles.

    scores[r][c] is a finite rank score, higher means more important; it need
    not be a calibrated risk. Active tiles require a supplied score. Unknown
    flow (None or nonfinite) stays explicitly ineligible for pairing, but its
    singleton can still receive slow service based on the supplied score.
    valid_points and slow_costs are integers; active costs must be positive.
    Empty tiles have both counts zero and never receive service.

    flow_scale_to_pixels=(sx,sy) converts the caller's flow units to image
    pixels. Its direction/time/coordinate conventions remain caller contracts.
    horizontal_local pairs eligible even/odd columns without direction; it
    shares the nonzero/min-motion/missing-data eligibility filter. independent
    always returns singleton groups. Controls do not fabricate missing flow.

    Every node proposes at most one edge. Unique candidate edges are processed
    by descending max member score, then ascending pair IDs. Collisions produce
    singletons; there is no alternate-neighbor search or recursive propagation.
    """
    require(policy in POLICIES,'Unknown policy')
    _config(tile_size_pixels,association_mode,min_motion_pixels,max_distance_tiles)
    require(isinstance(flow_scale_to_pixels,(tuple,list)) and len(flow_scale_to_pixels)==2 and
            all(real(v) and v>0 for v in flow_scale_to_pixels),'Positive finite flow conversion required')
    h,w=_shape(scores,'scores')
    for grid,name in ((flow_dxdy,'flow'),(valid_points,'valid_points'),(slow_costs,'slow_costs')):
        _shape(grid,name,(h,w))
    tiles=[]
    for row in range(h):
        for col in range(w):
            n,cost=valid_points[row][col],slow_costs[row][col]
            require(type(n) is int and n>=0 and type(cost) is int and cost>=0,'Counts/costs must be nonnegative integers')
            require((n==0 and cost==0) or (n>0 and cost>0),'Active tiles need positive slow cost; empty tiles cost zero')
            score=scores[row][col]
            require(n==0 or real(score),'Active tile score missing/nonfinite; never impute risk zero')
            f=flow_dxdy[row][col]
            if f is None:dx,dy=None,None
            else:
                require(isinstance(f,(tuple,list)) and len(f)==2,'Flow entry must be (dx,dy) or None')
                dx,dy=f
                if real(dx):dx=dx*flow_scale_to_pixels[0]
                if real(dy):dy=dy*flow_scale_to_pixels[1]
            raw_offset,reason=_offset_and_reason(dx,dy,tile_size_pixels,'direction_one_hop',min_motion_pixels,max_distance_tiles)
            flow_eligible=raw_offset is not None
            tile={'tile_id':row*w+col,'row':row,'col':col,'valid_points':n,'slow_service_cost':cost,
                  'score':float(score) if n else None,'flow_eligible':flow_eligible,'flow_status':reason,
                  'proposed_partner':None,'offset_xy':None,'association_reason':None}
            if not n:tile['association_reason']='empty_tile'
            elif policy=='independent':tile['association_reason']='independent_control'
            elif not flow_eligible:tile['association_reason']=reason
            else:
                if policy=='horizontal_local':offset=(1 if col%2==0 else -1,0);reason='fixed_horizontal_local_control'
                else:
                    if policy=='rotated90':dx,dy=-dy,dx
                    offset,reason=_offset_and_reason(dx,dy,tile_size_pixels,association_mode,min_motion_pixels,max_distance_tiles)
                tile['association_reason']=reason
                if offset is not None:
                    ox,oy=offset;rr,cc=row+oy,col+ox
                    tile['offset_xy']=list(offset)
                    if not (0<=rr<h and 0<=cc<w):tile['association_reason']='partner_out_of_grid'
                    elif valid_points[rr][cc]==0:tile['association_reason']='partner_empty'
                    else:tile['proposed_partner']=rr*w+cc
            tiles.append(tile)
    # A node with no valid own proposal (missing/zero/too small/out of grid)
    # remains singleton even if another node points into it.
    candidates={}
    for tile in tiles:
        other=tile['proposed_partner']
        if other is None:continue
        if tiles[other]['proposed_partner'] is None:
            tile['association_reason']='partner_not_pairable';continue
        edge=tuple(sorted((tile['tile_id'],other)))
        candidates[edge]=max(tiles[i]['score'] for i in edge)
    ordered=sorted(candidates,key=lambda edge:(-candidates[edge],edge))
    used=set();pairs=[];rejected=[]
    for edge in ordered:
        if any(i in used for i in edge):rejected.append(list(edge));continue
        pairs.append(edge);used.update(edge)
    members=pairs+[(t['tile_id'],) for t in tiles if t['valid_points'] and t['tile_id'] not in used]
    members.sort(key=lambda ids:ids)
    groups=[]
    for index,ids in enumerate(members):
        groups.append({'group_id':index,'members':list(ids),'score':max(tiles[i]['score'] for i in ids),
            'valid_points':sum(tiles[i]['valid_points'] for i in ids),
            'slow_service_cost':sum(tiles[i]['slow_service_cost'] for i in ids)})
    reasons={}
    for tile in tiles:
        key=tile['association_reason'];reasons[key]=reasons.get(key,0)+1
    return {'schema':SCHEMA,'shape':[h,w],'policy':policy,'association_mode':association_mode,
        'config':{'tile_size_pixels_xy':list(tile_size_pixels),'flow_scale_to_pixels_xy':list(flow_scale_to_pixels),
            'min_motion_pixels':min_motion_pixels,'max_distance_tiles':max_distance_tiles,
            'direction_tie_order_xy':[list(d) for d in DIRECTIONS],'aggregation':'max_member_score',
            'pair_tie_rule':'descending max score, then lexicographic member IDs',
            'rotation':'(-dy,dx) in image pixels before tile scaling' if policy=='rotated90' else None},
        'tiles':tiles,'groups':groups,'pairs':[list(p) for p in pairs],
        'metadata':{'active_tiles':sum(t['valid_points']>0 for t in tiles),'candidate_unique_edges':len(candidates),
            'accepted_pairs':len(pairs),'singleton_groups':sum(len(g['members'])==1 for g in groups),
            'conflict_rejected_edges':rejected,'proposal_reason_counts':reasons,
            'max_outgoing_proposals_per_tile':1,'recursive_propagation':False,
            'cost_kind':'caller_supplied_integer_slow_service_units_not_cycles',
            'GT_used_or_imputed':False,'motion_correspondence_proven':False,
            'feature_merging_performed':False,'temporal_execution':'caller must preserve complete T10'}}


def select_groups(proposal,slow_cost_cap):
    """Greedy max-score group selection under an exact nonnegative integer cap.

    Never splits a pair or exceeds the cap. An unaffordable high-score group
    is skipped and later groups are considered. Unused budget is explicit;
    this is not an optimal knapsack solver or an equal-realized-cost claim.
    """
    require(type(slow_cost_cap) is int and slow_cost_cap>=0,'Slow cost cap must be a nonnegative integer')
    require(proposal.get('schema')==SCHEMA,'Foreign proposal schema')
    h,w=proposal['shape'];tiles=proposal['tiles'];groups=proposal['groups']
    require(type(h) is int and type(w) is int and h>0 and w>0 and len(tiles)==h*w,'Bad proposal shape')
    require([t['tile_id'] for t in tiles]==list(range(h*w)),'Tile IDs are not raster ordered')
    for i,tile in enumerate(tiles):
        require(tile['row']==i//w and tile['col']==i%w,'Tile coordinates differ from raster ID')
        n,cost=tile['valid_points'],tile['slow_service_cost']
        require(type(n) is int and type(cost) is int and
                ((n==0 and cost==0) or (n>0 and cost>0)),'Bad tile validity/slow cost')
        require(n==0 or real(tile['score']),'Active tile has invalid score')
    seen=set();group_ids=set()
    for group in groups:
        ids=group['members']
        require(type(group['group_id']) is int and group['group_id'] not in group_ids,'Duplicate group ID')
        group_ids.add(group['group_id'])
        require(isinstance(ids,list) and len(ids) in (1,2) and ids==sorted(set(ids)) and
                all(type(i) is int and 0<=i<h*w and tiles[i]['valid_points']>0 and i not in seen for i in ids),
                'Groups overlap or contain invalid/empty members')
        require(type(group['slow_service_cost']) is int and group['slow_service_cost']>0 and
                group['slow_service_cost']==sum(tiles[i]['slow_service_cost'] for i in ids), 'Group slow cost differs')
        require(real(group['score']) and group['score']==max(tiles[i]['score'] for i in ids),'Group max score differs')
        seen.update(ids)
    require(seen=={t['tile_id'] for t in tiles if t['valid_points']>0},'Groups omit active tiles')
    order=sorted(groups,key=lambda g:(-g['score'],tuple(g['members'])))
    used=0;selected=[];skipped=[];mask=[[False]*w for _ in range(h)]
    for group in order:
        if used+group['slow_service_cost']>slow_cost_cap:
            skipped.append(group['group_id']);continue
        used+=group['slow_service_cost'];selected.append(group['group_id'])
        for i in group['members']:mask[i//w][i%w]=True
    selected_ids=[t['tile_id'] for t in tiles if mask[t['row']][t['col']]]
    require(used==sum(tiles[i]['slow_service_cost'] for i in selected_ids) and used<=slow_cost_cap,'Internal slow-cap error')
    return {'schema':'flow_motion_group_selection_v1','policy':proposal['policy'],
        'association_mode':proposal['association_mode'],'slow_mask':mask,'selected_group_ids':selected,
        'selected_tile_ids':selected_ids,'cap_skipped_group_ids':skipped,'slow_service_cost':used,
        'slow_cost_cap':slow_cost_cap,'unused_slow_cost':slow_cost_cap-used,
        'available_slow_service_cost':sum(t['slow_service_cost'] for t in tiles),
        'selected_valid_points':sum(tiles[i]['valid_points'] for i in selected_ids),
        'selected_tile_count':len(selected_ids),'selected_group_count':len(selected),
        'budget_semantics':'hard cap on supplied slow-service units; fast/scoring/selection costs excluded',
        'selection_rule':'descending group max score, first-fit; fixed member-ID tiebreak',
        'optimality_claim':False,'cycle_or_energy_measurement':False}
