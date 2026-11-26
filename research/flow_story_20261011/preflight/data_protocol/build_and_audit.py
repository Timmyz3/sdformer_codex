#!/usr/bin/env python3
"""Build a common DSEC target manifest and audit explicitly named flow data roots.

CPU/read-only with respect to source repositories and data. Outputs are confined
to --output-dir. No torch, CUDA, training, checkpoint loading, or network access.
Python 3.6+; NumPy is optional for the availability audit.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import socket
from collections import Counter
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
DEFAULT_OFFICIAL = HERE.parent / 'of_official_source'
DEFAULT_SD = Path('/home/zhumd/work/sdformer_codex/SDformer')
PINNED_OF_COMMIT = 'ff9406130acab6dcd3073cbe18f7f764806b0153'
PINNED_CSV = {
    'train': '91b9b4f3acc4f329419b562d970c1d6371ff6cb4c809e1b8955288a3cf10ddad',
    'valid': '5ed39eb5f7011a6190f4b3593f2cc31c3d3b246e1870d215a0ef6bdcc077c855',
}
ROLE_SEED = 'flow-a-bn-pool-v2:'
GATE_DEV_SEQUENCES = {'zurich_city_01_a', 'zurich_city_05_a'}
np = None
if importlib.util.find_spec('numpy'):
    import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows({k: r.get(k) for k in fields} for r in rows)


def parse_name(name):
    if Path(name).name != name or not name.endswith('.npy'):
        raise ValueError('Unsafe or unexpected filename: ' + name)
    seq, index = name[:-4].rsplit('_', 1)
    return seq, int(index)


def build_records(official):
    seqroot = official / 'data/dataset/saved_flow_data/sequence_lists'
    sources, rows = {}, []
    for split in ['train', 'valid']:
        p = seqroot / (split + '_split_doubleseq.csv')
        digest = sha(p)
        if digest != PINNED_CSV[split]:
            raise ValueError('Official split differs from audited pinned source: ' + str(p))
        source_rows = [r for r in csv.reader(p.open()) if r]
        sources[split] = {'path': str(p.resolve()), 'sha256': digest, 'rows': len(source_rows)}
        for source_row, pair in enumerate(source_rows, 1):
            if len(pair) != 2:
                raise ValueError('Expected exactly context,target columns')
            context, target = pair
            seq, n = parse_name(target)
            prevseq, prevn = parse_name(context)
            if seq != prevseq or n != prevn + 1:
                raise ValueError('Pair is not adjacent within a sequence')
            rows.append({
                'protocol_split': split, 'official_source_row_1based': source_row,
                'sequence': seq, 'target_id': target[:-4], 'target_index_1based': n,
                'of_context_filename': context, 'of_target_filename': target,
                'v2_target_filename': target, 'gt_filename': target, 'mask_filename': target,
                'timestamp_row_0based_candidate': n-1,
                'timestamp_mapping_status': 'unverified_raw_metadata_required',
                'target_start_us': None, 'target_end_us': None,
                'context_start_us': None, 'context_end_us': None,
                'raw_gt_png_filename': None, 'gt_sha256': None, 'mask_sha256': None,
                'v2_input_contract': 'one target interval, 10-bin signed voxel; encoding provenance pending',
                'of_input_contract': 'context+target 11-frame clips, last 21 frames; encoding provenance pending',
                'shared_gt_contract_status': 'pending_timestamp_rectification_and_content_checks',
            })
    train = [r for r in rows if r['protocol_split'] == 'train']
    valid = [r for r in rows if r['protocol_split'] == 'valid']
    trseq, vaseq = {r['sequence'] for r in train}, {r['sequence'] for r in valid}
    train_targets, valid_targets = {r['target_id'] for r in train}, {r['target_id'] for r in valid}
    train_clips = {r[k] for r in train for k in ['of_context_filename','of_target_filename']}
    valid_clips = {r[k] for r in valid for k in ['of_context_filename','of_target_filename']}
    checks = {
        'train_rows_6000': len(train) == 6000, 'valid_rows_2152': len(valid) == 2152,
        'train_sequences_13': len(trseq) == 13, 'valid_sequences_5': len(vaseq) == 5,
        'unique_target_each_split': len(train_targets) == len(train) and len(valid_targets) == len(valid),
        'train_valid_targets_disjoint': not (train_targets & valid_targets),
        'train_valid_sequences_disjoint': not (trseq & vaseq),
        'train_valid_input_context_clips_disjoint': not (train_clips & valid_clips),
    }
    if not all(checks.values()):
        raise ValueError('Membership checks failed: ' + repr(checks))
    return rows, sources, checks


def training_roles(rows):
    train = [r for r in rows if r['protocol_split'] == 'train']
    seqs = sorted({r['sequence'] for r in train})
    if not GATE_DEV_SEQUENCES <= set(seqs):
        raise ValueError('Frozen gate-development sequences missing from official train')
    candidates = [r for r in train if r['sequence'] not in GATE_DEV_SEQUENCES]
    digests = {r['target_id']: hashlib.sha256((ROLE_SEED+r['target_id']).encode()).hexdigest()
               for r in candidates}
    ranks = {}
    for seq in sorted(set(seqs) - GATE_DEV_SEQUENCES):
        members = sorted([r['target_id'] for r in candidates if r['sequence'] == seq],
                         key=lambda target: (digests[target], target))
        ranks.update({target: rank for rank, target in enumerate(members, 1)})
    result = []
    for r in train:
        rank = ranks.get(r['target_id'])
        role = ('gate_development' if r['sequence'] in GATE_DEV_SEQUENCES else
                'bn_calibration' if rank <= 64 else 'gate_fit')
        result.append({'target_id': r['target_id'], 'sequence': r['sequence'],
            'of_context_filename': r['of_context_filename'], 'target_filename': r['of_target_filename'],
            'backbone_training_member': True, 'auxiliary_role': role,
            'bn_selection_sha256': digests.get(r['target_id']),
            'bn_rank_within_sequence': rank,
            'bn_initial_member': rank is not None and rank <= 16,
            'bn_expanded_pool_member': rank is not None and rank <= 64})
    roles = sorted({r['auxiliary_role'] for r in result})
    target_sets = {role: {r['target_id'] for r in result if r['auxiliary_role'] == role}
                   for role in roles}
    clip_sets = {role: {r[k] for r in result if r['auxiliary_role'] == role
                       for k in ['of_context_filename', 'target_filename']} for role in roles}
    pairwise_target_disjoint = all(not (target_sets[a] & target_sets[b])
                                  for i, a in enumerate(roles) for b in roles[i+1:])
    initial_ids = {r['target_id'] for r in result if r['bn_initial_member']}
    if (not pairwise_target_disjoint or
            set.union(*target_sets.values()) != {r['target_id'] for r in train} or
            not initial_ids < target_sets['bn_calibration']):
        raise ValueError('Auxiliary role checks failed')
    return result, {'aux_policy_version': 'aux_policy_v2', 'seed': ROLE_SEED,
        'supersedes': 'aux_policy_v1: one-scene BN641, gate-dev724, gate-fit4635; planning-only, no training/calibration result produced',
        'change_reason': 'Use a deterministic multi-sequence BN pool to reduce single-scene calibration failure risk, and reconcile inconsistent per-sequence pool suggestions before experiments. Gate development and all official backbone/evaluation members remain unchanged.',
        'assignment_rule': 'Keep zurich_city_01_a and zurich_city_05_a entirely for gate development. In each remaining training sequence sort by SHA256(seed+target_id), target_id tie-break. First min(16,N) are the initial BN set; first min(64,N) are its reserved expanded pool; remaining targets fit gates.',
        'role_sequence_members': {role: sorted({r['sequence'] for r in result if r['auxiliary_role'] == role}) for role in roles},
        'role_target_counts': dict(Counter(r['auxiliary_role'] for r in result)),
        'bn_initial_target_count': len(initial_ids),
        'bn_initial_sequence_counts': dict(Counter(r['sequence'] for r in result if r['bn_initial_member'])),
        'bn_expanded_sequence_counts': dict(Counter(r['sequence'] for r in result if r['bn_expanded_pool_member'])),
        'bn_initial_is_strict_subset_of_expanded': initial_ids < target_sets['bn_calibration'],
        'backbone_train_members_unchanged': True,
        'roles_pairwise_disjoint_by_target_id': pairwise_target_disjoint,
        'roles_union_equals_official_train': True,
        'bn_and_gate_fit_sequences_overlap': True,
        'OF_context_clip_overlap_counts': {a+'__'+b: len(clip_sets[a] & clip_sets[b])
                                         for i, a in enumerate(roles) for b in roles[i+1:]},
        'context_overlap_scope': 'BN and gate-fit target IDs are disjoint, but neighboring OF clips and sequences can overlap. Do not call these scene- or context-independent holdouts. Gate development remains sequence-disjoint from both.',
        'scope': 'Holdouts for auxiliary gate fitting / BN calibration only. Backbone still trains on all 6000 official members. Gate-development data are NOT unseen-backbone generalization evidence.',
        'valid_usage': 'Five official validation sequences are final evaluation only; never choose thresholds, budget, BN statistics, gates, or architecture on them.',
        'bn_calibration_usage': 'Input-only statistics, no GT-based threshold or sample selection. Start with the initial set; if expansion is predeclared use the reserved expanded pool, with the same deterministic order and fresh capture. The entire expanded pool is excluded from gate fitting/development from the start.',
        'native_protocol_usage': 'Native paper reproduction is a separate protocol and result. These roles do not silently change the original backbone training membership.'}


def source_provenance(official, sd):
    paths = {
        'of_file_generator': official/'dsec_dataset_lite/data/file_generator.py',
        'of_loader': official/'data/dsec_dataset_lite_stereo_21x9.py',
        'of_sequence_metadata': official/'dsec_dataset_lite/main.py',
        'v2_preprocessor': sd/'tools/prepare_dsec_single_sequence.py',
        'v2_sequence_metadata': sd/'tools/prepare_dsec_full.py',
        'v2_loader': sd/'third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py',
    }
    entries = {k: {'path':str(p.resolve()),'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None} for k,p in paths.items()}
    try:
        commit = subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],universal_newlines=True).strip()
    except Exception:
        commit = None
    sequence_metadata = []
    p = paths['v2_sequence_metadata']
    if p.is_file():
        for node in ast.parse(p.read_text()).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_FLOW_SEQUENCES' for t in node.targets):
                sequence_metadata = ast.literal_eval(node.value)
    return {'official_repository':'https://github.com/J-Cuadrado/OF_EV_SNN',
            'expected_official_commit':PINNED_OF_COMMIT,'observed_official_commit':commit,
            'official_commit_matches':commit == PINNED_OF_COMMIT,'files':entries,
            'v2_default_sequence_names':sequence_metadata,
            'raw_mapping_rule_from_source':'target numeric suffix n -> timestamp row n-1 and n-th sorted forward PNG. A source-code rule, not proof about any existing NPY.',
            'rectification_source_findings':{
                'v2':'prepare_dsec_single_sequence.make_voxel_chunk passes rectified x/y to voxel conversion and filters image bounds.',
                'of':'pinned file_generator._load_events computes x_rect/y_rect but calls cumulate_spikes_into_frames(x,y,p) using the original coordinates.',
                'existing_of_preprocessed_arrays':'not verified; code difference is not proof of how author checkpoint inputs were generated.',
                'resolution':'must resolve and document before cross-model comparable evaluation; do not silently change the OF native preprocessing protocol.'}}


def inspect_array(path, expected_shape, cache):
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    r = {'path':key,'exists':path.is_file(),'sha256':None,'shape':None,'dtype':None,
         'shape_valid':False,'finite':None,'valid_pixel_count':None,'inspection_status':'missing'}
    if path.is_file():
        r['bytes'] = path.stat().st_size
        r['sha256'] = sha(path)
        if np is None:
            r['inspection_status'] = 'numpy_unavailable'
        else:
            try:
                a = np.load(str(path), mmap_mode='r', allow_pickle=False)
                r.update(shape=list(a.shape),dtype=str(a.dtype),shape_valid=tuple(a.shape)==tuple(expected_shape))
                r['finite'] = bool(np.isfinite(a).all())
                if tuple(expected_shape) == (480,640):
                    r['valid_pixel_count'] = int(np.count_nonzero(a))
                r['inspection_status'] = 'valid_shape' if r['shape_valid'] and r['finite'] else 'invalid_shape_or_nonfinite'
            except Exception as e:
                r['inspection_status'] = 'parse_error'
                r['error'] = type(e).__name__ + ': ' + str(e)
    cache[key] = r
    return r


def raw_metadata(root, sequence, family):
    if family == 'v2':
        flowdir = root/'train_optical_flow'/sequence/'flow'
        eventsdir = root/'train_events'/sequence/'events/left'
    else:
        flowdir = root/'train'/sequence/'flow'
        eventsdir = root/'train'/sequence/'events/left'
    timestamp = flowdir/'forward_timestamps.txt'
    pngs = sorted((flowdir/'forward').glob('*.png'))
    r = {'timestamp_path':str(timestamp.resolve()),'timestamp_exists':timestamp.is_file(),
         'timestamp_sha256':sha(timestamp) if timestamp.is_file() else None,
         'raw_forward_png_directory':str((flowdir/'forward').resolve()),
         'raw_forward_png_count':len(pngs),'flow_png_names':[p.name for p in pngs],
         'events_h5_path':str((eventsdir/'events.h5').resolve()),'events_h5_exists':(eventsdir/'events.h5').is_file(),
         'rectify_map_path':str((eventsdir/'rectify_map.h5').resolve()),'rectify_map_exists':(eventsdir/'rectify_map.h5').is_file(),
         'rectify_map_sha256':sha(eventsdir/'rectify_map.h5') if (eventsdir/'rectify_map.h5').is_file() else None,
         'timestamp_rows':[],'ordinal_png_mapping_ready':False,'status':'missing_raw_timestamp'}
    if timestamp.is_file():
        try:
            for line in timestamp.read_text().splitlines():
                line=line.strip()
                if not line or line.startswith('#'): continue
                vals=[int(v.strip()) for v in line.split(',')]
                if len(vals)!=2 or vals[0]>=vals[1]: raise ValueError('Expected increasing start,end microsecond pair')
                r['timestamp_rows'].append(vals)
            r['ordinal_png_mapping_ready']=len(pngs)==len(r['timestamp_rows']) and len(pngs)>0
            r['status']='raw_rows_and_png_count_match' if r['ordinal_png_mapping_ready'] else 'raw_png_timestamp_count_mismatch'
        except Exception as e:
            r['status']='timestamp_parse_error';r['error']=str(e)
    return r


def inventory(saved_root):
    result={}
    for kind, rel in [('gt','gt_tensors'),('mask','mask_tensors'),('events','event_tensors')]:
        paths=list((saved_root/rel).rglob('*.npy')) if (saved_root/rel).exists() else []
        result[kind]={'npy_count':len(paths),'sequence_counts':dict(Counter(parse_name(p.name)[0] for p in paths))}
    return result


def audit(rows, args):
    cache, missing, availability, mappings = {}, {}, [], []
    counters = Counter()
    raw={family:{seq:raw_metadata(root,seq,family) for seq in sorted({r['sequence'] for r in rows})}
         for family,root in [('v2',args.v2_raw_root),('of',args.of_raw_root)]}
    for r in rows:
        target, context, seq = r['of_target_filename'],r['of_context_filename'],r['sequence']
        requirements=[
            ('v2_gt',args.v2_saved_root/'gt_tensors'/target,(2,480,640)),
            ('v2_mask',args.v2_saved_root/'mask_tensors'/target,(480,640)),
            ('v2_events',args.v2_saved_root/'event_tensors/10bins/left'/seq/target,(10,480,640)),
            ('of_gt',args.of_saved_root/'gt_tensors'/target,(2,480,640)),
            ('of_mask',args.of_saved_root/'mask_tensors'/target,(480,640)),
        ]
        # The pinned loader expects 11frames/<filename>; the generator writes
        # 11frames/left/<filename>. Report both; never silently change loaders.
        for key,filename in [('of_context_events',context),('of_target_events',target)]:
            requirements.append((key,args.of_saved_root/args.of_event_subdir/filename,(11,2,480,640)))
        a={'protocol_split':r['protocol_split'],'target_id':r['target_id']}
        for key,p,shape in requirements:
            rec=inspect_array(p,shape,cache)
            ready=rec['inspection_status']=='valid_shape'
            a[key+'_ready']=ready
            counters[(r['protocol_split'],key,'exists')]+=int(rec['exists'])
            counters[(r['protocol_split'],key,'ready')]+=int(ready)
            if not ready:
                entry=missing.setdefault(key,{})
                entry[str(p.resolve())]=rec['inspection_status']
        a['v2_arrays_ready']=all(a[k+'_ready'] for k in ['v2_gt','v2_mask','v2_events'])
        a['of_arrays_ready']=all(a[k+'_ready'] for k in ['of_gt','of_mask','of_context_events','of_target_events'])
        m={'protocol_split':r['protocol_split'],'target_id':r['target_id'],'v2':None,'of':None,'target_timestamps_equal':None,
           'gt_content_equal_between_models':None,'mask_content_equal_between_models':None,
           'gt_npy_matches_raw_png':None,'gt_npy_raw_binding_status':'not_verified_by_this_audit'}
        for family in ['v2','of']:
            meta=raw[family][seq]
            ix=r['target_index_1based']-1
            if meta['ordinal_png_mapping_ready'] and ix < len(meta['timestamp_rows']):
                m[family]={'target_interval_us':meta['timestamp_rows'][ix],
                           'context_interval_us':meta['timestamp_rows'][ix-1],
                           'raw_gt_png_filename':meta['flow_png_names'][ix],
                           'binding_scope':'ordinal raw metadata only; stored NPY generation provenance remains unverified'}
        if m['v2'] is not None and m['of'] is not None:
            m['target_timestamps_equal']=m['v2']['target_interval_us']==m['of']['target_interval_us']
        for kind in ['gt','mask']:
            va=cache[str((args.v2_saved_root/(kind+'_tensors')/target).resolve())]
            oa=cache[str((args.of_saved_root/(kind+'_tensors')/target).resolve())]
            if va['inspection_status']=='valid_shape' and oa['inspection_status']=='valid_shape' and np is not None:
                left=np.load(va['path'],mmap_mode='r',allow_pickle=False)
                right=np.load(oa['path'],mmap_mode='r',allow_pickle=False)
                m[kind+'_content_equal_between_models']=bool(np.array_equal(left,right))
        a['cross_model_target_time_aligned']=m['target_timestamps_equal'] is True
        availability.append(a);mappings.append(m)
    summary={split:{key:{field:count for (s,k,field),count in counters.items() if s==split and k==key}
                    for key in sorted({k for s,k,f in counters if s==split})} for split in ['train','valid']}
    for family,sequences in raw.items():
        for seq,meta in sequences.items():
            for key,path_key,ready in [
                    ('timestamp','timestamp_path',meta['timestamp_exists']),
                    ('events_h5','events_h5_path',meta['events_h5_exists']),
                    ('rectify_map','rectify_map_path',meta['rectify_map_exists']),
                    ('forward_pngs','raw_forward_png_directory',meta['raw_forward_png_count']>0)]:
                if not ready:
                    missing.setdefault('raw_'+family+'_'+key,{})[meta[path_key]]='missing_or_empty'
    roots={k:str(getattr(args,k).resolve()) for k in ['v2_saved_root','of_saved_root','v2_raw_root','of_raw_root']}
    return {'roots':roots,'summary':summary,'availability':availability,'mappings':mappings,
            'missing':{k:[{'path':p,'status':status} for p,status in sorted(v.items())] for k,v in missing.items()},
            'existing_evidence':[v for v in cache.values() if v['exists']],
            'raw_metadata':raw,'inventory':{'v2':inventory(args.v2_saved_root),'of':inventory(args.of_saved_root)},
            'of_event_subdir':str(args.of_event_subdir),'numpy_available':np is not None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-source',type=Path,default=DEFAULT_OFFICIAL)
    p.add_argument('--sd-root',type=Path,default=DEFAULT_SD)
    p.add_argument('--v2-saved-root',type=Path)
    p.add_argument('--of-saved-root',type=Path)
    p.add_argument('--v2-raw-root',type=Path)
    p.add_argument('--of-raw-root',type=Path)
    p.add_argument('--of-event-subdir',type=Path,default=Path('event_tensors/11frames'))
    p.add_argument('--output-dir',type=Path,default=HERE/'artifacts')
    args=p.parse_args()
    args.v2_saved_root=args.v2_saved_root or args.sd_root/'data/Datasets/DSEC/saved_flow_data'
    args.of_saved_root=args.of_saved_root or args.official_source/'data/dataset/saved_flow_data'
    args.v2_raw_root=args.v2_raw_root or args.sd_root/'data/Datasets/DSEC'
    args.of_raw_root=args.of_raw_root or args.official_source/'data/dataset'
    if args.of_event_subdir.is_absolute() or '..' in args.of_event_subdir.parts:
        raise ValueError('--of-event-subdir must be a relative subdirectory')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    records,sources,checks=build_records(args.official_source)
    roles,role_policy=training_roles(records)
    provenance=source_provenance(args.official_source,args.sd_root)
    checks['local_sequence_metadata_covers_all_official_sequences']={r['sequence'] for r in records}==set(provenance['v2_default_sequence_names'])
    result=audit(records,args)
    fields=list(records[0])
    canonical={'schema_version':1,'protocol_id':'FLOW_A_OF_NATIVE_TARGET_MEMBERSHIP_V1',
       'membership_status':'fixed_official_OF_CSV_membership',
       'source_split_hashes':{k:v['sha256'] for k,v in sources.items()},
       'unknown_value_policy':'JSON null / blank CSV means unverified, never zero. Canonical fields are not populated from filename guesses.',
       'records':records}
    dump(out/'common_targets.json',canonical)
    write_csv(out/'common_targets.csv',records,fields)
    for split in ['train','valid']:
        selected=[r for r in records if r['protocol_split']==split]
        write_csv(out/('of_'+split+'_pairs.csv'),selected,['of_context_filename','of_target_filename'])
        write_csv(out/('v2_'+split+'_targets.csv'),selected,['v2_target_filename'])
        # Headerless exports are the drop-in file-list format of the original loaders.
        with (out/('of_'+split+'_pairs_noheader.csv')).open('w',newline='') as f:
            csv.writer(f,lineterminator='\n').writerows([r['of_context_filename'],r['of_target_filename']] for r in selected)
        with (out/('v2_'+split+'_targets_noheader.csv')).open('w',newline='') as f:
            csv.writer(f,lineterminator='\n').writerows([r['v2_target_filename']] for r in selected)
    write_csv(out/'train_auxiliary_roles.csv',roles,list(roles[0]))
    dump(out/'train_auxiliary_policy.json',role_policy)
    bn_schedule={'aux_policy_version':'aux_policy_v2','seed':ROLE_SEED,'stages':{}}
    for stage,flag in [('initial','bn_initial_member'),('expanded','bn_expanded_pool_member')]:
        selected=sorted([r for r in roles if r[flag]],
                        key=lambda r:(r['bn_rank_within_sequence'],r['sequence']))
        write_csv(out/('bn_calibration_'+stage+'_targets.csv'),selected,list(roles[0]))
        bn_schedule['stages'][stage]={'target_count':len(selected),
            'target_ids':[r['target_id'] for r in selected],
            'per_sequence_counts':dict(Counter(r['sequence'] for r in selected))}
    bn_schedule['execution_order']='Round-robin by per-sequence SHA rank, then sequence name. The initial target ID list is an exact prefix of the expanded list.'
    bn_schedule['initial_is_expanded_prefix']=(bn_schedule['stages']['initial']['target_ids']==
        bn_schedule['stages']['expanded']['target_ids'][:bn_schedule['stages']['initial']['target_count']])
    if not bn_schedule['initial_is_expanded_prefix']:
        raise ValueError('Initial calibration execution order is not an expanded prefix')
    dump(out/'bn_calibration_schedule.json',bn_schedule)
    write_csv(out/'local_availability.csv',result['availability'],list(result['availability'][0]))
    dump(out/'missing_artifacts.json',result['missing'])
    dump(out/'timestamp_gt_mapping_audit.json',result['mappings'])
    dump(out/'existing_array_evidence.json',result['existing_evidence'])
    dump(out/'raw_metadata_audit.json',result['raw_metadata'])
    dump(out/'source_provenance.json',{'sources':sources,'provenance':provenance,'checks':checks})
    summary_record={k:result[k] for k in ['roots','summary','inventory','of_event_subdir','numpy_available']}
    summary_record['hostname']=socket.gethostname()
    summary_record['scope']='Only the explicitly configured data roots, not a whole-machine absence claim.'
    dump(out/'availability_summary.json',summary_record)
    # Source disagreement + no NPY-to-PNG content binding means this script alone
    # deliberately cannot certify cross-model comparability or training readiness.
    readiness={
        'schema_version':1,'hostname':socket.gethostname(),'membership_manifest_ready':all(checks.values()),
        'aux_policy_version':role_policy['aux_policy_version'],
        'official_csv_hashes_match':True,'official_commit_matches':provenance['official_commit_matches'],
        'train_valid_membership_checks':checks,
        'v2_required_arrays_ready':all(r['v2_arrays_ready'] for r in result['availability']),
        'of_required_arrays_ready':all(r['of_arrays_ready'] for r in result['availability']),
        'all_target_timestamps_cross_model_aligned':all(r['cross_model_target_time_aligned'] for r in result['availability']),
        'gt_npy_to_raw_png_content_binding_ready':False,
        'preprocessing_rectification_agreement_ready':False,
        'cross_model_comparable_dataset_ready':False,'common_training_data_ready':False,
        'native_paper_reproduction_ready':False,'gpu_training_ready':False,
        'blocking_reasons':['required_preprocessed_arrays_incomplete_or_invalid' if not all(r['v2_arrays_ready'] and r['of_arrays_ready'] for r in result['availability']) else 'raw_metadata_and_provenance_still_required',
            'raw_timestamp_and_gt_binding_not_complete',
            'OF_loader_generator_event_path_disagreement_requires_explicit_resolved_adapter',
            'OF_v2_rectification_contracts_require_explicit_resolution'],
        'separate_research_pending':['native_paper_protocol_reference_not_verified',
            'any_future_warm_start_checkpoint_requires_training_provenance; no checkpoint used in this audit'],
        'no_training_started':True,'no_gpu_used':True,
        'existing_v2_complete_target_count':sum(r['v2_arrays_ready'] for r in result['availability']),
        'existing_of_complete_target_count':sum(r['of_arrays_ready'] for r in result['availability']),
        'note':'Manifest readiness proves membership only. GPU readiness here covers data prerequisites only, not server availability or authorization. Paper accuracy reproduction is a separate later gate, not a prerequisite to training that reproduction. Use a distinct output directory when auditing another machine.'}
    dump(out/'ready_status.json',readiness)
    artifact_hashes={f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name!='artifact_sha256.json'}
    dump(out/'artifact_sha256.json',artifact_hashes)
    print(json.dumps({'output_dir':str(out),'members':dict(Counter(r['protocol_split'] for r in records)),
          'roles':role_policy['role_target_counts'],'membership_manifest_ready':readiness['membership_manifest_ready'],
          'v2_complete_targets':readiness['existing_v2_complete_target_count'],
          'of_complete_targets':readiness['existing_of_complete_target_count'],
          'gpu_training_ready':False},indent=2))


if __name__=='__main__': main()
