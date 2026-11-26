#!/usr/bin/env python3
"""Executable, deliberately serial BN service contract. Not a fast BN design.

All arithmetic latencies are explicit assumptions. Expanded small cases perform
FP32 roundings; real-shape results are exact repeated-motif transaction/cycle
counts under this serial policy, NOT a simulation of 976M values.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
LEDGER = HERE.parents[1] / 'flow_story_20261014/damage_allocation_probe/hardware_contract_complete_bn_audit.json'
LEDGER_SHA = '9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def config():
    return dict(schema='serial_bn_service_config_v1', status='DECLARED_ASSUMPTIONS_NOT_RTL_TIMING',
                scratch_words_per_region=4096, scratch_regions=2, banks=16,
                read_ports_per_bank=1, write_ports_per_bank=1, state_words=4096,
                state_read_ports=1, state_write_ports=1, operand_register_words=8,
                external_outstanding=1, external_transaction_words=1, external_capacity_words=300000000,
                arithmetic_lanes=1, max_active_reduction_commands=1,
                latency=dict(external_read=33, external_write=33, scratch_read=2,
                             scratch_write=1, state_read=1, state_write=1,
                             add=4, sub=4, mul=4, rsqrt=16, copy=1,
                             command_accept=1, command_retire=1, chunk_config=1,
                             chunk_finish=1, mean_barrier=1, affine_barrier=1),
                initiation_interval=dict(add=1, sub=1, mul=1, rsqrt=8, copy=1),
                epsilon=1e-5)


def validate(cfg):
    assert cfg['scratch_regions'] == 2 and cfg['scratch_words_per_region'] > 0
    assert cfg['banks'] > 0 and cfg['read_ports_per_bank'] == cfg['write_ports_per_bank'] == 1
    assert cfg['state_read_ports'] == cfg['state_write_ports'] == 1
    assert cfg['operand_register_words'] >= 8
    assert cfg['external_outstanding'] == cfg['external_transaction_words'] == 1
    assert type(cfg['external_capacity_words']) is int and cfg['external_capacity_words'] > 0
    assert cfg['arithmetic_lanes'] == cfg['max_active_reduction_commands'] == 1
    assert set(cfg['latency']) == set(config()['latency'])
    assert all(type(x) is int and x > 0 for x in cfg['latency'].values())
    assert all(type(x) is int and x > 0 for x in cfg['initiation_interval'].values())


def duration(kind, cfg):
    return max(cfg['latency'][kind], cfg['initiation_interval'].get(kind, 1))


def geometry(record):
    shape = record['native_input_shape']
    strides = record['observed']['input_calls'][0]['stride']
    ca = record['channel_axis']; c = record['channels']
    assert len(shape) == len(strides) and shape[ca] == c
    size = 1
    for n in shape:
        size *= n
    assert size == record['elements']
    # Independently prove the native view is a contiguous permutation. Size-1
    # dimensions are allowed arbitrary stride. No free transpose is introduced.
    running = 1
    for st, n in sorted((st, n) for n, st in zip(shape, strides) if n > 1):
        assert st == running, ('Unsupported non-dense view', shape, strides)
        running *= n
    if strides[ca] == 1:
        mode = 'channel_contiguous'; gc = c; groups = 1
    else:
        mode = 'channel_major'; gc = 1; groups = c
        assert strides[-1] == 1 and strides[-2] == shape[-1]
    return dict(shape=shape, strides=strides, channel_axis=ca, channels=c,
                samples=size // c, elements=size, mode=mode,
                group_channels=gc, groups=groups)


def native_address(geo, sample, channel):
    assert 0 <= channel < geo['channels'] and 0 <= sample < geo['samples']
    result = channel * geo['strides'][geo['channel_axis']]
    axes = [i for i in range(len(geo['shape'])) if i != geo['channel_axis']]
    for i in reversed(axes):
        sample, coord = divmod(sample, geo['shape'][i])
        result += coord * geo['strides'][i]
    assert sample == 0
    return result


def reduction_commands(n, channels, region_words):
    assert n > 0 and channels > 0 and n * channels <= region_words
    source, dest, level = 0, region_words, 0
    while n > 1:
        for p in range(n // 2):
            for c in range(channels):
                yield dict(level=level, kind='pair', src=[source + 2*p*channels+c,
                      source + (2*p+1)*channels+c], dst=dest+p*channels+c, channel=c)
        if n % 2:
            for c in range(channels):
                yield dict(level=level, kind='carry', src=[source+(n-1)*channels+c],
                           dst=dest+(n//2)*channels+c, channel=c)
        n = (n + 1) // 2
        source, dest = dest, source; level += 1


def reduce_shape(n, channels, region_words):
    pairs = carries = 0; last_base = 0; levels = 0
    while n > 1:
        pairs += (n // 2) * channels
        carries += (n % 2) * channels
        n = (n + 1) // 2; levels += 1
        last_base = region_words if last_base == 0 else 0
    return pairs, carries, last_base, levels


def compressed(record, cfg):
    validate(cfg); geo = geometry(record)
    m, c, gc, groups = [geo[k] for k in ('samples','channels','group_channels','groups')]
    assert cfg['state_words'] >= 4 * gc
    capacity = cfg['scratch_words_per_region'] // gc
    assert capacity > 0
    q, tail = divmod(m, capacity)
    patterns = [(capacity, q)] if q else []
    if tail:
        patterns.append((tail, 1))
    chunks = q + bool(tail)
    p = chunks * c
    pair = carry = 0
    motifs = []
    for n, repeats in patterns:
        a, b, base, levels = reduce_shape(n, gc, cfg['scratch_words_per_region'])
        pair += a * repeats * groups * 2
        carry += b * repeats * groups * 2
        motifs.append(dict(samples=n, channels=gc, repeats_per_pass_per_group=repeats,
                           pairs=a, carry=b, levels=levels, result_base=base))
    e = geo['elements']
    assert 2*e+2*c <= cfg['external_capacity_words']
    cnt = Counter(external_read=3*e+2*c, external_write=e,
                  scratch_read=2*pair+carry+2*p,
                  scratch_write=2*e+pair+carry,
                  state_read=3*e+2*p+c, state_write=2*p+3*c,
                  add=pair+2*(p-c)+e+c, sub=e+c,
                  mul=2*e+4*c, rsqrt=c, copy=carry,
                  command_accept=pair+carry, command_retire=pair+carry,
                  chunk_config=2*chunks*groups, chunk_finish=2*chunks*groups,
                  mean_barrier=groups, affine_barrier=groups)
    assert cnt['add'] + cnt['sub'] == 4*e
    assert pair == 2*(e-p)
    cycles = {k:v*duration(k,cfg) for k,v in cnt.items()}
    return dict(name=record['name'], family=record['family'], geometry=geo,
                chunk_samples=capacity, chunks_per_group_per_pass=chunks,
                chunk_motifs=motifs, transactions_and_operations=dict(cnt),
                cycles_by_operation=cycles, conditional_serial_service_cycles=sum(cycles.values()),
                external_input_reads=3*e, external_parameter_reads=2*c,
                external_output_writes=e, pair_commands=pair, carry_commands=carry,
                local_storage_reserved_words=2*cfg['scratch_words_per_region']+cfg['state_words'],
                active_state_words=4*gc, external_required_words=2*e+2*c,
                external_input_base=0, external_gamma_base=e, external_beta_base=e+c, external_output_base=e+2*c,
                phases=['mean_chunks','mean_barrier','variance_chunks','affine','affine_barrier','normalize'],
                original_input_resident=False, full_network_cycles=None)


class Literal:
    """Fully expanded executable tiny case, with actual addresses and FP32 values.

    No overlap is attempted. Every primitive retires before the next issues;
    this guarantees all finite-port/resource constraints but intentionally does
    not represent a competitive high-throughput design.
    """
    def __init__(self, cfg, external):
        validate(cfg); self.cfg=cfg; self.external=dict(external); self.output={}
        self.scratch={}; self.state={}; self.now=0; self.events=[]; self.count=Counter()
        self.phase=None

    def event(self, kind, **info):
        item=dict(kind=kind, issue=self.now, retire=self.now+duration(kind,self.cfg), phase=self.phase)
        item.update(info); self.events.append(item)
        self.count[kind]+=1; self.now=item['retire']; return item

    def read(self, space, address, parameter=None):
        if space=='external':
            assert 0 <= address < self.cfg['external_capacity_words']
            value = self.external[address]
        else:
            memory=getattr(self,space); assert address in memory
            assert 0 <= address < (2*self.cfg['scratch_words_per_region'] if space=='scratch' else self.cfg['state_words'])
            value=memory[address]
        info=dict(address=address, value=value)
        if space=='scratch': info.update(bank=address%self.cfg['banks'], row=address//self.cfg['banks'])
        if parameter: info['parameter']=parameter
        self.event(space+'_read',**info); return value

    def write(self, space, address, value):
        value=f32(value)
        if space=='external':
            assert 0 <= address < self.cfg['external_capacity_words']
            assert address not in self.output
            self.output[address]=value
        else:
            assert 0 <= address < (2*self.cfg['scratch_words_per_region'] if space=='scratch' else self.cfg['state_words'])
            getattr(self,space)[address]=value
        info=dict(address=address,value=value)
        if space=='scratch': info.update(bank=address%self.cfg['banks'],row=address//self.cfg['banks'])
        self.event(space+'_write',**info)

    def op(self,kind,*values):
        if kind=='add': y=values[0]+values[1]
        elif kind=='sub': y=values[0]-values[1]
        elif kind=='mul': y=values[0]*values[1]
        elif kind=='rsqrt': y=1.0/math.sqrt(values[0])
        elif kind=='copy': y=values[0]
        else: raise ValueError(kind)
        y=f32(y); self.event(kind,inputs=list(values),value=y); return y

    def reduce(self,n,c):
        self.event('chunk_config',samples=n,channels=c)
        for command in reduction_commands(n,c,self.cfg['scratch_words_per_region']):
            self.event('command_accept',command=command)
            values=[self.read('scratch',a) for a in command['src']]
            y=self.op('add' if command['kind']=='pair' else 'copy',*values)
            self.write('scratch',command['dst'],y)
            self.event('command_retire',command=command)
        self.event('chunk_finish')
        return reduce_shape(n,c,self.cfg['scratch_words_per_region'])[2]


def literal(record, cfg, values=None, gamma=None, beta=None):
    geo=geometry(record); m=geo['samples']; c=geo['channels']; gc=geo['group_channels']
    assert cfg['state_words'] >= 4*gc
    values=values or [[f32(((s*17+k*3)%23)-11)/4 for k in range(c)] for s in range(m)]
    gamma=gamma or [f32(1+k/8) for k in range(c)]
    beta=beta or [f32(k/16-0.125) for k in range(c)]
    assert len(values)==m and all(len(x)==c for x in values)
    ext={native_address(geo,s,k): f32(values[s][k]) for s in range(m) for k in range(c)}
    assert len(ext)==geo['elements']
    for k in range(c): ext[geo['elements']+k]=f32(gamma[k]); ext[geo['elements']+c+k]=f32(beta[k])
    mach=Literal(cfg,ext); capacity=cfg['scratch_words_per_region']//gc
    inv_m=f32(1.0/m)
    for group in range(geo['groups']):
        first_channel=group if gc==1 else 0
        for pass_id in (0,1):
            mach.phase='mean' if pass_id==0 else 'variance'
            for start in range(0,m,capacity):
                n=min(capacity,m-start)
                for s in range(n):
                    for k in range(gc):
                        x=mach.read('external',native_address(geo,start+s,first_channel+k))
                        if pass_id:
                            mu=mach.read('state',k)
                            x=mach.op('sub',x,mu); x=mach.op('mul',x,x)
                        mach.write('scratch',s*gc+k,x)
                base=mach.reduce(n,gc)
                for k in range(gc):
                    partial=mach.read('scratch',base+k)
                    slot=pass_id*gc+k
                    if start:
                        partial=mach.op('add',mach.read('state',slot),partial)
                    mach.write('state',slot,partial)
            if pass_id==0:
                for k in range(gc):
                    mu=mach.op('mul',mach.read('state',k),inv_m)
                    mach.write('state',k,mu)
                mach.event('mean_barrier')
        mach.phase='affine'
        for k in range(gc):
            v=mach.op('mul',mach.read('state',gc+k),inv_m)
            inv=mach.op('rsqrt',mach.op('add',v,f32(cfg['epsilon'])))
            g=mach.read('external',geo['elements']+first_channel+k,parameter='gamma')
            a=mach.op('mul',g,inv); mach.write('state',2*gc+k,a)
            amu=mach.op('mul',a,mach.read('state',k))
            b=mach.op('sub',mach.read('external',geo['elements']+c+first_channel+k,parameter='beta'),amu)
            mach.write('state',3*gc+k,b)
        mach.event('affine_barrier'); mach.phase='normalize'
        for s in range(m):
            for k in range(gc):
                addr=native_address(geo,s,first_channel+k)
                x=mach.read('external',addr)
                a=mach.read('state',2*gc+k); b=mach.read('state',3*gc+k)
                y=mach.op('add',mach.op('mul',x,a),b)
                mach.write('external',geo['elements']+2*c+addr,y)
    expected=compressed(record,cfg)
    assert dict(mach.count)=={k:v for k,v in expected['transactions_and_operations'].items() if v}, (mach.count,expected)
    assert mach.now==expected['conditional_serial_service_cycles']
    return dict(record=record,config=cfg,cycles=mach.now,counters=dict(mach.count),
                input=values,gamma=gamma,beta=beta,
                output=[[mach.output[geo['elements']+2*c+native_address(geo,s,k)] for k in range(c)] for s in range(m)],
                events=mach.events)


def make_record(samples,channels,layout='channel_contiguous'):
    # shape [sample, C, 1, 1], either sample-major or channel-major storage.
    stride=[channels,1,1,1] if layout=='channel_contiguous' else [1,samples,1,1]
    return dict(name='toy_{}x{}_{}'.format(samples,channels,layout),family='toy',
                native_input_shape=[samples,channels,1,1],channel_axis=1,channels=channels,
                elements=samples*channels,reduction_elements_per_channel=samples,
                observed=dict(input_calls=[dict(stride=stride)]))


def all_real(cfg):
    assert sha(LEDGER)==LEDGER_SHA
    records=json.loads(LEDGER.read_text())['external_BN']['records']
    results=[compressed(r,cfg) for r in records]
    total=Counter(); fam={}
    for r in results:
        total.update(r['transactions_and_operations'])
        f=fam.setdefault(r['family'],dict(records=0,elements=0,cycles=0))
        f['records']+=1; f['elements']+=r['geometry']['elements']; f['cycles']+=r['conditional_serial_service_cycles']
    assert len(results)==54 and sum(x['geometry']['elements'] for x in results)==976584960
    assert sum(x['geometry']['channels'] for x in results)==19056
    assert total['add']+total['sub']==3906339840
    assert total['mul']==1953246144 and total['rsqrt']==19056
    return dict(schema='serial_complete_bn_service_v1', ledger_path=str(LEDGER),ledger_sha=LEDGER_SHA,
                parameter_status=cfg['status'],config=cfg,record_count=len(results),
                total_elements=976584960,total_channels=19056,
                aggregate_transactions_and_operations=dict(total),
                aggregate_cycles_by_operation={k:v*duration(k,cfg) for k,v in total.items()},
                conditional_independent_BN_serial_cycles=sum(x['conditional_serial_service_cycles'] for x in results),
                by_family=fam,records=results,full_network_cycles=None,hardware_benefit=None,
                RTL_full_service_validated=False,FP32_native_equivalence_validated=False,
                real_elements_numerically_executed=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    path=Path(args.output); assert not path.exists(), 'Refuse overwriting evidence'
    path.write_text(json.dumps(all_real(config()),indent=2,sort_keys=True)+'\n')

if __name__=='__main__': main()
