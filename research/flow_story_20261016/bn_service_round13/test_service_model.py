#!/usr/bin/env python3
"""Independent mathematical values and event replay, plus compressed-vs-literal.

This imports the model under test but does not call its address/reduction/golden
helpers from the replay or direct mathematical output reference.
"""
import copy
import json
import math
from pathlib import Path
import struct
import unittest
import service_model as model


def fp(x):
    return struct.unpack('f',struct.pack('f',x))[0]


def address(shape,stride,axis,s,c):
    coords=[0]*len(shape); coords[axis]=c
    for i in reversed(range(len(shape))):
        if i==axis: continue
        coords[i]=s%shape[i]; s//=shape[i]
    assert s==0
    return sum(x*y for x,y in zip(coords,stride))


def replay(run):
    r=run['record']; cfg=run['config']; shape=r['native_input_shape']; axis=r['channel_axis']
    stride=r['observed']['input_calls'][0]['stride']; c=r['channels']; e=r['elements']
    memory={'external':{},'scratch':{},'state':{},'output':{}}
    for s,row in enumerate(run['input']):
        for k,x in enumerate(row): memory['external'][address(shape,stride,axis,s,k)]=fp(x)
    for k in range(c):
        memory['external'][e+k]=fp(run['gamma'][k]); memory['external'][e+c+k]=fp(run['beta'][k])
    previous=0; active=None; config_shape=None; finished=False; counters={}
    for event in run['events']:
        kind=event['kind']; counters[kind]=counters.get(kind,0)+1
        assert event['issue']==previous
        assert event['retire']-event['issue']==max(cfg['latency'][kind],cfg['initiation_interval'].get(kind,1))
        previous=event['retire']
        if kind=='chunk_config':
            assert active is None
            config_shape=(event['samples'],event['channels']); finished=False
            assert config_shape[0]*config_shape[1]<=cfg['scratch_words_per_region']
        elif kind=='command_accept':
            assert config_shape and not finished and active is None
            active=event['command']; assert len(active['src'])==(2 if active['kind']=='pair' else 1)
        elif kind=='command_retire':
            assert active==event['command']; active=None
        elif kind=='chunk_finish': assert active is None; finished=True
        elif kind.endswith('_read') or kind.endswith('_write'):
            space,direction=kind.rsplit('_',1); addr=event['address']
            if space=='external': assert 0<=addr<cfg['external_capacity_words']
            if space in ('scratch','state'):
                assert 0<=addr<(2*cfg['scratch_words_per_region'] if space=='scratch' else cfg['state_words'])
            if space=='scratch':
                assert event['bank']==addr%cfg['banks'] and event['row']==addr//cfg['banks']
            if direction=='read': assert event['value']==memory[space][addr]
            else:
                if space=='external':
                    assert addr not in memory['output']; memory['output'][addr]=event['value']
                else: memory[space][addr]=event['value']
        elif kind in ('add','sub','mul','copy','rsqrt'):
            x=event['inputs']
            if kind=='add': result=x[0]+x[1]
            elif kind=='sub': result=x[0]-x[1]
            elif kind=='mul': result=x[0]*x[1]
            elif kind=='copy': result=x[0]
            else: result=1/math.sqrt(x[0])
            assert event['value']==fp(result)
    assert active is None and previous==run['cycles'] and counters==run['counters']
    assert len(memory['output'])==e
    for s,row in enumerate(run['output']):
        for k,y in enumerate(row): assert y==memory['output'][e+2*c+address(shape,stride,axis,s,k)]
    return len(run['events'])


def direct_reference(run):
    x=run['input']; m=len(x); c=len(x[0]); errors=[]
    for k in range(c):
        mean=math.fsum(row[k] for row in x)/m
        var=math.fsum((row[k]-mean)**2 for row in x)/m
        for s in range(m):
            y=(x[s][k]-mean)/math.sqrt(var+run['config']['epsilon'])*run['gamma'][k]+run['beta'][k]
            errors.append(abs(y-run['output'][s][k]))
    return max(errors)


class ServiceChecks(unittest.TestCase):
    def test_literal_all_phases_and_layouts(self):
        cfg=model.config(); cfg['scratch_words_per_region']=24; cfg['state_words']=64
        max_error=0; events=0; cases=0
        for layout in ('channel_major','channel_contiguous'):
            for n in (1,2,3,5,7,9,25,49):
                for c in (1,3,8):
                    run=model.literal(model.make_record(n,c,layout),cfg)
                    events+=replay(run); max_error=max(max_error,direct_reference(run)); cases+=1
        self.assertLess(max_error,2e-5)
        self.__class__.literal_summary=dict(cases=cases,events=events,max_abs_error_vs_float64_math=max_error,
                                          native_pytorch_equivalence=False)

    def test_constant_input(self):
        cfg=model.config(); cfg['scratch_words_per_region']=12
        run=model.literal(model.make_record(17,3),cfg,values=[[2.0]*3 for _ in range(17)])
        replay(run)
        self.assertLess(direct_reference(run),1e-4)

    def test_replay_rejects_corruption(self):
        run=model.literal(model.make_record(5,3),model.config()); bad=copy.deepcopy(run)
        index=next(i for i,e in enumerate(bad['events']) if e['kind']=='scratch_read')
        bad['events'][index]['value']+=1
        with self.assertRaises(AssertionError): replay(bad)
        bad=copy.deepcopy(run); bad['events'][0]['retire']+=1
        with self.assertRaises(AssertionError): replay(bad)

    def test_capacity_and_unknown_stride(self):
        cfg=model.config(); cfg['state_words']=1
        with self.assertRaises(AssertionError): model.compressed(model.make_record(5,3),cfg)
        record=model.make_record(5,3); record['observed']['input_calls'][0]['stride'][0]=4
        with self.assertRaises(AssertionError): model.geometry(record)

    def test_all_real_shape_conservation(self):
        result=model.all_real(model.config()); records=result['records']
        pairs=carry=templates=0; patterns={}
        for row in records:
            g=row['geometry']; c=g['channels']; m=g['samples']; gc=g['group_channels']
            # Independent arithmetic identity for this centered two-pass schedule.
            ops=row['transactions_and_operations']; e=g['elements']
            self.assertEqual(ops['add']+ops['sub'],4*e)
            self.assertEqual(ops['mul'],2*e+4*c)
            self.assertEqual(ops['external_read'],3*e+2*c)
            self.assertEqual(ops['external_write'],e)
            self.assertEqual(sum(x['samples']*x['repeats_per_pass_per_group'] for x in row['chunk_motifs']),m)
            for chunk in row['chunk_motifs']:
                key=(chunk['samples'],gc); patterns[key]=chunk
            # Native address shape proof + endpoint checks, rather than a billion-word enumeration.
            for s in sorted(set([0,m//2,m-1])):
                for k in sorted(set([0,c//2,c-1])):
                    self.assertEqual(model.native_address(g,s,k),address(g['shape'],g['strides'],g['channel_axis'],s,k))
        # Independent interval-tag reduction in finite RAM, all actual chunk classes.
        for (n,c), chunk in sorted(patterns.items()):
            memory={s*c+k:(k,frozenset([s])) for s in range(n) for k in range(c)}
            bank_words=4096; cmds=list(model.reduction_commands(n,c,bank_words))
            for cmd in cmds:
                tokens=[memory[a] for a in cmd['src']]
                self.assertTrue(all(t[0]==cmd['channel'] for t in tokens))
                if len(tokens)==2:
                    self.assertFalse(tokens[0][1]&tokens[1][1]); pairs+=1
                    token=(tokens[0][0],tokens[0][1]|tokens[1][1])
                else: token=tokens[0]; carry+=1
                self.assertTrue(all(0<=a<8192 for a in cmd['src']+[cmd['dst']]))
                memory[cmd['dst']]=token
            base=chunk['result_base']
            for k in range(c): self.assertEqual(memory[base+k],(k,frozenset(range(n))))
            templates+=1
        self.assertEqual(templates,11)
        self.__class__.real_summary=dict(records=54,actual_chunk_classes=templates,
                                         expanded_class_pair_commands=pairs,expanded_class_carry_commands=carry,
                                         real_elements_executed=False)

    def test_parameter_sensitivity_count_invariance(self):
        base=model.all_real(model.config()); changed=model.config(); changed['latency']['external_read']=65
        slower=model.all_real(changed)
        self.assertEqual(base['aggregate_transactions_and_operations'],slower['aggregate_transactions_and_operations'])
        self.assertEqual(slower['conditional_independent_BN_serial_cycles']-base['conditional_independent_BN_serial_cycles'],
                         32*base['aggregate_transactions_and_operations']['external_read'])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ServiceChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(tests_run=result.testsRun,success=result.wasSuccessful(),
                 literal_summary=getattr(ServiceChecks,'literal_summary',None),
                 real_summary=getattr(ServiceChecks,'real_summary',None),
                 scope='Implementer CPU tests with independent math/replay; not an independent-agent review')
    out=Path(__file__).with_name('cpu_checks.json'); assert not out.exists()
    out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
