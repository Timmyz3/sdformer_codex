"""Compile each new trained source and execute the existing common source RTL."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','DA_DEFAULT_THREADS'):
    os.environ[name]='1'
import sys,json,argparse,subprocess
import numpy as np
from importlib.metadata import version

HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[1]
BASE=OPEN.parent
LIFT=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40'
sys.path.insert(0,str(LIFT))
import ordinary_source_cmvm as cmvm
sys.path.insert(0,str(LIFT/'schedule_compare_same_port/two_stage_writeback'))
import run_compare as sched
PREV=OPEN/'stage_20260912'
KINDS={'nop':0,'load':1,'addsub':2,'norm24':3,'gate':4,'commit':5}


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')


def compile_graph(parameters,structure,out):
    assert version('da4ml')=='0.6.0'
    nodes=[dict(id=t,kind='input',source=t,parents=[]) for t in range(10)]
    ref,shifted=sched.base.ref,sched.base.shifted
    graphs={}
    def add(kind,parents,**kw):
        i=len(nodes);nodes.append(dict(id=i,kind=kind,parents=parents,**kw));return ref(i)
    def cmatrix(matrix,inputs,label):
        matrix=np.asarray(matrix,dtype=np.int64)
        options=dict(method0='wmc',method1='auto',hard_dc=-1,decompose_dc=-2,
            qintervals=[(float(cmvm.LOW),float(cmvm.HIGH),1.)]*10,
            adder_size=1,carry_size=1,search_all_decompose_dc=True)
        pipeline=cmvm.solve(np.ascontiguousarray(matrix.T,dtype=np.float32),**options)
        assert np.array_equal(pipeline.kernel,matrix.T)
        graph=cmvm.flatten(pipeline,matrix,-cmvm.LOW,np.array([cmvm.LOW]),np.array([cmvm.HIGH]))
        cmvm.precise_domain(graph);graphs[label]=graph
        aliases={}
        for n in graph['nodes']:
            if n['kind']=='input':aliases[n['id']]=inputs[n['source']]
            else:
                aliases[n['id']]=add('addsub',[
                    shifted(aliases[n['lhs']],n['lhs_shift']),
                    shifted(aliases[n['rhs']],n['rhs_shift'],-1 if n['subtract'] else 1)])
        return [shifted(aliases[n['node']],n['shift'],n['sign']) for n in graph['outputs']]
    if structure!='lifting40':
        matrix=parameters['As_q16'];outputs=cmatrix(matrix,[ref(t) for t in range(10)],'As')
        post=cmvm.compile_postprocess(parameters,matrix,int(parameters['As_exponent']))
        for value,row in zip(outputs,post['rows']):
            add('gate',[value],output_t=row['t'],threshold=row['direct_dot_cutoff'],
                direction=row['direction'],constant=row['constant_gate'])
    else:
        current=[ref(t) for t in range(10)];last=set()
        for layer in range(4):
            pairs=parameters['lifting_matchings'][layer]
            for half in (0,1):
                matrix=np.zeros((5,10),np.int64)
                for i,pair in enumerate(pairs):
                    matrix[i,pair[half]]=4096
                    matrix[i,pair[1-half]]=parameters['lifting_q12'][layer,i,half]
                outputs=cmatrix(matrix,current,'half'+str(2*layer+half))
                for value,pair in zip(outputs,pairs):
                    coordinate=int(pair[half])
                    if (layer,half)==(3,1):
                        current[coordinate]=value;last.add(coordinate)
                    else:current[coordinate]=add('norm24',[value],rne_shift=12)
        for t,coordinate in enumerate(parameters['source_permutation']):
            k,d,c=(int(parameters['source_'+name][t]) for name in ('threshold','direction','constant'))
            if coordinate in last:k,c=sched.base.folded_cutoff(k,d,c,12)
            add('gate',[current[coordinate]],output_t=t,threshold=k,direction=d,constant=c)
    program,accounting=sched.compile_program(nodes,'last_use_pressure')
    axis='lifting_raw' if structure=='lifting40' else 'ordinary'
    checks=sched.validate_program(axis,program,{k:v.tolist() for k,v in parameters.items()})
    dump(out/'graphs.json',graphs);dump(out/'program.json',program)
    dump(out/'compilation.json',dict(da4ml=version('da4ml'),accounting=accounting,
        checks=checks,source_structure=structure,constant_function='New trained endpoint; old source service not inherited.',
        full_halfstep_RNEs=sum(n['kind']=='norm24' for n in nodes)))
    return program,accounting


def literal_gate(x,p,structure):
    shape=x.shape;value=x.reshape(10,-1).astype(np.int64)
    if structure=='lifting40':
        value=value.copy()
        for layer in range(4):
            for half in (0,1):
                for j,pair in enumerate(p['lifting_matchings'][layer]):
                    a,b=int(pair[half]),int(pair[1-half]);q=int(p['lifting_q12'][layer,j,half])
                    value[a]=cmvm.rne_integer(4096*value[a]+q*value[b],12).clip(cmvm.LOW,cmvm.HIGH)
        value=value[p['source_permutation']]
    else:value=cmvm.rne_integer(p['As_q16'].astype(np.int64)@value,int(p['As_exponent'])).clip(cmvm.LOW,cmvm.HIGH)
    return cmvm.state_gate(value,p).reshape(shape)


def fields(program,path):
    rows=[]
    for i in program:
        ops=i.get('operands',[])+[dict(reg=0,shift=0,sign=1)]*2
        row=[KINDS[i['kind']],i.get('dst',95 if i['kind']=='gate' else 0),
            ops[0]['reg'],ops[1]['reg'],ops[0]['shift'],ops[1]['shift'],
            int(ops[0]['sign']<0),int(ops[1]['sign']<0),i.get('rne_shift',0),
            i.get('output_t',i.get('source_t',0)),i.get('threshold',0),
            int(i.get('direction',1)<0),i.get('constant',-1)+1]
        rows.append(' '.join(map(str,row)))
    path.write_text('\n'.join(rows)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('structure',choices=['dense','contiguous34','lifting40'])
    args=parser.parse_args();out=HERE/args.structure;out.mkdir(exist_ok=True)
    endpoint=HERE.parent/'algorithm/matched_training'/args.structure/'stage320'
    p=dict(np.load(endpoint/'deployed_constants.npz'))
    program,accounting=compile_graph(p,args.structure,out)
    fields(program,out/'program.txt')
    inputs=PREV/'hardware/source_rtl_inputs'
    cases=[r for r in json.loads((inputs/'manifest.json').read_text())['cases'] if r['axis']=='ordinary']
    binary=PREV/'hardware/rtl_source/obj_dir/Vtemporal_source'
    report=dict(structure=args.structure,endpoint=str(endpoint.relative_to(OPEN)),
        quality=json.loads((endpoint/'quality.json').read_text()),accounting=accounting,rows=[],
        evidence='Existing common source RTL, Verilator source-only arithmetic/handshake/cycles. No VCS/DC/PT/FM or PPA.',
        same_inputs='Unchanged upstream ordinary parent, actual two I24 halos. Gold recomputed from each new endpoint.',
        inherited_downstream_or_frame_service=False)
    for case in cases:
        label=case['window'];input_file=inputs/case['input_file']
        tiles=np.fromfile(input_file,dtype='<i4').reshape(-1,10,8)
        gate=literal_gate(tiles.transpose(1,0,2),p,args.structure)
        gold=sum(gate[t].astype(np.uint16)<<t for t in range(10))
        gold_file=out/(label+'_gates.bin');gold.astype('<u2').tofile(gold_file)
        for mode in ['ready','stress']:
            row=json.loads(subprocess.run([str(binary),str(out/'program.txt'),str(input_file),str(gold_file),mode],
                check=True,capture_output=True,text=True).stdout)
            row.update(window=label,source_input_values=int(tiles.size),new_gate_ones=int(gate.sum()),
                common_input=str(input_file.relative_to(OPEN)))
            report['rows'].append(row)
    dump(out/'results.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
