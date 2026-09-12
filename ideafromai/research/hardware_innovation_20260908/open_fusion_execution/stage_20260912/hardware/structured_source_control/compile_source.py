"""Whole-matrix official CSE, then the same two-stage finite-RF source ISA."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','DA_DEFAULT_THREADS'):
    os.environ[name] = '1'
import sys
import json
from importlib.metadata import version
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[3]
LIFT = BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40'
sys.path.insert(0, str(LIFT))
import ordinary_source_cmvm as cmvm
sys.path.insert(0, str(LIFT/'schedule_compare_same_port/two_stage_writeback'))
import run_compare


def main():
    p = dict(np.load(HERE/'deployed_constants.npz'))
    a, exponent = p['As_q16'].astype(np.int64), int(p['As_exponent'])
    assert version('da4ml') == '0.6.0'
    options = dict(method0='wmc', method1='auto', hard_dc=-1, decompose_dc=-2,
        qintervals=[(float(cmvm.LOW), float(cmvm.HIGH), 1.)]*10,
        adder_size=1, carry_size=1, search_all_decompose_dc=True)
    pipeline = cmvm.solve(np.ascontiguousarray(a.T, dtype=np.float32), **options)
    assert np.array_equal(pipeline.kernel, a.T)
    pipeline.save(HERE/'source34.official_pipeline.json')
    graph = cmvm.flatten(pipeline, a, -cmvm.LOW, np.array([cmvm.LOW]), np.array([cmvm.HIGH]))
    cmvm.precise_domain(graph)
    cmvm.save(HERE/'source34.integer_dag.json', graph)
    interpreter = cmvm.normalized_interpreter(graph)
    interpreter.save_binary(HERE/'source34.dais')
    post = cmvm.compile_postprocess(p, a, exponent)
    nodes = [dict(id=t,kind='input',source=t,parents=[]) for t in range(10)]
    aliases = {}
    ref = run_compare.base.ref
    shifted = run_compare.base.shifted
    def add(kind, parents, **kwargs):
        n = len(nodes)
        nodes.append(dict(id=n,kind=kind,parents=parents,**kwargs))
        return ref(n)
    for item in graph['nodes']:
        if item['kind'] == 'input': aliases[item['id']] = ref(item['source'])
        else:
            aliases[item['id']] = add('addsub', [
                shifted(aliases[item['lhs']],item['lhs_shift']),
                shifted(aliases[item['rhs']],item['rhs_shift'],-1 if item['subtract'] else 1)])
    for out, row in zip(graph['outputs'], post['rows']):
        parent = shifted(aliases[out['node']],out['shift'],out['sign'])
        add('gate',[parent],output_t=row['t'],threshold=row['direct_dot_cutoff'],
            direction=row['direction'],constant=row['constant_gate'])
    # Fixed before execution: current ordinary uses last_use_pressure.
    program, accounting = run_compare.compile_program(nodes, 'last_use_pressure')
    checks = run_compare.validate_program('ordinary', program, {k:v.tolist() for k,v in p.items()})
    rng = np.random.default_rng(912)
    corners = np.where(((np.arange(1024)[:,None] >> np.arange(10)) & 1) != 0,cmvm.HIGH,cmvm.LOW)
    x = np.concatenate([corners, rng.integers(cmvm.LOW,cmvm.HIGH+1,(4096,10),dtype=np.int64)]).T
    got = cmvm.eval_graph(graph,x,np.zeros((len(graph['nodes']),2),np.int64))
    expected = a@x
    assert np.array_equal(got,expected)
    assert np.array_equal(interpreter.predict(np.ascontiguousarray(x.T,dtype=np.float64),n_threads=1).T,expected)
    (HERE/'source34_program.json').write_text(json.dumps(program,indent=2)+'\n')
    (HERE/'source34_nodes.json').write_text(json.dumps(nodes,indent=2)+'\n')
    report=dict(official_version=version('da4ml'),official_options=options,
        arithmetic_nodes=len(graph['nodes'])-10,program=accounting,whole_matrix_symbolic_equal=True,
        independent_integer_and_official_DAIS_values=int(expected.size),integer_and_DAIS_differences=0,
        two_stage_register_lifetime_checks=checks,postprocess=post,
        compiler_policy='Same fixed last_use_pressure used by existing ordinary baseline; no extra policy sweep.',
        evidence='Complete official CSE and CPU arithmetic/ISA verification, not service/RTL/PPA.')
    cmvm.save(HERE/'compilation.json',report)
    print(json.dumps(dict(arithmetic_nodes=report['arithmetic_nodes'],program=accounting,checks=checks),indent=2),flush=True)


if __name__ == '__main__': main()
