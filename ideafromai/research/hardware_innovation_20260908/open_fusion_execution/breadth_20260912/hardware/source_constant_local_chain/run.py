"""Reuse the same local Machine with only the two declared source replacements."""
from pathlib import Path
import argparse,importlib.util,inspect,json

HERE=Path(__file__).resolve().parent
BREADTH=HERE.parents[1]
spec=importlib.util.spec_from_file_location('matched_chain_pot2_common',HERE.parent/'matched_local_chain/run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('structure',choices=['dense','lifting40']);ap.add_argument('window',choices=['corner','interior']);a=ap.parse_args()
    original=inspect.getsource(base.run)
    q="BREADTH/'algorithm/matched_training'/structure/'stage320/deployed_constants.npz'"
    program="BREADTH/'source_execution'/structure/'program.json'"
    assert q in original and program in original
    updated=original.replace(q,"CANDIDATE_ROOT/structure/'deployed_constants.npz'").replace(program,"CANDIDATE_ROOT/structure/'program.json'")
    space=dict(base.__dict__);space.update(HERE=HERE,CANDIDATE_ROOT=BREADTH/'source_constant_probe')
    # The preview reference remains the exact imported function and original
    # source file, so changing output directory cannot change its arithmetic.
    exec(compile(updated,str(HERE/'same_machine_binding'), 'exec'),space)
    print('POT2_LOCAL_START',a.structure,a.window,flush=True)
    r=space['run'](a.structure,a.window)
    parent=json.loads((HERE.parent/'matched_local_chain'/(a.structure+'_'+a.window+'.json')).read_text())
    r.update(source_constant_projection='Fixed two signed powers, same published rule for dense and lifting40.',
        actual_parameter_directory=str(BREADTH/'source_constant_probe'/a.structure),
        original_stage320_parameters_inherited_except_source=True,new_stage320_constants=False,
        source_parameters_changed_since_stage320=True,new_GPU_endpoint_capture=False,
        parent_service_slots=parent['service_slots'],reduction_vs_own_unquantized_source_parent=1-r['service_slots']/parent['service_slots'],
        new_function_requires_fresh_AEE=True)
    (HERE/(a.structure+'_'+a.window+'.json')).write_text(json.dumps(r,indent=2)+'\n')
    print('POT2_LOCAL_COMPLETE',a.structure,a.window,r['service_slots'],r['reduction_vs_own_unquantized_source_parent'],r['gate_activity'],flush=True)


if __name__=='__main__':main()
