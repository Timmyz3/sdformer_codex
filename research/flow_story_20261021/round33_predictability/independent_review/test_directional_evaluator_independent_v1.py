"""Independent dual-source evaluator delta checks; synthetic arrays only."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
BRANCH = ROUND/"directional_branch"
sys.path.insert(0,str(BRANCH))
spec=importlib.util.spec_from_file_location("directional_eval_independent",BRANCH/"evaluate.py")
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    output=HERE/"DIRECTIONAL_EVALUATOR_INDEPENDENT_CPU_V1.json"
    assert not output.exists()
    trees=[]
    for p in [ROUND/"evaluate.py",BRANCH/"evaluate.py"]:
        trees.append({n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)})
    same=["policy_weights","build_policies","mixed_metrics","operator_cost","summarize_policies","comparison","uniform_tail_risk","evaluate_data"]
    assert all(trees[0][name]==trees[1][name] for name in same)
    rng=np.random.default_rng(216033)
    original=dict(x=rng.normal(size=(3,128)),ids=np.asarray(["a","b","c"]),
                  roles=np.asarray(["confirm"]*3),delta_aee=rng.normal(size=(3,6)),valid_pixels=np.asarray([1,2,3]))
    augmented={k:v.copy() for k,v in original.items()}
    augmented["x"]=np.column_stack([original["x"],rng.normal(size=(3,32))])
    e.validate_augmented_arrays(augmented,original)
    failures=[]
    for case in ["label_change","prefix_change","metadata_order","dtype_change","extra_key","wrong_width"]:
        changed={k:v.copy() for k,v in augmented.items()}
        if case=="label_change":changed["delta_aee"][0,0]+=1
        if case=="prefix_change":changed["x"][0,17]+=1
        if case=="metadata_order":changed["ids"]=changed["ids"][::-1].copy()
        if case=="dtype_change":changed["valid_pixels"]=changed["valid_pixels"].astype(np.float64)
        if case=="extra_key":changed["extra"]=np.zeros(1)
        if case=="wrong_width":changed["x"]=changed["x"][:,:159]
        try:e.validate_augmented_arrays(changed,original)
        except ValueError:failures.append(case)
        else:raise AssertionError(case)
    out=dict(schema="flow_r33_directional_evaluator_independent_cpu_v1",status="PASS",
        unchanged_aggregation_functions_ast_identical=same,
        dual_source_array_mutations_rejected=failures,
        source_sha256={"directional_evaluate":sha(BRANCH/"evaluate.py"),"base_evaluate":sha(ROUND/"evaluate.py"),"independent_test":sha(__file__)},
        real_quality_read=False,new_GT_read=False,GPU_used=False)
    output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"status":"PASS","unchanged_functions":len(same),"mutations_rejected":len(failures)}))


if __name__=="__main__":main()
