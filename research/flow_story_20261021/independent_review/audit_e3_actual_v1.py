"""E3 execution-only audit entry; no numerical GT access or quality computation."""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import reduce_e3_rne_probe_v1 as reviewer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run","e0-run","e0-audit","contract","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a = p.parse_args()
    assert not a.output.exists()
    *_,audit = reviewer.bind(a)
    out = dict(schema="flow_e3_rne_actual_execution_audit_v1",**audit,
               quality_analysis_allowed=True,GT_values_read=False,
               auditor_sha256=reviewer.sha(__file__),
               binding_code_sha256=reviewer.sha(reviewer.__file__))
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:out[k] for k in ("status","model_forwards","new_quality_predictions",
                                       "new_action_labels","engineering_dense_references","optimizer_steps")},indent=2))


if __name__=="__main__":
    main()
