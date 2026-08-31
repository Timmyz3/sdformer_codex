#!/usr/bin/env python3
"""Static checker for M981 source; executes no real decoder prefix or EDA."""
import argparse,ast,hashlib,importlib.util,json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;HW=HERE.parent.parent
DRIVER=HERE/"execute_m981_m977_decoder_d2d3_10k_atomic_evidence_r1.py"
RUNNER=HERE/"run_m985_m981_decoder_d2d3_10k_atomic_evidence_one_shot.sh"
TEST=HW/"system_simulator/tests/test_m981_m977_decoder_atomic_evidence_source.py"
PYTHON=Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA="9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"

def sha(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as h:
        for b in iter(lambda:h.read(1<<20),b""):d.update(b)
    return d.hexdigest()

def load():
    s=importlib.util.spec_from_file_location("m981_static",DRIVER)
    m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
    return m

def check(contract):
    if Path(sys.executable).resolve()!=PYTHON or sha(PYTHON)!=PYTHON_SHA:
        raise RuntimeError("M981 checker interpreter drift")
    driver=DRIVER.read_text();runner=RUNNER.read_text();tree=ast.parse(driver)
    imports={a.name.split('.')[0] for n in ast.walk(tree)
             if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names}
    forbidden=sorted(imports&{"subprocess","socket","requests","urllib",
                              "torch","tensorflow","cupy"})
    if forbidden:raise RuntimeError("forbidden import")
    for stale in ("m973_","m974_","m975_","M973_","M974_","M975_"):
        if stale in driver or stale in runner:
            raise RuntimeError("stale M972 chain token: "+stale)
    required=("m981_","m982_","m983_","m984_","m985_",
              "M985_WORK_RETAINED_NOT_MOVED","--quarantine-work",
              "--run-row D2","--run-row D3")
    missing=[x for x in required if x not in (driver+runner)]
    if missing:raise RuntimeError("missing chain/durability token: "+str(missing))
    if '[[ ! -f "${m985_work}/SHA256SUMS"' in runner:
        raise RuntimeError("cleanup can skip on lone manifest")
    module=load();validation=module.validate_source_contract(contract,RUNNER)
    directed=module.source_self_test()
    return {"schema":"m981_atomic_evidence_source_static_check_v1",
      "status":"PASS_M981_STATIC_SOURCE__NO_REAL_10K",
      "driver_sha256":sha(DRIVER),"runner_sha256":sha(RUNNER),
      "test_sha256":sha(TEST),"contract_sha256":sha(contract),
      "source_status":validation["status"],"forbidden_imports":forbidden,
      "atomic_outer_bundle":directed["atomic_outer_bundle"],
      "partial_manifest_recovered":directed["partial_manifest_recovered"],
      "lone_manifest_does_not_skip_cleanup":
        directed["lone_legacy_manifest_does_not_skip_cleanup"],
      "failed_cleanup_retains_unmoved_work":
        directed["failed_cleanup_retains_unmoved_work"],
      "real_10k_executed":False,"eda_gpu_remote_used":False}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--contract",type=Path,required=True)
    print(json.dumps(check(p.parse_args().contract),sort_keys=True))
