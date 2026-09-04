#!/usr/bin/env bash
set -euo pipefail

# M602 source-only exact runner for the bounded M597/M593 generated-macro
# component-energy analyzer.  --preflight-only is the only currently admitted
# mode.  --execute requires a future, separately sealed M604 true-launch
# admission whose M603 runner hammer has P0=P1=0.  This file itself grants no
# launch authority.

PYTHON_BIN="/usr/libexec/platform-python3.6"
PYTHON_SHA256="9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f"
ANALYZER_REL="system_simulator/scripts/analyze_m597_m593_m528_parent_scratch_generated_macro_energy_r2.py"
ANALYZER_SHA256="6896c8a406dc3274926e6c7d958136aca47b9df9afa3522d6c2539a142ea9cf9"
SOURCE_CONTRACT_REL="contracts/m597_m593_m528_parent_scratch_generated_macro_energy_source_contract_r2_20260828.json"
SOURCE_CONTRACT_SHA256="90399b6c932e28f6eac38f3408af0374b23beb369e1fd4e57e3b98d92d28b1bf"
RESULT_REL="results/m597_m593_m528_parent_scratch_generated_macro_energy_r2_20260828"
ATTEMPT_REL="results/m597_m593_m528_parent_scratch_generated_macro_energy_r2_20260828.attempt"
CONSUMED_REL="results/m597_m593_m528_parent_scratch_generated_macro_energy_r2_20260828.attempt.consumed"
FUTURE_AUTH_REL="contracts/m604_m602_m593_parent_scratch_energy_true_launch_admission_r1_20260828.json"
FUTURE_HAMMER_REL="reviews/m603_m602_m593_parent_scratch_energy_exact_runner_static_hammer_r1_20260828/review.json"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
HW_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd -P)"
REPO_ROOT="$(CDPATH= cd -- "$HW_ROOT/.." && pwd -P)"
RUNNER="$(realpath -e -- "$0")"
ANALYZER="$HW_ROOT/$ANALYZER_REL"
SOURCE_CONTRACT="$HW_ROOT/$SOURCE_CONTRACT_REL"
RESULT_DIR="$HW_ROOT/$RESULT_REL"
ATTEMPT_DIR="$HW_ROOT/$ATTEMPT_REL"
CONSUMED_DIR="$HW_ROOT/$CONSUMED_REL"
RESULTS_PARENT="$HW_ROOT/results"
FUTURE_AUTH="$HW_ROOT/$FUTURE_AUTH_REL"

sha_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

lexists() {
  "$PYTHON_BIN" - "$1" <<'PY'
import os
import sys
raise SystemExit(0 if os.path.lexists(sys.argv[1]) else 1)
PY
}

rename_noreplace() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import ctypes
import os
import sys

source = os.fsencode(sys.argv[1])
target = os.fsencode(sys.argv[2])
libc = ctypes.CDLL(None, use_errno=True)
try:
    renameat2 = libc.renameat2
except AttributeError:
    raise RuntimeError("renameat2 is unavailable; fail closed")
renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                      ctypes.c_uint]
renameat2.restype = ctypes.c_int
if renameat2(-100, source, -100, target, 1) != 0:  # AT_FDCWD, RENAME_NOREPLACE
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error), sys.argv[2])
PY
}

verify_static_identity() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$RUNNER" "$RUNNER_SHA_START" <<'PY'
import hashlib
import json
import math
import os
import pathlib
import stat
import sys

repo = pathlib.Path(sys.argv[1])
runner = pathlib.Path(sys.argv[2])
runner_sha_start = sys.argv[3]

def require(value, message):
    if not value:
        raise RuntimeError(message)

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def strict(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=pairs,
            parse_constant=lambda raw: (_ for _ in ()).throw(RuntimeError(raw)),
        )
    require(isinstance(value, dict), "top-level JSON must be object")
    def finite(node):
        if isinstance(node, float):
            require(math.isfinite(node), "non-finite JSON number")
        elif isinstance(node, dict):
            for child in node.values(): finite(child)
        elif isinstance(node, list):
            for child in node: finite(child)
    finite(value)
    return value

def exact(rel, kind="file"):
    require(not os.path.isabs(rel), "absolute frozen relative path")
    path = repo / rel
    normalized = os.path.normpath(rel)
    require(normalized == rel and not rel.startswith("../"), "non-canonical/traversing path: " + rel)
    current = repo
    for part in pathlib.Path(rel).parts:
        current = current / part
        require(os.path.lexists(str(current)), "missing frozen path: " + str(current))
        require(not stat.S_ISLNK(os.lstat(str(current)).st_mode), "symlink frozen path: " + str(current))
    mode = os.lstat(str(path)).st_mode
    if kind == "file": require(stat.S_ISREG(mode), "not regular file: " + str(path))
    else: require(stat.S_ISDIR(mode), "not directory: " + str(path))
    return path

def verify_sealed_dir(rel, expected_manifest, expected_outer):
    directory = exact(rel, "dir")
    manifest = exact(rel + "/SHA256SUMS")
    outer = exact(rel + "/SHA256SUMS.seal.sha256")
    require(sha(manifest) == expected_manifest, "manifest SHA drift: " + rel)
    require(sha(outer) == expected_outer, "outer file SHA drift: " + rel)
    tokens = outer.read_text(encoding="utf-8").strip().split()
    require(tokens == [expected_manifest, "SHA256SUMS"], "outer content drift: " + rel)
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        tokens = line.split(None, 1)
        require(len(tokens) == 2, "malformed manifest: " + rel)
        digest, name = tokens
        name = name.lstrip("*")
        require(name not in seen, "duplicate manifest member: " + name)
        seen.add(name)
        member_rel = os.path.normpath(rel + "/" + name)
        require(member_rel == rel + "/" + name.lstrip("./"), "manifest path alias/traversal: " + name)
        member = exact(member_rel)
        require(sha(member) == digest, "manifest member drift: " + member_rel)
    return seen

expected_inputs = {
    "m504_result": {
        "path": "hw_autoresearch_nts07/results/m504_h67_single_port_parent_scratch_r3_20260827/m504_h67_single_port_parent_scratch_result_r3.json",
        "sha256": "a0d2234a3a660df42bb87be04d42085c6c19025e55bdc35a1d61b9c48a54634b",
        "directory": "hw_autoresearch_nts07/results/m504_h67_single_port_parent_scratch_r3_20260827",
        "manifest_sha256": "f682a43c35847fa1fd2d9234bff9f225943ed582db7c65bb3590fb634b51212c",
        "outer_seal_file_sha256": "87f3af91debc5dff7fa8510bd8bf91abc57884b996452d8157d7ab51c369568c"},
    "m504_result_hammer": {
        "path": "hw_autoresearch_nts07/reviews/m504_r3_result_hammer_r1_20260827/m504_r3_result_hammer_r1_20260827.json",
        "sha256": "ac3a961a41a4c1b6511275c9c98fcdf5669f9c0ed98399f2afcd2ded075389a1",
        "directory": "hw_autoresearch_nts07/reviews/m504_r3_result_hammer_r1_20260827",
        "manifest_sha256": "766305f189ffe95e03ac54d1bc1a79e8f199aa5532901034d4e38d0877908545",
        "outer_seal_file_sha256": "4b13077464eb96e21091663a0bd4598af7340c29d1920c5e4a2561075fb70f4d"},
    "m528_result": {
        "path": "hw_autoresearch_nts07/results/m528_h67_single_port_same_ledger_recompute_r4_20260827/m528_h67_single_port_same_ledger_recompute_result_r1.json",
        "sha256": "778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1",
        "directory": "hw_autoresearch_nts07/results/m528_h67_single_port_same_ledger_recompute_r4_20260827",
        "manifest_sha256": "4556a3383507e81ad9883f59bb55bb3d4fd08e7ec03977b215108b5ce4565073",
        "outer_seal_file_sha256": "02abbf7f9209d9a41d803c9942bfb43550be0d40945e3c094c1e457bda0db053"},
    "m528_result_hammer": {
        "path": "hw_autoresearch_nts07/reviews/m528_r4_result_hammer_r1_20260827/review.json",
        "sha256": "4f70610dcb5c0778fd7874b8f70239f9139c5f98732ae439ab246129ede53d6e",
        "directory": "hw_autoresearch_nts07/reviews/m528_r4_result_hammer_r1_20260827",
        "manifest_sha256": "678a0541702b9804691a5700a55fb4dc8c07f524ee5b6176800196371ebe3b56",
        "outer_seal_file_sha256": "ec442c74ca4dee305178e863a97e976940e0f5d6b98a0ad57e52cd298c01653e"},
    "generated_macro_mapping": {
        "path": "hw_autoresearch_nts07/reviews/tsmc28_sram_macro_audit_r1_20260827/tsmc28_sram_mapping_r1.json",
        "sha256": "68017fb51773713dd7dbee9463ec60d1dcdac9dea6e56588463e7f4ded96be4d",
        "directory": "hw_autoresearch_nts07/reviews/tsmc28_sram_macro_audit_r1_20260827",
        "manifest_sha256": "34be39b31afc57b0f22775590a7977c3b42f5277c52e8062c8b1b3bc0d648321",
        "outer_seal_file_sha256": "7832fea23f44038be1528c1480bfeed705c7c9705d1e727d367d678ae9720df4"},
    "m595_failed_review": {
        "path": "hw_autoresearch_nts07/reviews/m595_m593_parent_scratch_energy_source_static_hammer_r1_20260828/review.json",
        "sha256": "b8db95dbe045025fb815c2a6513cf258b519faa334446f5c3b4ccb8d2e23f875",
        "directory": "hw_autoresearch_nts07/reviews/m595_m593_parent_scratch_energy_source_static_hammer_r1_20260828",
        "manifest_sha256": "200c8d1ac338ff2746e540e7243514dbb6b704ed18a2c5c2620bf9c363c674da",
        "outer_seal_file_sha256": "921fef583a8dd4a3e3b19e4ca97059fa53504e049d7edd59c9d0ac0703ac071f"},
    "docs359": {
        "path": "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md",
        "sha256": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"},
}

fixed_files = {
    "hw_autoresearch_nts07/system_simulator/scripts/analyze_m597_m593_m528_parent_scratch_generated_macro_energy_r2.py": "6896c8a406dc3274926e6c7d958136aca47b9df9afa3522d6c2539a142ea9cf9",
    "hw_autoresearch_nts07/contracts/m597_m593_m528_parent_scratch_generated_macro_energy_source_contract_r2_20260828.json": "90399b6c932e28f6eac38f3408af0374b23beb369e1fd4e57e3b98d92d28b1bf",
    "hw_autoresearch_nts07/reviews/m597_m593_parent_scratch_energy_r2_source_author_handoff_20260828/handoff.json": "48c3e6605c2bc868ba0848984a3975891b9ab91355146a8496d94d84347e4479",
    "hw_autoresearch_nts07/reviews/m599_m597_m593_parent_scratch_energy_r2_source_static_hammer_r1_20260828/review.json": "56ac7aafd7b603d437efe267ee2875909a365072181d0abd9101fd5d601497b1",
}
for rel, expected in fixed_files.items():
    require(sha(exact(rel)) == expected, "fixed file SHA drift: " + rel)
require(sha(runner) == runner_sha_start, "runner SHA changed")

verify_sealed_dir(
    "hw_autoresearch_nts07/reviews/m597_m593_parent_scratch_energy_r2_source_author_handoff_20260828",
    "1a0fcee6d9858bd977f8a25641f0193f88c28d93ecf355c4748cea054a60f045",
    "82692ac417e05d9bab9e0763ba3cddef111e2f197c1369d7d57d4d28d17b939a")
verify_sealed_dir(
    "hw_autoresearch_nts07/reviews/m599_m597_m593_parent_scratch_energy_r2_source_static_hammer_r1_20260828",
    "ee81f87282ab10d83721350d1f214fb88466142bada524bca26bf9f8af1aa9cf",
    "55955df0787706695f11550a71873d7a2b4f7454c8ce50c19d58a4c91771cd64")

for name, entry in expected_inputs.items():
    require(sha(exact(entry["path"])) == entry["sha256"], name + " SHA drift")
    if "directory" in entry:
        members = verify_sealed_dir(entry["directory"], entry["manifest_sha256"], entry["outer_seal_file_sha256"])
        member = os.path.relpath(entry["path"], entry["directory"]).replace(os.sep, "/")
        require(member in members or "./" + member in members, name + " absent from manifest")

contract = strict(exact("hw_autoresearch_nts07/contracts/m597_m593_m528_parent_scratch_generated_macro_energy_source_contract_r2_20260828.json"))
require(contract.get("contract_id") == "m597_m593_m528_parent_scratch_generated_macro_energy_source_contract_r2_20260828", "contract id drift")
require(contract.get("frozen_inputs") == expected_inputs, "contract frozen-input map drift")
handoff = strict(exact("hw_autoresearch_nts07/reviews/m597_m593_parent_scratch_energy_r2_source_author_handoff_20260828/handoff.json"))
require(handoff.get("identity", {}).get("analyzer", {}).get("sha256") == fixed_files["hw_autoresearch_nts07/system_simulator/scripts/analyze_m597_m593_m528_parent_scratch_generated_macro_energy_r2.py"], "handoff analyzer binding drift")
review = strict(exact("hw_autoresearch_nts07/reviews/m599_m597_m593_parent_scratch_energy_r2_source_static_hammer_r1_20260828/review.json"))
require(review.get("status") == "PASS_SOURCE_STATIC_WITH_ONE_P2_RUNNER_HARDENING__AUTHOR_EXACT_RUNNER_CHAIN_ONLY", "M599 status drift")
require((review.get("p0_count"), review.get("p1_count")) == (0, 0), "M599 P0/P1 drift")
require(review.get("authorization", {}).get("exact_runner_authoring_allowed") is True, "M599 runner authorization missing")
require(review.get("authorization", {}).get("formal_analyzer_execution_allowed") is False, "M599 execution boundary drift")
print("PASS_M602_STATIC_IDENTITY")
PY
}

assert_coordinate_policy() {
  "$PYTHON_BIN" - "$HW_ROOT" "$RESULT_DIR" "$ATTEMPT_DIR" "$CONSUMED_DIR" "$STAGING_DIR" <<'PY'
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
expected_parent = root / "results"
paths = [pathlib.Path(value) for value in sys.argv[2:]]

def require(value, message):
    if not value: raise RuntimeError(message)

current = root
for part in pathlib.Path("results").parts:
    current = current / part
    require(os.path.lexists(str(current)), "missing result parent")
    require(stat.S_ISDIR(os.lstat(str(current)).st_mode), "result parent not plain directory")
    require(not stat.S_ISLNK(os.lstat(str(current)).st_mode), "result parent symlink")
for path in paths:
    require(path.parent == expected_parent, "coordinate not same-parent: " + str(path))
    require(not os.path.lexists(str(path)), "coordinate already exists including dangling symlink: " + str(path))
print("PASS_M602_COORDINATE_POLICY")
PY
}

verify_future_authorization() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$FUTURE_AUTH" "$RUNNER" "$RUNNER_SHA_START" "$RESULT_REL" "$ATTEMPT_REL" "$FUTURE_HAMMER_REL" <<'PY'
import hashlib
import json
import math
import os
import pathlib
import stat
import sys

repo = pathlib.Path(sys.argv[1]); auth = pathlib.Path(sys.argv[2]); runner = pathlib.Path(sys.argv[3])
runner_sha = sys.argv[4]; result_rel = sys.argv[5]; attempt_rel = sys.argv[6]; hammer_rel = sys.argv[7]

def require(v, m):
    if not v: raise RuntimeError(m)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def strict(path):
    def pairs(items):
        d={}
        for k,v in items:
            require(k not in d,"duplicate JSON key: "+k); d[k]=v
        return d
    with path.open("r",encoding="utf-8") as f:
        x=json.load(f,object_pairs_hook=pairs,parse_constant=lambda raw: (_ for _ in ()).throw(RuntimeError(raw)))
    require(isinstance(x,dict),"top level is not object")
    return x
def plain(path, directory=False):
    require(os.path.lexists(str(path)),"missing path: "+str(path))
    mode=os.lstat(str(path)).st_mode
    require(not stat.S_ISLNK(mode),"symlink path: "+str(path))
    require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode),"wrong path type: "+str(path))

expected_auth = repo / "hw_autoresearch_nts07/contracts/m604_m602_m593_parent_scratch_energy_true_launch_admission_r1_20260828.json"
require(auth == expected_auth, "future authorization path drift")
plain(auth); plain(pathlib.Path(str(auth)+".sha256")); plain(pathlib.Path(str(auth)+".sha256.seal.sha256"))
auth_sha=sha(auth); side=pathlib.Path(str(auth)+".sha256"); outer=pathlib.Path(str(auth)+".sha256.seal.sha256")
require(side.read_text(encoding="utf-8").strip().split() == [auth_sha,auth.name],"authorization sidecar drift")
side_sha=sha(side)
require(outer.read_text(encoding="utf-8").strip().split() == [side_sha,side.name],"authorization outer seal drift")
payload=strict(auth)
require(set(payload) == {"admission_id","date","status","launch_now","release","runner","source_static_hammer","upstream","canonical","claim_boundary"},"authorization key-set drift")
require(payload["admission_id"] == "m604_m602_m593_parent_scratch_energy_true_launch_admission_r1_20260828","authorization id drift")
require(payload["status"] == "TRUE_LAUNCH_ADMISSION__FRESH_M603_P0_P1_ZERO_REQUIRED","authorization status drift")
require(payload["launch_now"] is True and payload["release"] is True,"authorization not true/released")
require(payload["runner"] == {"path":"hw_autoresearch_nts07/system_simulator/scripts/run_m602_m593_parent_scratch_generated_macro_energy_r2_exact_sha.sh","sha256":runner_sha},"authorization runner drift")
require(sha(runner)==runner_sha,"runner changed")
require(payload["canonical"] == {"result_dir":result_rel,"attempt_dir":attempt_rel},"authorization coordinates drift")
up=payload["upstream"]
require(up.get("m597_contract_sha256") == "90399b6c932e28f6eac38f3408af0374b23beb369e1fd4e57e3b98d92d28b1bf","M597 contract admission drift")
require(up.get("m597_analyzer_sha256") == "6896c8a406dc3274926e6c7d958136aca47b9df9afa3522d6c2539a142ea9cf9","M597 analyzer admission drift")
require(up.get("m599_review_sha256") == "56ac7aafd7b603d437efe267ee2875909a365072181d0abd9101fd5d601497b1","M599 admission drift")
hammer=payload["source_static_hammer"]
require(hammer.get("path") == "hw_autoresearch_nts07/"+hammer_rel,"future hammer path drift")
hammer_path=repo/hammer["path"]
plain(hammer_path)
require(sha(hammer_path)==hammer.get("sha256"),"future hammer SHA drift")
hammer_dir=hammer_path.parent; plain(hammer_dir,True)
manifest=hammer_dir/"SHA256SUMS"; outer_h=hammer_dir/"SHA256SUMS.seal.sha256"; plain(manifest); plain(outer_h)
manifest_sha=sha(manifest)
require(outer_h.read_text(encoding="utf-8").strip().split()==[manifest_sha,"SHA256SUMS"],"future hammer outer drift")
member=None
for line in manifest.read_text(encoding="utf-8").splitlines():
    tokens=line.split(None,1)
    if len(tokens)==2 and tokens[1].lstrip("*").lstrip("./")==hammer_path.name: member=tokens[0]
require(member==hammer.get("sha256"),"future hammer manifest membership drift")
review=strict(hammer_path)
require(review.get("schema") == "m603_m602_m593_parent_scratch_energy_exact_runner_static_hammer_v1","future hammer schema drift")
require((review.get("p0_count"),review.get("p1_count"))==(0,0),"future hammer P0/P1 nonzero")
require(review.get("authorization",{}).get("true_launch_admission_authoring_allowed") is True,"future hammer did not authorize true launch")
boundary=payload["claim_boundary"]
require(boundary.get("component_only") is True and boundary.get("paper_data") is False and boundary.get("system_energy") is False,"authorization claim boundary drift")
print(auth_sha)
PY
}

verify_analyzer_result_tree() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib, json, math, os, pathlib, stat, sys
directory=pathlib.Path(sys.argv[1])
def req(v,m):
    if not v: raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def pairs(items):
    d={}
    for k,v in items: req(k not in d,"duplicate JSON key"); d[k]=v
    return d
req(os.path.lexists(str(directory)),"missing analyzer output")
req(stat.S_ISDIR(os.lstat(str(directory)).st_mode) and not stat.S_ISLNK(os.lstat(str(directory)).st_mode),"output not plain dir")
expected={"m597_m593_m528_parent_scratch_generated_macro_energy_result_r2.json","m597_parent_scratch_energy_rows_r2.csv","RUN_COMPLETE.txt","SHA256SUMS","SHA256SUMS.seal.sha256"}
actual=set(p.name for p in directory.iterdir())
req(actual==expected,"analyzer output set drift: "+repr(actual))
for p in directory.iterdir(): req(stat.S_ISREG(os.lstat(str(p)).st_mode) and not stat.S_ISLNK(os.lstat(str(p)).st_mode),"nonregular member")
manifest=directory/"SHA256SUMS"; outer=directory/"SHA256SUMS.seal.sha256"; msha=sha(manifest)
req(outer.read_text().strip().split()==[msha,"SHA256SUMS"],"outer drift")
members={}
for line in manifest.read_text().splitlines():
    dig,name=line.split(None,1); name=name.lstrip("*"); req(name not in members,"duplicate manifest member"); members[name]=dig
req(set(members)==expected-{"SHA256SUMS","SHA256SUMS.seal.sha256"},"manifest set drift")
for n,d in members.items(): req(sha(directory/n)==d,"member SHA drift")
with (directory/"m597_m593_m528_parent_scratch_generated_macro_energy_result_r2.json").open("r",encoding="utf-8") as f:
    result=json.load(f,object_pairs_hook=pairs,parse_constant=lambda raw: (_ for _ in ()).throw(RuntimeError(raw)))
req(result.get("schema")=="m597_m593_m528_parent_scratch_generated_macro_energy_result_v2","result schema drift")
req(result.get("status")=="PASS_BOUNDED_GENERATED_MACRO_COMPONENT_MODEL__PENDING_FRESH_INDEPENDENT_RESULT_HAMMER","result status drift")
req(result.get("identity",{}).get("source_contract",{}).get("sha256")=="90399b6c932e28f6eac38f3408af0374b23beb369e1fd4e57e3b98d92d28b1bf","result contract drift")
req([r.get("design") for r in result.get("rows",[])]==["m504_all_write_1rw_parent_scratch","m528_dead_write_only_1rw_parent_scratch"],"result rows drift")
req(result.get("conservation",{}).get("all_equalities_pass") is True,"result conservation drift")
ab=result.get("ablation",{})
req(math.isclose(ab.get("dead_write_only_parent_scratch_component_energy_reduction_percent"),38.228307918921945,rel_tol=0,abs_tol=1e-12),"reduction drift")
req(math.isclose(ab.get("dead_write_only_parent_scratch_component_energy_saved_mj_per_frozen_sampled_inference"),1.2622562286593053,rel_tol=0,abs_tol=1e-12),"saved energy drift")
cb=result.get("claim_boundary",{})
req(cb.get("result_hammer_pending") is True and cb.get("date_headline") is False and cb.get("system_energy") is False,"result boundary drift")
print("PASS_M602_ANALYZER_RESULT_TREE")
PY
}

verify_final_result_tree() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib, json, os, pathlib, stat, sys
d=pathlib.Path(sys.argv[1])
def req(v,m):
    if not v: raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
req(os.path.lexists(str(d)),"missing final tree")
req(stat.S_ISDIR(os.lstat(str(d)).st_mode) and not stat.S_ISLNK(os.lstat(str(d)).st_mode),"final tree not plain dir")
members={"m597_m593_m528_parent_scratch_generated_macro_energy_result_r2.json","m597_parent_scratch_energy_rows_r2.csv","RUN_COMPLETE.txt","production_stdout.log","production_stderr.log","m602_terminal_rehash_receipt.json"}
expected=members|{"SHA256SUMS","SHA256SUMS.seal.sha256"}
req(set(p.name for p in d.iterdir())==expected,"final member set drift")
for p in d.iterdir(): req(stat.S_ISREG(os.lstat(str(p)).st_mode) and not stat.S_ISLNK(os.lstat(str(p)).st_mode),"nonregular final member")
manifest=d/"SHA256SUMS"; outer=d/"SHA256SUMS.seal.sha256"; msha=sha(manifest)
req(outer.read_text().strip().split()==[msha,"SHA256SUMS"],"final outer drift")
listed={}
for line in manifest.read_text().splitlines():
    dig,name=line.split(None,1); name=name.lstrip("*"); req(name not in listed,"duplicate final member"); listed[name]=dig
req(set(listed)==members,"final manifest set drift")
for n,digest in listed.items(): req(sha(d/n)==digest,"final member SHA drift: "+n)
def pairs(items):
    out={}
    for key,value in items:
        req(key not in out,"duplicate terminal JSON key"); out[key]=value
    return out
receipt=json.loads((d/"m602_terminal_rehash_receipt.json").read_text(encoding="utf-8"),object_pairs_hook=pairs,parse_constant=lambda raw: (_ for _ in ()).throw(RuntimeError(raw)))
req(receipt.get("status")=="PASS_M602_TERMINAL_IDENTITY_AND_OUTPUT_REHASH","terminal receipt drift")
for name,digest in receipt.get("output_members_preseal",{}).items():
    req(name in members-{"m602_terminal_rehash_receipt.json"},"unexpected terminal-bound member")
    req(sha(d/name)==digest,"terminal receipt member SHA drift: "+name)
result=json.loads((d/"m597_m593_m528_parent_scratch_generated_macro_energy_result_r2.json").read_text(encoding="utf-8"),object_pairs_hook=pairs,parse_constant=lambda raw: (_ for _ in ()).throw(RuntimeError(raw)))
req(result.get("schema")==receipt.get("output_schema"),"post-seal output schema drift")
req(result.get("status")==receipt.get("output_status"),"post-seal output status drift")
req(result.get("claim_boundary",{}).get("result_hammer_pending") is True,"post-seal result-hammer boundary drift")
print("PASS_M602_FINAL_RESULT_TREE")
PY
}

seal_tree_exact() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib, os, pathlib, stat, sys
d=pathlib.Path(sys.argv[1])
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
files=[]
for root, dirs, names in os.walk(str(d), topdown=True, followlinks=False):
    for name in list(dirs):
        p=pathlib.Path(root)/name
        if stat.S_ISLNK(os.lstat(str(p)).st_mode): raise RuntimeError("symlink directory in seal")
    for name in names:
        p=pathlib.Path(root)/name
        if p.name in ("SHA256SUMS","SHA256SUMS.seal.sha256") and p.parent==d: continue
        mode=os.lstat(str(p)).st_mode
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode): raise RuntimeError("nonregular member in seal")
        files.append(p)
files.sort(key=lambda p: str(p.relative_to(d)))
manifest=d/"SHA256SUMS"; outer=d/"SHA256SUMS.seal.sha256"
if os.path.lexists(str(manifest)) or os.path.lexists(str(outer)): raise RuntimeError("seal target exists")
with manifest.open("x",encoding="utf-8") as f:
    for p in files: f.write("%s  %s\n"%(sha(p),str(p.relative_to(d))))
with outer.open("x",encoding="utf-8") as f: f.write("%s  SHA256SUMS\n"%sha(manifest))
PY
}

MODE="preflight"
AUTH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight-only) MODE="preflight"; shift ;;
    --execute) MODE="execute"; shift ;;
    --authorization)
      [[ $# -ge 2 ]] || { echo "missing --authorization value" >&2; exit 64; }
      AUTH="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

[[ -x "$PYTHON_BIN" && ! -L "$PYTHON_BIN" ]] || { echo "frozen Python missing/symlink" >&2; exit 65; }
[[ "$(sha_file "$PYTHON_BIN")" == "$PYTHON_SHA256" ]] || { echo "frozen Python SHA drift" >&2; exit 66; }
[[ -f "$ANALYZER" && ! -L "$ANALYZER" ]] || { echo "M597 analyzer missing/symlink" >&2; exit 65; }
[[ "$(sha_file "$ANALYZER")" == "$ANALYZER_SHA256" ]] || { echo "M597 analyzer SHA drift" >&2; exit 66; }
[[ -f "$SOURCE_CONTRACT" && ! -L "$SOURCE_CONTRACT" ]] || { echo "M597 contract missing/symlink" >&2; exit 65; }
[[ "$(sha_file "$SOURCE_CONTRACT")" == "$SOURCE_CONTRACT_SHA256" ]] || { echo "M597 contract SHA drift" >&2; exit 66; }
RUNNER_SHA_START="$(sha_file "$RUNNER")"
STAGING_DIR="${RESULT_DIR}.staging.$$"
verify_static_identity >/dev/null
assert_coordinate_policy >/dev/null

if [[ "$MODE" == "preflight" ]]; then
  [[ -z "$AUTH" ]] || { echo "preflight must not receive authorization" >&2; exit 64; }
  "$PYTHON_BIN" "$ANALYZER" --source-contract "$SOURCE_CONTRACT" --self-test >/dev/null
  [[ "$(sha_file "$RUNNER")" == "$RUNNER_SHA_START" ]] || { echo "runner changed during preflight" >&2; exit 66; }
  verify_static_identity >/dev/null
  echo "PASS_M602_M593_SOURCE_PREFLIGHT_ONLY__NO_RESULT_ATTEMPT_OR_LAUNCH"
  exit 0
fi

[[ -n "$AUTH" ]] || { echo "--execute requires future authorization" >&2; exit 64; }
AUTH="$(realpath -e -- "$AUTH")"
[[ "$AUTH" == "$FUTURE_AUTH" ]] || { echo "future authorization coordinate drift" >&2; exit 67; }
AUTH_SHA_START="$(verify_future_authorization)"
[[ "$(sha_file "$RUNNER")" == "$RUNNER_SHA_START" ]] || { echo "runner changed after authorization" >&2; exit 66; }
verify_static_identity >/dev/null
assert_coordinate_policy >/dev/null

SUCCESS=0
STAGE="trap_installed_before_attempt"
SIGNAL_CAUGHT="none"
cleanup() {
  rc=$?
  trap - EXIT INT TERM HUP
  if [[ "$SUCCESS" -ne 1 ]]; then
    [[ "$rc" -ne 0 ]] || rc=1
    set +e
    if lexists "$ATTEMPT_DIR" || lexists "$STAGING_DIR" || "$PYTHON_BIN" - "$RESULTS_PARENT" "$(basename "$STAGING_DIR")" <<'PY'
import os,sys
parent=sys.argv[1]; prefix="."+sys.argv[2]+".m597_staging_"
raise SystemExit(0 if any(e.name.startswith(prefix) for e in os.scandir(parent)) else 1)
PY
    then
      stamp="$(date -u +%Y%m%dT%H%M%SZ)"
      quarantine_stage="$RESULTS_PARENT/.m597_m593_energy.failed_quarantine.staging.${stamp}.$$"
      quarantine_final="$RESULTS_PARENT/m597_m593_energy.failed_or_incomplete.${stamp}.$$"
      if lexists "$quarantine_stage" || lexists "$quarantine_final"; then
        echo "quarantine collision; state left fail-closed" >&2
        exit "$rc"
      fi
      mkdir -- "$quarantine_stage"
      if lexists "$ATTEMPT_DIR"; then rename_noreplace "$ATTEMPT_DIR" "$quarantine_stage/attempt"; fi
      if lexists "$STAGING_DIR"; then rename_noreplace "$STAGING_DIR" "$quarantine_stage/runner_staging"; fi
      internal_index=0
      while IFS= read -r -d '' internal; do
        internal_index=$((internal_index+1))
        rename_noreplace "$internal" "$quarantine_stage/analyzer_internal_staging_${internal_index}"
      done < <("$PYTHON_BIN" - "$RESULTS_PARENT" "$(basename "$STAGING_DIR")" <<'PY'
import os,sys
parent=sys.argv[1]; prefix="."+sys.argv[2]+".m597_staging_"
for entry in sorted(os.scandir(parent),key=lambda e:e.name):
    if entry.name.startswith(prefix):
        sys.stdout.buffer.write(os.fsencode(entry.path)+b"\0")
PY
      )
      "$PYTHON_BIN" - "$quarantine_stage" "$rc" "$STAGE" "$SIGNAL_CAUGHT" "$RUNNER" "$RUNNER_SHA_START" "$ANALYZER" "$ANALYZER_SHA256" "$SOURCE_CONTRACT" "$SOURCE_CONTRACT_SHA256" "$AUTH" "$AUTH_SHA_START" "$RESULT_DIR" <<'PY'
import hashlib,json,os,pathlib,sys
d=pathlib.Path(sys.argv[1])
def observed(p):
    p=pathlib.Path(p)
    if not p.is_file() or p.is_symlink(): return None
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
payload={
 "schema":"m602_m593_energy_failed_attempt_quarantine_v1",
 "status":"FAILED_OR_INTERRUPTED_ATTEMPT_AND_STAGING_QUARANTINED",
 "exit_code":int(sys.argv[2]),"failure_stage":sys.argv[3],"signal":sys.argv[4],
 "runner":{"path":sys.argv[5],"expected_sha256":sys.argv[6],"observed_sha256":observed(sys.argv[5])},
 "analyzer":{"path":sys.argv[7],"expected_sha256":sys.argv[8],"observed_sha256":observed(sys.argv[7])},
 "source_contract":{"path":sys.argv[9],"expected_sha256":sys.argv[10],"observed_sha256":observed(sys.argv[9])},
 "authorization":{"path":sys.argv[11],"expected_sha256":sys.argv[12],"observed_sha256":observed(sys.argv[11])},
 "canonical_result_lexists":os.path.lexists(sys.argv[13]),
}
(d/"failure_receipt.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
      seal_tree_exact "$quarantine_stage"
      rename_noreplace "$quarantine_stage" "$quarantine_final"
    fi
  fi
  exit "$rc"
}
on_signal() { SIGNAL_CAUGHT="$1"; exit "$2"; }
trap cleanup EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

STAGE="attempt_mkdir"
mkdir -- "$ATTEMPT_DIR"
STAGE="attempt_marker"
"$PYTHON_BIN" - "$ATTEMPT_DIR" "$RUNNER_SHA_START" "$AUTH" "$AUTH_SHA_START" <<'PY'
import json,pathlib,sys
d=pathlib.Path(sys.argv[1])
payload={"schema":"m602_m593_energy_attempt_v1","status":"ATTEMPT_CONSUMED","runner_sha256_start":sys.argv[2],"authorization_path":sys.argv[3],"authorization_sha256_start":sys.argv[4]}
(d/"ATTEMPT_CONSUMED.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY

STAGE="formal_analyzer"
"$PYTHON_BIN" "$ANALYZER" --source-contract "$SOURCE_CONTRACT" --output-dir "$STAGING_DIR" >"$ATTEMPT_DIR/production_stdout.log" 2>"$ATTEMPT_DIR/production_stderr.log"

STAGE="terminal_static_identity"
[[ "$(sha_file "$RUNNER")" == "$RUNNER_SHA_START" ]]
[[ "$(sha_file "$AUTH")" == "$AUTH_SHA_START" ]]
verify_future_authorization >/dev/null
verify_static_identity >/dev/null
verify_analyzer_result_tree "$STAGING_DIR" >/dev/null
cp -- "$ATTEMPT_DIR/production_stdout.log" "$STAGING_DIR/production_stdout.log"
cp -- "$ATTEMPT_DIR/production_stderr.log" "$STAGING_DIR/production_stderr.log"

STAGE="terminal_receipt"
rm -- "$STAGING_DIR/SHA256SUMS" "$STAGING_DIR/SHA256SUMS.seal.sha256"
"$PYTHON_BIN" - "$STAGING_DIR" "$RUNNER" "$RUNNER_SHA_START" "$ANALYZER" "$ANALYZER_SHA256" "$SOURCE_CONTRACT" "$SOURCE_CONTRACT_SHA256" "$AUTH" "$AUTH_SHA_START" <<'PY'
import hashlib,json,pathlib,sys
d=pathlib.Path(sys.argv[1])
def sha(p):
    p=pathlib.Path(p); h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
payload={
 "schema":"m602_m593_energy_terminal_rehash_receipt_v1",
 "status":"PASS_M602_TERMINAL_IDENTITY_AND_OUTPUT_REHASH",
 "runner":{"path":sys.argv[2],"sha256":sha(sys.argv[2]),"expected_sha256":sys.argv[3]},
 "analyzer":{"path":sys.argv[4],"sha256":sha(sys.argv[4]),"expected_sha256":sys.argv[5]},
 "source_contract":{"path":sys.argv[6],"sha256":sha(sys.argv[6]),"expected_sha256":sys.argv[7]},
 "authorization":{"path":sys.argv[8],"sha256":sha(sys.argv[8]),"expected_sha256":sys.argv[9]},
 "output_schema":"m597_m593_m528_parent_scratch_generated_macro_energy_result_v2",
 "output_status":"PASS_BOUNDED_GENERATED_MACRO_COMPONENT_MODEL__PENDING_FRESH_INDEPENDENT_RESULT_HAMMER",
 "output_members_preseal":{
   name:sha(d/name) for name in ["m597_m593_m528_parent_scratch_generated_macro_energy_result_r2.json","m597_parent_scratch_energy_rows_r2.csv","RUN_COMPLETE.txt","production_stdout.log","production_stderr.log"]},
 "claim":"component-only per-frozen-sampled-inference; pending independent result hammer; not paper data"
}
for key in ("runner","analyzer","source_contract","authorization"):
    if payload[key]["sha256"]!=payload[key]["expected_sha256"]: raise RuntimeError(key+" terminal drift")
(d/"m602_terminal_rehash_receipt.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
seal_tree_exact "$STAGING_DIR"
verify_final_result_tree "$STAGING_DIR" >/dev/null

STAGE="pre_publish_rehash"
[[ "$(sha_file "$RUNNER")" == "$RUNNER_SHA_START" ]]
[[ "$(sha_file "$AUTH")" == "$AUTH_SHA_START" ]]
verify_future_authorization >/dev/null
verify_static_identity >/dev/null
verify_final_result_tree "$STAGING_DIR" >/dev/null

STAGE="publish_result_noreplace"
rename_noreplace "$STAGING_DIR" "$RESULT_DIR"
STAGE="post_publish_canonical_rehash"
verify_final_result_tree "$RESULT_DIR" >/dev/null
verify_static_identity >/dev/null

STAGE="seal_success_attempt"
"$PYTHON_BIN" - "$ATTEMPT_DIR" "$RUNNER_SHA_START" "$AUTH_SHA_START" <<'PY'
import json,pathlib,sys
d=pathlib.Path(sys.argv[1])
(d/"ATTEMPT_COMPLETION.json").write_text(json.dumps({"schema":"m602_m593_energy_attempt_completion_v1","status":"RESULT_PUBLISHED_AND_CANONICAL_REHASH_PASS","runner_sha256_start":sys.argv[2],"authorization_sha256_start":sys.argv[3]},indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
seal_tree_exact "$ATTEMPT_DIR"

STAGE="consume_attempt_noreplace"
rename_noreplace "$ATTEMPT_DIR" "$CONSUMED_DIR"
SUCCESS=1
trap - EXIT INT TERM HUP
echo "PASS_M602_M593_ATOMIC_COMPONENT_RESULT_PENDING_INDEPENDENT_RESULT_HAMMER $RESULT_DIR"
