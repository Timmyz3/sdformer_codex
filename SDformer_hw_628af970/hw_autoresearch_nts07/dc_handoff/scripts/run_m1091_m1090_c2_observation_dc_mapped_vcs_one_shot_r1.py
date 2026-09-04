#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh M1090 observation-wrapper DC then one mapped case-0 short window.

Release source only until a different-author M1092 seal authorizes one M1091
attempt.  This flow never requests power activity or register initialization.
"""
from __future__ import annotations

import fcntl, hashlib, json, os, re, signal, stat, subprocess, sys
from pathlib import Path

DESIGN="m1090_c2_k1_observation_wrapper"
DC_ROOT=Path(__file__).resolve().parent.parent
HW=DC_ROOT.parent
RUNNER=Path(__file__).resolve()
CONTRACT=HW/"contracts/m1090_c2_k1_observation_dc_mapped_vcs_source_contract_r1_20260830.json"
RELEASE=HW/"contracts/m1090_c2_k1_observation_dc_mapped_vcs_release_r1_20260830.json"
M1092=HW/"reviews/m1092_m1090_c2_observation_source_hammer_r1_20260830"
M1088=HW/"reviews/m1088_m1080_c2_mapped_gate_failure_audit_r1_20260830"
M1080_ATTEMPT=HW/"results/.m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_attempt_consumed"
M1080_FAILURE=HW/"results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.2746017.quarantine"
DOCS359=HW/"docs/359_DATE终局冻结_20260813.md"
FILELIST=HW/"dc_handoff/filelists/date_m1090_c2_k1_observation_logic_only_dc.f"
TB=HW/"dc_handoff/tb/tb_m1090_c2_k1_observation_mapped_case0_short.sv"
MEMORY=HW/"tb_m349/m349_fc2_scalar_bank_memory_model.sv"
DC_TCL=HW/"dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC=HW/"dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
SLOW=Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
FAST=Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db")
CELL=Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
DC_HOME=Path("/opt/synopsys/syn/V-2023.12-SP3")
VCS_HOME=Path("/opt/synopsys/vcs/V-2023.12-SP1")
DC_SHELL=DC_HOME/"bin/dc_shell"; DC_TARGET=DC_HOME/"bin/snps_shell"
VCS=VCS_HOME/"bin/vcs"; LMUTIL=Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
RESULT=HW/"results/m1091_m1090_c2_observation_dc_mapped_vcs_r1_20260830"
ATTEMPT=HW/"results/.m1091_m1090_c2_observation_dc_mapped_vcs_attempt_consumed"
WORK=HW/f"results/.m1091_m1090_c2_observation_dc_mapped_vcs_work.{os.getpid()}"
FAILURE=HW/f"results/m1091_m1090_c2_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.{os.getpid()}.quarantine"
LOCK=Path("/tmp/m1091_m1090_c2_observation_eda.lock")
phase="SOURCE_PREFLIGHT"; attempted=False; complete=False

class GateFailure(RuntimeError): pass
def fail(message): raise GateFailure(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def verify_flat(directory,outer_sha):
    directory=Path(directory); manifest=directory/"SHA256SUMS"; outer=directory/"SHA256SUMS.seal.sha256"
    if not directory.is_dir() or directory.is_symlink():fail("sealed directory absent")
    for line in manifest.read_text().splitlines():
        digest,name=line.split(None,1); target=directory/name.lstrip("*")
        if not target.is_file() or target.is_symlink() or sha(target)!=digest:fail("sealed member drift")
    if outer.read_text().split()!=[sha(manifest),"SHA256SUMS"] or sha(outer)!=outer_sha:fail("outer seal drift")
def verify_double(path,outer_sha):
    path=Path(path); side=Path(str(path)+".sha256"); outer=Path(str(path)+".sha256.seal.sha256")
    if side.read_text().split()!=[sha(path),path.relative_to(HW).as_posix()]:fail("sidecar drift")
    if outer.read_text().split()!=[sha(side),side.relative_to(HW).as_posix()] or sha(outer)!=outer_sha:fail("double seal drift")
def seal(directory):
    members=sorted(p for p in Path(directory).rglob("*") if p.is_file() and p.name not in {"SHA256SUMS","SHA256SUMS.seal.sha256"})
    manifest=Path(directory)/"SHA256SUMS"
    manifest.write_text("".join(f"{sha(p)}  {p.relative_to(directory).as_posix()}\n" for p in members))
    (Path(directory)/"SHA256SUMS.seal.sha256").write_text(f"{sha(manifest)}  SHA256SUMS\n")
def run(argv,log,timeout,env):
    with Path(log).open("w") as out:return subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT,timeout=timeout,env=env,check=False).returncode
def static_gate():
    contract=load(CONTRACT); release=load(RELEASE)
    if contract["status"]!="M1090_OBSERVATION_SOURCE_ONLY__M1092_REQUIRED__NO_EDA" or contract["launch_now"] is not False:fail("contract boundary")
    if release["status"]!="M1090_RELEASE_FROZEN__M1092_REQUIRED__NO_EDA" or release["launch_now"] is not False:fail("release boundary")
    verify_double(CONTRACT,release["contract_outer_seal_file_sha256"])
    verify_double(RELEASE,os.environ.get("M1091_EXPECTED_RELEASE_OUTER_SHA256", ""))
    if release["runner_sha256"]!=sha(RUNNER) or os.environ.get("M1091_EXPECTED_RUNNER_SHA256")!=sha(RUNNER):fail("runner pin")
    if sha(DOCS359)!="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4":fail("docs359 drift")
    verify_flat(M1088,"fb3f208dc704c7663769422ad9f27b17851cc86b11826727fe0c0c795260bd5f")
    verify_flat(M1080_ATTEMPT,"21944247a673bda71a1d3f8cce2cf567b91e51a661b88d5028ed89b70d3a8f7c")
    verify_flat(M1080_FAILURE,"2e3367c239cda08987027a55a01f65b0cbebbd1c0dd907a9a945aa12f5cea89d")
    if load(M1088/"review.json")["status"]!="PASS_M1088_M1080_FAILURE_AUDIT__M1080_DO_NOT_RETRY":fail("M1080 retry boundary")
    for rel,digest in contract["source_sha256"].items():
        path=HW/rel
        if not path.is_file() or path.is_symlink() or sha(path)!=digest:fail("source drift "+rel)
    text=TB.read_text(); forbidden="+vcs+"+"".join(chr(x) for x in (105,110,105,116,114,101,103))
    if forbidden in text or "$toggle" in text:fail("forbidden simulation option/activity")
    verify_flat(M1092,os.environ.get("M1091_EXPECTED_M1092_OUTER_SHA256",""))
    review=load(M1092/"review.json")
    if review["status"]!="PASS_M1092_M1090_OBSERVATION_SOURCE_HAMMER__GO_ONE_M1091_ATTEMPT" or review["authorization"]["one_m1091_attempt"] is not True:fail("M1092 no GO")
def resource_gate():
    info={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key,raw=line.split(":",1)
        if key in {"MemAvailable","CommitLimit","Committed_AS"}:info[key]=int(raw.split()[0])
    if info["MemAvailable"]<8*1024*1024 or info["CommitLimit"]-info["Committed_AS"]<8*1024*1024:fail("resource gate")
def collision_gate():
    uid=str(os.getuid())
    for name in ("vcs","vcs1","vlogan","dc_shell","dc_shell-t","fm_shell","pt_shell","simv"):
        if subprocess.run(["/usr/bin/pgrep","-u",uid,"-x",name],stdout=subprocess.DEVNULL,check=False).returncode==0:fail("EDA collision "+name)
def license_gate():
    route=os.environ.get("SNPSLMD_LICENSE_FILE") or os.environ.get("LM_LICENSE_FILE")
    if not route or subprocess.run([str(LMUTIL),"lmstat","-a","-c",route],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60,check=False).returncode:fail("license gate")
def flow():
    global phase,attempted,complete
    static_gate()
    if any(p.exists() or p.is_symlink() for p in (RESULT,ATTEMPT,WORK)):fail("namespace collision")
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open("a+") as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:fail("lock busy")
        collision_gate();resource_gate();license_gate()
        phase="ATTEMPT_CONSUME_BEFORE_EDA";ATTEMPT.mkdir();attempted=True
        write_json(ATTEMPT/"attempt.json",{"status":"M1091_ATTEMPT_CONSUMED","runner_sha256":sha(RUNNER),"contract_sha256":sha(CONTRACT),"m1092_outer":os.environ["M1091_EXPECTED_M1092_OUTER_SHA256"],"dc_attempts":1,"mapped_cases":1,"activity_files":0,"random_initialization":False})
        seal(ATTEMPT);WORK.mkdir()
        env=os.environ.copy();env.update({"VCS_HOME":str(VCS_HOME),"PATH":f"{VCS_HOME}/bin:/usr/bin:/bin"})
        phase="FRESH_DC_M1090_OBSERVATION_TOP";dc=WORK/"dc";dc.mkdir()
        dcenv=env.copy();dcenv.update({"DESIGN_NAME":DESIGN,"HW_ROOT":str(HW),"RTL_FILELIST":str(FILELIST),"LIB_DB":str(SLOW),"MIN_LIB_DB":str(FAST),"SDC_FILE":str(SDC),"OUTPUT_DIR":str(dc),"ELAB_PARAMETERS":"","OPERATING_CONDITION":"ssg0p9v125c"})
        rc=run([str(DC_SHELL),"-f",str(DC_TCL)],dc/"dc.log",21600,dcenv);(dc/"dc.rc").write_text(str(rc)+"\n")
        if rc or not (dc/"TCL_PASS_TERMINAL.txt").is_file():fail("fresh DC failed")
        netlist=dc/f"netlist/{DESIGN}_mapped.v"
        if not netlist.is_file() or not netlist.stat().st_size:fail("mapped netlist absent")
        phase="FRESH_MAPPED_VCS_CASE0_COMPILE";mapped=WORK/"mapped_vcs";mapped.mkdir();simv=mapped/"simv"
        rc=run([str(VCS),"-full64","-sverilog","+v2k","-timescale=1ns/1ps",f"-Mdir={mapped/'csrc'}",str(CELL),str(netlist),str(MEMORY),str(TB),"-top","tb_m1090_c2_k1_observation_mapped_case0_short","-o",str(simv)],mapped/"compile.log",1800,env)
        (mapped/"compile.rc").write_text(str(rc)+"\n")
        if rc or not simv.is_file():fail("mapped compile failed")
        phase="FRESH_MAPPED_VCS_CASE0_SHORT_128";rc=run([str(simv),"-no_save"],mapped/"case0.log",300,env);(mapped/"case0.rc").write_text(str(rc)+"\n")
        text=(mapped/"case0.log").read_text(errors="replace")
        if rc or "PASS_M1090_OBSERVATION_SHORT_WINDOW cycles=128 raw_seen=1 no_unknown=1 diagnostic_only=1" not in text:fail("short observation window found X/stall; inspect quarantine case0.log")
        write_json(WORK/"receipt.json",{"status":"PASS_M1091_FRESH_DC_MAPPED_OBSERVATION_SHORT_WINDOW","mapped_netlist_sha256":sha(netlist),"stage_lines":len(re.findall(r"^M1090_STAGE",text,re.M)),"window_cycles":128,"unknowns":0,"diagnostic_only":True,"paper_citable":False})
        (WORK/"RUN_COMPLETE.txt").write_text("PASS_M1091_FRESH_DC_MAPPED_OBSERVATION_SHORT_WINDOW\n");seal(WORK);os.rename(WORK,RESULT);complete=True
def quarantine(message):
    if attempted and not complete:
        WORK.mkdir(parents=True,exist_ok=True);write_json(WORK/"failure.json",{"status":"FAILED_DIAGNOSTIC_DO_NOT_CITE","phase":phase,"message":message,"m1080_retry":False});seal(WORK);os.rename(WORK,FAILURE)
def handler(signum,_frame):raise GateFailure("signal "+str(signum))
for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,handler)
try:flow()
except (GateFailure,OSError,subprocess.TimeoutExpired,KeyError,ValueError,json.JSONDecodeError) as exc:
    quarantine(str(exc));print("M1091 failure: "+str(exc),file=sys.stderr);raise SystemExit(3)
print("PASS_M1091_FRESH_DC_MAPPED_OBSERVATION_SHORT_WINDOW")
