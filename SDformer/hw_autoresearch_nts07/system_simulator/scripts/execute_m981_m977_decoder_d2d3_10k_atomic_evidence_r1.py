#!/usr/bin/env python3
"""M981 additive decoder wrapper with atomic two-file seal publication.

M981 freezes M946/M896 and all inputs. It only repairs M972's milestone
namespace and failure durability. A seal is built in a sibling staging
directory, fsynced, then atomically renamed as `.m981_atomic_seal`; observers
therefore see either no seal or a complete manifest+outer-seal bundle.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple


HERE=Path(__file__).resolve().parent
HW=HERE.parent.parent
REPO=HW.parent
PYTHON=Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA="9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
M946_PATH=HERE/"analyze_m946_decoder_multilayer_bounded_prefix_source_candidate.py"
M946_SHA="0ffd1ee810f24d1a95b0df33ffe8eae43240920e12a2fccb86c947d2be51b6ac"
M896_PATH=HERE/"analyze_m896_decoder_run_gtls_source_candidate.py"
M896_SHA="c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39"
CONTRACT=HW/"contracts/m981_m977_decoder_d2d3_10k_atomic_evidence_source_contract_r1_20260829.json"
SOURCE_HAMMER=HW/"reviews/m982_m981_decoder_d2d3_10k_atomic_evidence_source_hammer_r1_20260829"
RELEASE=HW/"contracts/m983_m981_decoder_d2d3_10k_atomic_evidence_release_r1_20260829.json"
RELEASE_HAMMER=HW/"reviews/m984_m983_m981_decoder_d2d3_10k_atomic_evidence_release_hammer_r1_20260829"
RESULT=HW/"results/m985_m981_decoder_d2d3_10k_atomic_evidence_r1_20260829"
ATTEMPT=HW/"results/.m985_m981_decoder_d2d3_10k_atomic_evidence_attempt_consumed"
FAILURE_PREFIX=RESULT.name+".failed_or_incomplete."
DOCS359_SHA="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOURCE_SCHEMA="m981_m977_decoder_d2d3_10k_atomic_evidence_source_contract_v1"
RELEASE_SCHEMA="m983_m981_decoder_d2d3_10k_atomic_evidence_release_v1"
PREFIX=10000
GEOMETRY={"D2":{"source_bytes":231600,"source_fetch_requests":1207},
          "D3":{"source_bytes":465600,"source_fetch_requests":2425}}
SEAL_DIR=".m981_atomic_seal"
SEAL_MANIFEST="SHA256SUMS"
SEAL_OUTER="SHA256SUMS.seal.sha256"


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition:bool,message:str)->None:
    if not condition: raise RuntimeError(message)


def strict_json(path:Path):
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,"duplicate JSON key: "+key)
            value[key]=item
        return value
    with Path(path).open("r",encoding="utf-8") as handle:
        return json.load(handle,object_pairs_hook=pairs,
                         parse_constant=lambda x:(_ for _ in ()).throw(
                             ValueError("nonfinite JSON: "+x)))


def write_exclusive(path:Path,data:bytes)->None:
    with Path(path).open("xb") as handle:
        handle.write(data);handle.flush();os.fsync(handle.fileno())


def append_fsync(path:Path,text:str)->None:
    with Path(path).open("ab") as handle:
        handle.write(text.encode());handle.flush();os.fsync(handle.fileno())


def fsync_dir(path:Path)->None:
    descriptor=os.open(str(path),os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def validate_interpreter()->Dict[str,object]:
    executable=Path(sys.executable).resolve()
    require(executable==PYTHON and sha256(executable)==PYTHON_SHA and
            tuple(sys.version_info[:3])==(3,10,18),"M981 Python drift")
    return {"path":str(executable),"sha256":PYTHON_SHA,"version":[3,10,18]}


validate_interpreter()


def load_m946():
    require(sha256(M946_PATH)==M946_SHA and sha256(M896_PATH)==M896_SHA,
            "M981 frozen M946/M896 drift")
    spec=importlib.util.spec_from_file_location("m981_frozen_m946",M946_PATH)
    require(spec is not None and spec.loader is not None,"cannot load M946")
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module);return module


M946=load_m946()


def canonical_paths()->Dict[str,str]:
    return {
        "source_contract":str(CONTRACT.relative_to(REPO)),
        "source_hammer":str(SOURCE_HAMMER.relative_to(REPO)),
        "release":str(RELEASE.relative_to(REPO)),
        "release_hammer":str(RELEASE_HAMMER.relative_to(REPO)),
        "run_result":str(RESULT.relative_to(REPO)),
        "run_attempt":str(ATTEMPT.relative_to(REPO)),
        "run_failure_prefix":"hw_autoresearch_nts07/results/"+FAILURE_PREFIX,
    }


def safe_result_sibling(path:Path,prefix:str,
                        allowed_parent:Optional[Path]=None)->None:
    path=Path(path)
    parent=RESULT.parent if allowed_parent is None else Path(allowed_parent)
    require(path.parent.resolve()==parent.resolve() and
            path.name.startswith(prefix) and not path.is_symlink(),
            "M981 unsafe result sibling")


def payload_files(directory:Path):
    output=[]
    for item in sorted(Path(directory).rglob("*")):
        rel=item.relative_to(directory)
        if rel.parts and rel.parts[0]==SEAL_DIR: continue
        if item.is_file() and not item.is_symlink(): output.append(item)
        elif item.is_symlink(): raise RuntimeError("M981 seal refuses symlink")
    return output


def partial_seal_stages(directory:Path):
    prefix=Path(directory).name+".m981_seal_stage."
    return sorted(item for item in Path(directory).parent.iterdir()
                  if item.name.startswith(prefix))


def recover_partial_seal_stages(directory:Path)->int:
    directory=Path(directory);stages=partial_seal_stages(directory)
    if not stages:return 0
    recovery=directory/"PARTIAL_SEAL_ATTEMPTS"
    recovery.mkdir(mode=0o700,exist_ok=True)
    for index,stage in enumerate(stages):
        target=recovery/("attempt_%03d"%index)
        require(not target.exists(),"M981 partial recovery collision")
        os.rename(stage,target)
    fsync_dir(recovery);fsync_dir(directory)
    return len(stages)


def verify_atomic_seal(directory:Path)->Dict[str,object]:
    directory=Path(directory);bundle=directory/SEAL_DIR
    require(bundle.is_dir() and not bundle.is_symlink(),"M981 seal absent")
    manifest=bundle/SEAL_MANIFEST;outer=bundle/SEAL_OUTER
    require(manifest.is_file() and outer.is_file() and
            not manifest.is_symlink() and not outer.is_symlink(),
            "M981 incomplete atomic seal bundle")
    require(outer.read_text(encoding="utf-8")==
            sha256(manifest)+"  "+SEAL_MANIFEST+"\n","M981 outer mismatch")
    listed={}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest,rel=line.split("  ",1)
        require(rel not in listed,"M981 duplicate manifest member")
        member=directory/rel
        require(member.resolve().is_relative_to(directory.resolve()) and
                member.is_file() and not member.is_symlink() and
                sha256(member)==digest,"M981 member drift: "+rel)
        listed[rel]=digest
    actual={item.relative_to(directory).as_posix() for item in
            payload_files(directory)}
    require(set(listed)==actual,"M981 atomic manifest coverage drift")
    return {"manifest_sha256":sha256(manifest),
            "outer_seal_file_sha256":sha256(outer),
            "member_count":len(actual),"atomic_bundle":SEAL_DIR}


def atomic_seal(directory:Path,inject_fault:str="")->Dict[str,object]:
    directory=Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "M981 seal target invalid")
    if (directory/SEAL_DIR).exists(): return verify_atomic_seal(directory)
    recover_partial_seal_stages(directory)
    members=payload_files(directory);require(members,"M981 empty seal target")
    stage=directory.parent/(directory.name+".m981_seal_stage.%d.%d"%
                            (os.getpid(),time.time_ns()))
    stage.mkdir(mode=0o700)
    lines=[sha256(item)+"  "+item.relative_to(directory).as_posix()
           for item in members]
    write_exclusive(stage/SEAL_MANIFEST,("\n".join(lines)+"\n").encode())
    if inject_fault=="after_manifest":
        raise RuntimeError("M981 injected interruption after manifest")
    write_exclusive(stage/SEAL_OUTER,
                    (sha256(stage/SEAL_MANIFEST)+"  "+SEAL_MANIFEST+"\n").encode())
    fsync_dir(stage)
    if inject_fault=="before_atomic_rename":
        raise RuntimeError("M981 injected interruption before atomic rename")
    os.rename(stage,directory/SEAL_DIR);fsync_dir(directory)
    return verify_atomic_seal(directory)


def source_geometry(layer:str)->Dict[str,int]:
    require(layer in ("D2","D3"),"M981 only D2/D3")
    index=M946.MODULE_BY_LAYER[layer]
    cin,_,hin,win,_,_=M946.M785.MODULE_GEOMETRY[index]
    source_bytes=math.ceil(cin*hin*win/8)
    transaction=M946.M785._source_read("m981_geometry","m981_geometry",
        "A1_OSG",index,0,source_bytes)
    value={"source_bytes":source_bytes,
           "source_fetch_requests":int(transaction.count)}
    require(value==GEOMETRY[layer],"M981 generated geometry drift")
    return value


def summarize_row(row:Mapping[str,object])->Dict[str,object]:
    identity=row["row_identity"];layer=str(identity["layer"])
    exact=row["exact_miter"]
    require(layer in ("D2","D3") and
            identity["numerical_route"]=="EXACT_BINARY_SUPPORT" and
            int(row["prefix"])==PREFIX and
            exact["status"]=="PASS_M768_M861_M890_M896_EXACT_MITER" and
            int(exact["expanded_request_count"])==PREFIX,
            "M981 exact row drift")
    geometry=source_geometry(layer)
    return {"layer":layer,"prefix_requests":PREFIX,**geometry,
        "requests_beyond_first_source_fetch":
            PREFIX-geometry["source_fetch_requests"],
        "observed_compressed_transaction_count":
            int(exact["compressed_transaction_count"]),
        "observed_commit_requests_in_prefix":
            int(exact["commit_requests_in_prefix"]),
        "automatic_100k_authorized":False,"full_row_authorized":False}


def safe_work(work:Path,allowed_parent:Optional[Path]=None)->None:
    safe_result_sibling(work,RESULT.name+".work.",allowed_parent)


def run_row(layer:str,stage:Path,
            producer:Callable[...,Mapping[str,object]]=M946.run_bounded_prefix):
    stage=Path(stage)
    require(layer in ("D2","D3") and stage.name==layer and
            stage.parent.name.startswith(RESULT.name+".work."),
            "M981 row stage drift")
    require(not stage.exists(),"M981 row stage collision")
    stage.mkdir(mode=0o700)
    write_exclusive(stage/"ROW_STARTED.json",(json.dumps({
        "status":"STARTED_BEFORE_MODEL_CALL","layer":layer,"prefix":PREFIX
    },sort_keys=True)+"\n").encode())
    write_exclusive(stage/"row.log",b"stage created before model call\n")
    try:
        row=producer(layer,0,"A1_OSG",0,PREFIX);summary=summarize_row(row)
        payload={"schema":"m981_decoder_10k_row_v1",
                 "status":"PASS_M981_ROW_EXACT__FRESH_HAMMER_REQUIRED",
                 "row":row,"summary":summary,
                 "claim_boundary":{"paper_citable":False,
                    "automatic_100k_authorized":False,
                    "full_row_authorized":False}}
        write_exclusive(stage/"row.json",(json.dumps(payload,indent=2,
            sort_keys=True,allow_nan=False)+"\n").encode())
        append_fsync(stage/"row.log","row complete; atomic sealing\n")
        write_exclusive(stage/"ROW_COMPLETE.txt",b"M981_ROW_COMPLETE\n")
        return {"payload":payload,"seal":atomic_seal(stage)}
    except BaseException as error:
        write_exclusive(stage/"traceback.log",traceback.format_exc().encode())
        write_exclusive(stage/"failure.json",(json.dumps({
            "status":"FAILED__QUARANTINE_REQUIRED","layer":layer,
            "exception_type":type(error).__name__,
            "exception_message":str(error)},sort_keys=True)+"\n").encode())
        append_fsync(stage/"row.log","exception persisted; atomic sealing\n")
        write_exclusive(stage/"ROW_FAILED.txt",b"M981_ROW_FAILED\n")
        atomic_seal(stage);raise


def quarantine_work(work:Path,quarantine:Path,return_code:int,
                    inject_fault:str="",
                    allowed_parent:Optional[Path]=None)->Dict[str,object]:
    work=Path(work);quarantine=Path(quarantine);safe_work(work,allowed_parent)
    safe_result_sibling(quarantine,FAILURE_PREFIX,allowed_parent)
    require(work.is_dir() and not quarantine.exists(),
            "M981 quarantine namespace drift")
    if (work/SEAL_DIR).exists():
        seal=verify_atomic_seal(work)
        os.rename(work,quarantine);fsync_dir(quarantine.parent)
        require(verify_atomic_seal(quarantine)==seal,
                "M981 sealed-root quarantine drift")
        return {"status":"PASS_M981_PRESEALED_ROOT_QUARANTINED_UNMODIFIED",
                "path":str(quarantine),"seal":seal}
    # Never trust a lone legacy top-level SHA256SUMS as completion. It is a
    # normal payload member and is covered by the atomic seal bundle below.
    for layer in ("D2","D3"):
        row=work/layer
        if row.is_dir() and not (row/SEAL_DIR).exists():
            if not (row/"ROW_INTERRUPTED.txt").exists():
                write_exclusive(row/"ROW_INTERRUPTED.txt",
                                b"M981_ROW_INTERRUPTED\n")
            atomic_seal(row)
    if not (work/"failure.json").exists():
        write_exclusive(work/"failure.json",(json.dumps({
            "schema":"m981_failure_root_v1",
            "status":"FAILED_OR_INTERRUPTED__NO_RETRY",
            "return_code":int(return_code)},sort_keys=True)+"\n").encode())
    if inject_fault=="before_root_seal":
        raise RuntimeError("M981 injected cleanup failure; work retained")
    seal=atomic_seal(work,inject_fault if inject_fault in
                     ("after_manifest","before_atomic_rename") else "")
    verify_atomic_seal(work)
    os.rename(work,quarantine);fsync_dir(quarantine.parent)
    require(verify_atomic_seal(quarantine)==seal,
            "M981 quarantine publication drift")
    return {"status":"PASS_M981_DOUBLE_SEALED_QUARANTINE",
            "path":str(quarantine),"seal":seal}


def verify_flat_review(directory:Path,identity:Tuple[str,str,str],label:str):
    sealed=M946.M785.verify_sealed_directory(directory)
    require(sha256(directory/"review.json")==identity[0] and
            sealed["manifest_sha256"]==identity[1] and
            sealed["outer_seal_file_sha256"]==identity[2],
            label+" identity drift")
    return sealed


def validate_source_contract(contract:Path,runner:Path,
                             require_fresh:bool=True)->Dict[str,object]:
    require(Path(contract).resolve()==CONTRACT.resolve(),"contract path drift")
    value=strict_json(contract)
    require(value.get("schema")==SOURCE_SCHEMA and value.get("launch_now") is False
            and value.get("canonical")==canonical_paths(),
            "M981 source contract drift")
    for name,item in value["source_identity"].items():
        path=HW/item["path"]
        require(path.is_file() and not path.is_symlink() and
                sha256(path)==item["sha256"],"source drift: "+name)
    require(Path(runner).resolve()==
            (HW/value["source_identity"]["m985_runner"]["path"]).resolve(),
            "M985 runner path drift")
    require(sha256(HW/"docs/359_DATE终局冻结_20260813.md")==DOCS359_SHA,
            "docs359 drift")
    require({layer:source_geometry(layer) for layer in ("D2","D3")}==GEOMETRY,
            "geometry contract drift")
    if require_fresh:
        require(not RESULT.exists() and not ATTEMPT.exists(),
                "M985 run namespace not fresh")
    return {"status":"PASS_M981_SOURCE__NO_10K_EXECUTED",
            "contract_sha256":sha256(contract),"runner_sha256":sha256(runner)}


def validate_authority(runner:Path,expected_release_sha:str,
        source_identity:Tuple[str,str,str],release_identity:Tuple[str,str,str]):
    source=validate_source_contract(CONTRACT,runner,require_fresh=False)
    require(RELEASE.is_file() and sha256(RELEASE)==expected_release_sha,
            "M983 release identity drift")
    verify_flat_review(SOURCE_HAMMER,source_identity,"M982")
    source_review=strict_json(SOURCE_HAMMER/"review.json")
    require(source_review.get("status")==
            "PASS_M982_M981_ATOMIC_EVIDENCE_SOURCE_HAMMER" and
            source_review.get("verdict")=="GO_AUTHOR_M983_RELEASE_ONLY",
            "M982 source hammer authority drift")
    release=strict_json(RELEASE)
    require(release.get("schema")==RELEASE_SCHEMA and
            release.get("status")=="AUTHORIZE_ONE_M985_D2_THEN_D3_10K_RUN" and
            release.get("release") is True and release.get("launch_now") is False
            and release.get("max_attempts")==1,
            "M983 release authority drift")
    require(release.get("exact_rows")==[
        {"layer":"D2","sample_id":0,"config":"A1_OSG","timestep":0,
         "expanded_prefix":10000},
        {"layer":"D3","sample_id":0,"config":"A1_OSG","timestep":0,
         "expanded_prefix":10000}],"M983 row order drift")
    binding=release.get("source_binding",{})
    require(binding.get("m981_contract_sha256")==source["contract_sha256"] and
            binding.get("m981_driver_sha256")==sha256(Path(__file__)) and
            binding.get("m985_runner_sha256")==sha256(runner) and
            binding.get("m982_review_sha256")==source_identity[0] and
            binding.get("m946_sha256")==M946_SHA and
            binding.get("m896_sha256")==M896_SHA,
            "M983 source binding drift")
    auth=release.get("authorization",{})
    require(auth.get("one_m985_d2_then_d3_10k")==True and
            all(auth.get(key) is False for key in
                ("retry","d2_or_d3_100k","full_row","production",
                 "eda_gpu_remote")),"M983 authorization expansion")
    verify_flat_review(RELEASE_HAMMER,release_identity,"M984")
    release_review=strict_json(RELEASE_HAMMER/"review.json")
    require(release_review.get("status")==
            "PASS_M984_M983_M981_ATOMIC_EVIDENCE_RELEASE_HAMMER" and
            release_review.get("verdict")=="GO_ONE_M985_RUN_ONLY" and
            release_review.get("release_sha256")==expected_release_sha,
            "M984 release hammer authority drift")
    return {"status":"PASS_M981_M985_ONE_RUN_AUTHORITY",
            "release_sha256":expected_release_sha,
            "source_hammer_review_sha256":source_identity[0],
            "release_hammer_review_sha256":release_identity[0]}


def consume_attempt(stage:Path,authority:Mapping[str,object]):
    safe_result_sibling(stage,ATTEMPT.name+".stage.")
    require(not stage.exists() and not ATTEMPT.exists() and not RESULT.exists(),
            "M985 attempt namespace collision")
    stage.mkdir(mode=0o700)
    receipt={"schema":"m985_attempt_v1",
             "status":"CONSUMED_BEFORE_D2_MODEL_CALL","max_attempts":1,
             "release_sha256":authority["release_sha256"],
             "release_hammer_review_sha256":
                 authority["release_hammer_review_sha256"],
             "retry":False,"d2_or_d3_100k_authorized":False,
             "full_row_authorized":False}
    write_exclusive(stage/"attempt.json",(json.dumps(receipt,sort_keys=True)+
                                           "\n").encode())
    seal=atomic_seal(stage);os.rename(stage,ATTEMPT);fsync_dir(ATTEMPT.parent)
    require(verify_atomic_seal(ATTEMPT)==seal,"M985 attempt publish drift")
    return {"receipt":receipt,"seal":seal}


def validate_attempt(authority:Mapping[str,object]):
    seal=verify_atomic_seal(ATTEMPT);receipt=strict_json(ATTEMPT/"attempt.json")
    require(receipt.get("status")=="CONSUMED_BEFORE_D2_MODEL_CALL" and
            receipt.get("release_sha256")==authority["release_sha256"] and
            receipt.get("release_hammer_review_sha256")==
                authority["release_hammer_review_sha256"] and
            receipt.get("retry") is False,"M985 attempt drift")
    return {"receipt":receipt,"seal":seal}


def assemble(work:Path,authority:Mapping[str,object]):
    safe_work(work);validate_attempt(authority)
    rows=[];row_seals={}
    for layer in ("D2","D3"):
        row_stage=work/layer;row_seals[layer]=verify_atomic_seal(row_stage)
        payload=strict_json(row_stage/"row.json")
        require(payload.get("status")==
                "PASS_M981_ROW_EXACT__FRESH_HAMMER_REQUIRED" and
                payload.get("summary",{}).get("layer")==layer,
                "M985 row/order drift")
        rows.append(payload)
    result={"schema":"m985_decoder_d2d3_10k_atomic_result_v1",
            "status":"PASS_M985_D2_THEN_D3_10K__RESULT_HAMMER_REQUIRED",
            "release_sha256":authority["release_sha256"],"rows":rows,
            "row_seals":row_seals,
            "claim_boundary":{"paper_citable":False,
                "automatic_100k_authorized":False,"full_row_authorized":False,
                "decoder_complete":False,"table_a_row":False,
                "system_speedup":False}}
    write_exclusive(work/"result.json",(json.dumps(result,indent=2,
        sort_keys=True,allow_nan=False)+"\n").encode())
    write_exclusive(work/"RUN_COMPLETE.txt",b"M985_COMPLETE__HAMMER_REQUIRED\n")
    return {"result":result,"seal":atomic_seal(work)}


def publish(work:Path):
    safe_work(work);seal=verify_atomic_seal(work)
    require(strict_json(work/"result.json").get("status")==
            "PASS_M985_D2_THEN_D3_10K__RESULT_HAMMER_REQUIRED" and
            not RESULT.exists(),"M985 publish drift")
    os.rename(work,RESULT);fsync_dir(RESULT.parent)
    require(verify_atomic_seal(RESULT)==seal,"M985 result publish drift")
    return {"status":"PASS_M985_ATOMIC_RESULT_PUBLICATION","seal":seal}


def source_self_test()->Dict[str,object]:
    def fake(layer,*unused):
        return {"row_identity":{"layer":layer,
                 "numerical_route":"EXACT_BINARY_SUPPORT"},"prefix":PREFIX,
                "exact_miter":{"status":"PASS_M768_M861_M890_M896_EXACT_MITER",
                  "expanded_request_count":PREFIX,
                  "compressed_transaction_count":157,
                  "commit_requests_in_prefix":9}}
    with tempfile.TemporaryDirectory(prefix="m981_static_") as temp:
        parent=Path(temp)
        work=parent/(RESULT.name+".work.test");work.mkdir()
        write_exclusive(work/"WORK_STARTED.txt",b"before D2\n")
        run_row("D2",work/"D2",fake)
        # Simulate termination after manifest write. Work remains in place;
        # retry recovers the partial sibling stage as evidence and seals anew.
        try: atomic_seal(work,inject_fault="after_manifest")
        except RuntimeError: pass
        require(work.exists() and partial_seal_stages(work),
                "injected partial seal not retained")
        quarantine=parent/(FAILURE_PREFIX+"test")
        result=quarantine_work(work,quarantine,143,allowed_parent=parent)
        require(not work.exists() and quarantine.exists(),
                "atomic quarantine publication failure")
        verify_atomic_seal(quarantine)
        # Legacy lone manifest cannot suppress cleanup.
        work2=parent/(RESULT.name+".work.legacy");work2.mkdir()
        write_exclusive(work2/"SHA256SUMS",b"legacy partial\n")
        quarantine2=parent/(FAILURE_PREFIX+"legacy")
        quarantine_work(work2,quarantine2,130,allowed_parent=parent)
        verify_atomic_seal(quarantine2)
        # Cleanup failure must not move the work root.
        work3=parent/(RESULT.name+".work.retain");work3.mkdir()
        write_exclusive(work3/"payload",b"retain\n")
        quarantine3=parent/(FAILURE_PREFIX+"retain")
        try: quarantine_work(work3,quarantine3,137,"before_root_seal",
                             allowed_parent=parent)
        except RuntimeError: pass
        require(work3.exists() and not quarantine3.exists(),
                "failed cleanup moved unsealed work")
    return {"status":"PASS_M981_ATOMIC_SOURCE_SELF_TEST__NO_REAL_PREFIX",
        "atomic_outer_bundle":True,"partial_manifest_recovered":True,
        "lone_legacy_manifest_does_not_skip_cleanup":True,
        "failed_cleanup_retains_unmoved_work":True,
        "real_10k_executed":False,"eda_gpu_remote_used":False}


def main(argv:Optional[Sequence[str]]=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--validate-source",action="store_true")
    parser.add_argument("--quarantine-work",action="store_true")
    parser.add_argument("--validate-authority",action="store_true")
    parser.add_argument("--consume-attempt",action="store_true")
    parser.add_argument("--run-row",choices=("D2","D3"))
    parser.add_argument("--assemble",action="store_true")
    parser.add_argument("--publish",action="store_true")
    parser.add_argument("--contract",type=Path,default=CONTRACT)
    parser.add_argument("--runner",type=Path)
    parser.add_argument("--work",type=Path);parser.add_argument("--quarantine",type=Path)
    parser.add_argument("--return-code",type=int,default=1)
    parser.add_argument("--attempt-stage",type=Path)
    parser.add_argument("--row-stage",type=Path)
    parser.add_argument("--expected-release-sha",default="")
    for prefix in ("source","release"):
        parser.add_argument("--expected-%s-review-sha"%prefix,default="")
        parser.add_argument("--expected-%s-manifest-sha"%prefix,default="")
        parser.add_argument("--expected-%s-outer-sha"%prefix,default="")
    args=parser.parse_args(argv)
    modes=(args.self_test,args.validate_source,args.quarantine_work,
           args.validate_authority,args.consume_attempt,args.run_row is not None,
           args.assemble,args.publish)
    require(sum(bool(item) for item in modes)==1,
            "M981 requires one explicit source-safe mode")
    if args.self_test:value=source_self_test()
    elif args.validate_source:
        require(args.runner is not None,"runner required")
        value=validate_source_contract(args.contract,args.runner)
    else:
        if args.quarantine_work:
            require(args.work is not None and args.quarantine is not None,
                    "work/quarantine required")
            value=quarantine_work(args.work,args.quarantine,args.return_code)
        else:
            require(args.runner is not None and args.expected_release_sha,
                    "M985 authority args required")
            source_identity=(args.expected_source_review_sha,
                args.expected_source_manifest_sha,args.expected_source_outer_sha)
            release_identity=(args.expected_release_review_sha,
                args.expected_release_manifest_sha,args.expected_release_outer_sha)
            require(all(source_identity+release_identity),
                    "M982/M984 identities required")
            authority=validate_authority(args.runner,args.expected_release_sha,
                                         source_identity,release_identity)
            if args.validate_authority:value=authority
            elif args.consume_attempt:
                require(args.attempt_stage is not None,"attempt stage required")
                value=consume_attempt(args.attempt_stage,authority)
            else:
                validate_attempt(authority)
                if args.run_row is not None:
                    require(args.row_stage is not None,"row stage required")
                    value=run_row(args.run_row,args.row_stage)
                elif args.assemble:
                    require(args.work is not None,"work required")
                    value=assemble(args.work,authority)
                else:
                    require(args.work is not None,"work required")
                    value=publish(args.work)
    print(json.dumps(value,sort_keys=True,allow_nan=False));return 0


if __name__=="__main__":raise SystemExit(main())
