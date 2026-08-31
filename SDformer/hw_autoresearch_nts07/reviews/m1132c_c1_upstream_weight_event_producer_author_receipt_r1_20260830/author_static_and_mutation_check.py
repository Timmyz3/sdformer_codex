#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Author-side M1132C source/contract/bounded mutation check only."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1132c_c1_upstream_weight_event_producer_source.py"
CONTRACT = HW / "contracts/m1132c_c1_upstream_weight_event_producer_source_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
SOURCE_SHA = "d6b077fc71d7433f194d497834babd530e0939ca1166dab9376546c670bbdc5f"
CONTRACT_ID = ("8218699210c481a5a8d2ddfc7b2fe1091b24ef36b004716dc530d9b193acec91",
               "be85e9a08684691c964c78f0b441a85a43a61c69a3d4014ae608a7c123526b4f",
               "9592d136ea18b86c722fb69af3422ef8106d5d5d628d8badbf1e5b079f8d9f07")
DOCS_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
checks = 0; attacks = []
class Reject(RuntimeError): pass
def require(v, m):
    global checks; checks += 1
    if not v: raise Reject(m)
def rejected(label, action):
    try: action()
    except Exception: attacks.append(label); return
    raise Reject("accepted " + label)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def load_subject():
    spec=importlib.util.spec_from_file_location("m1132c_author_subject",SOURCE)
    require(spec and spec.loader,"spec"); mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod
def main():
    before={p:sha(p) for p in (SOURCE,CONTRACT,DOCS359)}
    require(before=={SOURCE:SOURCE_SHA,CONTRACT:CONTRACT_ID[0],DOCS359:DOCS_SHA},"identity")
    require(sha(Path(str(CONTRACT)+".sha256"))==CONTRACT_ID[1] and
            sha(Path(str(CONTRACT)+".sha256.seal.sha256"))==CONTRACT_ID[2],"double seal")
    contract=json.loads(CONTRACT.read_text()); text=SOURCE.read_text(); tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="PerBeatAddressedWeightRefillProducer")
    emit=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="emit_refill_event")
    kw=[x.arg for x in emit.args.kwonlyargs]
    require(kw==contract["producer_supplied_event_fields"],"17 keyword-only fields")
    require(not emit.args.args[1:] and emit.args.vararg is None and emit.args.kwarg is None,"no positional/unknown fields")
    emit_text=ast.get_source_segment(text,emit)
    require('op == "WRITE"' in emit_text and "event.validate()" in emit_text and
            "self._sink(event)" in emit_text and not any(x in kw for x in
            ("count","weight_beat_first","start","beats","capacity")),"direct emitter")
    iterator=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and
                  n.name=="iter_canonical_upstream_weight_refill_events")
    iterator_text=ast.get_source_segment(text,iterator)
    require('audit["canonical_ready"] is True' in iterator_text and
            iterator_text.index('audit["canonical_ready"] is True') < iterator_text.index("yield None"),"canonical stop")
    mod=load_subject(); oracle=mod.source_small_oracle()
    require(oracle["status"]=="PASS_ADDITIVE_PER_BEAT_PRODUCER_SYNTHETIC__CANONICAL_STOP" and
            oracle["synthetic"]=={"producer_write_events":6,"unique_exact_once_write_ids":6,
                                  "post_schedule_stalled_transactions":3,
                                  "post_schedule_native_1rw_conflicts":0} and
            oracle["canonical_rows"]==oracle["canonical_events"]==0,"oracle")
    m=mod.load_m1130(); sink=[]; producer=mod.PerBeatAddressedWeightRefillProducer(sink.append)
    base=dict(axis="candidate",task_id=0,source_local_ordinal=0,requested_cycle=0,op="WRITE",
              logical_bank=0,half_slot=0,logical_row=0,local_row=0,native_slices=tuple(range(8)),
              bytes=128,byte_enable_per_slice=(0xffff,)*8,native_macro_activations=8,
              service_beat_ordinal=0,store_transaction_ordinal=0,
              service_event_exact_once_id=m.exact_once_id("candidate",0,0,0,0),
              source_row_provenance_sha256="0"*64)
    producer.emit_refill_event(**base); require(len(sink)==producer.emitted==1,"one call one event")
    for name in ("axis","task_id","source_local_ordinal","requested_cycle","op","logical_bank","half_slot",
                 "logical_row","local_row","native_slices","bytes","byte_enable_per_slice",
                 "native_macro_activations","service_beat_ordinal","store_transaction_ordinal",
                 "service_event_exact_once_id","source_row_provenance_sha256"):
        bad=dict(base); bad.pop(name); rejected("missing_"+name,lambda bad=bad: mod.PerBeatAddressedWeightRefillProducer(lambda _e:None).emit_refill_event(**bad))
    for name in ("count","weight_beat_first","start","beats","capacity"):
        bad=dict(base); bad[name]=1; rejected("aggregate_"+name,lambda bad=bad: mod.PerBeatAddressedWeightRefillProducer(lambda _e:None).emit_refill_event(**bad))
    for name,value in (("op","READ"),("logical_bank",1),("local_row",1),("native_slices",tuple(range(1,9))),
                       ("bytes",127),("byte_enable_per_slice",(0xfffe,)*8),
                       ("native_macro_activations",7),("service_event_exact_once_id","0"*64)):
        bad=dict(base);bad[name]=value;rejected("invalid_"+name,lambda bad=bad:mod.PerBeatAddressedWeightRefillProducer(lambda _e:None).emit_refill_event(**bad))
    rejected("duplicate_exact_id_beat_transaction",lambda:producer.emit_refill_event(**base))
    def bad_sink(_event): raise OSError("sink failure")
    failed=mod.PerBeatAddressedWeightRefillProducer(bad_sink)
    rejected("sink_exception_propagation",lambda:failed.emit_refill_event(**base))
    require(failed.emitted==0,"sink failure not admitted")
    rejected("canonical_before_hook",lambda:next(mod.iter_canonical_upstream_weight_refill_events()))
    require(before=={p:sha(p) for p in before},"no mutation")
    result={"schema":"m1132c_author_static_mutation_checks_v1",
            "status":"PASS_M1132C_AUTHOR_SOURCE_AND_BOUNDED_SYNTHETIC__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
            "checks":checks,"attacks_rejected":len(attacks),"attack_labels":attacks,
            "synthetic":oracle["synthetic"],"canonical_rows":0,"canonical_events":0,
            "full_51840000":False,"eda_rtl_gpu_remote":False}
    (HERE/"mechanical_checks.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    (HERE/"small_synthetic_oracle.json").write_text(json.dumps(oracle,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],"checks":checks,"attacks":len(attacks)},sort_keys=True))
if __name__=="__main__":main()
