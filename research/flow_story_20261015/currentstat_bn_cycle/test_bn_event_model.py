#!/usr/bin/env python3
"""Small CPU checks including an independent one-clock-at-a-time scheduler."""
import copy
import json
from pathlib import Path
import random
import unittest
import bn_event_model as model


def fixture():
    unit = dict(ports=1,lanes=4,ii=1,latency=3,max_outstanding=3)
    memory = dict(ports=1,words_per_cycle=4,latency=2,max_outstanding=2)
    return dict(schema="flow_bn_event_parameters_v1",parameter_status="synthetic_sensitivity_only",
        word_bytes=4,chunk_words=4,buffer_slots=2,local_capacity_words=128,scalar_register_words=16,
        retention="spill",add=dict(unit),multiply=dict(unit),rsqrt=dict(unit),
        external=dict(read=dict(memory),write=dict(memory),shared_read_write=False),
        local=dict(read=dict(memory),write=dict(memory),shared_read_write=False))


def tick_golden(graph,cfg):
    """No model.Resource/run_graph calls; deterministic literal clock stepping."""
    states={}; aliases={}
    for name in ("add","multiply","rsqrt"):
        states[name]=dict(spec=cfg[name],kind="vector",ports=[0]*cfg[name]["ports"],flying=[])
        aliases[name]=name
    for group in ("external","local"):
        for direction in ("read","write"):
            alias=group+"_"+direction
            name=group+"_shared" if cfg[group]["shared_read_write"] else alias
            aliases[alias]=name
            if name not in states:
                spec=cfg[group][direction]
                states[name]=dict(spec=spec,kind="memory",ports=[0]*spec["ports"],flying=[])
    tasks=graph.tasks; issued={}; retired=set(); starts={}; now=0
    while len(retired)<len(tasks):
        retired.update(i for i,end in issued.items() if end<=now)
        for state in states.values():state["flying"]=[end for end in state["flying"] if end>now]
        for i,task in enumerate(tasks):
            if i in issued or any(p not in retired for p in task["parents"]):continue
            state=states[aliases[task["resource"]]]; spec=state["spec"]
            ports=[k for k,t in enumerate(state["ports"]) if t<=now]
            if not ports or len(state["flying"])>=spec["max_outstanding"]:continue
            p=min(ports,key=lambda k:(state["ports"][k],k))
            width=spec["lanes"] if state["kind"]=="vector" else spec["words_per_cycle"]
            groups=(task["units"]+width-1)//width
            duration=groups*spec["ii"] if state["kind"]=="vector" else groups
            latency=(groups-1)*spec["ii"]+spec["latency"] if state["kind"]=="vector" else groups+spec["latency"]
            state["ports"][p]=now+duration; end=now+latency
            state["flying"].append(end);issued[i]=end;starts[i]=now
        if len(retired)==len(tasks):break
        now+=1
        if now>20000:raise AssertionError("Golden did not finish")
    return max([max(issued.values())]+[max(s["ports"]) for s in states.values()]),starts,issued


class Checks(unittest.TestCase):
    def test_manual_pipeline_and_credit(self):
        cfg=fixture();cfg["add"].update(lanes=4,ii=2,latency=5,max_outstanding=2)
        g=model.Graph();g.task("a","add",4);g.task("b","add",4)
        x=model.run_graph(g,cfg);self.assertEqual(x["starts"],[0,2]);self.assertEqual(x["cycles"],7)
        cfg["add"]["max_outstanding"]=1
        x=model.run_graph(g,cfg);self.assertEqual(x["starts"],[0,5]);self.assertEqual(x["cycles"],10)

    def test_shared_memory_backpressure(self):
        cfg=fixture();g=model.Graph();g.task("r","external_read",8);g.task("w","external_write",8)
        a=model.run_graph(g,cfg)
        cfg["external"]["shared_read_write"]=True;b=model.run_graph(g,cfg)
        self.assertEqual(a["cycles"],4);self.assertEqual(b["cycles"],6)

    def test_independent_clock_golden(self):
        rng=random.Random(20261015)
        for repeat in range(8):
            cfg=fixture();cfg["buffer_slots"]=rng.choice((1,2,3));cfg["chunk_words"]=rng.choice((2,3,4))
            cfg["local"]["shared_read_write"]=bool(repeat%2)
            cfg["retention"]=rng.choice(("spill","auto_channel_resident"))
            cfg["rsqrt"]["latency"]=rng.randint(2,7)
            graph,_=model.channel_graph(10,5,cfg)
            event=model.run_graph(graph,cfg);cycle,starts,ends=tick_golden(graph,cfg)
            self.assertEqual(event["cycles"],cycle)
            self.assertEqual(event["starts"],[starts[i] for i in range(len(graph.tasks))])
            self.assertEqual(event["ends"],[ends[i] for i in range(len(graph.tasks))])

    def test_service_tail_and_barriers(self):
        cfg=fixture();x=model.simulate_channel(15,5,cfg,trace_limit=10)
        self.assertEqual(x["layout"]["chunks"],6)
        self.assertEqual(x["scalar_ops"],dict(add=60,multiply=34,rsqrt=1))
        self.assertEqual(x["transferred_words"]["external_read"],47)
        self.assertEqual(x["transferred_words"]["external_write"],15)
        self.assertGreaterEqual(x["first_normalize_input_issue"],x["barrier_cycles"]["affine"])
        self.assertLess(x["barrier_cycles"]["mean"],x["barrier_cycles"]["variance"])
        self.assertLess(x["barrier_cycles"]["variance"],x["barrier_cycles"]["rsqrt"])

    def test_resident_capacity_and_immutability(self):
        cfg=fixture();before=copy.deepcopy(cfg)
        spill=model.simulate_channel(15,5,cfg)
        self.assertEqual(cfg,before)
        cfg["retention"]="auto_channel_resident";resident=model.simulate_channel(15,5,cfg)
        self.assertTrue(resident["layout"]["resident"])
        self.assertEqual(resident["transferred_words"]["external_read"],17)
        self.assertEqual(resident["scalar_ops"],spill["scalar_ops"])
        cfg["local_capacity_words"]=8;fallback=model.simulate_channel(15,5,cfg)
        self.assertFalse(fallback["layout"]["resident"])
        cfg["retention"]="require_channel_resident"
        with self.assertRaises(ValueError):model.simulate_channel(15,5,cfg)

    def test_one_element_and_bad_configs(self):
        cfg=fixture();x=model.simulate_channel(1,1,cfg)
        self.assertEqual(x["scalar_ops"],dict(add=4,multiply=6,rsqrt=1))
        for field,value in (("word_bytes",1),("chunk_words",0),("buffer_slots",False),("scalar_register_words",1)):
            wrong=copy.deepcopy(cfg);wrong[field]=value
            with self.assertRaises(ValueError):model.validate_config(wrong)

    def test_channel_contiguous_group(self):
        cfg=fixture();cfg.update(chunk_words=6,scalar_register_words=64)
        graph,_=model.channel_graph(10,5,cfg,channels_per_group=3)
        event=model.run_graph(graph,cfg);cycles,starts,ends=tick_golden(graph,cfg)
        self.assertEqual(event["cycles"],cycles)
        self.assertEqual(event["starts"],[starts[i] for i in range(len(graph.tasks))])
        self.assertEqual(dict(graph.scalar_ops),dict(add=120,multiply=72,rsqrt=3))
        result=model.simulate_channel(10,5,cfg,channels_per_group=3)
        self.assertEqual(result["transferred_words"]["external_read"],96)
        self.assertEqual(result["transferred_words"]["external_write"],30)

    def test_observed_layout_no_free_transpose(self):
        record=dict(name="fixture",native_input_shape=[2,1,3,2,5],channel_axis=2,channels=3,
                    observed=dict(input_calls=[dict(shape=[2,1,3,2,5],stride=[30,30,10,5,1])]))
        self.assertEqual(model.observed_layout(record),("native_channel_major",1))
        record["observed"]["input_calls"][0]["stride"]=[30,30,1,15,3]
        self.assertEqual(model.observed_layout(record),("native_channel_contiguous",3))
        record["observed"]["input_calls"][0]["stride"]=[32,32,1,16,3]
        with self.assertRaises(ValueError):model.observed_layout(record)


if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status="PASS" if result.wasSuccessful() else "FAIL",tests_run=result.testsRun,
                failed=len(result.failures),errors=len(result.errors),model_inference=False,GPU_used=False,
                numerical_BN_datapath_verified=False,independent_literal_cycle_golden_cases=9,
                source_sha256={str(Path(__file__).resolve()):model.sha(__file__),str(Path(model.__file__).resolve()):model.sha(model.__file__)})
    print(json.dumps(report,sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
