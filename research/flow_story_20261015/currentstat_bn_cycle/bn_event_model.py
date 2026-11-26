#!/usr/bin/env python3
"""Parameterized discrete-event BN schedule, NOT measured chip/full-net cycles.

Only stdlib. Explicit bounded resources and dependence DAG; vector/burst groups
reserve issue slots, retire with latency, and hold outstanding credits. The
reference serializes native channel-major domains; native channel-contiguous
layouts process all channels as a group, without a free transpose. Independent
BN modules are serialized. Numerical FP reduction/rsqrt equivalence is unverified.
"""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "flow_story_20261014/damage_allocation_probe/hardware_contract_complete_bn_audit.json"
SOURCE_SHA = "9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(x):
    return type(x) is int and x > 0


class Resource:
    def __init__(self, name, spec, kind):
        self.name, self.spec, self.kind = name, spec, kind
        self.ports = [0] * spec["ports"]
        self.inflight = []
        self.busy = self.jobs = self.units = self.peak = 0
        self.last_done = 0

    def earliest(self, now):
        start = max(now, min(self.ports))
        active = [end for end in self.inflight if end > start]
        if len(active) >= self.spec["max_outstanding"]:
            start = min(active)
        return start

    def dispatch(self, now, units):
        require(now >= self.earliest(now), "Resource dispatched before credit/port")
        while self.inflight and self.inflight[0] <= now:
            heapq.heappop(self.inflight)
        index = min(range(len(self.ports)), key=lambda i: (self.ports[i], i))
        require(self.ports[index] <= now, "Busy resource port")
        width = self.spec["lanes"] if self.kind == "vector" else self.spec["words_per_cycle"]
        groups = (units + width - 1) // width
        if self.kind == "vector":
            occupancy = groups * self.spec["ii"]
            finish = now + (groups - 1) * self.spec["ii"] + self.spec["latency"]
        else:
            occupancy = groups
            finish = now + groups + self.spec["latency"]
        self.ports[index] = now + occupancy
        heapq.heappush(self.inflight, finish)
        self.busy += occupancy; self.jobs += 1; self.units += units
        self.peak = max(self.peak, len(self.inflight)); self.last_done = max(self.last_done, finish)
        return finish


def validate_config(cfg):
    require(cfg.get("schema") == "flow_bn_event_parameters_v1", "Config schema")
    require(cfg.get("parameter_status") in ("synthetic_sensitivity_only", "externally_characterized"), "Parameter status")
    for key in ("word_bytes", "chunk_words", "buffer_slots", "local_capacity_words", "scalar_register_words"):
        require(positive(cfg.get(key)), "Positive integer " + key)
    require(cfg["word_bytes"] == 4, "Current graph is FP32; other precision needs separate numerical acceptance")
    require(cfg["local_capacity_words"] >= cfg["chunk_words"] * cfg["buffer_slots"], "Chunk buffers exceed local capacity")
    require(cfg["scalar_register_words"] >= 8 + cfg["buffer_slots"], "Insufficient named scalar/partial registers")
    require(cfg.get("retention") in ("spill", "auto_channel_resident", "require_channel_resident"), "Retention policy")
    for key in ("add", "multiply", "rsqrt"):
        spec = cfg[key]
        require(set(spec) == {"ports", "lanes", "ii", "latency", "max_outstanding"}, "Vector specification " + key)
        require(all(positive(v) for v in spec.values()), "Vector parameters " + key)
    for name in ("external", "local"):
        mem = cfg[name]
        require(type(mem["shared_read_write"]) is bool, "Explicit duplex mode")
        for direction in ("read", "write"):
            spec = mem[direction]
            require(set(spec) == {"ports", "words_per_cycle", "latency", "max_outstanding"}, "Memory specification")
            require(all(positive(v) for k, v in spec.items() if k != "latency") and
                    type(spec["latency"]) is int and spec["latency"] >= 0, "Memory parameters")
        require(not mem["shared_read_write"] or mem["read"] == mem["write"], "Shared ports require identical read/write spec")


class Graph:
    def __init__(self):
        self.tasks = []
        self.scalar_ops = Counter()
        self.words = Counter()

    def task(self, name, resource, units, parents=(), scalar_category=None):
        require(positive(units), "Empty/noninteger task")
        index = len(self.tasks)
        parents = tuple(sorted(set(p for p in parents if p is not None)))
        require(all(0 <= p < index for p in parents), "Noncausal DAG")
        self.tasks.append(dict(name=name, resource=resource, units=units, parents=parents))
        if scalar_category: self.scalar_ops[scalar_category] += units
        if resource.startswith(("external_", "local_")): self.words[resource] += units
        return index


def channel_graph(samples, contiguous_span, cfg, channels_per_group=1):
    """Independent BN domains in one physical layout group; no free transpose."""
    validate_config(cfg)
    require(positive(samples) and positive(contiguous_span) and samples % contiguous_span == 0, "Domain geometry")
    group = channels_per_group
    require(positive(group) and cfg["chunk_words"] >= group, "Working buffer cannot hold one grouped sample")
    require(cfg["scalar_register_words"] >= group*(8+cfg["buffer_slots"]), "Grouped reduction registers exceed capacity")
    resident = cfg["retention"] != "spill" and samples*group + cfg["chunk_words"] * cfg["buffer_slots"] <= cfg["local_capacity_words"]
    require(cfg["retention"] != "require_channel_resident" or resident, "Channel input does not fit local memory")
    graph = Graph(); t = graph.task
    chunk_samples = cfg["chunk_words"]//group
    chunks = [min(chunk_samples, contiguous_span - offset)
              for _ in range(samples // contiguous_span)
              for offset in range(0, contiguous_span, chunk_samples)]
    coeff = t("gamma_beta_fetch", "external_read", 2*group)

    def vector(name, resource, count, dep, scalar_category, reads=None, writes=None):
        dep = t(name + ".scratch_read", "local_read", count if reads is None else reads, (dep,))
        dep = t(name, resource, count, (dep,), scalar_category)
        return t(name + ".scratch_write", "local_write", count if writes is None else writes, (dep,))

    def tree(n, dep, name):
        level = 0
        while n > 1:
            pairs = n // 2
            dep = vector(name + ".tree" + str(level), "add", pairs*group, dep, "add", reads=2*pairs*group, writes=pairs*group)
            n = (n + 1) // 2; level += 1
        return t(name + ".partial_to_register", "local_read", group, (dep,))

    def traversal(label, barrier, kind):
        retired = []; accumulated = None
        for index, n in enumerate(chunks):
            previous_slot = retired[index - cfg["buffer_slots"]] if index >= cfg["buffer_slots"] else None
            deps = (barrier, previous_slot)
            source = "local_read" if resident and kind != "sum" else "external_read"
            words = n*group
            dep = t(label + ".load." + str(index), source, words, deps)
            # A retained input and destructive working buffer are distinct.
            dep = t(label + ".buffer_fill." + str(index), "local_write", words * (2 if resident and kind == "sum" else 1), (dep,))
            if kind == "variance":
                dep = vector(label + ".center." + str(index), "add", words, dep, "add")
                dep = vector(label + ".square." + str(index), "multiply", words, dep, "multiply")
            if kind in ("sum", "variance"):
                dep = tree(n, dep, label + ".reduce." + str(index))
                if accumulated is not None:
                    dep = t(label + ".accumulate." + str(index), "add", group, (dep, accumulated), "add")
                accumulated = dep
            else:
                dep = vector(label + ".scale." + str(index), "multiply", words, dep, "multiply")
                dep = vector(label + ".offset." + str(index), "add", words, dep, "add")
                dep = t(label + ".output_read." + str(index), "local_read", words, (dep,))
                dep = t(label + ".output_write." + str(index), "external_write", words, (dep,))
            retired.append(dep)
        return accumulated if accumulated is not None else retired

    total = traversal("sum", None, "sum")
    mean = t("mean_ready", "multiply", group, (total,), "multiply")
    variance = traversal("variance", mean, "variance")
    var_scale = t("variance_ready", "multiply", group, (variance,), "multiply")
    epsilon = t("variance_plus_eps", "add", group, (var_scale,), "add")
    invstd = t("rsqrt_ready", "rsqrt", group, (epsilon,), "rsqrt")
    scale = t("affine_scale_ready", "multiply", group, (invstd, coeff), "multiply")
    mu_scale = t("mean_times_scale", "multiply", group, (scale, mean), "multiply")
    affine = t("affine_offset_ready", "add", group, (mu_scale, coeff), "add")
    outputs = traversal("normalize", affine, "normalize")
    require(graph.scalar_ops == Counter(add=4*samples*group, multiply=(2*samples+4)*group, rsqrt=group), "Mathematical service mismatch")
    return graph, dict(samples=samples, contiguous_span=contiguous_span, channels_per_group=group, chunks=len(chunks), resident=resident,
        local_reserved_words=cfg["chunk_words"]*cfg["buffer_slots"]+(samples*group if resident else 0),
        barriers=dict(mean=mean, variance=var_scale, rsqrt=invstd, affine=affine), output_tasks=outputs)


def resources(cfg):
    result = {key: Resource(key, cfg[key], "vector") for key in ("add", "multiply", "rsqrt")}
    for name in ("external", "local"):
        mem = cfg[name]
        read = Resource(name + ("_shared" if mem["shared_read_write"] else "_read"), mem["read"], "memory")
        result[name + "_read"] = read
        result[name + "_write"] = read if mem["shared_read_write"] else Resource(name+"_write",mem["write"],"memory")
    return result


def run_graph(graph, cfg, trace_limit=0):
    """Jump between issue/retirement events; no per-cycle busy polling."""
    units = resources(cfg); tasks = graph.tasks; children = [[] for _ in tasks]
    pending = [len(task["parents"]) for task in tasks]
    for i, task in enumerate(tasks):
        for parent in task["parents"]: children[parent].append(i)
    ready = {i for i, n in enumerate(pending) if n == 0}
    events = []; now = done_count = 0; starts = [None]*len(tasks); ends = [None]*len(tasks)
    trace = []
    while done_count < len(tasks):
        while events and events[0][0] <= now:
            end, i = heapq.heappop(events); ends[i] = end; done_count += 1
            for child in children[i]:
                pending[child] -= 1
                if pending[child] == 0: ready.add(child)
        if not ready:
            require(events or done_count == len(tasks), "DAG deadlock")
            if events: now = events[0][0]
            continue
        start, i = min((units[tasks[i]["resource"]].earliest(now), i) for i in ready)
        if events and events[0][0] <= start:
            now = events[0][0]; continue
        now = start; ready.remove(i); task = tasks[i]
        end = units[task["resource"]].dispatch(now,task["units"]); starts[i] = now
        heapq.heappush(events,(end,i))
        if len(trace) < trace_limit:
            trace.append(dict(task_id=i,name=task["name"],resource=task["resource"],units=task["units"],
                              parents=list(task["parents"]),issue=now,retire=end))
    unique = {resource.name: resource for resource in units.values()}
    finish = max([max(ends)] + [max(r.ports) for r in unique.values()])
    for i, task in enumerate(tasks):
        require(all(starts[i] >= ends[p] for p in task["parents"]), "Dependency violation")
    return dict(cycles=finish,task_count=len(tasks),starts=starts,ends=ends,trace=trace,
        resource_usage={name: dict(issue_occupied_port_cycles=r.busy,jobs=r.jobs,units=r.units,
                                   peak_outstanding=r.peak,last_retirement=r.last_done) for name,r in unique.items()})


def simulate_channel(samples, contiguous_span, cfg, trace_limit=0, channels_per_group=1):
    graph, layout = channel_graph(samples,contiguous_span,cfg,channels_per_group)
    result = run_graph(graph,cfg,trace_limit)
    result["barrier_cycles"] = {name:result["ends"][i] for name,i in layout["barriers"].items()}
    result["first_normalize_input_issue"] = min(result["starts"][i] for i,t in enumerate(graph.tasks) if t["name"].startswith("normalize.load."))
    result["last_output_retire"] = max(result["ends"][i] for i in layout["output_tasks"])
    require(result["first_normalize_input_issue"] >= result["barrier_cycles"]["affine"], "BN affine barrier")
    result.update(layout=layout,scalar_ops=dict(graph.scalar_ops),transferred_words=dict(graph.words))
    del result["starts"]; del result["ends"]
    return result


def observed_layout(record):
    shape=record["native_input_shape"]; axis=record["channel_axis"]
    call=record["observed"]["input_calls"][0]; stride=call["stride"]
    require(call["shape"]==shape and len(shape)==len(stride),"Observed stride shape")
    channels=record["channels"]
    def matches(order):
        expected=1
        for dim in reversed(order):
            if shape[dim]>1 and stride[dim]!=expected:return False
            expected*=shape[dim]
        return True
    if matches(list(range(len(shape)))):return "native_channel_major",1
    if matches([i for i in range(len(shape)) if i!=axis]+[axis]):return "native_channel_contiguous",channels
    raise ValueError("Observed strided layout is unsupported; no implicit gather/transpose: "+record["name"])


def run_inventory(cfg, source=SOURCE, trace_limit=0):
    validate_config(cfg); require(sha(source)==SOURCE_SHA,"Pinned BN inventory changed")
    inventory=json.loads(Path(source).read_text())["external_BN"]
    require(inventory["inventory"]==54 and inventory["total_elements"]==976584960,"BN inventory")
    memo={}; records=[]; scalar=Counter(); traffic=Counter(); total=0
    for record in inventory["records"]:
        shape=record["native_input_shape"]; axis=record["channel_axis"]; span=1
        for dim in shape[axis+1:]: span*=dim
        samples=record["reduction_elements_per_channel"]; channels=record["channels"]
        require(samples*channels==record["elements"] and shape[axis]==channels,"BN native shape")
        layout,group=observed_layout(record); groups=channels//group
        key=(samples,span,group)
        if key not in memo: memo[key]=simulate_channel(samples,span,cfg,trace_limit,channels_per_group=group)
        channel=memo[key]; cycles=channel["cycles"]*groups; total+=cycles
        for name,n in channel["scalar_ops"].items(): scalar[name]+=groups*n
        for name,n in channel["transferred_words"].items(): traffic[name]+=groups*n
        records.append(dict(name=record["name"],family=record["family"],shape=shape,channel_axis=axis,channels=channels,
                            samples_per_channel=samples,contiguous_span=span,group_cycles=channel["cycles"],
                            observed_strides=record["observed"]["input_calls"][0]["stride"],layout=layout,
                            channels_per_group=group,serialized_groups=groups,
                            serialized_BN_cycles=cycles,geometry_key="{}:{}:{}".format(*key),
                            input_ready_assumption="entire input already exists in external memory at cycle0",
                            output_ready_after="all groups normalized and written; groups deliberately serialized"))
    require(scalar==Counter(add=3906339840,multiply=1953246144,rsqrt=19056),"Inventory arithmetic differs")
    return dict(schema="flow_currentstat_bn_event_sensitivity_v1",status="COMPLETE_PARAMETERIZED_REFERENCE",
        source_sha256={str(source):sha(source),str(Path(__file__)):sha(__file__)},parameters=cfg,
        BN_count=54,channel_domains=19056,scalar_ops=dict(scalar),transferred_words=dict(traffic),
        serialized_isolated_BN_cycle_sum=total,records=records,
        unique_channel_schedules={"{}:{}:{}".format(*k):v for k,v in memo.items()},
        full_network_cycles=None,measured_chip_cycles=None,RTL_validated=False,native_FP32_equivalence_validated=False,
        limits=["Functional population-BN graph retained; tree/chunk reduction and rsqrt precision have no numerical acceptance yet",
                "Native channel-major channels and BN modules serialized; native channel-contiguous domains use full-C groups; not an optimal-engine lower bound",
                "Each native contiguous HW span splits bursts; banking, DRAM rows and bus alignment are not characterized",
                "Input producer, private BN, MLP/attention/conv/decoder/residual contention and output consumer are not scheduled",
                "Dedicated add/mul/rsqrt resource sharing is only as declared; no full-engine area/frequency/power claim",
                "No MLP static issue count is added to these cycles; no quality or net-PE conclusion"])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trace-limit",type=int,default=0)
    args=parser.parse_args(argv)
    require(not args.output.exists(),"New output path required")
    require(0<=args.trace_limit<=200,"Trace limit0..200")
    cfg=json.loads(args.config.read_text()); result=run_inventory(cfg,trace_limit=args.trace_limit)
    result["source_sha256"][str(args.config)]=sha(args.config)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(status=result["status"],BN_count=54,cycles=result["serialized_isolated_BN_cycle_sum"],output=str(args.output))))


if __name__=="__main__":main()
