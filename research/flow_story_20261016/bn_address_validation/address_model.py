#!/usr/bin/env python3
"""Address/port reference for a ping-pong grouped reduction, not a BN engine.

Standard library, integer data path. Timings are declared synthetic parameters.
Every odd carry is a real read/copy/write; no implicit compaction or transpose.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'flow_story_20261014/damage_allocation_probe/hardware_contract_complete_bn_audit.json'
SOURCE_SHA = '9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_config():
    return dict(schema='flow_reduction_address_parameters_v1', parameter_status='synthetic_only',
                chunk_words=4096, local_capacity_words=8192, register_words=96,
                banks=16, read_ports_per_bank=1, write_ports_per_bank=1,
                shared_read_write=False, read_latency=2, write_latency=1,
                add_lanes=8, add_ii=1, add_latency=3, copy_lanes=8,
                copy_latency=1, max_active_operations=32)


def validate_config(cfg):
    require(set(cfg) == set(default_config()), 'Unexpected or missing configuration key')
    require(cfg['schema'] == 'flow_reduction_address_parameters_v1', 'Schema')
    require(cfg['parameter_status'] == 'synthetic_only', 'No characterized hardware is provided')
    for key, value in cfg.items():
        if key not in ('schema', 'parameter_status', 'shared_read_write'):
            require(type(value) is int and value > 0, 'Positive integer ' + key)
    require(type(cfg['shared_read_write']) is bool, 'Explicit port mode')
    require(not cfg['shared_read_write'] or cfg['read_ports_per_bank'] == cfg['write_ports_per_bank'], 'Shared port counts must match')
    require(cfg['local_capacity_words'] >= 2 * cfg['chunk_words'], 'Two distinct scratch regions must fit')
    require(cfg['register_words'] >= 3 * cfg['max_active_operations'], 'Three reserved operand/result words per active operation')


def sample_value(sample, channel):
    return ((sample * 17 + channel * 31) % 257) - 128


def token(value, channel, coverage):
    return (value, channel, coverage)


class Machine:
    """Literal per-tick scheduler. Level barriers make buffer reuse explicit."""
    def __init__(self, cfg):
        validate_config(cfg)
        self.cfg = cfg
        self.memory = {}
        self.now = 0
        self.trace = []
        self.jobs = []
        self.levels = []
        self.peak_active = 0
        self.stalls = Counter()
        self.add_lanes_available = [0] * cfg['add_lanes']

    def _event(self, kind, op, latency, **kwargs):
        event = dict(event_id=len(self.trace), kind=kind, operation=op['id'],
                     phase=op['phase'], issue=self.now, retire=self.now + latency)
        event.update(kwargs)
        self.trace.append(event)
        return event

    def run_jobs(self, jobs, phase):
        """All jobs retire before return; no cross-level free storage movement."""
        start = self.now
        unread = Counter(addr for job in jobs for addr in job.get('src', []))
        inflight_reads = Counter()
        queue = []
        for job in jobs:
            job = dict(job, id=len(self.jobs), phase=phase)
            self.jobs.append(dict(job))
            queue.append(dict(job, issued_reads=0, values={}, state='new', compute_event=None,
                              result=job.get('value'), result_event=None))
        active, pending = [], []
        head = done = 0
        while done < len(jobs):
            # All retirements precede this tick's new issues.
            still_pending = []
            for event, job in pending:
                if event['retire'] > self.now:
                    still_pending.append((event, job))
                    continue
                kind = event['kind']
                if kind == 'read':
                    inflight_reads[event['address']] -= 1
                    job['values'][event['operand']] = tuple(event['data'])
                elif kind in ('add', 'copy'):
                    job['result'] = tuple(event['data'])
                    job['result_event'] = event['event_id']
                elif kind == 'write':
                    self.memory[event['address']] = tuple(event['data'])
                    job['state'] = 'done'
                    self.jobs[job['id']]['retire'] = self.now
                    done += 1
                elif kind == 'export':
                    inflight_reads[event['address']] -= 1
                    job['result'] = tuple(event['data'])
                    job['state'] = 'done'
                    self.jobs[job['id']]['retire'] = self.now
                    done += 1
            pending = still_pending
            active = [job for job in active if job['state'] != 'done']
            while head < len(queue) and len(active) < self.cfg['max_active_operations']:
                self.jobs[queue[head]['id']]['admit'] = self.now
                active.append(queue[head]); head += 1
            self.peak_active = max(self.peak_active, len(active))
            read_used, write_used, shared_used = Counter(), Counter(), Counter()
            copy_used = 0

            def claim(direction, address):
                require(0 <= address < self.cfg['local_capacity_words'], 'Address exceeds finite SRAM')
                bank = address % self.cfg['banks']
                if self.cfg['shared_read_write']:
                    if shared_used[bank] >= self.cfg['read_ports_per_bank']:
                        self.stalls['shared_bank_port_requests_denied'] += 1
                        return False
                    shared_used[bank] += 1
                else:
                    used = read_used if direction == 'read' else write_used
                    if used[bank] >= self.cfg[direction + '_ports_per_bank']:
                        self.stalls[direction + '_bank_port_requests_denied'] += 1
                        return False
                    used[bank] += 1
                return True

            # Issue completed results first. This policy is declared, not optimal.
            for job in active:
                if job['state'] == 'writing' or job['kind'] == 'export' or job['result'] is None:
                    continue
                address = job['dst']
                require(unread[address] == 0 and inflight_reads[address] == 0,
                        'Read-before-overwrite violation at address ' + str(address))
                if claim('write', address):
                    event = self._event('write', job, self.cfg['write_latency'], address=address,
                                        bank=address % self.cfg['banks'], row=address // self.cfg['banks'],
                                        data=list(job['result']), parent=job['result_event'])
                    pending.append((event, job)); job['state'] = 'writing'
            for job in active:
                if job['state'] in ('writing', 'computing') or job['result'] is not None:
                    continue
                if job['kind'] == 'export':
                    if job['state'] == 'reading':
                        continue
                    address = job['src'][0]
                    if claim('read', address):
                        require(address in self.memory, 'Uninitialized export')
                        event = self._event('export', job, self.cfg['read_latency'], address=address,
                                            bank=address % self.cfg['banks'], row=address // self.cfg['banks'],
                                            data=list(self.memory[address]))
                        unread[address] -= 1; inflight_reads[address] += 1
                        pending.append((event, job)); job['state'] = 'reading'
                    continue
                if len(job['values']) == len(job.get('src', [])):
                    values = [job['values'][i] for i in range(len(job['src']))]
                    if job['kind'] == 'pair':
                        free = [i for i, ready in enumerate(self.add_lanes_available) if ready <= self.now]
                        if not free:
                            self.stalls['add_lane_requests_denied'] += 1
                            continue
                        left, right = values
                        require(left[1] == right[1] and left[2] & right[2] == 0, 'Mixed domain or duplicate coverage')
                        result = token(left[0] + right[0], left[1], left[2] | right[2])
                        lane = free[0]; self.add_lanes_available[lane] = self.now + self.cfg['add_ii']
                        event = self._event('add', job, self.cfg['add_latency'], data=list(result), lane=lane)
                    else:
                        require(job['kind'] == 'carry', 'Unexpected compute job')
                        if copy_used >= self.cfg['copy_lanes']:
                            self.stalls['copy_lane_requests_denied'] += 1
                            continue
                        event = self._event('copy', job, self.cfg['copy_latency'], data=list(values[0]), lane=copy_used)
                        copy_used += 1
                    pending.append((event, job)); job['state'] = 'computing'
                else:
                    for operand in range(job['issued_reads'], len(job['src'])):
                        address = job['src'][operand]
                        if not claim('read', address):
                            break
                        require(address in self.memory, 'Read before initialization')
                        event = self._event('read', job, self.cfg['read_latency'], address=address,
                                            bank=address % self.cfg['banks'], row=address // self.cfg['banks'],
                                            operand=operand, data=list(self.memory[address]))
                        unread[address] -= 1; inflight_reads[address] += 1
                        pending.append((event, job)); job['issued_reads'] += 1
            if done < len(jobs):
                self.now += 1
        require(not pending and not any(unread.values()) and not any(inflight_reads.values()), 'Unretired level')
        self.levels.append(dict(phase=phase, start=start, retire=self.now, jobs=len(jobs)))
        return queue


def reduce_group(samples, channels, cfg=None, sample_start=0, channel_start=0, values=None):
    cfg = cfg or default_config()
    validate_config(cfg)
    require(type(samples) is int and samples > 0 and type(channels) is int and channels > 0, 'Geometry')
    require(samples * channels <= cfg['chunk_words'], 'Chunk exceeds scratch region')
    if values is None:
        values = [[sample_value(sample_start + s, channel_start + c) for c in range(channels)] for s in range(samples)]
    require(len(values) == samples and all(len(row) == channels for row in values), 'Value shape')
    require(all(type(v) is int for row in values for v in row), 'Integer golden only, no FP32 equivalence')
    machine = Machine(cfg)
    machine.run_jobs([dict(kind='load', src=[], dst=s * channels + c,
                           value=token(values[s][c], c, 1 << s))
                      for s in range(samples) for c in range(channels)], 'load')
    n, source, destination, level = samples, 0, cfg['chunk_words'], 0
    while n > 1:
        jobs = [dict(kind='pair', src=[source + 2 * p * channels + c, source + (2 * p + 1) * channels + c],
                     dst=destination + p * channels + c)
                for p in range(n // 2) for c in range(channels)]
        if n % 2:
            jobs.extend(dict(kind='carry', src=[source + (n - 1) * channels + c],
                             dst=destination + (n // 2) * channels + c) for c in range(channels))
        machine.run_jobs(jobs, 'reduce_' + str(level))
        source, destination = destination, source
        n = (n + 1) // 2; level += 1
    exports = machine.run_jobs([dict(kind='export', src=[source + c]) for c in range(channels)], 'export')
    result = [job['result'][0] for job in exports]
    expected = [sum(values[s][c] for s in range(samples)) for c in range(channels)]
    require(result == expected, 'Independent integer sum golden mismatch')
    for c, job in enumerate(exports):
        require(job['result'][1:] == (c, (1 << samples) - 1), 'Missing/duplicated samples or mixed channels')
    counts = Counter(event['kind'] for event in machine.trace)
    require(counts['add'] == (samples - 1) * channels, 'Addition count')
    return dict(schema='flow_reduction_address_run_v1', samples=samples, channels=channels,
                sample_start=sample_start, channel_start=channel_start, parameters=cfg,
                synthetic_component_cycles=machine.now, output=result, expected=expected,
                transaction_counts=dict(counts), logical_external_input_words=samples * channels,
                logical_external_output_words=channels, external_transfer_cycles=None,
                peak_active_operations=machine.peak_active, reserved_register_words=3 * cfg['max_active_operations'],
                reserved_local_words=2 * cfg['chunk_words'], bank_conflict_requests=dict(machine.stalls),
                levels=machine.levels, jobs=machine.jobs, trace=machine.trace,
                native_FP32_equivalence_validated=False, RTL_validated=False, full_network_cycles=None)


def validate_trace(result):
    """Independent transaction replay: port limits, lifetime, operands and values.

    Does not invoke the scheduler. A deliberately corrupted trace must fail.
    """
    cfg, trace, jobs = result['parameters'], result['trace'], result['jobs']
    validate_config(cfg)
    ports = Counter(); add_slots = {}; copy_slots = Counter()
    reads = {}; compute = {}; writes = {}; memory = {}; pending_reads = Counter()
    unread_by_phase = {}
    levels = {level['phase']: level for level in result['levels']}
    for job in jobs:
        unread_by_phase.setdefault(job['phase'], Counter()).update(job['src'])
    event_order = []; events_by_op = {}
    for event in trace:
        events_by_op.setdefault(event['operation'], []).append(event)
        require(event['retire'] > event['issue'], 'Nonpositive latency')
        require(event['phase'] == jobs[event['operation']]['phase'], 'Transaction changed phase')
        bounds = levels[event['phase']]
        require(bounds['start'] <= event['issue'] < event['retire'] <= bounds['retire'], 'Transaction crossed level barrier')
        event_order.extend([(event['issue'], 1, event['event_id'], event), (event['retire'], 0, event['event_id'], event)])
        kind = event['kind']; op = event['operation']
        if kind in ('read', 'write', 'export'):
            address = event['address']; bank = address % cfg['banks']
            require(0 <= address < cfg['local_capacity_words'], 'Out of capacity')
            require(event['bank'] == bank and event['row'] == address // cfg['banks'], 'Bank/row mismatch')
            direction = 'write' if kind == 'write' else 'read'
            port_kind = 'shared' if cfg['shared_read_write'] else direction
            key = (event['issue'], bank, port_kind); ports[key] += 1
            require(ports[key] <= cfg[direction + '_ports_per_bank'], 'Port collision')
            require(event['retire'] - event['issue'] == cfg[direction + '_latency'], 'Memory latency')
            if kind in ('read', 'export'): reads.setdefault(op, []).append(event)
            else: require(op not in writes, 'Duplicate write'); writes[op] = event
        elif kind == 'add':
            require(0 <= event['lane'] < cfg['add_lanes'], 'Add lane')
            for tick in range(event['issue'], event['issue'] + cfg['add_ii']):
                key = (tick, event['lane']); require(key not in add_slots, 'Add II collision'); add_slots[key] = op
            require(event['retire'] - event['issue'] == cfg['add_latency'], 'Add latency')
            require(op not in compute, 'Duplicate compute'); compute[op] = event
        elif kind == 'copy':
            copy_slots[event['issue']] += 1
            require(copy_slots[event['issue']] <= cfg['copy_lanes'], 'Copy lane collision')
            require(event['retire'] - event['issue'] == cfg['copy_latency'], 'Copy latency')
            require(op not in compute, 'Duplicate compute'); compute[op] = event
        else:
            raise ValueError('Unknown transaction')
    # Replay memory at issue and retirement; pending reads prohibit writes.
    for tick, edge, event_id, event in sorted(event_order):
        kind = event['kind']; address = event.get('address')
        if edge == 1 and kind in ('read', 'export'):
            require(address in memory and list(memory[address]) == event['data'], 'Read value/source version mismatch')
            unread_by_phase[event['phase']][address] -= 1
            pending_reads[address] += 1
        elif edge == 1 and kind == 'write':
            require(pending_reads[address] == 0, 'Write while read outstanding')
            require(unread_by_phase[event['phase']][address] == 0, 'Write before a required read was issued')
        elif edge == 0 and kind in ('read', 'export'):
            pending_reads[address] -= 1
        elif edge == 0 and kind == 'write':
            require(pending_reads[address] == 0, 'Retiring write over outstanding read')
            memory[address] = tuple(event['data'])
    # Every declared source and destination, and every arithmetic/carry result.
    active_intervals = []
    for job in jobs:
        op = job['id']; rr = reads.get(op, [])
        own_events = events_by_op[op]
        require(job['admit'] <= min(event['issue'] for event in own_events), 'Operation used registers before admission')
        require(job['retire'] == max(event['retire'] for event in own_events), 'Registers freed before final operation retirement')
        require(sorted(e['address'] for e in rr) == sorted(job['src']), 'Missing or duplicated operand transaction')
        if job['kind'] == 'export':
            require(op not in writes and op not in compute, 'Export has side effect')
            active_intervals.append((job['admit'], job['retire']))
            continue
        require(op in writes and writes[op]['address'] == job['dst'], 'Missing or wrong destination')
        wr = writes[op]
        if job['kind'] == 'load':
            require(not rr and op not in compute and wr['data'] == list(job['value']), 'Load changed input')
            active_intervals.append((job['admit'], job['retire']))
        else:
            require(op in compute, 'No add/copy for operation')
            cc = compute[op]
            require(all(e['retire'] <= cc['issue'] for e in rr), 'Compute before operands retire')
            require(cc['retire'] <= wr['issue'], 'Write before compute retires')
            require(wr['parent'] == cc['event_id'], 'Wrong result dependency')
            rr.sort(key=lambda e: e['operand'])
            if job['kind'] == 'pair':
                left, right = [e['data'] for e in rr]
                require(cc['kind'] == 'add' and left[1] == right[1] and left[2] & right[2] == 0, 'Pair domain/coverage')
                expected = [left[0] + right[0], left[1], left[2] | right[2]]
            else:
                require(job['kind'] == 'carry' and cc['kind'] == 'copy', 'Carry is explicit copy')
                expected = rr[0]['data']
            require(cc['data'] == expected and wr['data'] == expected, 'Arithmetic or carry value mismatch')
            active_intervals.append((job['admit'], job['retire']))
    # Register reservations include admitted operations waiting for bank ports.
    edges = []
    for start, end in active_intervals: edges.extend([(start, 1), (end, -1)])
    active = peak = 0
    for _, change in sorted(edges):
        active += change; peak = max(peak, active)
    require(peak <= cfg['max_active_operations'], 'Too many live operations')
    require(3 * peak <= cfg['register_words'], 'Register capacity exceeded')
    require(peak == result['peak_active_operations'], 'Scheduler register occupancy disagrees with independent replay')
    for previous, current in zip(result['levels'], result['levels'][1:]):
        require(current['start'] >= previous['retire'], 'Buffer reused before level barrier')
    require(not any(pending_reads.values()), 'Pending read not retired')
    require(all(not any(v.values()) for v in unread_by_phase.values()), 'Unissued required read')
    return dict(status='PASS_INDEPENDENT_TRANSACTION_REPLAY', events=len(trace), jobs=len(jobs),
                checked_bank_cycles=len(ports), exact_peak_reserved_operations=peak,
                scheduler_peak_active_operations=result['peak_active_operations'])


def observed_layout(record):
    shape = record['native_input_shape']; axis = record['channel_axis']
    call = record['observed']['input_calls'][0]; stride = call['stride']
    require(call['shape'] == shape, 'Shape/stride source mismatch')
    def matches(order):
        expected = 1
        for dim in reversed(order):
            if shape[dim] > 1 and stride[dim] != expected: return False
            expected *= shape[dim]
        return True
    if matches(list(range(len(shape)))): return 'native_channel_major', 1
    if matches([i for i in range(len(shape)) if i != axis] + [axis]): return 'native_channel_contiguous', record['channels']
    raise ValueError('Unsupported native layout; no free transpose')


def source_address(record, sample, channel):
    shape = record['native_input_shape']; axis = record['channel_axis']
    stride = record['observed']['input_calls'][0]['stride']
    require(0 <= sample < record['reduction_elements_per_channel'] and 0 <= channel < record['channels'], 'Source index')
    address = channel * stride[axis]
    for dim in reversed([i for i in range(len(shape)) if i != axis]):
        coordinate = sample % shape[dim]; sample //= shape[dim]
        address += coordinate * stride[dim]
    require(sample == 0 and 0 <= address < record['elements'], 'Native source address')
    return address


def run_inventory(cfg, source=SOURCE):
    validate_config(cfg); require(sha(source) == SOURCE_SHA, 'Pinned actual BN ledger changed')
    inventory = json.loads(Path(source).read_text())['external_BN']
    require(inventory['inventory'] == 54 and inventory['total_elements'] == 976584960, 'Actual inventory')
    schedules, records = {}, []
    for record in inventory['records']:
        layout, group = observed_layout(record)
        require(group <= cfg['chunk_words'], 'Cannot hold one complete channel-contiguous sample')
        shape, axis = record['native_input_shape'], record['channel_axis']
        span = 1
        for dim in shape[axis + 1:]: span *= dim
        samples = record['reduction_elements_per_channel']
        require(samples % span == 0, 'Native span divides domain')
        chunk_samples = cfg['chunk_words'] // group
        sizes = sorted(set(min(chunk_samples, span - offset) for offset in range(0, span, chunk_samples)))
        for count in sizes:
            key = '{}:{}'.format(count, group)
            if key not in schedules:
                result = reduce_group(count, group, cfg)
                replay = validate_trace(result)
                schedules[key] = dict(samples=count, channels=group,
                    synthetic_component_cycles=result['synthetic_component_cycles'],
                    transaction_counts=result['transaction_counts'], bank_conflict_requests=result['bank_conflict_requests'],
                    peak_active_operations=result['peak_active_operations'], replay=replay,
                    trace_sha256=hashlib.sha256(json.dumps(result['trace'], sort_keys=True).encode()).hexdigest())
        # Verify each real chunk shape at first and last native span, and first/last
        # channel-major domains. Address enumeration is bounded, not a tensor load.
        address_cases = []
        spans_to_check = sorted(set([0, samples - span]))
        channels_to_check = [0] if group > 1 else sorted(set([0, record['channels'] - 1]))
        offsets = sorted(set([0, ((span - 1) // chunk_samples) * chunk_samples]))
        for base in spans_to_check:
            for offset in offsets:
                count = min(chunk_samples, span - offset)
                for channel_base in channels_to_check:
                    addresses = [source_address(record, base + offset + s, channel_base + c)
                                 for s in range(count) for c in range(group)]
                    require(len(set(addresses)) == len(addresses), 'Duplicated native source word')
                    require(all(b == a + 1 for a, b in zip(addresses, addresses[1:])), 'Claimed contiguous segment is not contiguous')
                    address_cases.append(dict(sample_start=base + offset, samples=count, channel_start=channel_base,
                        group=group, first_native_word_address=addresses[0], last_native_word_address=addresses[-1],
                        input_words=len(addresses), local_load_destination_range=[0, len(addresses)-1],
                        native_address_sha256=hashlib.sha256(json.dumps(addresses).encode()).hexdigest()))
        records.append(dict(name=record['name'], shape=shape, stride=record['observed']['input_calls'][0]['stride'],
            channel_axis=axis, channels=record['channels'], layout=layout, channels_per_group=group,
            reduction_samples_per_channel=samples, native_contiguous_span=span, chunk_samples=chunk_samples,
            chunk_size_classes=sizes, schedule_keys=['{}:{}'.format(count, group) for count in sizes],
            checked_native_address_cases=address_cases))
    return dict(schema='flow_bn_address_inventory_v1', status='COMPLETE_ADDRESS_COMPONENT_VALIDATION',
        source_sha256={str(source): sha(source), str(Path(__file__)): sha(__file__)}, parameters=cfg,
        actual_BN_count=54, actual_channel_domains=19056, actual_input_elements=976584960,
        unique_reduction_chunk_shapes=len(schedules), records=records, schedules=schedules,
        full_BN_cycles=None, full_network_cycles=None, calibrated_hardware_cycles=None,
        native_FP32_equivalence_validated=False, RTL_validated=False,
        limits=['Only local chunk reduction is scheduled, not the full BN mathematical graph.',
                'Native input address enumeration covers first/last spans and full/tail classes; not all 976584960 addresses.',
                'Integer exact reduction validates addresses and domain conservation, not native FP32 reduction order.',
                'External data availability is assumed; logical input/output words are recorded but DMA latency is not modeled.',
                'Input buffering, interchunk accumulators, variance/rsqrt/normalize, producer/consumer and whole-engine contention remain open.',
                'Synthetic SRAM ports/add/copy timings are not SRAM macro or RTL/DC characterization.',
                'Same reduction implementation must be offered to all dense/static/dynamic baselines; no policy speedup is reported.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output must be new')
    cfg = json.loads(args.config.read_text()) if args.config else default_config()
    result = run_inventory(cfg)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=result['status'], BN=result['actual_BN_count'], chunk_shapes=result['unique_reduction_chunk_shapes'])))


if __name__ == '__main__': main()
