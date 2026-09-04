#!/usr/bin/env python3
"""M1556 successor source for one ep34 decoder non-product calibration call.

M1549 is deliberately a *source* gate.  It binds the independently hammered
M1539 request kernel, fixes the calibration population to call zero (D0,
sample 10, all ten timesteps), and demonstrates a bounded-memory execution
shape.  It does not expose an actual-payload pilot CLI and cannot launch the
120-call production population.

The future launch hammer may import :func:`stream_actual_call` after pinning
this source byte-for-byte.  That is the only executable entry point: it accepts
only a non-product configuration and opens the canonical call internally.
There is deliberately no public helper that accepts a plane, module or call
ordinal.  At clean import it snapshots call-0's member, shape and SHA directly
into the entry closure.  The implementation hashes and fstat's the actual
opened descriptor, copies its compact bit plane into immutable bytes, closes
the descriptor, emits and schedules requests destination by destination, and
retires dependency tokens after each destination.  Bank
calendars, outstanding queues, address digests, the nine-tile weight cache and
cycle state are never reset inside a call.

Python syntax is compatible with CPython 3.6.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1539_SOURCE = HERE / "build_m1539_ep34_decoder_nonproduct_address_timed_replay_successor_source.py"
M1539_SOURCE_SHA256 = "9acc4d316061b1791f0ad49793d2f2a7a79eb24fdf0d0c5867cde6648a64b4b4"
M1542 = HW / "reviews/m1542_m1539_decoder_nonproduct_address_timed_source_independent_hammer_r1_20260831"
M1542_REVIEW_SHA256 = "b85014ca32604b7b2659a7ba962bfb873bdb4c330dc011ff94d263ee6898c970"
M1542_OUTER_FILE_SHA256 = "3d1f38281d106b040340e56e210698fa245020eff874783702e8901718207a3a"

SCHEMA = "m1556_ep34_decoder_nonproduct_streaming_single_call_pilot_immutable_snapshot_source_r4_v1"
STATUS = "M1556_SOURCE_ONLY__CLOSURE_ROW_AND_IMMUTABLE_PLANE_SNAPSHOT__EXECUTION_AND_PRODUCTION_BLOCKED"
PILOT_CALL_ORDINAL = 0
PILOT_SAMPLE_ID = 10
PILOT_MODULE = 0
PILOT_TIMESTEPS = 10
PEAK_RSS_LIMIT_KIB = 8 * 1024 * 1024
FORBIDDEN_CONFIG = "PRODUCT_CAPTURE_TYPED_K8"


class M1543Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1543Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_open_stream(stream):
    """Hash the opened object which will back mmap, never a second pathname."""
    position = stream.tell()
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        digest.update(block)
    stream.seek(position)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise M1543Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " is not a regular file")
    require(sha256(path) == expected, label + " SHA drift")


def verify_flat_seal(path, expected_review, expected_outer):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "M1542 directory drift")
    review = path / "review.json"
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    regular_exact(review, expected_review, "M1542 review")
    regular_exact(outer, expected_outer, "M1542 outer seal")
    require(outer.read_text(encoding="utf-8").split() ==
            [sha256(manifest), "SHA256SUMS"], "M1542 outer content drift")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and fields[1] not in expected,
                "M1542 manifest malformed or duplicated")
        expected[fields[1]] = fields[0]
    for name, digest in expected.items():
        require("/" not in name and ".." not in name,
                "M1542 non-flat member")
        regular_exact(path / name, digest, "M1542 member " + name)
    actual = set(item.name for item in path.iterdir()
                 if item.is_file() and item.name not in
                 ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(actual == set(expected), "M1542 seal coverage drift")
    payload = json.loads(review.read_text(encoding="utf-8"))
    require(payload.get("status") ==
            "PASS_M1542_M1539_INDEPENDENT_SOURCE_HAMMER__DISTINCT_STREAMING_RUNNER_AUTHORING_ALLOWED__NO_PRODUCTION" and
            payload.get("authorization", {}).get(
                "distinct_streaming_runner_authoring") is True and
            payload.get("authorization", {}).get("production_execution") is False and
            payload.get("authorization", {}).get("product_configuration") is False,
            "M1542 authorization drift")
    return {"members": len(expected), "manifest_sha256": sha256(manifest)}


def load_m1539():
    regular_exact(M1539_SOURCE, M1539_SOURCE_SHA256, "M1539 source")
    spec = importlib.util.spec_from_file_location("m1543_bound_m1539", M1539_SOURCE)
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M1539")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(tuple(module.CONFIGS) == (
        "DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8") and
        module.FORBIDDEN_CONFIG == FORBIDDEN_CONFIG,
        "M1539 configuration boundary drift")
    return module


M = load_m1539()


def peak_rss_kib():
    # Linux reports ru_maxrss in KiB.  This project runs only on the Linux
    # Synopsys host; the explicit limit is part of the pilot contract.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def memory_gate():
    value = peak_rss_kib()
    require(value < PEAK_RSS_LIMIT_KIB,
            "streaming pilot exceeded the strict 8 GiB RSS limit")
    return value


class ImmutableLittleBitPlane(object):
    """Read a verified compact bit plane from an immutable byte snapshot."""
    def __init__(self, path, shape, expected_sha256=None):
        self.path = Path(path)
        self.expected_sha256 = expected_sha256
        require(len(shape) == 5 and int(shape[1]) == 1,
                "bit-plane shape must be T,1,C,H,W")
        self.timesteps = int(shape[0]); self.channels = int(shape[2])
        self.height = int(shape[3]); self.width = int(shape[4])
        self.elements = (self.timesteps * self.channels * self.height * self.width)
        self.bytes = (self.elements + 7) // 8
        try:
            path_stat = self.path.lstat()
        except FileNotFoundError as error:
            raise M1543Error("missing pilot payload") from error
        require(stat.S_ISREG(path_stat.st_mode) and not self.path.is_symlink(),
                "bad pilot payload file")
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(str(self.path), flags)
        except OSError as error:
            raise M1543Error("cannot safely open pilot payload") from error
        self._stream = os.fdopen(descriptor, "rb")
        opened_stat = os.fstat(self._stream.fileno())
        require(stat.S_ISREG(opened_stat.st_mode),
                "opened pilot payload is not regular")
        require((opened_stat.st_dev, opened_stat.st_ino, opened_stat.st_size) ==
                (path_stat.st_dev, path_stat.st_ino, path_stat.st_size),
                "pilot payload changed while opening")
        require(opened_stat.st_size == self.bytes,
                "pilot payload byte count drift")
        self.opened_device = int(opened_stat.st_dev)
        self.opened_inode = int(opened_stat.st_ino)
        self.opened_size = int(opened_stat.st_size)
        self.opened_sha256 = sha256_open_stream(self._stream)
        if expected_sha256 is not None:
            require(self.opened_sha256 == expected_sha256,
                    "opened pilot payload SHA drift")
        self._stream.seek(0)
        payload = self._stream.read()
        require(len(payload) == self.bytes and
                hashlib.sha256(payload).hexdigest() == self.opened_sha256,
                "immutable pilot snapshot drift")
        self._snapshot = bytes(payload)
        self._stream.close()
        self._stream = None

    def bit(self, timestep, channel, y, x):
        timestep = int(timestep); channel = int(channel)
        y = int(y); x = int(x)
        require(0 <= timestep < self.timesteps and
                0 <= channel < self.channels and
                0 <= y < self.height and 0 <= x < self.width,
                "bit-plane index out of range")
        index = (((timestep * self.channels + channel) * self.height + y) *
                 self.width + x)
        require(self._snapshot is not None, "pilot snapshot already closed")
        return (self._snapshot[index >> 3] >> (index & 7)) & 1

    def close(self):
        if self._snapshot is not None:
            self._snapshot = None
        if self._stream is not None:
            self._stream.close(); self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, _kind, _value, _traceback):
        self.close()


class StreamingCallScheduler(object):
    """Bounded-token facade over the frozen address/port scheduler."""
    def __init__(self, config):
        M.validate_config(config)
        self.config = config
        self.scheduler = M.AddressTimedScheduler(config)
        self.max_live_tokens = 0
        self.max_live_outstanding = 0
        self.destinations = 0
        self.timesteps = 0

    def one(self, row):
        receipt = self.scheduler.schedule_one(row)
        self.max_live_tokens = max(self.max_live_tokens,
                                   len(self.scheduler.tokens))
        self.max_live_outstanding = max(
            self.max_live_outstanding,
            sum(len(values) for values in self.scheduler.outstanding.values()))
        # getrusage is intentionally amortized: an actual D0 call emits a very
        # large request stream, so checking every row would measure syscall
        # overhead rather than the hardware schedule.  Destination retirement
        # also checks RSS below, bounding the interval much more tightly than
        # the 65,536-row fallback for normal execution.
        if (self.scheduler.requests & 0xffff) == 0:
            memory_gate()
        return receipt

    def retire_destination(self, retained_tokens):
        retained = set(retained_tokens)
        require(retained.issubset(set(self.scheduler.tokens)),
                "retained token was never produced")
        self.scheduler.tokens = dict((key, self.scheduler.tokens[key])
                                     for key in retained)
        self.destinations += 1
        memory_gate()

    def finish(self):
        require(self.scheduler.requests > 0 and self.destinations > 0,
                "empty streaming schedule")
        return {"configuration": self.config,
                "resource_manifest_sha256": M.validate_resource(),
                "total_cycles": self.scheduler.last_cycle + 1,
                "request_count": self.scheduler.requests,
                "kind_counts": dict(self.scheduler.kind_counts),
                "byte_counts": dict(self.scheduler.byte_counts),
                "transaction_address_sha256":
                    self.scheduler.address_digest.hexdigest(),
                "commit_sequence_sha256":
                    self.scheduler.commit_digest.hexdigest(),
                "streaming": {"materialized_transaction_list": False,
                    "destinations": self.destinations,
                    "timesteps": self.timesteps,
                    "max_live_dependency_tokens": self.max_live_tokens,
                    "max_live_outstanding_entries": self.max_live_outstanding,
                    "peak_rss_kib": peak_rss_kib(),
                    "peak_rss_limit_kib": PEAK_RSS_LIMIT_KIB}}


def selected_pilot_record():
    manifest = M.strict_json(M.M1521_MANIFEST)
    M.validate_population_manifest(manifest)
    row = manifest["records"][PILOT_CALL_ORDINAL]
    require(row["global_call_ordinal"] == PILOT_CALL_ORDINAL and
            row["global_sample_id"] == PILOT_SAMPLE_ID and
            row["module_ordinal"] == PILOT_MODULE and
            tuple(row["shape"]) == M.INPUT_SHAPES[PILOT_MODULE],
            "single-call pilot selection drift")
    return row


def _build_canonical_streamer():
    """Capture the verified types and kernel; expose no injectable plane seam."""
    bound_m = M
    plane_type = ImmutableLittleBitPlane
    scheduler_type = StreamingCallScheduler
    canonical_root = M.M1521_ROOT.resolve()
    module = PILOT_MODULE
    call_ordinal = PILOT_CALL_ORDINAL
    # Freeze values, not a selector function whose globals could be replaced
    # after import.  No future call consults selected_pilot_record or M1521_ROOT.
    frozen = selected_pilot_record()
    payload_member = str(frozen["positive_output"])
    payload_sha256 = str(frozen["positive_output_sha256"])
    payload_shape = tuple(int(value) for value in frozen["shape"])
    sample_id = int(frozen["global_sample_id"])
    frozen_module = int(frozen["module_ordinal"])
    frozen_call = int(frozen["global_call_ordinal"])
    canonical_path = (canonical_root / payload_member).resolve()
    require(canonical_path.parent == (canonical_root / "payloads").resolve() and
            sample_id == PILOT_SAMPLE_ID and frozen_module == module and
            frozen_call == call_ordinal,
            "clean-import pilot closure snapshot drift")

    def schedule_verified(config, plane):
        require(type(plane) is plane_type,
                "canonical streamer requires the exact immutable plane type")
        require(plane.path.resolve() == canonical_path and
                plane.expected_sha256 == payload_sha256 and
                plane.opened_sha256 == payload_sha256 and
                plane._stream is None and plane._snapshot is not None,
                "pilot snapshot is not bound to canonical opened bytes")
        cin, cout, hin, win, hout, wout = bound_m.GEOMETRY[module]
        require((plane.timesteps, plane.channels, plane.height, plane.width) ==
                (PILOT_TIMESTEPS, cin, hin, win),
                "pilot tensor geometry drift")
        runner = scheduler_type(config)
        cache = bound_m.WeightTileCache()
        output_blocks = (cout + 95) // 96
        persistent_control = "{}:c{}:control_done".format(config, call_ordinal)
        first_source = "{}:c{}:t0:source_done".format(config, call_ordinal)
        source_bytes = (cin * hin * win + 7) // 8
        runner.one(bound_m.request(
            "{}:c{}:t0:source".format(config, call_ordinal), config,
            "external_read", [(1 << 60) | (call_ordinal << 28)], [0],
            source_bytes, produces=first_source))
        control_read = "{}:c{}:control_read_done".format(config, call_ordinal)
        runner.one(bound_m.request(
            "{}:c{}:control_read".format(config, call_ordinal), config,
            "external_read", [(5 << 60) | (call_ordinal << 12)], [0], 144,
            [first_source], control_read))
        runner.one(bound_m.request(
            "{}:c{}:control_write".format(config, call_ordinal), config,
            "external_write", [(5 << 60) | (call_ordinal << 12)], [0], 144,
            [control_read], persistent_control))
        barrier = persistent_control
        for timestep in range(PILOT_TIMESTEPS):
            if timestep:
                barrier = "{}:c{}:t{}:source_done".format(
                    config, call_ordinal, timestep)
                runner.one(bound_m.request(
                    "{}:c{}:t{}:source".format(config, call_ordinal, timestep),
                    config, "external_read",
                    [(1 << 60) | (call_ordinal << 28) | (timestep << 20)],
                    [0], source_bytes, produces=barrier))
            getter = lambda channel, y, x, t=timestep: plane.bit(
                t, channel, y, x)
            for oy in range(hout):
                for ox in range(wout):
                    destination = oy * wout + ox
                    contributors = bound_m.contributors_for_destination(
                        getter, config, cin, hin, win, oy, ox)
                    for output_block in range(output_blocks):
                        last = ""
                        rows = bound_m.destination_transactions(
                            config, module, timestep, destination,
                            output_block, contributors, barrier, cache)
                        for request_row in rows:
                            runner.one(request_row)
                            if request_row["kind"] == "psum_write":
                                last = request_row["produces"]
                        commit_id = "{}:c{}:t{}:commit:{}:{}".format(
                            config, call_ordinal, timestep, destination,
                            output_block)
                        commit_address = (
                            (4 << 60) | (module << 52) | (timestep << 44) |
                            ((destination * output_blocks + output_block) *
                             bound_m.OUTPUT_COMMIT_BYTES))
                        runner.one(bound_m.request(
                            commit_id, config, "commit", [commit_address],
                            [0], bound_m.OUTPUT_COMMIT_BYTES,
                            [last] if last else [barrier]))
                    runner.retire_destination((barrier, persistent_control))
            runner.timesteps += 1
        result = runner.finish()
        result.update({"schema": SCHEMA,
            "pilot_call_ordinal": call_ordinal,
            "module_ordinal": module, "timesteps": PILOT_TIMESTEPS,
            "diagnostic_only": True, "paper_result": False,
            "product_capture": False, "production": False,
            "payload_fd_sha256": plane.opened_sha256,
            "payload_fd_size": plane.opened_size})
        return result

    def canonical_entry(config):
        bound_m.validate_config(config)
        require(config != FORBIDDEN_CONFIG,
                "product configuration is forbidden")
        with plane_type(canonical_path, payload_shape,
                        payload_sha256) as plane:
            return schedule_verified(config, plane)
    return canonical_entry


stream_actual_call = _build_canonical_streamer()


def validate_authorities(full_payload=False):
    regular_exact(M1539_SOURCE, M1539_SOURCE_SHA256, "M1539 source")
    seal = verify_flat_seal(M1542, M1542_REVIEW_SHA256,
                            M1542_OUTER_FILE_SHA256)
    upstream = M.validate_authorities(bool(full_payload))
    selected_pilot_record()
    return {"m1542": seal, "m1539": upstream,
            "pilot_call_ordinal": PILOT_CALL_ORDINAL,
            "pilot_execution": False, "production": False}


def pilot_release(_token=None):
    raise M1543Error(
        "M1543 authoring source cannot execute the actual pilot; a distinct "
        "independent launch hammer must pin this source first")


def production_release(_token=None):
    raise M1543Error(
        "M1543 is single-call-pilot source only; 120-call production and "
        "PRODUCT_CAPTURE_TYPED_K8 are independently forbidden")


def synthetic_self_test():
    # Exercise little-bit order and bounded-token scheduling without opening
    # any canonical ep34 payload or executing the actual pilot.
    import tempfile
    shape = (10, 1, 8, 2, 2)
    raw = bytearray((10 * 8 * 2 * 2 + 7) // 8)
    for index in (0, 1, 31, 32, 319):
        raw[index >> 3] |= 1 << (index & 7)
    with tempfile.TemporaryDirectory(prefix="m1543_source_test.") as directory:
        path = Path(directory) / "plane.bitpack"
        path.write_bytes(bytes(raw))
        with ImmutableLittleBitPlane(path, shape) as plane:
            require(plane.bit(0, 0, 0, 0) == 1 and
                    plane.bit(0, 0, 0, 1) == 1 and
                    plane.bit(0, 7, 1, 1) == 1 and
                    plane.bit(1, 0, 0, 0) == 1 and
                    plane.bit(9, 7, 1, 1) == 1 and
                    plane.bit(2, 0, 0, 0) == 0,
                    "little-bit-order mmap test failed")
    # The upstream synthetic transaction stream is consumed one row at a time
    # through the same facade; no list of transactions is built.
    bits = [[[0 for _x in range(2)] for _y in range(2)] for _c in range(8)]
    bits[0][0][0] = 1; bits[1][0][0] = 1
    results = []
    for config in M.CONFIGS:
        runner = StreamingCallScheduler(config)
        for row in M.synthetic_config_transactions(config, bits):
            runner.one(row)
        runner.destinations = 16; runner.timesteps = 1
        results.append(runner.finish())
    require(results[0]["kind_counts"]["compute"] >
            results[2]["kind_counts"]["compute"] and
            results[1]["kind_counts"]["compute"] ==
            results[2]["kind_counts"]["compute"] and
            results[1]["byte_counts"]["external_read"] >
            results[2]["byte_counts"]["external_read"] and
            len(set(row["commit_sequence_sha256"] for row in results)) == 1,
            "streaming synthetic comparator drift")
    return {"schema": SCHEMA,
            "status": "PASS_M1556_IMMUTABLE_SNAPSHOT_STREAMING_SOURCE_SYNTHETIC_TEST__NO_PILOT_NO_PRODUCTION",
            "configurations": list(M.CONFIGS), "results": results,
            "pilot_execution": False, "production": False,
            "product_capture": False}


def describe():
    return {"schema": SCHEMA, "status": STATUS,
            "configurations": list(M.CONFIGS),
            "forbidden_configuration": FORBIDDEN_CONFIG,
            "pilot": {"call_ordinal": PILOT_CALL_ORDINAL,
                "sample_id": PILOT_SAMPLE_ID, "module_ordinal": PILOT_MODULE,
                "timesteps": PILOT_TIMESTEPS, "execution": False},
            "streaming": {"mmap_one_payload": False,
                "immutable_compact_plane_snapshot": True,
                "materialized_transaction_list": False,
                "retire_dependency_tokens_per_destination": True,
                "preserve_bank_calendars": True,
                "preserve_weight_cache": True,
                "peak_rss_limit_kib": PEAK_RSS_LIMIT_KIB},
            "source_capabilities": {"fast_preflight": True,
                "synthetic_test": True, "actual_pilot_engine": True,
                "pilot_cli": False, "production_cli": False,
                "external_plane_parameter": False,
                "opened_fd_hash_binding": True,
                "closure_row_snapshot": True,
                "mutable_file_backing_during_schedule": False},
            "claim_boundary": {"source_only": True,
                "pilot_executed": False, "production": False,
                "transactions": False, "cycles": False, "traffic": False,
                "speedup": False, "system_speedup": False,
                "energy": False, "rtl": False, "eda": False,
                "ppa": False, "table_a": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--synthetic-self-test", action="store_true")
    parser.add_argument("--verify-payload-members", action="store_true")
    args = parser.parse_args(argv)
    if args.describe:
        require(not args.verify_payload_members, "describe cannot hash payloads")
        value = describe()
    elif args.preflight:
        value = {"schema": SCHEMA,
            "status": "PASS_M1556_IMMUTABLE_SNAPSHOT_STREAMING_SOURCE_PREFLIGHT__NO_PILOT_NO_PRODUCTION",
            "authorities": validate_authorities(args.verify_payload_members),
            "pilot_execution": False, "production": False}
    else:
        require(not args.verify_payload_members,
                "synthetic test cannot hash canonical payloads")
        value = synthetic_self_test()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
