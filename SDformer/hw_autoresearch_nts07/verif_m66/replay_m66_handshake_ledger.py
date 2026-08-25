#!/usr/bin/env python3
"""Stream-replay the M66 accepted-handshake ledger without loading it in RAM."""

import argparse
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_kv(line):
    fields = {}
    for token in line.strip().split()[1:]:
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(str(path), "rt", encoding="utf-8")
    return Path(path).open("r", encoding="utf-8")


def uncompressed_sha256(path):
    digest = hashlib.sha256()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay(args):
    manifest = json.loads(args.schedule_manifest.read_text(encoding="utf-8"))
    expected = manifest["m53_exact_reconstruction"]
    metadata = deque()
    expected_meta = 0
    requests = responses = outputs = event_lines = 0
    last_cycle = -1
    next_request_tag = next_output_tag = 0
    observed_max_meta = observed_max_complete = observed_max_context = 0
    begin = end = None
    with open_text(args.ledger) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("BEGIN "):
                require(begin is None and event_lines == 0, "duplicate/late BEGIN")
                begin = parse_kv(line)
                continue
            if line.startswith("END "):
                require(end is None, "duplicate END")
                end = parse_kv(line)
                continue
            require(line.startswith("E "), "unknown ledger line {}".format(line_number))
            tokens = line.strip().split()
            if len(tokens) == 7 and "=" not in tokens[1]:
                cycle = int(tokens[1])
                flags = int(tokens[2], 16)
                occupancy = int(tokens[3], 16)
                req, rsp, out = flags & 1, (flags >> 1) & 1, (flags >> 2) & 1
                meta = occupancy & 0x1f
                complete = (occupancy >> 5) & 0x1f
                context = (occupancy >> 10) & 0x1f
                reqtag, rsptag, outtag = tokens[4:7]
            else:
                fields = parse_kv(line)
                cycle = int(fields["cycle"])
                req, rsp, out = int(fields["req"]), int(fields["rsp"]), int(fields["out"])
                meta, complete, context = int(fields["meta"]), int(fields["complete"]), int(fields["context"])
                reqtag, rsptag, outtag = fields["reqtag"], fields["rsptag"], fields["outtag"]
            require(cycle > last_cycle, "nonmonotonic or duplicate event cycle")
            require((req, rsp, out) != (0, 0, 0) or meta == 16 or complete == 16,
                    "event line without event/full state")
            require(meta == expected_meta, "metadata occupancy mismatch cycle {}".format(cycle))
            require(0 <= meta <= 16 and 0 <= complete <= 16 and 0 <= context <= 16,
                    "finite queue/context bound violation")
            observed_max_meta = max(observed_max_meta, meta)
            observed_max_complete = max(observed_max_complete, complete)
            observed_max_context = max(observed_max_context, context)
            if rsp:
                require(metadata, "response accepted with empty replay FIFO")
                require(int(rsptag, 16) == metadata.popleft(),
                        "response tag FIFO mismatch cycle {}".format(cycle))
                responses += 1
            if req:
                request_tag = int(reqtag, 16)
                require(request_tag == next_request_tag,
                        "request tag sequence mismatch cycle {}".format(cycle))
                metadata.append(request_tag)
                next_request_tag = (next_request_tag + 1) & 0xffff
                requests += 1
            expected_meta += req - rsp
            require(expected_meta == len(metadata) and 0 <= expected_meta <= 16,
                    "metadata FIFO conservation failure")
            if out:
                require(int(outtag, 16) == next_output_tag,
                        "output order mismatch cycle {}".format(cycle))
                next_output_tag += 1
                outputs += 1
            event_lines += 1
            last_cycle = cycle

    require(begin is not None and end is not None, "ledger missing BEGIN/END")
    require(int(begin["sample"]) == manifest["sample_id"] and
            int(begin["groups"]) == expected["fusion_groups"] and
            int(begin["commands"]) == expected["descriptor_commands"] and
            int(begin["model_cycles"]) == expected["model_integrated_cycles"],
            "BEGIN does not bind schedule manifest")
    require(requests == responses == expected["source_issue_cycles"] and
            outputs == expected["descriptor_commands"] and expected_meta == 0 and not metadata,
            "accepted-handshake totals/conservation mismatch")
    numeric_end = {key: int(value.split(",")[0]) for key, value in end.items()
                   if key != "parent"}
    require(numeric_end["sample"] == manifest["sample_id"] and
            numeric_end["commands"] == expected["descriptor_commands"] and
            numeric_end["groups"] == expected["fusion_groups"] and
            numeric_end["requests"] == requests and numeric_end["responses"] == responses and
            numeric_end["outputs"] == outputs and numeric_end["mismatches"] == 0,
            "END totals mismatch")
    require(numeric_end["max_meta"] == observed_max_meta and
            numeric_end["max_ctx"] == observed_max_context and
            numeric_end["max_complete"] == observed_max_complete,
            "END maximum occupancy mismatch")
    parents = [int(value) for value in end["parent"].split(",")]
    stream_parent = expected["parent_descriptor_count_stream_x8"]
    require(parents == [stream_parent["local_zero"], stream_parent["left"],
                        stream_parent["up"], stream_parent["previous_timestep"]] and
            numeric_end["add"] == expected["signed_add_updates"] and
            numeric_end["sub"] == expected["signed_subtract_updates"],
            "parent/signed arithmetic ledger mismatch")
    return {
        "schema": "m66_accepted_handshake_ledger_replay_v1",
        "status": "PASS_M66_STREAMING_FIFO_TAG_ARITHMETIC_REPLAY",
        "identity": {"ledger_file_sha256": sha256_path(args.ledger),
                     "ledger_uncompressed_sha256": uncompressed_sha256(args.ledger),
                     "schedule_manifest_sha256": sha256_path(args.schedule_manifest)},
        "sample_id": manifest["sample_id"], "event_lines": event_lines,
        "accepted_requests": requests, "accepted_responses": responses,
        "accepted_outputs": outputs,
        "maximum_metadata_occupancy": numeric_end["max_meta"],
        "maximum_context_occupancy": numeric_end["max_ctx"],
        "maximum_complete_occupancy": numeric_end["max_complete"],
        "response_tag_wraps": numeric_end["tag_wrap"],
        "context_reuses": numeric_end["reuse"],
        "rtl_cycles": numeric_end["rtl_cycles"],
        "m53_transaction_model_cycles": numeric_end["model_cycles"],
        "rtl_minus_m53_transaction_cycles": numeric_end["rtl_cycles"] - numeric_end["model_cycles"],
        "seamless_launches": numeric_end["seamless"],
        "stalls": {"command": numeric_end["cmd_stall"], "launch": numeric_end["launch_stall"],
                   "request": numeric_end["req_stall"], "response": numeric_end["rsp_stall"],
                   "output": numeric_end["out_stall"], "schedule_late_accumulated": numeric_end["late"],
                   "schedule_late_groups": numeric_end.get("late_groups", 0)},
        "launch_phase": {"direct_groups": numeric_end.get("phase_direct", 0),
                         "aligned_groups": numeric_end.get("phase_aligned", 0),
                         "prelaunch_artificial_bubbles": numeric_end.get("prelaunch_artificial_bubbles", -1)},
        "functional_mismatch_count": numeric_end["mismatches"],
        "metadata_fifo_final_occupancy": expected_meta,
        "system_or_full_network_cycles_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing replay output overwrite")
    payload = replay(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M66 replay sample={} requests={} outputs={} rtl_cycles={} seamless={}".format(
        payload["sample_id"], payload["accepted_requests"], payload["accepted_outputs"],
        payload["rtl_cycles"], payload["seamless_launches"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FAIL M66 ledger replay: {}".format(error))
        raise SystemExit(1)
