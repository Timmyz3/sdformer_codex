#!/usr/bin/env python3
"""Independent fail-closed validator and tamper audit for M66 additive r2."""

from __future__ import print_function

import argparse
import copy
from contextlib import redirect_stdout
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REVIEW = HERE / "m66_independent_hammer_review.json"
RECEIPT = HERE / "m66_independent_hammer_validation_receipt.json"
CONTRACT = HW / "contracts/m66_s00_additive_release_contract_r2_20260823.json"
PRODUCER_VALIDATOR = HW / "verif_m66/validate_m66_s00_additive_release_r2.py"
COMPAT_VALIDATOR = HW / "verif_m66/validate_m66_s00_exact_sha_release_r2.py"
RELEASE = HW / "results/m66_s00_additive_release_r2_20260823"
R1_RUN = HW / "results/m66_h67_k4c16_temporal_vcs_s00_lookahead_exact_sha_r1_20260823"
COMPILE = HW / "dc_handoff/runs/m66_s00_lookahead_exact_sha_compile_r3_20260823"


PATHS = {
    "additive_contract": CONTRACT,
    "producer_validator": PRODUCER_VALIDATOR,
    "compat_validator": COMPAT_VALIDATOR,
    "producer_receipt": RELEASE / "m66_s00_additive_release_receipt_r2.json",
    "run_complete": RELEASE / "RUN_COMPLETE.txt",
    "release_output_manifest": RELEASE / "output_manifest.sha256",
    "release_validator_log": RELEASE / "validator.raw.log",
    "r1_sim_log": R1_RUN / "sim.raw.log",
    "r1_ledger_gzip": R1_RUN / "m66_s00_handshake_ledger.compact.log.gz",
    "r1_replay": R1_RUN / "m66_s00_ledger_replay.json",
    "r1_receipt": R1_RUN / "m66_s00_exact_sha_vcs_receipt.json",
    "r1_failed_marker": R1_RUN / "FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt",
    "r1_validator_log": R1_RUN / "validator.raw.log",
    "r1_validator_rc": R1_RUN / "validator.rc",
    "core_rtl": HW / "rtl_m66/qfit_k4_parent_delta_p8_l96_ctx16_lookahead.sv",
    "bridge_rtl": HW / "rtl_m66/qfit_m66_m53_schedule_bridge_lookahead.sv",
    "inherited_sva": HW / "verif_m54/qfit_k4_parent_delta_p8_l96_ctx16_assertions.sv",
    "lookahead_sva": HW / "verif_m66/qfit_k4_parent_delta_lookahead_assertions.sv",
    "testbench": HW / "tb_m66/tb_m66_m53_schedule_bridge_lookahead.sv",
    "filelist": HW / "dc_handoff/filelists/date_m66_m53_schedule_bridge_lookahead_vcs.f",
    "simv": COMPILE / "simv",
    "precompile_manifest": COMPILE / "precompile_input.sha256",
    "prelaunch_manifest": R1_RUN / "prelaunch_input.sha256",
}


EXPECTED_SHA = {
    "additive_contract": "d65aa56b44bf41ba3d1e43a501456b9264dd07f62c9b2c7c9afa12f86704727e",
    "producer_validator": "cb4aaef440adbc6417c7b5cf7ea2f032052dbda6ade172c00680f91cd0fd1e32",
    "compat_validator": "2c717014a6369980d5e9694739bd029b25ee21611daf6ce54216014aac228d5d",
    "producer_receipt": "559b43a3e1de2ad23d1dc041f66dec70bdb8e6619a3bf9e10a23f53b948b521b",
    "run_complete": "a358a4351a31da6990046d4bac471627f786a3ef734909394e5655f91ad75382",
    "release_output_manifest": "c00623230e82d6f596eff25d0b917e5e74a1174d5cc97243da97660da81d44c1",
    "release_validator_log": "98dc030613ca2b458ed3dd51387fe732c40b4e1761c2c76e004bfacab1d0e77e",
    "r1_sim_log": "f5a3c9bd29a6fb67a977c5e7e920540af85e5e67022dafb8af1c1eb6cc87befc",
    "r1_ledger_gzip": "e6b5d54015960c76890b825642a15219992d66d93726a2d0521380bca5c1e262",
    "r1_replay": "73ad4f40c63fadc24fcff7317a9548ee7e9f4c515dd2f69d370a2df9bbb2a17b",
    "r1_receipt": "340d5db574005d6690e3548f87b7efbd887034cd8405d7de67cea70345013f20",
    "r1_failed_marker": "6478a78b963c0e0d25b7317d066c009337478327cb3fafea6a82543f4e82ba2f",
    "r1_validator_log": "055fa5d0743140bd724152b812abf081757eba7a0e9443af7f6275685a3c2d8b",
    "r1_validator_rc": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
    "core_rtl": "b9a2064ab73764534415f2dc54aa134807a147c6b8528f0fb041e3afc5d13f4d",
    "bridge_rtl": "d1020823c328c528c5e9693cc85bd973667e143a335de2fa7a1f081f19e7c7af",
    "inherited_sva": "1338421c3ee3d12f70fb2b2299e76d6651c297500920b1ffb70989c90cc2a267",
    "lookahead_sva": "e522c849411ab89e59037825764410e617cc642a158d3a488472272131fb3973",
    "testbench": "67d6f76182c1566ffbda9274cdbc0f01cbca19a34668290f3c086e4730c32771",
    "filelist": "1a6bea2c3bc7b9a83fa69b875739f21bcb896021bc4cddcbd4089dbea311af03",
    "simv": "839d599287f63b7a973688253c815d8549448a1a0f8078e9185d6f3d098333cf",
    "precompile_manifest": "15dc732bc79c6245835c8e1db65067a1a97c3a81445773dc5fdb1895e2aed88c",
    "prelaunch_manifest": "fd707765e226ed04a3326942defd2c0b25f55352a6588f0132e1193d9662cfb7",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant: " + value)))


def validate_hashes():
    for name, expected in EXPECTED_SHA.items():
        observed = sha256_path(PATHS[name])
        require(observed == expected,
                "{} SHA drift {} != {}".format(name, observed, expected))


def validate_release_manifest():
    manifest = PATHS["release_output_manifest"].read_text(
        encoding="utf-8").splitlines()
    require(len(manifest) == 5, "release output manifest count drift")
    observed = {}
    for line in manifest:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.]+)", line)
        require(match is not None, "malformed release manifest line")
        expected, relative = match.groups()
        require(relative not in observed, "duplicate release manifest path")
        target = RELEASE / relative
        require(target.is_file(), "release manifest target missing")
        require(sha256_path(target) == expected,
                "release manifest target SHA drift: " + relative)
        observed[relative] = expected
    require(set(observed) == {
        "RUN_COMPLETE.txt", "m66_s00_additive_release_receipt_r2.json",
        "prelaunch.sha256", "validator.raw.log", "validator.rc"},
        "release output manifest path set drift")


def rerun_producer_validator():
    process = subprocess.run([
        sys.executable, str(PRODUCER_VALIDATOR), "--repo", str(HW.parent),
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
       universal_newlines=True)
    require(process.returncode == 0, "producer validator rerun failed")
    require(process.stdout ==
            "PASS M66 additive r2 exact-SHA release; r1 failed marker preserved\n",
            "producer validator terminal drift")
    return process.stdout.strip()


def validate_metrics_and_claims():
    receipt = strict_json(PATHS["producer_receipt"])
    replay = strict_json(PATHS["r1_replay"])
    r1_receipt = strict_json(PATHS["r1_receipt"])
    log = PATHS["r1_sim_log"].read_text(encoding="utf-8", errors="strict")
    released = receipt["released_evidence"]
    require(receipt["status"] ==
            "PASS_M66_ADDITIVE_R2_EXACT_SHA_RELEASE_R1_FAILED_MARKER_PRESERVED",
            "producer receipt status drift")
    require(receipt["original_r1"] == {
        "hardware_sim_rc": 0, "gzip_rc": 0, "ledger_replay_rc": 0,
        "validator_rc": 1, "failed_or_incomplete_marker_preserved": True,
        "failure_class": "Python_3p6_subprocess_run_text_keyword_incompatibility",
        "failed_run_may_be_relabelled_pass": False,
    }, "original r1 failure preservation drift")
    require(released["m66_rtl_cycles"] == replay["rtl_cycles"] == 8117392,
            "M66 cycle drift")
    require(released["m57_phase_safe_rtl_cycles"] == 8791654 and
            released["cycles_saved_vs_m57"] == 674262,
            "M57-to-M66 comparison drift")
    require(released["functional_mismatch_count"] == 0 and
            replay["accepted_requests"] == replay["accepted_responses"] == 7011032 and
            replay["accepted_outputs"] == 2592000,
            "functional/conservation drift")
    covers = released["lookahead_cover_matches"]
    require(sum(covers["seam_k{}".format(k)] for k in range(1, 5)) ==
            released["seamless_launches"] == 697471,
            "K1..K4 seam partition drift")
    require(covers["zero_next_wait"] == 17104 and
            covers["seam_with_output_accept"] == 33674 and
            covers["seam_with_completion_push"] == 697471 and
            covers["seam_with_command_accept"] == 0 and
            covers["seam_with_command_and_output"] == 0,
            "seam concurrency cover drift")
    require(log.count("M66_LOOKAHEAD_ASSERTION_MODULE_ACTIVE=1") == 1 and
            log.count("M54_ASSERTION_MODULE_ACTIVE=1") == 1,
            "SVA activation drift")
    require(not re.search(
        r"(?i)(assertion failed|error-|fatal:|\$error|\$fatal)", log),
        "simulation failure signature")
    require(all(receipt["claim_boundary"][name] is False for name in (
        "system_speedup_admitted", "full_network_cycles_admitted",
        "online_scheduler_admitted", "memory_system_admitted",
        "power_or_energy_admitted", "paper_ppa_ready")),
        "producer receipt claim widened")
    require(r1_receipt["claim_boundary"]["system_speedup_admitted"] is False,
            "r1 receipt claim widened")
    speed = Fraction(8791654, 8117392)
    reduction = Fraction(674262, 8791654)
    return {
        "speedup_numerator": speed.numerator,
        "speedup_denominator": speed.denominator,
        "speedup_float": float(speed),
        "cycle_reduction_fraction_numerator": reduction.numerator,
        "cycle_reduction_fraction_denominator": reduction.denominator,
        "cycle_reduction_percent": 100.0 * float(reduction),
        "seam_partition_sum": sum(covers["seam_k{}".format(k)]
                                  for k in range(1, 5)),
    }


def load_producer_module():
    name = "m66_additive_validator_attack_" + next(tempfile._get_candidate_names())
    spec = importlib.util.spec_from_file_location(name, str(PRODUCER_VALIDATOR))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke_with_contract(attack_contract):
    module = load_producer_module()
    original_load = module.load_json

    def attack_load(path):
        if Path(path).name == CONTRACT.name:
            return copy.deepcopy(attack_contract)
        return original_load(path)

    module.load_json = attack_load
    old_argv = sys.argv
    stream = io.StringIO()
    try:
        sys.argv = [str(PRODUCER_VALIDATOR), "--repo", str(HW.parent)]
        with redirect_stdout(stream):
            module.main()
        return 0, stream.getvalue()
    except Exception as error:
        return 1, str(error)
    finally:
        sys.argv = old_argv


def redirect_entry(contract, original_suffix, new_path):
    found = False
    for entry in contract["entries"]:
        if entry["path"].endswith(original_suffix):
            entry["path"] = str(new_path)
            entry["sha256"] = sha256_path(new_path)
            found = True
            break
    require(found, "attack entry not found: " + original_suffix)


def semantic_failure_attack(field):
    base_contract = strict_json(CONTRACT)
    with tempfile.TemporaryDirectory(prefix="m66_hammer_failure_") as temporary:
        run = Path(temporary) / "failed_r1"
        run.mkdir()
        required = [
            "m66_s00_exact_sha_vcs_receipt.json", "sim.rc", "gzip.rc",
            "replay.rc", "validator.rc", "validator.raw.log",
            "FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt",
        ]
        for name in required:
            shutil.copy2(str(R1_RUN / name), str(run / name))
        if field == "failed_marker":
            target = run / "FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
            target.write_text("PASS\n", encoding="utf-8")
        elif field == "validator_rc":
            target = run / "validator.rc"
            target.write_text("0\n", encoding="utf-8")
        elif field == "validator_diagnostic":
            target = run / "validator.raw.log"
            target.write_text("PASS\n", encoding="utf-8")
        else:
            raise ValueError("unknown semantic attack")
        attack = copy.deepcopy(base_contract)
        attack["paths"]["failed_r1_run"] = str(run)
        for name in required:
            suffix = "/results/m66_h67_k4c16_temporal_vcs_s00_lookahead_exact_sha_r1_20260823/" + name
            matching = [entry for entry in attack["entries"]
                        if entry["path"].endswith(suffix)]
            if matching:
                redirect_entry(attack, suffix, run / name)
        rc, diagnostic = invoke_with_contract(attack)
        require(rc == 1, field + " linked-resign attack survived")
        return diagnostic


def linked_resign_attack():
    attack = strict_json(CONTRACT)
    attack["claim_boundary"]["system_speedup_admitted"] = True
    attack["claim_boundary"]["paper_ppa_ready"] = True
    with tempfile.TemporaryDirectory(prefix="m66_hammer_linked_") as temporary:
        fake_ledger = Path(temporary) / "ledger.gz"
        fake_replay = Path(temporary) / "replay.json"
        fake_receipt = Path(temporary) / "receipt.json"
        fake_ledger.write_bytes(b"linked-resigned fake ledger\n")
        fake_replay.write_text('{"linked_resigned": true}\n', encoding="utf-8")
        fake_receipt.write_text('{"system_speedup_admitted": true}\n',
                                encoding="utf-8")
        redirect_entry(attack, "/m66_s00_handshake_ledger.compact.log.gz",
                       fake_ledger)
        redirect_entry(attack, "/m66_s00_ledger_replay.json", fake_replay)
        redirect_entry(attack, "/m66_s00_exact_sha_vcs_receipt.json",
                       fake_receipt)
        rc, terminal = invoke_with_contract(attack)
        require(rc == 0, "expected linked-resign weakness was rejected")
        require(terminal ==
                "PASS M66 additive r2 exact-SHA release; r1 failed marker preserved\n",
                "linked-resign terminal drift")
    producer_source = PRODUCER_VALIDATOR.read_text(encoding="utf-8")
    require("m66_s00_additive_release_receipt_r2.json" not in producer_source,
            "producer unexpectedly validates additive receipt")
    return {
        "contract_claim_widening_survived": True,
        "contract_entry_ledger_replay_receipt_path_substitution_survived": True,
        "additive_receipt_is_not_consumed_by_producer_validator": True,
        "impact": "Current bytes are correct, but producer validation is not a root trust seal against linked resigning of the additive wrapper.",
    }


def direct_tamper_checks():
    checked = {}
    for name in (
            "r1_failed_marker", "r1_validator_rc", "r1_validator_log",
            "r1_ledger_gzip", "r1_replay", "r1_receipt", "core_rtl",
            "bridge_rtl", "lookahead_sva", "producer_receipt"):
        original = PATHS[name].read_bytes()
        mutated = original + b"\nINDEPENDENT_TAMPER\n"
        require(sha256_bytes(mutated) != EXPECTED_SHA[name],
                "direct tamper unexpectedly retained SHA")
        checked[name] = "REJECTED_BY_INDEPENDENT_ROOT_SHA"
    return checked


def validate_review(metrics, semantic_attacks, linked_attack, direct_attacks):
    review = strict_json(REVIEW)
    require(review["schema"] == "m66_independent_hammer_review_v1",
            "review schema drift")
    require(review["status"] ==
            "RTL_AND_LEGAL_TRACE_PASS_RELEASE_WRAPPER_P1_AND_DATE_NO_GO_HEADLINE",
            "review status drift")
    require(review["producer_evidence_modified"] is False,
            "review claims producer modification")
    require(review["scores"]["m66_milestone_quality"] == 82 and
            review["scores"]["date_paper_completeness"] == 61 and
            review["scores"]["hardware_novelty"] == 43,
            "review score drift")
    require(review["exact_reconstruction"]["same_trace_kernel_speedup"] ==
            metrics["speedup_float"], "review speedup drift")
    require(review["tamper_audit"]["survived"] == [
        "linked_resign_additive_contract_claim_widening",
        "linked_resign_contract_entry_path_substitution",
        "linked_resign_additive_receipt_claim_widening_not_consumed",
    ], "review survived-attack disclosure drift")
    require(len(review["issues"]["P0"]) == 0 and
            len(review["issues"]["P1"]) == 4 and
            len(review["issues"]["P2"]) == 3,
            "review issue count drift")
    require(linked_attack["contract_claim_widening_survived"] and
            len(semantic_attacks) == 3 and len(direct_attacks) == 10,
            "attack execution drift")
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RECEIPT)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing independent receipt overwrite")
    validate_hashes()
    validate_release_manifest()
    producer_terminal = rerun_producer_validator()
    metrics = validate_metrics_and_claims()
    semantic_attacks = {
        name: semantic_failure_attack(name)
        for name in ("failed_marker", "validator_rc", "validator_diagnostic")
    }
    direct_attacks = direct_tamper_checks()
    linked_attack = linked_resign_attack()
    review = validate_review(metrics, semantic_attacks, linked_attack,
                             direct_attacks)
    payload = {
        "schema": "m66_independent_hammer_validation_receipt_v1",
        "status": "PASS_CURRENT_BYTES_WITH_DISCLOSED_LINKED_RESIGN_P1",
        "review_sha256": sha256_path(REVIEW),
        "independent_validator_sha256": sha256_path(Path(__file__)),
        "producer_contract_sha256": EXPECTED_SHA["additive_contract"],
        "producer_receipt_sha256": EXPECTED_SHA["producer_receipt"],
        "producer_validator_terminal": producer_terminal,
        "exact_reconstruction": metrics,
        "semantic_linked_resign_rejections": semantic_attacks,
        "direct_tamper_rejections": direct_attacks,
        "linked_resign_survival": linked_attack,
        "issue_counts": {name: len(review["issues"][name])
                         for name in ("P0", "P1", "P2")},
        "producer_evidence_modified": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M66 independent hammer current-bytes; linked-resign P1 disclosed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FAIL M66 independent hammer: {}".format(error))
        raise SystemExit(1)
