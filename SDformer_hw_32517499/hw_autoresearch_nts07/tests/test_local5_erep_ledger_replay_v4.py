from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from scripts.local5_erep_capacity_baselines_v4 import evaluate_c4_oracle
from scripts.local5_erep_command_schedule_v4 import evaluate_window
from scripts.local5_erep_ledger_replay_v4 import (
    HEAD_LEDGER_SCHEMA,
    PROJECTION_CONTRACT_SHA256,
    SELECTION_PLAN_SHA256,
    canonical_sha,
    encode_phase,
    encode_window_fixture,
    replay_ledger_document,
    validate_replayed_ledgers,
)
from tests.test_local5_erep_command_schedule_v4 import fixture


def synthetic_head_ledger() -> dict[str, object]:
    window = fixture()
    window_row, head_rows = encode_window_fixture(
        window,
        sample=0,
        stage=0,
        block=0,
        selected_window=7,
        weight=440,
    )
    return {
        "schema": HEAD_LEDGER_SCHEMA,
        "evidence_level": "synthetic_fixture_not_formal",
        "selection_plan_sha256": SELECTION_PLAN_SHA256,
        "formal_manifest_sha256": "1" * 64,
        "projection_contract_sha256": PROJECTION_CONTRACT_SHA256,
        "rtl_trace_archive_file": "rtl_trace_archive.npz",
        "rtl_trace_archive_sha256": "2" * 64,
        "acc32_miter_archive_file": "acc32_miter_archive.npz",
        "acc32_miter_archive_sha256": "3" * 64,
        "windows": [window_row],
        "heads": head_rows,
    }


class Local5ErepLedgerReplayV4Test(unittest.TestCase):
    def test_head_ledger_has_no_candidate_scalar_and_replays_all_candidates(self) -> None:
        ledger = synthetic_head_ledger()
        serialized = str(ledger)
        for candidate in ("'c0'", "'c1'", "'c2'", "'c3'", "'c4'"):
            self.assertNotIn(candidate, serialized)
        windows, commands = replay_ledger_document(ledger)
        self.assertEqual(len(windows["rows"]), 1)
        self.assertEqual(len(commands["rows"]), 1)
        expected = evaluate_window(fixture())
        row = commands["rows"][0]
        self.assertEqual(row["c0"], expected["c0_direct_serial"].cycles)
        self.assertEqual(row["c1"], expected["c1_reuse_only_s2"].cycles)
        self.assertEqual(row["c2"], expected["c2_overlap_only"].cycles)
        self.assertEqual(row["c3"], expected["c3_erep_s2"].cycles)
        self.assertEqual(row["c4"], evaluate_c4_oracle(fixture()).cycles)

    def test_coherent_scalar_forgery_is_rejected_by_phase_replay(self) -> None:
        ledger = synthetic_head_ledger()
        windows, commands = replay_ledger_document(ledger)
        forged_windows = copy.deepcopy(windows)
        forged_commands = copy.deepcopy(commands)
        schedule = forged_windows["rows"][0]
        schedule["tail_cycles"]["c3"] += 1
        schedule["candidates"]["c3"]["cycles"] += 1
        body = {key: value for key, value in schedule.items() if key != "window_schedule_sha256"}
        schedule["window_schedule_sha256"] = canonical_sha(body)
        command = forged_commands["rows"][0]
        command["c3"] += 1
        command["window_schedule_sha256"] = schedule["window_schedule_sha256"]
        command_body = {
            key: value for key, value in command.items() if key != "command_ledger_sha256"
        }
        command["command_ledger_sha256"] = canonical_sha(command_body)
        forged_commands["window_schedule_ledger_canonical_sha256"] = canonical_sha(
            forged_windows
        )
        with self.assertRaisesRegex(ValueError, "independent phase replay"):
            validate_replayed_ledgers(ledger, forged_windows, forged_commands)

    def test_phase_mutation_changes_replay_and_stale_summaries_fail(self) -> None:
        ledger = synthetic_head_ledger()
        windows, commands = replay_ledger_document(ledger)
        modified = copy.deepcopy(ledger)
        phase = modified["heads"][0]["direct_by_tile"][0]
        phase["duration"] += 1
        phase["phase_event_sha256"] = canonical_sha(
            {
                "duration": phase["duration"],
                "resource_events": phase["resource_events"],
            }
        )
        changed_windows, changed_commands = replay_ledger_document(modified)
        self.assertEqual(changed_commands["rows"][0]["c0"], commands["rows"][0]["c0"] + 1)
        self.assertNotEqual(
            changed_windows["rows"][0]["window_schedule_sha256"],
            windows["rows"][0]["window_schedule_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "independent phase replay"):
            validate_replayed_ledgers(modified, windows, commands)

    def test_phase_digest_collision_and_acc32_mismatch_fail_closed(self) -> None:
        for mutation in ("digest", "mismatch", "duplicate_cycle"):
            ledger = synthetic_head_ledger()
            if mutation == "digest":
                ledger["heads"][0]["fill"]["phase_event_sha256"] = "0" * 64
                message = "digest mismatch"
            elif mutation == "mismatch":
                ledger["windows"][0]["acc32_mismatch_count"] = 1
                message = "nonzero Acc32"
            else:
                events = ledger["heads"][0]["fill"]["resource_events"][
                    "relation_workspace_1rw"
                ]
                events.append(dict(events[-1]))
                body = {
                    "duration": ledger["heads"][0]["fill"]["duration"],
                    "resource_events": ledger["heads"][0]["fill"]["resource_events"],
                }
                ledger["heads"][0]["fill"]["phase_event_sha256"] = canonical_sha(body)
                message = "unique in-phase order"
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                replay_ledger_document(ledger)

    def test_event_identity_permutation_is_persisted_and_rejected(self) -> None:
        source = fixture().heads[0].fill
        epoch_indices = [
            index
            for index, command in enumerate(source.commands)
            if command.resource.value == "epoch_slot_1rw"
        ]
        commands = list(source.commands)
        left, right = epoch_indices
        left_identity = commands[left].identity
        right_identity = commands[right].identity
        commands[left] = replace(commands[left], identity=right_identity)
        commands[right] = replace(commands[right], identity=left_identity)
        permuted = type(source)(source.duration, tuple(commands))
        self.assertNotEqual(encode_phase(source, "fill"), encode_phase(permuted, "fill"))

        ledger = synthetic_head_ledger()
        phase = ledger["heads"][0]["fill"]
        events = phase["resource_events"]["epoch_slot_1rw"]
        events[0]["identity"], events[1]["identity"] = (
            events[1]["identity"], events[0]["identity"],
        )
        events.sort(key=lambda event: (event["cycle"], event["identity"]))
        phase["phase_event_sha256"] = canonical_sha(
            {"duration": phase["duration"], "resource_events": phase["resource_events"]}
        )
        with self.assertRaisesRegex(ValueError, "identity.*mismatch"):
            replay_ledger_document(ledger)

        duplicate = synthetic_head_ledger()
        direct = duplicate["heads"][0]["direct_by_tile"][0]
        acc_events = next(
            events for events in direct["resource_events"].values() if len(events) >= 2
        )
        acc_events[1]["identity"] = acc_events[0]["identity"]
        direct["phase_event_sha256"] = canonical_sha(
            {"duration": direct["duration"], "resource_events": direct["resource_events"]}
        )
        with self.assertRaisesRegex(ValueError, "identities are not unique"):
            replay_ledger_document(duplicate)

    def test_float_equal_upper_ledger_values_do_not_bypass(self) -> None:
        ledger = synthetic_head_ledger()
        windows, commands = replay_ledger_document(ledger)
        forged_windows = copy.deepcopy(windows)
        forged_commands = copy.deepcopy(commands)
        forged_windows["rows"][0]["tail_cycles"]["c0"] = float(
            forged_windows["rows"][0]["tail_cycles"]["c0"]
        )
        forged_commands["rows"][0]["c0"] = float(
            forged_commands["rows"][0]["c0"]
        )
        with self.assertRaisesRegex(ValueError, "independent phase replay"):
            validate_replayed_ledgers(ledger, forged_windows, forged_commands)

    def test_short_synthetic_ledger_cannot_enter_formal_replay(self) -> None:
        ledger = synthetic_head_ledger()
        ledger["evidence_level"] = "formal_t450_rtl_phase_ledger"
        with self.assertRaisesRegex(ValueError, "1200 windows/13800 heads"):
            replay_ledger_document(ledger, formal=True, plan_records=[])


if __name__ == "__main__":
    unittest.main()
