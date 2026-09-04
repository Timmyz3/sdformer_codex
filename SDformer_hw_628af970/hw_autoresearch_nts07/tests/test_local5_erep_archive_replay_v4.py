from __future__ import annotations

import copy
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from scripts.local5_erep_archive_replay_v4 import (
    FORMAL_ACC32_VALUE_COUNT,
    FORMAL_HEAD_COUNT,
    FORMAL_PHASE_COUNT,
    FORMAL_WINDOW_COUNT,
    ROLE_TO_CODE,
    encode_miter_fixture,
    encode_trace_fixture,
    validate_archive_contents,
    validate_archive_files,
)
from tests.test_local5_erep_ledger_replay_v4 import synthetic_head_ledger


def synthetic_archives() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    miter, bound = encode_miter_fixture(synthetic_head_ledger())
    trace = encode_trace_fixture(bound)
    return trace, miter, bound


class Local5ErepArchiveReplayV4Test(unittest.TestCase):
    def test_strict_archives_reconstruct_head_and_window_ledgers(self) -> None:
        trace, miter, ledger = synthetic_archives()
        result = validate_archive_contents(trace, miter, ledger)
        self.assertEqual(result["window_count"], 1)
        self.assertEqual(result["head_count"], 3)
        self.assertEqual(result["phase_count"], 27)
        self.assertEqual(result["acc32_value_count"], 3 * 450 * 32)
        self.assertEqual(result["acc32_mismatch_count"], 0)

    def test_npz_member_dtype_shape_and_phase_order_are_frozen(self) -> None:
        trace, miter, ledger = synthetic_archives()
        mutations = []
        extra = dict(trace)
        extra["shadow"] = np.asarray([1], dtype=np.uint8)
        mutations.append((extra, miter, "member set"))
        dtype = dict(trace)
        dtype["phase_role"] = dtype["phase_role"].astype(np.int16)
        mutations.append((dtype, miter, "dtype"))
        shape = dict(miter)
        shape["schema_version"] = shape["schema_version"].reshape(1, 1)
        mutations.append((trace, shape, "one-dimensional"))
        order = {key: value.copy() for key, value in trace.items()}
        order["phase_role"][0], order["phase_role"][3] = (
            order["phase_role"][3], order["phase_role"][0]
        )
        mutations.append((order, miter, "canonical role/head/tile order"))
        for changed_trace, changed_miter, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_archive_contents(changed_trace, changed_miter, ledger)

    def test_raw_event_change_cannot_match_stale_head_ledger(self) -> None:
        trace, miter, ledger = synthetic_archives()
        changed = {key: value.copy() for key, value in trace.items()}
        phase_index = next(
            index
            for index, role in enumerate(changed["phase_role"])
            if int(role) == ROLE_TO_CODE["direct"]
            and changed["phase_event_offsets"][index + 1]
            > changed["phase_event_offsets"][index]
        )
        event_index = int(changed["phase_event_offsets"][phase_index])
        changed["event_identity"][event_index] = b"changed_acc_identity"
        with self.assertRaisesRegex(ValueError, "parsed RTL trace archive"):
            validate_archive_contents(changed, miter, ledger)

    def test_expected_actual_mismatch_is_recomputed_not_self_reported(self) -> None:
        trace, miter, ledger = synthetic_archives()
        changed = {key: value.copy() for key, value in miter.items()}
        changed["actual_acc32"][123] += 1
        ledger["windows"][0]["acc32_mismatch_count"] = 0
        with self.assertRaisesRegex(ValueError, "nonzero mismatch"):
            validate_archive_contents(trace, changed, ledger)

    def test_real_npz_files_use_allow_pickle_false_and_reject_corruption(self) -> None:
        trace, miter, ledger = synthetic_archives()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path = root / "rtl_trace_archive.npz"
            miter_path = root / "acc32_miter_archive.npz"
            np.savez_compressed(trace_path, **trace)
            np.savez_compressed(miter_path, **miter)
            result = validate_archive_files(trace_path, miter_path, ledger)
            self.assertEqual(result["acc32_mismatch_count"], 0)
            trace_path.write_bytes(b"not-an-npz")
            with self.assertRaisesRegex(ValueError, "cannot be parsed safely"):
                validate_archive_files(trace_path, miter_path, ledger)

    def test_duplicate_raw_zip_member_is_rejected_before_dict_collapse(self) -> None:
        trace, miter, ledger = synthetic_archives()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path = root / "rtl_trace_archive.npz"
            miter_path = root / "acc32_miter_archive.npz"
            np.savez_compressed(trace_path, **trace)
            np.savez_compressed(miter_path, **miter)
            with zipfile.ZipFile(trace_path, "r") as archive:
                duplicate = archive.read("schema_version.npy")
            with zipfile.ZipFile(trace_path, "a") as archive:
                with self.assertWarns(UserWarning):
                    archive.writestr("schema_version.npy", duplicate)
            with self.assertRaisesRegex(ValueError, "cannot be parsed safely"):
                validate_archive_files(trace_path, miter_path, ledger)

    def test_raw_zip_order_directory_comment_extra_and_codec_are_rejected(self) -> None:
        trace, miter, ledger = synthetic_archives()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.npz"
            miter_path = root / "acc32_miter_archive.npz"
            np.savez_compressed(base, **trace)
            np.savez_compressed(miter_path, **miter)
            with zipfile.ZipFile(base, "r") as archive:
                members = [
                    (info.filename, archive.read(info.filename))
                    for info in archive.infolist()
                ]

            def write_variant(
                name: str,
                *,
                reverse: bool = False,
                directory: bool = False,
                archive_comment: bool = False,
                member_extra: bool = False,
                compression: int = zipfile.ZIP_DEFLATED,
            ) -> Path:
                path = root / name
                ordered = list(reversed(members)) if reverse else members
                with zipfile.ZipFile(path, "w") as archive:
                    if archive_comment:
                        archive.comment = b"forbidden"
                    for index, (filename, payload) in enumerate(ordered):
                        info = zipfile.ZipInfo(filename)
                        info.compress_type = compression
                        if member_extra and index == 0:
                            info.extra = b"\x01\x00\x00\x00"
                        archive.writestr(info, payload)
                    if directory:
                        archive.writestr("shadow/", b"")
                return path

            variants = (
                write_variant("reordered.npz", reverse=True),
                write_variant("directory.npz", directory=True),
                write_variant("comment.npz", archive_comment=True),
                write_variant("extra.npz", member_extra=True),
                write_variant("bzip2.npz", compression=zipfile.ZIP_BZIP2),
            )
            for trace_path in variants:
                with self.subTest(trace_path=trace_path.name):
                    with self.assertRaisesRegex(ValueError, "cannot be parsed safely"):
                        validate_archive_files(trace_path, miter_path, ledger)

    def test_short_fixture_cannot_claim_formal_archive(self) -> None:
        trace, miter, ledger = synthetic_archives()
        with self.assertRaisesRegex(ValueError, "exactly 1200 windows"):
            validate_archive_contents(trace, miter, ledger, formal=True)

    def test_formal_topology_constants_are_exact(self) -> None:
        self.assertEqual(FORMAL_WINDOW_COUNT, 1200)
        self.assertEqual(FORMAL_HEAD_COUNT, 13_800)
        self.assertEqual(FORMAL_PHASE_COUNT, 462_600)
        self.assertEqual(FORMAL_ACC32_VALUE_COUNT, 198_720_000)

    def test_stage_heads_weight_and_identity_width_fail_closed(self) -> None:
        trace, miter, ledger = synthetic_archives()
        bad_weight = {key: value.copy() for key, value in trace.items()}
        bad_weight["window_weight"][0] += 1
        with self.assertRaisesRegex(ValueError, "invalid stage/H/weight"):
            validate_archive_contents(bad_weight, miter, ledger)

        long_identity = copy.deepcopy(ledger)
        phase = long_identity["heads"][0]["direct_by_tile"][0]
        resource = next(
            name for name, events in phase["resource_events"].items() if events
        )
        phase["resource_events"][resource][0]["identity"] = "x" * 65
        with self.assertRaisesRegex(ValueError, "within S64"):
            encode_trace_fixture(long_identity)


if __name__ == "__main__":
    unittest.main()
