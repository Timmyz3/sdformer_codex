from __future__ import annotations

import math
import unittest

from scripts.local5_erep_identity_service_v3 import (
    DEFAULT_SEED,
    REQUIRED_FIELDS_BY_KIND,
    SCHEMA,
    IdentityService,
    Transaction,
    canonical_json_bytes,
    compare_candidate_ledgers,
    ledger_digests,
    make_transaction,
    transaction_digest,
    transaction_hash_material,
)


# These are frozen outputs, not values recomputed by a second copy of the service.
GOLDEN_VECTORS = (
    {
        "kind": "relation",
        "identity": {
            "window": 4,
            "sample": "seq0",
            "source_id": 17,
            "block": 2,
            "stage": 1,
            "input_head": 3,
        },
        "canonical_identity": (
            '{"block":2,"input_head":3,"sample":"seq0",'
            '"source_id":17,"stage":1,"window":4}'
        ),
        "digest": "cb35c14b5d0d41226421503c3b5d8db48e6fdd4aa239e6923f657eef3bf9f2c0",
        "delay": 2,
    },
    {
        "kind": "epoch_read",
        "identity": {
            "source_id": 449,
            "output_tile": 10,
            "input_head": 5,
            "stripe": 5,
            "window": 8,
            "block": 6,
            "stage": 2,
            "sample": "seq|1",
            "occurrence": 1,
            "binding": {"config": "x:y", "checkpoint": "ab|cd"},
        },
        "canonical_identity": (
            '{"binding":{"checkpoint":"ab|cd","config":"x:y"},"block":6,'
            '"input_head":5,"occurrence":1,"output_tile":10,"sample":"seq|1",'
            '"source_id":449,"stage":2,"stripe":5,"window":8}'
        ),
        "digest": "d49b23aa3ba8615d9398e003f529f9e50c2965361c4a28219c1bb7fad64bd3c7",
        "delay": 1,
    },
    {
        "kind": "weight",
        "identity": {
            "out": 31,
            "lane": 4,
            "output_tile": 7,
            "input_head": 2,
            "window": 3,
            "block": 1,
            "stage": 0,
            "sample": "\u5e8f\u5217A",
        },
        "canonical_identity": (
            '{"block":1,"input_head":2,"lane":4,"out":31,"output_tile":7,'
            '"sample":"\u5e8f\u5217A","stage":0,"window":3}'
        ),
        "digest": "6c56847b2cb6354aa7d5c1db658dff6563d3e7d31a5097518ba158412c976eb6",
        "delay": 2,
    },
    {
        "kind": "final",
        "identity": {
            "out": 0,
            "source_id": 0,
            "output_tile": 0,
            "window": 0,
            "block": 0,
            "stage": 0,
            "sample": "sample0",
        },
        "canonical_identity": (
            '{"block":0,"out":0,"output_tile":0,"sample":"sample0",'
            '"source_id":0,"stage":0,"window":0}'
        ),
        "digest": "ce1b16e9dcbcebf91a07633981c4694b1f047497847f5b28ccef771331a5fdf6",
        "delay": 1,
    },
)

GOLDEN_ORDERED_LEDGER_DIGEST = (
    "4d3c4a0962997161d59ab0aee0b53b743888b52f84c91b8174af611a72ea763d"
)
GOLDEN_MULTISET_DIGEST = (
    "92796c26aa9b9a6b5ccca2ac5a1340ac34652245d4c001e7cfc664a8fe669a95"
)


def golden_transactions() -> tuple[Transaction, ...]:
    return tuple(
        make_transaction(vector["kind"], vector["identity"])
        for vector in GOLDEN_VECTORS
    )


def decode_length_frame(material: bytes) -> tuple[bytes, ...]:
    parts: list[bytes] = []
    cursor = 0
    while cursor < len(material):
        if cursor + 8 > len(material):
            raise AssertionError("truncated frame length")
        length = int.from_bytes(material[cursor : cursor + 8], "big")
        cursor += 8
        end = cursor + length
        if end > len(material):
            raise AssertionError("truncated frame payload")
        parts.append(material[cursor:end])
        cursor = end
    return tuple(parts)


class Local5ErepIdentityServiceV3Test(unittest.TestCase):
    def test_four_frozen_golden_vectors(self) -> None:
        self.assertGreaterEqual(len(GOLDEN_VECTORS), 4)
        for vector in GOLDEN_VECTORS:
            with self.subTest(kind=vector["kind"]):
                row = make_transaction(vector["kind"], vector["identity"])
                self.assertEqual(
                    row.identity_json,
                    vector["canonical_identity"].encode("utf-8"),
                )
                self.assertEqual(row.digest, vector["digest"])
                self.assertEqual(row.delay, vector["delay"])
                self.assertIn(row.delay, range(4))

    def test_hash_frame_explicitly_contains_schema_seed_kind_and_full_identity(self) -> None:
        vector = GOLDEN_VECTORS[1]
        material = transaction_hash_material(vector["kind"], vector["identity"])
        self.assertEqual(
            decode_length_frame(material),
            (
                b"schema",
                SCHEMA.encode("utf-8"),
                b"seed",
                str(DEFAULT_SEED).encode("ascii"),
                b"kind",
                b"epoch_read",
                b"identity",
                vector["canonical_identity"].encode("utf-8"),
            ),
        )

    def test_full_identity_and_each_service_component_affect_digest(self) -> None:
        identity = dict(GOLDEN_VECTORS[0]["identity"])
        baseline = transaction_digest("relation", identity)
        identity["checkpoint_sha256"] = "0" * 64
        self.assertNotEqual(baseline, transaction_digest("relation", identity))
        self.assertNotEqual(
            baseline,
            transaction_digest("relation", GOLDEN_VECTORS[0]["identity"], seed=1),
        )
        self.assertNotEqual(
            baseline,
            transaction_digest(
                "relation",
                GOLDEN_VECTORS[0]["identity"],
                schema="local5_erep_identity_service_v3-test",
            ),
        )

        all_fields = {
            "sample": 0,
            "stage": 0,
            "block": 0,
            "window": 0,
            "input_head": 0,
            "source_id": 0,
            "stripe": 0,
            "output_tile": 0,
            "lane": 0,
            "out": 0,
        }
        self.assertNotEqual(
            transaction_digest("relation", all_fields),
            transaction_digest("final", all_fields),
        )

    def test_canonical_json_is_compact_sorted_and_utf8(self) -> None:
        value = {"z": "\u5e8f\u5217", "a": [3, {"y": 2, "x": 1}]}
        expected = '{"a":[3,{"x":1,"y":2}],"z":"\u5e8f\u5217"}'.encode(
            "utf-8"
        )
        self.assertEqual(canonical_json_bytes(value), expected)
        self.assertNotIn(b"\\u5e8f", expected)
        with self.assertRaises(ValueError):
            canonical_json_bytes({"bad": math.nan})
        with self.assertRaises(ValueError):
            canonical_json_bytes({1: "non-string key"})

    def test_every_kind_requires_its_frozen_fields(self) -> None:
        identities = {
            vector["kind"]: dict(vector["identity"])
            for vector in GOLDEN_VECTORS
        }
        for kind, required_fields in REQUIRED_FIELDS_BY_KIND.items():
            for field in required_fields:
                broken = dict(identities[kind])
                broken.pop(field)
                with self.subTest(kind=kind, missing=field):
                    with self.assertRaisesRegex(ValueError, field):
                        make_transaction(kind, broken)
        with self.assertRaisesRegex(ValueError, "unknown transaction kind"):
            make_transaction("global", identities["relation"])

    def test_global_transaction_indices_are_forbidden_at_any_depth(self) -> None:
        for field in (
            "transaction_index",
            "global_transaction_index",
            "global_index",
        ):
            identity = dict(GOLDEN_VECTORS[0]["identity"])
            identity["metadata"] = {field: 9}
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "forbidden global index"):
                    make_transaction("relation", identity)

    def test_repeated_logical_identity_requires_unique_explicit_occurrence(self) -> None:
        service = IdentityService()
        base = dict(GOLDEN_VECTORS[3]["identity"])
        first = service.transaction("final", base)
        second = service.transaction("final", dict(base))
        with self.assertRaisesRegex(ValueError, "explicit occurrence"):
            service.ledger_digests((first, second))

        occurrence_zero = dict(base, occurrence=0)
        occurrence_one = dict(base, occurrence=1)
        rows = (
            service.transaction("final", occurrence_zero),
            service.transaction("final", occurrence_one),
        )
        self.assertEqual(service.ledger_digests(rows).transaction_count, 2)
        with self.assertRaisesRegex(ValueError, "occurrence values must be unique"):
            service.ledger_digests((rows[0], rows[0]))

        for invalid in (-1, 1.0, True, "0"):
            with self.subTest(occurrence=invalid):
                with self.assertRaisesRegex(ValueError, "occurrence"):
                    service.transaction("final", dict(base, occurrence=invalid))

    def test_ordered_and_multiset_ledger_golden_digests(self) -> None:
        rows = golden_transactions()
        forward = ledger_digests(rows)
        reverse = ledger_digests(reversed(rows))
        self.assertEqual(forward.ordered_digest, GOLDEN_ORDERED_LEDGER_DIGEST)
        self.assertEqual(forward.multiset_digest, GOLDEN_MULTISET_DIGEST)
        self.assertNotEqual(forward.ordered_digest, reverse.ordered_digest)
        self.assertEqual(forward.multiset_digest, reverse.multiset_digest)
        self.assertEqual(
            forward.as_dict(),
            {
                "transaction_count": 4,
                "ordered_ledger_digest": GOLDEN_ORDERED_LEDGER_DIGEST,
                "unordered_multiset_digest": GOLDEN_MULTISET_DIGEST,
            },
        )

    def test_candidate_common_transactions_compare_despite_different_order(self) -> None:
        relation, epoch_read, weight, final = golden_transactions()
        left = (relation, weight, final)
        right = (epoch_read, weight, relation)
        comparison = compare_candidate_ledgers(left, right)
        self.assertEqual(comparison.common_transaction_count, 2)
        self.assertEqual(comparison.left_only_transaction_count, 1)
        self.assertEqual(comparison.right_only_transaction_count, 1)
        self.assertTrue(comparison.common_delays_match)
        self.assertEqual(comparison.delay_mismatches, ())

        service = IdentityService()
        left_map = service.comparable_map(left)
        right_map = service.comparable_map(right)
        common = set(left_map).intersection(right_map)
        self.assertEqual(
            {key: left_map[key] for key in common},
            {key: right_map[key] for key in common},
        )

    def test_candidate_comparison_rejects_inconsistent_occurrence_encoding(self) -> None:
        service = IdentityService()
        base = dict(GOLDEN_VECTORS[3]["identity"])
        implicit = service.transaction("final", base)
        explicit = service.transaction("final", dict(base, occurrence=0))
        with self.assertRaisesRegex(ValueError, "inconsistent occurrence"):
            service.compare_candidates((implicit,), (explicit,))

    def test_transaction_copies_identity_and_rejects_forged_delay(self) -> None:
        identity = dict(GOLDEN_VECTORS[0]["identity"])
        row = make_transaction("relation", identity)
        identity["source_id"] = 99
        self.assertEqual(row.identity["source_id"], 17)
        self.assertEqual(row.response_latency_cycles, row.delay + 1)
        self.assertEqual(row.response_cycle(11), 12 + row.delay)
        self.assertIn(row.response_latency_cycles, range(1, 5))
        for invalid_cycle in (-1, 1.5, True):
            with self.subTest(accept_cycle=invalid_cycle):
                with self.assertRaisesRegex(ValueError, "accept_cycle"):
                    row.response_cycle(invalid_cycle)
        with self.assertRaisesRegex(ValueError, "delay does not match"):
            Transaction(row.schema, row.seed, row.kind, row.identity_json, 3 - row.delay)


if __name__ == "__main__":
    unittest.main()
