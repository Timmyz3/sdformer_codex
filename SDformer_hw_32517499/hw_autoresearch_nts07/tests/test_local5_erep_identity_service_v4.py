from __future__ import annotations

import math
import unittest

from scripts.local5_erep_identity_service_v4 import (
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


# These are frozen outputs, not values recomputed by a second implementation.
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
        "digest": "262ddf636d7ea80dd605efea5726657130f6d7f06201e54af21adf874b8e2103",
        "delay": 1,
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
        },
        "canonical_identity": (
            '{"block":6,"input_head":5,"output_tile":10,"sample":"seq|1",'
            '"source_id":449,"stage":2,"stripe":5,"window":8}'
        ),
        "digest": "42a1d292fa48e4af0ef6158b17316203846a5b13659babf6360c84d77d443ed3",
        "delay": 3,
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
        "digest": "f62dfdf22c204356037fdc982e9712809d9b083687a348e59dd4b46de2344ccb",
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
        "digest": "025eefd9c46cedf4eff2621c234f2d6946b7bfaf2573ae919a04d89c699851e8",
        "delay": 0,
    },
)

GOLDEN_ORDERED_LEDGER_DIGEST = (
    "d2c5e0413df8c70cebf6fccfdf061b60c8e2925bd0a5c2c1b98c823731aa0758"
)
GOLDEN_MULTISET_DIGEST = (
    "71f5f2858e3e4e2b59bacac56f262e144dcfc93eca8c422e850f8e28ea90d965"
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


class Local5ErepIdentityServiceV4Test(unittest.TestCase):
    def test_four_frozen_golden_vectors(self) -> None:
        self.assertEqual(len(GOLDEN_VECTORS), 4)
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

    def test_hash_frame_is_length_prefixed_canonical_utf8(self) -> None:
        vector = GOLDEN_VECTORS[2]
        material = transaction_hash_material(vector["kind"], vector["identity"])
        self.assertEqual(
            decode_length_frame(material),
            (
                b"schema",
                SCHEMA.encode("utf-8"),
                b"seed",
                str(DEFAULT_SEED).encode("ascii"),
                b"kind",
                b"weight",
                b"identity",
                vector["canonical_identity"].encode("utf-8"),
            ),
        )

    def test_canonical_json_is_compact_sorted_and_strict_utf8(self) -> None:
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

    def test_every_kind_requires_exactly_its_frozen_fields(self) -> None:
        identities = {
            vector["kind"]: dict(vector["identity"]) for vector in GOLDEN_VECTORS
        }
        for kind, fields in REQUIRED_FIELDS_BY_KIND.items():
            self.assertEqual(set(identities[kind]), set(fields))
            for field in fields:
                broken = dict(identities[kind])
                broken.pop(field)
                with self.subTest(kind=kind, missing=field):
                    with self.assertRaisesRegex(ValueError, "missing fields"):
                        make_transaction(kind, broken)
            with self.subTest(kind=kind, extra="metadata"):
                with self.assertRaisesRegex(ValueError, "unexpected fields: metadata"):
                    make_transaction(kind, dict(identities[kind], metadata="x"))
        with self.assertRaisesRegex(ValueError, "unknown transaction kind"):
            make_transaction("global", identities["relation"])

    def test_candidate_private_occurrence_and_global_index_bypasses_are_rejected(self) -> None:
        base = dict(GOLDEN_VECTORS[0]["identity"])
        for field in (
            "candidate_private",
            "occurrence",
            "transaction_index",
            "global_transaction_index",
            "global_index",
        ):
            with self.subTest(field=field, depth="top"):
                with self.assertRaisesRegex(ValueError, field):
                    make_transaction("relation", dict(base, **{field: 0}))
            with self.subTest(field=field, depth="nested"):
                identity = dict(base, metadata={field: 0})
                with self.assertRaisesRegex(ValueError, field):
                    make_transaction("relation", identity)

    def test_sample_and_hardware_integer_types_are_strict(self) -> None:
        for vector in GOLDEN_VECTORS:
            kind = vector["kind"]
            base = dict(vector["identity"])
            for invalid_sample in ("", 0, False, 1.0, None):
                with self.subTest(kind=kind, sample=invalid_sample):
                    with self.assertRaisesRegex(ValueError, "non-empty string"):
                        make_transaction(kind, dict(base, sample=invalid_sample))
            integer_field = next(field for field in base if field != "sample")
            for invalid_integer in (True, False, 1.0, -1, "1"):
                with self.subTest(
                    kind=kind, field=integer_field, value=invalid_integer
                ):
                    with self.assertRaisesRegex(ValueError, "non-negative integer"):
                        make_transaction(
                            kind, dict(base, **{integer_field: invalid_integer})
                        )

    def test_duplicate_logical_transactions_are_allowed_and_delay_is_stable(self) -> None:
        service = IdentityService()
        identity = dict(GOLDEN_VECTORS[3]["identity"])
        first = service.transaction("final", identity)
        second = service.transaction("final", dict(reversed(tuple(identity.items()))))
        third = service.transaction("final", dict(identity))
        self.assertEqual({first.delay, second.delay, third.delay}, {0})
        self.assertEqual({first.digest, second.digest, third.digest}, {first.digest})

        ledger = service.ledger_digests((first, second, third))
        self.assertEqual(ledger.transaction_count, 3)
        self.assertEqual(ledger.identity_count, 1)
        self.assertEqual(ledger.identity_multiplicities[0].multiplicity, 3)
        self.assertEqual(list(ledger.multiplicity_by_identity.values()), [3])
        self.assertEqual(service.comparable_map((first, second, third)), {first.identity_key: 0})

    def test_ledger_golden_digests_and_per_identity_multiplicity(self) -> None:
        rows = golden_transactions()
        forward = ledger_digests(rows)
        reverse = ledger_digests(reversed(rows))
        self.assertEqual(forward.ordered_digest, GOLDEN_ORDERED_LEDGER_DIGEST)
        self.assertEqual(forward.multiset_digest, GOLDEN_MULTISET_DIGEST)
        self.assertNotEqual(forward.ordered_digest, reverse.ordered_digest)
        self.assertEqual(forward.multiset_digest, reverse.multiset_digest)
        self.assertEqual(forward.identity_count, 4)
        self.assertEqual(
            [audit.multiplicity for audit in forward.identity_multiplicities],
            [1, 1, 1, 1],
        )
        payload = forward.as_dict()
        self.assertEqual(payload["transaction_count"], 4)
        self.assertEqual(payload["ordered_ledger_digest"], GOLDEN_ORDERED_LEDGER_DIGEST)
        self.assertEqual(
            payload["unordered_multiset_digest"], GOLDEN_MULTISET_DIGEST
        )
        self.assertEqual(len(payload["identity_multiplicities"]), 4)

    def test_multiset_digest_and_multiplicity_are_order_independent(self) -> None:
        relation, _, _, final = golden_transactions()
        left = ledger_digests((relation, final, final))
        right = ledger_digests((final, relation, final))
        self.assertNotEqual(left.ordered_digest, right.ordered_digest)
        self.assertEqual(left.multiset_digest, right.multiset_digest)
        self.assertEqual(left.multiplicity_by_identity, right.multiplicity_by_identity)
        self.assertEqual(left.multiplicity_by_identity[final.identity_key], 2)

    def test_candidate_comparison_matches_equal_multisets_in_any_order(self) -> None:
        relation, _, weight, final = golden_transactions()
        comparison = compare_candidate_ledgers(
            (relation, final, relation, weight),
            (weight, relation, final, relation),
        )
        self.assertEqual(comparison.common_identity_count, 3)
        self.assertEqual(comparison.common_transaction_count, 4)
        self.assertEqual(comparison.left_only_transaction_count, 0)
        self.assertEqual(comparison.right_only_transaction_count, 0)
        self.assertTrue(comparison.common_delays_match)
        self.assertTrue(comparison.multiplicities_match)

    def test_candidate_comparison_reports_exact_left_right_multiplicity(self) -> None:
        relation, _, weight, final = golden_transactions()
        comparison = compare_candidate_ledgers(
            (relation, relation, relation, weight),
            (relation, final, final),
        )
        self.assertEqual(comparison.common_identity_count, 1)
        self.assertEqual(comparison.common_transaction_count, 1)
        self.assertEqual(comparison.left_only_transaction_count, 3)
        self.assertEqual(comparison.right_only_transaction_count, 2)
        self.assertTrue(comparison.common_delays_match)

        differences = {
            difference.kind: (
                difference.left_multiplicity,
                difference.right_multiplicity,
                difference.delta,
            )
            for difference in comparison.multiplicity_differences
        }
        self.assertEqual(
            differences,
            {
                "relation": (3, 1, 2),
                "weight": (1, 0, 1),
                "final": (0, 2, -2),
            },
        )
        self.assertEqual(len(comparison.common_multiplicity_differences), 1)
        self.assertFalse(comparison.multiplicities_match)

    def test_candidate_comparison_checks_delay_for_common_identity(self) -> None:
        left = make_transaction("relation", GOLDEN_VECTORS[0]["identity"])
        forged = make_transaction("relation", GOLDEN_VECTORS[0]["identity"])
        object.__setattr__(forged, "delay", (forged.delay + 1) % 4)
        comparison = compare_candidate_ledgers((left,), (forged,))
        self.assertEqual(comparison.delay_mismatches, (left.identity_key,))
        self.assertFalse(comparison.common_delays_match)

    def test_identity_is_copied_and_registered_latency_is_delay_plus_one(self) -> None:
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

    def test_transaction_constructor_rejects_forged_delay(self) -> None:
        row = make_transaction("relation", GOLDEN_VECTORS[0]["identity"])
        for invalid_delay in (True, (row.delay + 1) % 4):
            with self.subTest(delay=invalid_delay):
                with self.assertRaisesRegex(ValueError, "delay does not match"):
                    Transaction(
                        row.schema,
                        row.seed,
                        row.kind,
                        row.identity_json,
                        invalid_delay,
                    )

    def test_schema_seed_kind_and_identity_all_affect_digest(self) -> None:
        identity = GOLDEN_VECTORS[0]["identity"]
        baseline = transaction_digest("relation", identity)
        self.assertNotEqual(baseline, transaction_digest("relation", identity, seed=1))
        self.assertNotEqual(
            baseline,
            transaction_digest(
                "relation", identity, schema="local5_erep_identity_service_v4-test"
            ),
        )
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            transaction_digest("relation", dict(identity, candidate="private"))


if __name__ == "__main__":
    unittest.main()
