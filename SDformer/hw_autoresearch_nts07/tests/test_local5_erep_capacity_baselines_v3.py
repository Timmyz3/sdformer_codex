from __future__ import annotations

import unittest

from scripts import local5_erep_capacity_baselines_v3 as model


def entry(
    head: int,
    records: int,
    *,
    accesses: int = 3,
    miss: int = 10,
    fill: int = 2,
    hit: int = 1,
    read: int = 2,
    tag_bits: int = 7,
) -> model.HeadEntry:
    return model.HeadEntry(
        head=head,
        records=records,
        accesses=accesses,
        miss_cycles=miss,
        fill_cycles=fill,
        hit_cycles=hit,
        read_cycles=read,
        tag_bits=tag_bits,
    )


class Local5ErepCapacityBaselinesV3Test(unittest.TestCase):
    def test_c4_payload_geometry_is_exact(self) -> None:
        self.assertEqual(model.C4_PAYLOAD_CAPACITY_RECORDS, 5014)
        self.assertEqual(model.C4_PAYLOAD_STATIC_UNUSED_BITS, 32)
        self.assertEqual(
            model.C4_PAYLOAD_CAPACITY_RECORDS * model.RELATION_RECORD_BITS + 32,
            model.C4_PAYLOAD_CAPACITY_BITS,
        )

    def test_oracle_maximizes_saved_cycles_and_has_fixed_tie_break(self) -> None:
        heads = (
            entry(10, 2, miss=10, fill=0, hit=0, read=5),  # saves 10
            entry(11, 2, miss=10, fill=0, hit=0, read=5),  # saves 10
            entry(12, 4, miss=10, fill=0, hit=0, read=5),  # saves 10
        )
        result = model.evaluate_c4_oracle(
            heads,
            capacity_bits=4 * model.RELATION_RECORD_BITS,
        )
        # Heads 10+11 save 20 cycles.  Equal single-entry ties prefer lower
        # input index; the tie contract itself is frozen as a public constant.
        self.assertEqual(result.admission, (True, True, False))
        self.assertIn("lexicographically smallest", model.ORACLE_TIE_BREAK)

        equal = model.evaluate_c4_oracle(
            (entry(20, 4), entry(21, 4)),
            capacity_bits=4 * model.RELATION_RECORD_BITS,
        )
        self.assertEqual(equal.admission, (True, False))

        zero_value_prefix = model.evaluate_c4_oracle(
            (
                entry(30, 0, accesses=2, miss=7, fill=2, hit=2, read=3),
                entry(31, 0, accesses=2, miss=3, fill=1, hit=0, read=1),
            ),
            capacity_bits=32,
        )
        self.assertEqual(zero_value_prefix.admission, (True, True))

    def test_first_fit_is_head_ordered_and_never_replaces(self) -> None:
        heads = (entry(0, 3), entry(1, 4), entry(2, 2))
        result = model.evaluate_c4_first_fit(
            heads,
            capacity_bits=5 * model.RELATION_RECORD_BITS,
        )
        self.assertEqual(result.admission, (True, False, True))
        self.assertEqual(result.admitted_heads, (0, 2))

        harmful = entry(
            3, 0, accesses=2, miss=1, fill=4, hit=2, read=2
        )
        self.assertEqual(
            model.head_order_first_fit_admission((harmful,), capacity_bits=32),
            (False,),
        )
        with self.assertRaises(ValueError):
            model.head_order_first_fit_admission((entry(2, 1), entry(1, 1)))

    def test_cycle_and_record_ledger_uses_all_explicit_costs(self) -> None:
        admitted = entry(
            0,
            2,
            accesses=3,
            miss=10,
            fill=2,
            hit=1,
            read=2,
            tag_bits=9,
        )
        rejected = entry(
            1,
            2,
            accesses=2,
            miss=7,
            fill=4,
            hit=3,
            read=5,
            tag_bits=11,
        )
        result = model.evaluate_admission(
            (admitted, rejected),
            (True, False),
            baseline="c4_first_fit",
            capacity_bits=2 * model.RELATION_RECORD_BITS,
        )
        self.assertEqual(result.cycles, (10 + 2 + 2 * (1 + 2)) + 2 * 7)
        self.assertEqual(result.miss_path_cycles, 24)
        self.assertEqual(result.fill_path_cycles, 2)
        self.assertEqual(result.hit_path_cycles, 2)
        self.assertEqual(result.read_path_cycles, 4)
        self.assertEqual(
            result.cycles,
            result.miss_path_cycles
            + result.fill_path_cycles
            + result.hit_path_cycles
            + result.read_path_cycles,
        )
        self.assertEqual(result.record_writes, 2)
        self.assertEqual(result.record_reads, 4)
        self.assertEqual(result.entry_misses, 3)
        self.assertEqual(result.entry_fills, 1)
        self.assertEqual(result.entry_hits, 2)
        self.assertEqual(result.capacity_misses, 2)
        self.assertEqual(result.tag_bits, 9)
        self.assertEqual(result.valid_bits, 1)
        self.assertEqual(result.metadata_bits, 10)
        self.assertFalse(result.metadata_in_payload_capacity)
        self.assertEqual(
            result.allocated_state_bits,
            result.payload_capacity_bits + result.metadata_bits,
        )

    def test_zero_record_head_is_an_entry_but_has_no_payload_traffic(self) -> None:
        result = model.evaluate_c4_first_fit(
            (entry(3, 0, tag_bits=13),),
            capacity_bits=32,
        )
        self.assertEqual(result.admission, (True,))
        self.assertEqual(result.record_writes, 0)
        self.assertEqual(result.record_reads, 0)
        self.assertEqual(result.unused_bits, 32)
        self.assertEqual(result.metadata_bits, 14)

    def test_c4_capacity_boundary_keeps_32_unusable_bits(self) -> None:
        full = model.evaluate_c4_first_fit(
            (entry(0, model.C4_PAYLOAD_CAPACITY_RECORDS),)
        )
        self.assertEqual(full.admission, (True,))
        self.assertEqual(full.payload_used_records, 5014)
        self.assertEqual(full.unused_bits, 32)

        oversized = model.evaluate_c4_first_fit(
            (entry(0, model.C4_PAYLOAD_CAPACITY_RECORDS + 1),)
        )
        self.assertEqual(oversized.admission, (False,))
        self.assertEqual(oversized.capacity_misses, 3)
        self.assertEqual(oversized.record_writes, 0)

    def test_c5_has_no_capacity_miss_and_reports_stage3_worst_case(self) -> None:
        heads = tuple(entry(index, 450) for index in range(24))
        result = model.evaluate_c5_full(heads)
        self.assertTrue(all(result.admission))
        self.assertEqual(result.capacity_misses, 0)
        self.assertEqual(result.payload_capacity_bits, 1_209_600)
        self.assertEqual(result.payload_used_records, 10_800)
        self.assertEqual(result.unused_bits, 0)
        self.assertEqual(result.record_writes, 10_800)
        self.assertEqual(result.record_reads, 21_600)


if __name__ == "__main__":
    unittest.main()
