#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parent.parent / "scripts" /
          "build_m1323_ep34_decoder_capture_adapter_source.py")
SPEC = importlib.util.spec_from_file_location("m1323_source", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def retained_payload(sample, global_order, name):
    stem = "s{:02d}_o{:05d}_{}".format(
        sample, global_order, M.hashlib.sha256(name.encode()).hexdigest()[:12])
    return {
        "retained": True, "raw_fp32_sha256": "1" * 64,
        "compressed_fp32": "payloads/{}.fp32.zlib".format(stem),
        "compressed_sha256": "2" * 64,
        "support_sign": "payloads/{}.support_sign.le.bitpack".format(stem),
        "support_sign_sha256": "3" * 64, "positive_plane_bytes": 1,
        "negative_plane_bytes": 1,
    }


def input_stats(shape):
    elements = M.math.prod(shape)
    return {"shape": list(shape), "stride": [1] * len(shape), "dtype": "torch.float32",
            "elements": elements, "bytes": elements * 4, "active": 0,
            "positive": 0, "negative": 0, "nonfinite": 0}


def make_ordered():
    inventory = M.frozen_inventory_names()
    cohort = M.expected_cohort()
    module_rows = []
    for category, names in inventory.items():
        for name in names:
            module_rows.append((category, name))
    rows = []
    for sample in range(40):
        for category, name in module_rows:
            global_order = len(rows)
            shape = [1]
            if category == "decoder_convtranspose":
                shape = list(M.SHAPES[M.MODULES.index(name)])
            identity = cohort[sample]
            rows.append({
                "global_order": global_order, "global_sample_id": sample,
                "cohort": identity["cohort"], "sequence": identity["sequence"],
                "sample_key": identity["sample_key"],
                "source_sha256": identity["source_sha256"],
                "category": category, "name": name, "input": input_stats(shape),
                "payload": (retained_payload(sample, global_order, name) if category in
                            {"c1_conv3x3", "decoder_convtranspose"}
                            else dict(M.NONRETAINED_PAYLOAD)),
            })
    return rows, inventory, cohort


def valid_weight_rows():
    checkpoint = "a" * 64
    rows = []
    for ordinal, shape in enumerate(M.WEIGHT_SHAPES):
        rows.append({"module_ordinal": ordinal, "module": M.MODULES[ordinal],
                     "checkpoint_sha256": checkpoint,
                     "weight": {"shape": list(shape), "dtype": "torch.float32",
                                "layout": "C_ORDER_CONTIGUOUS", "byte_order": "little",
                                "content_bytes": M.math.prod(shape) * 4,
                                "content_sha256": ("%x" % (ordinal + 1)) * 64},
                     "bias": None})
    return rows, checkpoint


class FullOrderedPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.inventory, cls.cohort = make_ordered()

    def call(self, rows):
        return M.decoder_rows_from_ordered(rows, self.inventory, self.cohort)

    def test_exact_9880_population_passes_before_projection(self):
        selected, identity = self.call(self.rows)
        self.assertEqual(len(selected), 120)
        self.assertEqual([row["module_ordinal"] for row in selected[:4]], [0, 1, 2, 3])
        self.assertEqual((selected[0]["global_sample_id"],
                          selected[-1]["global_sample_id"]), (10, 39))
        self.assertTrue(identity["all_sample_sequences_equal"])
        self.assertEqual(identity["ordered_rows"], 9880)

    def test_bool_and_noncontiguous_global_order_rejected(self):
        rows = list(self.rows)
        rows[1] = dict(rows[1]); rows[1]["global_order"] = True
        with self.assertRaisesRegex(M.M1323Error, "exact file ordinal"):
            self.call(rows)
        rows = list(self.rows)
        rows[500] = dict(rows[500]); rows[500]["global_order"] = 499
        with self.assertRaisesRegex(M.M1323Error, "exact file ordinal"):
            self.call(rows)

    def test_ignored_duplicate_or_replacement_rejected(self):
        # Keep the file ordinal and count valid while duplicating an unretained row.
        rows = list(self.rows)
        source = rows[300]
        victim = rows[301]
        self.assertFalse(source["payload"]["retained"])
        replacement = copy.deepcopy(source)
        replacement["global_order"] = victim["global_order"]
        rows[301] = replacement
        with self.assertRaisesRegex(M.M1323Error, "duplicated/missing/replaced"):
            self.call(rows)

        rows = list(self.rows)
        rows[301] = copy.deepcopy(rows[301])
        rows[301]["name"] = "invented.ignored.module"
        with self.assertRaisesRegex(M.M1323Error, "not frozen"):
            self.call(rows)

    def test_cross_call_retained_payload_alias_rejected(self):
        rows = list(self.rows)
        retained = [index for index, row in enumerate(rows)
                    if row["payload"].get("retained") is True]
        victim, source = retained[1], retained[0]
        rows[victim] = copy.deepcopy(rows[victim])
        rows[victim]["payload"]["compressed_fp32"] = \
            rows[source]["payload"]["compressed_fp32"]
        rows[victim]["payload"]["support_sign"] = rows[source]["payload"]["support_sign"]
        with self.assertRaisesRegex(M.M1323Error, "exact call identity|alias"):
            self.call(rows)

    def test_sample_identity_and_structure_rejected_on_ignored_row(self):
        rows = list(self.rows)
        rows[300] = dict(rows[300]); rows[300]["source_sha256"] = "f" * 64
        with self.assertRaisesRegex(M.M1323Error, "differs from M1313"):
            self.call(rows)
        rows = list(self.rows)
        rows[300] = dict(rows[300]); rows[300]["extra"] = 1
        with self.assertRaisesRegex(M.M1323Error, "row keys"):
            self.call(rows)

    def test_sample_or_module_order_drift_rejected(self):
        rows = list(self.rows)
        rows[247] = dict(rows[247]); rows[247]["global_sample_id"] = 0
        with self.assertRaisesRegex(M.M1323Error, "contiguous"):
            self.call(rows)
        rows = list(self.rows)
        # Swap two unretained ATLIF rows while preserving exact file ordinals.
        a, b = 255, 256
        first, second = copy.deepcopy(rows[a]), copy.deepcopy(rows[b])
        first["global_order"], second["global_order"] = b, a
        rows[a], rows[b] = second, first
        with self.assertRaisesRegex(M.M1323Error, "execution order"):
            self.call(rows)


class ExactIntegerBoundaryTests(unittest.TestCase):
    def test_weight_bool_ordinal_rejected(self):
        rows, checkpoint = valid_weight_rows()
        rows[1]["module_ordinal"] = True
        with self.assertRaisesRegex(M.M1323Error, "exact integer"):
            M.validate_weight_identities(rows, checkpoint)

    def test_payload_bool_ordinal_rejected_before_io(self):
        with self.assertRaisesRegex(M.M1323Error, "exact integer"):
            M.audit_two_plane_payload(Path("absent"), Path("absent"), (1,), True)

    def test_valid_weight_identity_and_cli_boundary(self):
        rows, checkpoint = valid_weight_rows()
        self.assertEqual(len(M.validate_weight_identities(rows, checkpoint)), 4)
        with self.assertRaises(M.M1323Error):
            M.main([])


if __name__ == "__main__":
    unittest.main()
