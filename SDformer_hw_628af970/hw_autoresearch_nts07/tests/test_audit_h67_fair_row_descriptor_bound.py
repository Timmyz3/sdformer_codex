#!/usr/bin/env python3

import unittest

from scripts.audit_h67_fair_row_descriptor_bound import ROW_RE, SUM_RE


class FairRowDescriptorBoundParserTest(unittest.TestCase):
    def test_row_and_sum_parse(self) -> None:
        text = (
            "FAIR_ROW row=0 active=2 skip=0 fixed=10 rqtb=8 shared=8 "
            "fslots=450 rslots=300 equal=150\n"
            "FAIR_SUM rows=138 skip=33 fixed=112589 rqtb=94891 shared=87034 "
            "fpairs=31050 fslots=62100 fequal=28001 rpairs=31050 "
            "rslots=34099 requal=28001\n"
        )
        row = ROW_RE.search(text)
        summary = SUM_RE.search(text)
        self.assertIsNotNone(row)
        self.assertIsNotNone(summary)
        self.assertEqual(int(row["rslots"]), 300)
        self.assertEqual(int(summary["rqtb"]), 94891)


if __name__ == "__main__":
    unittest.main()
