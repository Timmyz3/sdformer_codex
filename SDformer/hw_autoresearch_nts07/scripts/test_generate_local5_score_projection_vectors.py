#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from scripts.generate_local5_score_projection_vectors import (
    pack_fields,
    read_memh,
)


class Local5ScoreProjectionVectorTest(unittest.TestCase):
    def test_pack_fields_places_candidate_zero_in_lsb(self) -> None:
        self.assertEqual(pack_fields([1, 2, 3], 4), 0x321)

    def test_read_memh_checks_entry_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.memh"
            path.write_text("01\n02\n", encoding="ascii")
            self.assertEqual(read_memh(path, (1, 2)).tolist(), [[1, 2]])
            with self.assertRaisesRegex(ValueError, "entries=2 expected=3"):
                read_memh(path, (1, 3))


if __name__ == "__main__":
    unittest.main()
