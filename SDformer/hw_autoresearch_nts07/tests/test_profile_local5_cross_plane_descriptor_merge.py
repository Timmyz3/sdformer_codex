import unittest

import numpy as np

from scripts.profile_local5_cross_plane_descriptor_merge import profile_arrays


class Local5CrossPlaneDescriptorMergeTest(unittest.TestCase):
    def test_equal_active_payload_saves_one_side_terms(self) -> None:
        result = profile_arrays(
            group_offsets=np.asarray([0, 4]),
            planes=np.asarray([0, 0, 1, 1]),
            ys=np.asarray([0, 0, 0, 0]),
            xs=np.asarray([0, 1, 0, 1]),
            k_bitmaps=np.asarray([3, 4, 3, 5], dtype=np.uint64),
            valid_masks=np.asarray([1, 1, 1, 1], dtype=np.uint8),
            gates=np.asarray([[8, 0, 0, 0, 0]] * 4, dtype=np.uint16),
            terms=np.asarray([2, 1, 2, 1], dtype=np.uint16),
            sources_per_plane=2,
        )
        self.assertEqual(result["payload_equal_active_pairs"], 1)
        self.assertEqual(result["theoretical_saved_source_terms"], 2)
        self.assertEqual(result["baseline_source_terms"], 6)

    def test_coordinate_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "coordinate map mismatch"):
            profile_arrays(
                group_offsets=np.asarray([0, 4]),
                planes=np.asarray([0, 0, 1, 1]),
                ys=np.asarray([0, 0, 0, 0]),
                xs=np.asarray([0, 1, 0, 2]),
                k_bitmaps=np.asarray([0, 0, 0, 0], dtype=np.uint64),
                valid_masks=np.asarray([0, 0, 0, 0], dtype=np.uint8),
                gates=np.zeros((4, 5), dtype=np.uint16),
                terms=np.zeros(4, dtype=np.uint16),
                sources_per_plane=2,
            )


if __name__ == "__main__":
    unittest.main()
