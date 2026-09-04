import unittest

from scripts.miter_local5_destination_coefficient_fusion import (
    coefficient_projection,
)
from scripts.miter_local5_source_owned_gate_quotient_rtl import (
    HEAD_DIM,
    OUT_DIM,
    ROLES,
    SOURCES,
    source_for,
)


class DestinationCoefficientFusionTest(unittest.TestCase):
    def test_matches_direct_projection(self) -> None:
        candidate_k = []
        valid_mask = []
        packed_gates = []
        for destination in range(SOURCES):
            k_word = 0
            mask = 0
            gates = 0
            for role in range(ROLES):
                if source_for(destination, role) is None:
                    continue
                mask |= 1 << role
                gate = 3 + role
                gates |= gate << (role * 9)
                if (destination + role) % 3 == 0:
                    k_word |= 1 << (role * HEAD_DIM + (destination % HEAD_DIM))
            candidate_k.append(k_word)
            valid_mask.append(mask)
            packed_gates.append(gates)
        weights = [[lane - 7, 5 - lane] for lane in range(HEAD_DIM)]
        observed = coefficient_projection(
            candidate_k=candidate_k,
            valid_mask=valid_mask,
            packed_gates=packed_gates,
            weights=weights,
        )
        expected = [[0 for _ in range(OUT_DIM)] for _ in range(SOURCES)]
        for destination in range(SOURCES):
            for role in range(ROLES):
                if not ((valid_mask[destination] >> role) & 1):
                    continue
                gate = (packed_gates[destination] >> (role * 9)) & 0x1FF
                k_bitmap = (candidate_k[destination] >> (role * HEAD_DIM)) & 0xFFFFFFFF
                for lane in range(HEAD_DIM):
                    if not ((k_bitmap >> lane) & 1):
                        continue
                    for out_index in range(OUT_DIM):
                        expected[destination][out_index] += gate * weights[lane][out_index]
        self.assertEqual(observed["acc"], expected)
        self.assertEqual(observed["invalid_nonzero_gates"], 0)


if __name__ == "__main__":
    unittest.main()
