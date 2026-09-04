import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_prosperity_motion_bittrace_probe import make_k_support_fc


class ProsperityMotionBittraceProbeTest(unittest.TestCase):
    def test_make_k_support_fc_concatenates_heads(self):
        k = np.zeros((2, 1, 2, 3, 4), dtype=np.uint8)
        k[0, 0, 0, 0, 1] = 1
        k[1, 0, 1, 2, 3] = 1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.npz"
            np.savez(
                path,
                k_shape=np.asarray(k.shape, dtype=np.int32),
                k_bits_packed=np.packbits(
                    k.reshape(-1),
                    bitorder="little",
                ),
                projection_weight_int8=np.zeros((8, 8), dtype=np.int8),
                gate_q17=np.full((1, 2, 6), 64, dtype=np.uint16),
            )
            operator, source = make_k_support_fc(
                {"name": "S0.B0.attn", "file": str(path)},
                Path(temp),
            )
        self.assertEqual(
            list(operator.activation_tensor.sparse_map.shape),
            [1, 2, 3, 8],
        )
        self.assertEqual(
            int(operator.activation_tensor.sparse_map.sum().item()),
            2,
        )
        self.assertEqual(source["weight_shape"], [8, 8])
        self.assertEqual(source["gate_unique_codes"], [64])


if __name__ == "__main__":
    unittest.main()
