"""CPU synthetic contracts; no native checkpoint, raw dataset, network or GPU."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import collect_fast_cache as c


class SamplingChecks(unittest.TestCase):
    def test_fixed_geometry_identity_seed_selection_and_no_GT_argument(self):
        for stage, shape in c.SHAPES.items():
            ids = c.sample_indices('fixed_target_0001', stage)
            self.assertEqual(ids, sorted(set(ids))); self.assertEqual(len(ids), 128)
            self.assertTrue(all(0 <= x < shape[2]*shape[3] for x in ids))
            self.assertEqual(ids, c.sample_indices('fixed_target_0001', stage))
            self.assertNotEqual(ids, c.sample_indices('fixed_target_0002', stage))
            # Independently rank binary SHA digests; collector ranks hexadecimal strings.
            keys = [(hashlib.sha256(('flow-fast-warmstart-spatial-v1-20261014:fixed_target_0001:stage'
                                    +str(stage)+':'+str(i)).encode()).digest(), i) for i in range(shape[2]*shape[3])]
            self.assertEqual(ids, sorted(i for _, i in sorted(keys)[:128]))
        with self.assertRaises(TypeError): c.sample_indices('fixed_target_0001', 0, GT='forbidden')
        with self.assertRaises(ValueError): c.sample_indices('fixed_target_0001', 1)

    def test_gather_preserves_allT_channels_and_logical_spatial_coordinates(self):
        shape = (10, 1, 12, 16, 3)
        value = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        indices = list(range(64))+list(range(128, 192))
        sampled = c.sample_tensor(value, indices, shape)
        self.assertEqual(list(sampled.shape), [10, 128, 3])
        for t in (0, 1, 9):
            for slot in (0, 63, 64, 127):
                y, x = divmod(indices[slot], 16)
                torch.testing.assert_close(sampled[t, slot], value[t, 0, y, x], rtol=0, atol=0)
        self.assertEqual(sampled[9, 127, 2].item(), 5759.)
        # A branch target is separately sampled, with no residual/input addition.
        branch = -2*value+3
        label = c.sample_tensor(branch, indices, shape)
        torch.testing.assert_close(label, -2*sampled+3, rtol=0, atol=0)
        self.assertFalse(torch.equal(label, -sampled+3))
        self.assertFalse(sampled.requires_grad)
        self.assertEqual(c.tensor_sha(sampled), hashlib.sha256(sampled.numpy().tobytes()).hexdigest())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'fixture.pt'; torch.save({'u': sampled, 'target': label, 'spatial_indices': indices}, path)
            restored = torch.load(path, map_location='cpu', weights_only=True)
            torch.testing.assert_close(restored['target'], label, rtol=0, atol=0)

    def test_shapes_nonfinite_duplicates_and_partialT_rejected(self):
        shape = (10, 1, 12, 16, 3); value = torch.zeros(shape); ids = list(range(128))
        cases = [(value[:9], ids, shape), (value.double(), ids, shape),
                 (value, [0]*128, shape), (value, list(range(127)), shape),
                 (value, list(range(127))+[192], shape)]
        bad = value.clone(); bad[9, 0, 11, 15, 2] = float('nan')
        cases.append((bad, ids, shape))  # Even an unsampled nonfinite producer is refused.
        for args in cases:
            with self.assertRaises(ValueError): c.sample_tensor(*args)


class PopulationAndInputs(unittest.TestCase):
    def fixture(self):
        fits = [{'target_id': 'fit_seq'+str(s)+'_'+str(i), 'sequence': 'fit_seq'+str(s),
                 'protocol_split': 'train'} for s in range(10) for i in range(2)]
        dev = [{'target_id': 'dev_seq'+str(s)+'_'+str(i), 'sequence': 'dev_seq'+str(s),
                'protocol_split': 'train'} for s in range(2) for i in range(4)]
        valid = [{'target_id': 'valid_seq_1', 'sequence': 'valid_seq', 'protocol_split': 'valid'}]
        roles = {r['target_id']: 'gate_fit' for r in fits}
        roles.update({r['target_id']: 'gate_development' for r in dev})
        ids = [r['target_id'] for r in fits]
        manifest = {'schema': 'flow_recovery_svs_fit20_v1', 'checkpoint_sha256': c.CHECKPOINT_SHA,
                    'checkpoint_optimizer_steps': 135930, 'target_ids': ids, 'target_ids_sha256': c.digest(ids),
                    'sequence_counts': {'fit_seq'+str(s): 2 for s in range(10)}}
        return manifest, fits+dev+valid, roles, dev

    def test_fit20_only_with_mixed_canonical_and_train_only_roles(self):
        manifest, canonical, roles, dev = self.fixture()
        rows = c.fit_members(manifest, canonical, roles, dev)
        self.assertEqual([r['target_id'] for r in rows], manifest['target_ids'])
        self.assertEqual(len(rows), 20)
        for kind in ('dev_in_fit', 'valid_in_fit', 'duplicate', 'checkpoint'):
            bad = copy.deepcopy(manifest)
            if kind == 'dev_in_fit': bad['target_ids'][0] = dev[0]['target_id']
            elif kind == 'valid_in_fit': bad['target_ids'][0] = 'valid_seq_1'
            elif kind == 'duplicate': bad['target_ids'][0] = bad['target_ids'][1]
            else: bad['checkpoint_optimizer_steps'] = 135931
            bad['target_ids_sha256'] = c.digest(bad['target_ids'])
            with self.assertRaises(ValueError): c.fit_members(bad, canonical, roles, dev)

    def test_event_only_read_succeeds_with_absent_GT_and_refuses_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); event_path = root/'events.npy'
            values = np.zeros((10, 480, 640), dtype=np.float32); values[9, -1, -1] = 2.5
            np.save(event_path, values)
            class Native:
                @staticmethod
                def target_files(row): return {'input': 'events.npy', 'gt': 'NEVER_CREATED_GT.npy', 'mask': 'NEVER_CREATED_MASK.npy'}
                @staticmethod
                def checked_path(root, rel): return root/rel
            hashes = {'events.npy': {'sha256': c.sha(event_path)}}
            loaded, paths, evidence = c.read_events(root, {'target_id': 'synthetic'}, Native, hashes)
            self.assertEqual(loaded[9, -1, -1], 2.5)
            self.assertEqual(set(evidence), {'events'})
            self.assertFalse((root/paths['gt']).exists()); self.assertFalse((root/paths['mask']).exists())
            with event_path.open('ab') as stream: stream.write(b'tamper')
            with self.assertRaisesRegex(ValueError, 'SHA'): c.read_events(root, {}, Native, hashes)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
