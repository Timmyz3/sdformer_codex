"""Retain six V1 interface tests and add exact original version-source check."""
import ast
from pathlib import Path
import unittest
import test_preflight_cpu as original
import preflight_readonly_v2 as corrected
original.p=corrected
PreflightTests=original.PreflightTests


class VersionSourceRegression(unittest.TestCase):
    def test_torchvision_version_matches_original_training_contract_api(self):
        tree=ast.parse(Path(corrected.__file__).read_text())
        entries=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Dict):
                entries += [value for key,value in zip(node.keys,node.values)
                            if isinstance(key,ast.Constant) and key.value=='torchvision']
        self.assertEqual(len(entries),1)
        self.assertEqual(ast.unparse(entries[0]),'str(torchvision.__version__)')
        # Original recorded API includes the CUDA build suffix. Distribution
        # metadata can report only 0.20.1 and is not an interchangeable source.
        import types
        module=types.SimpleNamespace(__version__='0.20.1+cu124')
        self.assertNotEqual(str(module.__version__),'0.20.1')


if __name__=='__main__':unittest.main(verbosity=2)
