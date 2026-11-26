"""CPU/read-only interface rejection tests; no native model or GPU forwards."""
import ast
import contextlib
import fcntl
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import preflight_readonly as p


class PreflightTests(unittest.TestCase):
    def test_fixed_original_native_cli_without_execution_flag(self):
        code=Path(__file__).resolve().parents[2]/'flow_story_20261012/baseline'
        sys.path.insert(0,str(code))
        for name,expected in p.SOURCES.items():self.assertEqual(p.sha(code/name),expected)
        spec=importlib.util.spec_from_file_location('_readonly_cli_test',code/'train_native_baseline.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        a=m.parse_args(p.resume_arguments())
        self.assertEqual((a.phase,a.max_steps,a.wall_hours,a.seed,a.workers,a.cpu_threads,a.device),('continue',480000,11.5,0,2,2,0))
        self.assertEqual((a.checkpoint_steps,a.keep_last,a.log_every),(250,2,10))
        self.assertFalse(a.run_gpu);self.assertEqual(a.scope,'canonical6000')
        self.assertEqual(a.resume.name,'checkpoint_step_000271482.pt')

    def test_no_execution_flag_in_new_script(self):
        with patch.object(sys,'argv',['preflight_readonly.py','--run-gpu']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:p.main()
        self.assertEqual(cm.exception.code,2)

    def test_local_paths_must_be_paired_before_remote_access(self):
        with patch.object(sys,'argv',['preflight_readonly.py','--local-checkpoint','missing']),patch.object(p,'remote_preflight') as remote:
            with self.assertRaisesRegex(ValueError,'supplied together'):p.main()
        remote.assert_not_called()

    def test_lock_probe_does_not_create_or_write(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'lock'
            self.assertEqual(p.readonly_lock(path),'MISSING_NOT_CREATED');self.assertFalse(path.exists())
            path.write_bytes(b'unchanged');before=(p.sha(path),path.stat().st_mtime_ns)
            self.assertEqual(p.readonly_lock(path),'NO_EXCLUSIVE_OWNER_AT_OBSERVATION')
            with path.open('rb') as owner:
                fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertEqual(p.readonly_lock(path),'EXCLUSIVE_OWNER_PRESENT')
            self.assertEqual((p.sha(path),path.stat().st_mtime_ns),before)

    def test_invalid_checkpoint_rejected_before_torch_deserialization(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.pt';path.write_bytes(b'not a trusted checkpoint')
            with patch.object(torch,'load') as loader:
                with self.assertRaisesRegex(ValueError,'checkpoint bytes'):p.checkpoint_audit(path,Path(d)/'missing.json')
            loader.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())

    def test_only_readonly_external_commands_and_no_training_call(self):
        tree=ast.parse(Path(p.__file__).read_text())
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('run_training',attrs);self.assertNotIn('main',attrs)
        self.assertNotIn('Popen',attrs);self.assertNotIn('run',attrs)
        external=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                  and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess']
        self.assertEqual(len(external),2)
        for n in external:
            self.assertEqual(n.func.attr,'check_output');self.assertEqual(n.args[0].elts[0].value,'nvidia-smi')


if __name__=='__main__':unittest.main(verbosity=2)
