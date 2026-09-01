#!/usr/bin/env python3
"""Source-only tests for M1754 interpreter-bound wrapper."""
from __future__ import print_function
import importlib.util, json, os
from pathlib import Path
import tempfile, unittest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_m1754_m1747_tsbg_interpreter_bound_one_shot.py"
SPEC = importlib.util.spec_from_file_location("m1754", str(SOURCE)); M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)

class TestM1754(unittest.TestCase):
    def seal_dir(self, root):
        names = sorted(p.name for p in root.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
        sums = root / "SHA256SUMS"; sums.write_text("".join("{}  {}\n".format(M.sha256(root/n), n) for n in names))
        (root/"SHA256SUMS.seal.sha256").write_text("{}  SHA256SUMS\n".format(M.sha256(sums)))
    def seal_file(self, path):
        side=Path(str(path)+".sha256"); outer=Path(str(path)+".sha256.seal.sha256")
        side.write_text("{}  {}\n".format(M.sha256(path),path.name)); outer.write_text("{}  {}\n".format(M.sha256(side),side.name))
    def authority(self, root):
        ids=M.source_identities(); review=root/"review"; review.mkdir()
        (review/"review.json").write_text(json.dumps({"schema":M.REVIEW_SCHEMA,"status":M.REVIEW_STATUS,"identity":ids,
            "authorization":{"m1756_release_may_be_created":True,"execution":False,"analysis_run":False},"claim_boundary":{"paper_result":False}},sort_keys=True))
        self.seal_dir(review); rb=M.validate_future_review(review,ids)
        release=root/"release.json"; rid=dict(ids); rid.update({"m1755_review_sha256":rb["review_sha256"],"m1755_review_outer_seal_file_sha256":rb["outer_seal_file_sha256"]})
        (release).write_text(json.dumps({"schema":M.RELEASE_SCHEMA,"status":M.RELEASE_STATUS,"identity":rid,
            "authorization":{"wrapper_runs":1,"interpreter_preflights":1,"execs":1,"analysis_runs":1,"capture_verifications":1,"result_publications":1,"automatic_retry":False,"gpu_runs":0,"eda_runs":0,"all_other_runs":0},"claim_boundary":{"paper_result":False}},sort_keys=True))
        self.seal_file(release); return review,release,ids
    def test_exact_static(self):
        M.validate_static(); self.assertEqual(M.sha256(M.M1747_SOURCE),M.M1747_SOURCE_SHA256)
    def test_failure_consumed(self):
        row=M.strict_json(M.FAILURE); self.assertTrue(row["absence_and_budget"]["m1749_authority_consumed"]); self.assertEqual(row["absence_and_budget"]["payload_replays"],0)
    def test_authority_valid(self):
        with tempfile.TemporaryDirectory() as t:
            r,l,i=self.authority(Path(t)); rb=M.validate_future_review(r,i); self.assertEqual(len(M.validate_future_release(l,rb,i)["release_sha256"]),64)
    def test_review_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            r,l,i=self.authority(Path(t)); p=r/"review.json"; d=json.loads(p.read_text()); d["identity"]["interpreter_sha256"]="0"*64; p.write_text(json.dumps(d)); self.seal_dir(r)
            with self.assertRaises(M.M1754Error): M.validate_future_review(r,i)
    def test_release_budget_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            r,l,i=self.authority(Path(t)); rb=M.validate_future_review(r,i); d=json.loads(l.read_text()); d["authorization"]["execs"]=2; l.write_text(json.dumps(d)); self.seal_file(l)
            with self.assertRaises(M.M1754Error): M.validate_future_release(l,rb,i)
    def test_wrong_interpreter_before_attempt(self):
        old=M.verify_authority; olda=M.ATTEMPT
        with tempfile.TemporaryDirectory() as t:
            M.ATTEMPT=Path(t)/"attempt"; M.verify_authority=lambda: ({},{},{})
            try:
                with self.assertRaises(M.M1754Error): M.run_execution()
                self.assertFalse(M.ATTEMPT.exists())
            finally: M.verify_authority=old; M.ATTEMPT=olda
    def test_authority_precedes_preflight_and_attempt(self):
        text=SOURCE.read_text(); body=text[text.index("def run_execution():"):text.index("def source_self_check():")]
        self.assertLess(body.index("verify_authority()"),body.index("interpreter_preflight()")); self.assertLess(body.index("interpreter_preflight()"),body.index("ATTEMPT.mkdir()")); self.assertLess(body.index("ATTEMPT.mkdir()"),body.index("os.execve"))
    def test_exec_exact_m1747(self):
        text=SOURCE.read_text(); self.assertIn("[str(INTERPRETER), str(M1747_SOURCE), \"--run-analysis\"]",text)
    def test_self_check_inert(self):
        row=M.source_self_check(); self.assertFalse(row["attempt_created"]); self.assertEqual(row["analysis_runs"],0)
    def test_no_network_or_analysis_in_self_check(self):
        text=SOURCE.read_text(); self.assertNotIn("import socket",text); self.assertNotIn("requests",text); self.assertFalse(os.path.lexists(str(M.ATTEMPT)))

if __name__ == "__main__": unittest.main()
