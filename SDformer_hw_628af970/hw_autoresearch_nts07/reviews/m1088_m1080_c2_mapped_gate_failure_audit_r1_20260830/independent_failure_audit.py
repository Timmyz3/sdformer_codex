#!/usr/bin/env python3
"""Read-only mechanical audit for the consumed M1080 quarantine."""
from pathlib import Path
import hashlib
import re

ROOT = Path(__file__).resolve().parents[2]
Q = ROOT / "results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.2746017.quarantine"

def text(path): return path.read_text(errors="replace")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

checks = []
checks.append(("quarantine recursive seal", sha(Q / "SHA256SUMS") == text(Q / "SHA256SUMS.seal.sha256").split()[0]))
checks.append(("fresh DC rc", text(Q / "dc/dc.rc").strip() == "0"))
checks.append(("mapped compile rc", text(Q / "mapped_vcs/compile.rc").strip() == "0"))
checks.append(("case0 process rc", text(Q / "mapped_vcs/case0.rc").strip() == "0"))
case = text(Q / "mapped_vcs/case0.log")
checks.append(("header accepted edge 3", "M979_SAIF_WINDOW_START axis=K1 case=0 edge=3" in case))
checks.append(("watchdog 300015000 ps", "at time 300015000 ps" in case and "M979 watchdog" in case))
checks.append(("no case PASS", "PASS M979 mapped replay" not in case))
area = text(Q / "dc/reports/area.rpt")
checks.append(("area diagnostic 124351.163170", "Total cell area:                124351.163170" in area))
timing = text(Q / "dc/reports/timing_setup.rpt")
checks.append(("setup MET +0.0007 ns", bool(re.search(r"slack \(MET\)\s+0\.0007", timing))))
loop = text(Q / "dc/reports/precompile_loop_gate.rpt")
checks.append(("precompile loop gate", "TIM-209=0" in loop and "OPT-150=0" in loop))
netlist = text(Q / "dc/netlist/m1058_fc2_reset_hygiene_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v")
for stem, count in (("fifo_tag_q_reg", 96), ("fifo_block_q_reg", 12),
                    ("fifo_bank_id_q_reg", 12), ("fifo_channel_q_reg", 48)):
    checks.append((f"mapped {stem} count {count}", netlist.count("g_k1_implementation_core_g_k1_service_" + stem) == count))
checks.append(("docs359 unchanged", sha(ROOT / "docs/359_DATE终局冻结_20260813.md") == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"))
for label, ok in checks:
    print(("PASS" if ok else "FAIL"), label)
raise SystemExit(0 if all(ok for _, ok in checks) else 1)
