"""Phase 2 auto-launcher: read Phase 1 ranking, pick top 5 attentions, launch Phase 2."""
import subprocess, sys, yaml
from pathlib import Path

PHASE1_TAG = "rapid_screen_h40_phase1_fast"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "generated"

# Find latest Phase 1 summary
phase1_dirs = sorted(RESULTS_DIR.glob(f"{PHASE1_TAG}_*"))
if not phase1_dirs:
    print("Phase 1 not found"); sys.exit(1)
summary = phase1_dirs[-1] / "summary.md"
if not summary.exists():
    print(f"Phase 1 not done yet: {summary}"); sys.exit(1)

# Parse ranking
rows = []
for line in open(summary):
    if not line.startswith("|") or "---" in line or "rank" in line:
        continue
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 12: continue
    name = parts[1]
    aee = float(parts[5])
    aae = float(parts[6])
    sops = float(parts[7])
    attn_id = name.split("_")[2][:2]
    rows.append((attn_id, aee, aae, sops, name))

# Score: lower AAE + lower SOPs = better
scored = sorted(rows, key=lambda r: r[2]*0.6 + r[3]*0.4)
top5 = list(dict.fromkeys(r[0] for r in scored))[:5]
print(f"Top 5 attention modes: {top5}")

# Find Phase 2 configs for these top 5
for ffn in ["S02", "S012", "N"]:
    cfgs = []
    for aid in top5:
        p = CONFIG_DIR / f"h40_p2_{aid}{ffn}_F.yml"
        if p.exists(): cfgs.append(str(p.relative_to(CONFIG_DIR.parent)))
    if cfgs:
        cmd = [
            sys.executable, str(Path(__file__).resolve().parents[1] / "entrypoints/rapid_screen.py"),
            f"--tag=rapid_screen_h40_phase2_{ffn}",
            "--steps", "80", "--valid-samples", "5",
            "--batch-size", "8", "--workers", "8", "--amp",
        ]
        for c in cfgs:
            cmd.extend(["--config", c])
        print(f"Launching Phase 2 {ffn}: {len(cfgs)} configs")
        subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[3])
