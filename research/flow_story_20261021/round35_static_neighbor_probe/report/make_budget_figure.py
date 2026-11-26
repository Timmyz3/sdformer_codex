"""Descriptive quality/service point and retained regional tradeoffs, R35."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
P = R.parent
O = Path(__file__).resolve().parent
sources = {
    "quality": R / "quality/actual_v1/SUMMARY_V1.json",
    "regions": R / "independent_review/STATIC360_RAW_QUALITY_INDEPENDENT_V1.json",
    "budget": P / "round34_deployability/hardware/actual_v2/STATIC_NEIGHBOR_SEARCH_V1.json",
    "static151": P / "round34_deployability/hardware/actual_v2/STATIC_151_V1.json",
    "static360": P / "round34_deployability/hardware/actual_v2/STATIC_360_V1.json",
    "q4": P / "round33_predictability/hardware/results/SUMMARY_V1.json",
}
a = {k: json.loads(p.read_text()) for k, p in sources.items()}
q = a["quality"]["results"]["confirm"]["original"]
folded = "X128__folded_constant_v__six"
budget = next(r for r in a["budget"]["rows"]
              if r["head"] == "X128_folded_constant_v" and r["policy"] == "fixed6")
q4 = next(r for r in a["q4"]["candidates"] if r["action"] == 6)
points = [
    ("Q4", 4*q4["static_affine_BN_ticks"], q["Q4"]["AEE"]),
    ("Fit static 151", 4*a["static151"]["result"]["affine_plus_BN_completion_ticks"], q["fit_static__six"]["AEE"]),
    ("Static 360", 4*a["static360"]["result"]["affine_plus_BN_completion_ticks"], q["static360"]["AEE"]),
    ("Response-selected 151", budget["serial_predictor_plus_four_isolated_conv_budget"], q[folded]["AEE"]),
]
cmp = a["regions"]["independent_quality"]["confirm"]["uncorrected"]["static360_vs_each_frozen_policy"][folded]["summaries"]["overall"]
regions = [(key, label, 100*(cmp[key]["reference_AEE"]/cmp[key]["AEE"]-1)) for key, label in
           [("all_valid", "All GT"), ("B", "Boundary"), ("L", "Large motion"),
            ("voxel_nonzero", "Nonzero voxel"), ("voxel_zero", "Zero voxel"), ("B_unknown", "Boundary unknown")]]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, (ax, ar) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
colors = ["#666666", "#666666", "#327aa0", "#15856e"]
offsets = [(7, 6), (-15, 10), (-12, 9), (-14, -19)]
for (name, cost, aee), color, offset in zip(points, colors, offsets):
    ax.scatter(cost/1e6, aee, color=color, marker="D" if "Response" in name else "o", s=60, zorder=3)
    ax.annotate(name, (cost/1e6, aee), xytext=offset, textcoords="offset points",
                ha="right" if offset[0] < 0 else "left", color=color)
ax.set_xlabel("Serial predictor + four isolated convs (M model ticks)")
ax.set_ylabel("All-valid-GT AEE (original outputs)")
ax.set_xlim(810, 1060)
ax.set_ylim(1.3555, 1.3745)
ax.grid(alpha=.18)
ax.set_title("One scoped quality-cost point", loc="left", fontweight="bold")
values = [v for _, _, v in regions]
ar.barh(range(len(regions)), values, color=["#15856e" if v < 0 else "#c58132" for v in values])
ar.set_yticks(range(len(regions)), [label for _, label, _ in regions])
ar.invert_yaxis()
ar.axvline(0, color="#444444", linewidth=.8)
ar.set_xlabel("Candidate AEE change vs static360 (%)")
ar.set_title("Regional tradeoffs remain", loc="left", fontweight="bold")
ar.set_xlim(-2.55, .65)
for i, v in enumerate(values):
    ar.text(v-.05 if v < 0 else v+.05, i, f"{v:+.3f}", ha="right" if v < 0 else "left", va="center", fontsize=9)
fig.suptitle("R35: paying the modeled selector cost leaves a candidate advantage", fontsize=13, fontweight="bold", x=.075, ha="left")
fig.text(.075, .017, "40 previously exposed identities / 5 sequences. Conditional service model; shared stem and rest of network excluded. "
         "No full-network speed or significance claim.", fontsize=8.5)
fig.tight_layout(rect=(0, .065, 1, .93))
for suffix in ("png", "pdf", "svg"):
    fig.savefig(O/f"QUALITY_COST_V1.{suffix}", dpi=180, bbox_inches="tight")
plt.close(fig)
(O/"FIGURE_SOURCES_V1.json").write_text(json.dumps({
    "sources": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in sources.items()},
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "points": points, "regional_changes": regions,
    "scope": "Descriptive subset; complete41 policies and16 regions remain in linked results.",
}, indent=2)+"\n")
print("Saved QUALITY_COST_V1.{png,pdf,svg}")
