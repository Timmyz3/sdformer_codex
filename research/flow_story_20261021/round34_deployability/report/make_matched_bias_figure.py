"""Source-backed descriptive figure; no fits, policy selection, or new evaluation."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "bias_control/extension_v1/actual_v1/SUMMARY_V1.json"
data = json.loads(SOURCE.read_text())
policies = [
    ("fit_static__six", "Fit static\n151 tiles"),
    ("uniform_six_expectation", "Uniform\nexpectation"),
    ("X128__direct_raw__six", "Direct AEE\nX128"),
    ("X128__direct_aug_g1__six", "Augmented\nX128"),
    ("X128__calibrated_ordinary__six", "Calibrated\nX128"),
    ("X160__signed_g1__six", "Online signed\nX160"),
    ("X128__folded_constant_v__six", "Folded response\nX128"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.6), sharex=True)
points = []
x = np.arange(len(policies))
for ax, role, label in zip(axes, ["tune", "confirm"],
                           ["Tune: 16 exposed identities", "Old confirmation: 40 exposed identities"]):
    for offset, condition, color, caption in [
        (-.19, "uncorrected", "#327aa0", "Original outputs"),
        (+.19, "same_bias_corrected", "#c58132", "Same fit-only bias correction for all actions"),
    ]:
        values = [100 * data["results"][role][condition][key]["relative_Q4"] for key, _ in policies]
        ax.bar(x + offset, values, width=.36, color=color, label=caption, zorder=3)
        points.extend({"role": role, "condition": condition, "policy": key, "relative_AEE_percent": value}
                      for (key, _), value in zip(policies, values))
    ax.axhline(0, color="#333333", linewidth=.8)
    ax.grid(axis="y", color="#dedede", linewidth=.6, zorder=0)
    ax.set_title(label, loc="left", fontweight="bold")
    ax.set_ylabel("AEE change vs matching Q4 (%)")
    ax.set_ylim(-1.3, .9)
axes[0].legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
axes[-1].set_xticks(x, [name for _, name in policies])
fig.suptitle("The fixed output-bias control does not remove the response-selector gain", x=.09,
             ha="left", fontsize=13, fontweight="bold", y=.98)
fig.text(.09, .015, "All valid GT pixels; fixed-six decisions unchanged. Lower is better. "
         "Small exposed development sets; no significance or whole-network speed claim.", fontsize=9)
fig.tight_layout(rect=(.01, .055, 1, .95))
for suffix in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"MATCHED_BIAS_V1.{suffix}", dpi=180, bbox_inches="tight")
plt.close(fig)
(OUT / "MATCHED_BIAS_FIGURE_SOURCES_V1.json").write_text(json.dumps({
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "points": points,
    "selection_scope": "Descriptive fixed subset; complete forty-policy table remains linked in milestone.",
}, indent=2) + "\n")
print("Saved MATCHED_BIAS_V1.{png,pdf,svg} and source receipt")
