# Fixed temporal pairing probe

1. Read the completed count8/psum overlay and temporal endpoint calibration. Use exactly `[8,2,6,3,9,4,1,5,7,0]`, with no search or refit.
2. Independently reconstruct native source events on calibration tiles 0–31, prior held tiles 128–191, and source-halo-disjoint tiles 4000–4063. Count source update blocks, live retirement blocks, packed/scalar retirement issues and all remaining cycle obligations. Charge one 40-bit configuration word and one cycle for the candidate.
3. Cross-check the original ordering against the measured 64-tile mode20 receipt. If the complete fixed-order prediction improves service, implement mode21 in this isolated directory: source-word bit permutation, existing positive count8 layout, inverse mapping only on output reads. Preserve mode20 behavior and do not introduce endpoints or a second z domain.
4. If implemented, run small correctness/corner/reconfiguration checks, then modes14/20/21 on both held 64-tile sets without reset between tiles. Check every raw output and all counted service obligations. Report state, mux/port changes and calibration scope; do not claim area, energy or first-in-literature novelty.

The original overlay files and mode20 results remain the reference. No hashes, archives, training, EDA, production edits or commits.
