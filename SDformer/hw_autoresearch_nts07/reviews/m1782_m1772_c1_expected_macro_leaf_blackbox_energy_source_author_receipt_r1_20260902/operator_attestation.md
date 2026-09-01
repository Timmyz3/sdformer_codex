# M1782 author attestation

M1772 remains a sealed failed campaign. Its mapped simulation passed, its 253-cycle SAIF contains 117,690 activity forms per tag with zero nonzero-TX entries, and its PTPX invocation stopped immediately after successful link at the predecessor's broad black-box gate. No power result exists.

M1782 changes only that gate. It accepts exactly nine leaf instances named `u_parent_scratch/g_slice_0__u_parent_sram` through `u_parent_scratch/g_slice_8__u_parent_sram`, all with reference `TS1N28HPCPHVTB128X128M4S`, `is_hierarchical=false`, and `is_black_box=true`. Any missing, extra, duplicated, hierarchical, wrong-reference, or other unresolved black box is fatal. The black-box check is not removed, and no selected-cell subtraction is introduced.

The successor is intentionally fresh: one compile, one mapped simulation, one SAIF, and one PTPX run. It may not reuse M1772's `simv`, SAIF, or unsealed private directory.

I did not launch VCS, `simv`, SAIF generation, PrimeTime PX, or a license query while authoring M1782. I created no M1782 attempt or result namespace and did not modify `docs/359`. A different author must complete M1783 review, and a separately pinned M1784 release must exist before the one-shot campaign can run. A future candidate still requires a different-author result hammer before any power or energy field is cited.
