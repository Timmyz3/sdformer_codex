M37-r8 immutable RTL source provenance
======================================

Artifact:
  qfit_atlif_csd_reconstruct_t10.sv
  SHA256 ab7d73a6a82f8547437919813d6cf9496d0672fc23f46cfaec0c3d9be46c8cbd

Identity:
  This is the exact historical M37-r8 RTL source consumed by the frozen r8
  Synopsys VCS input manifest.  The same SHA is bound by the r8 receipt,
  r8 VCS contract, and both historical and fresh r8 DC input anchors.

Reconstruction method:
  The live source had already advanced to r9 before this standalone admission
  was assembled.  This snapshot was reconstructed by reverse-applying only
  the exact r9 static-index delta to the live r9 source.  No functional edit,
  padding, message filter, or semantic invention was introduced.  The result
  was accepted only after its full-file SHA256 exactly matched every frozen
  r8 identity anchor above.

Auditable r8-to-r9 delta:
  1. r8 dynamic bias_q[selected_row] indexing is replaced in r9 by a bounded
     row_index loop and equality selection.
  2. r8 runtime phase_cycle_q indexing and coefficient-array indexing are
     replaced in r9 by bounded phase_index/coefficient_index loops and equality
     selection.
  3. r8 selected_intermediate and selected_lane dynamic variables are replaced
     in r9 by a direct compile-time rank/lane intermediate-array subscript.

Live successor at sealing time:
  hw_autoresearch_nts07/rtl_m37/qfit_atlif_csd_reconstruct_t10.sv
  revision r9
  SHA256 a5f42567fc5262a99152ef04699c9062cbedc70075c0a91397ce8d00dc4397ed

Claim boundary:
  This snapshot exists only to make the independent r8 VCS/source-intent
  admission reproducible after the live RTL advanced.  It is not a new r8 run,
  does not close r8 Formality, and is not DC/STA/PPA/power/energy/system proof.
