# Additive M1798 identity wrapper around the immutable, source-reviewed M1790
# PTPX implementation.  Only environment names change; the mapped netlist,
# SDC, zero-macro gates, annotation gates, and power boundary remain exact.
set ::env(M1790_TT_LIB_DB) $::env(M1798_TT_LIB_DB)
set ::env(M1790_MAPPED_NETLIST) $::env(M1798_MAPPED_NETLIST)
set ::env(M1790_MAPPED_SDC) $::env(M1798_MAPPED_SDC)
set ::env(M1790_GATE_SAIF) $::env(M1798_GATE_SAIF)
set ::env(M1790_OUTPUT_DIR) $::env(M1798_OUTPUT_DIR)
set ::env(M1790_SAIF_INSTANCE) $::env(M1798_SAIF_INSTANCE)
set ::env(M1790_MEASUREMENT_CYCLES) $::env(M1798_MEASUREMENT_CYCLES)
set ::env(M1790_SAIF_DURATION_NS) $::env(M1798_SAIF_DURATION_NS)
source /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_ptpx_m1790_c3_m1454_fixed_t10_mapped_energy.tcl
