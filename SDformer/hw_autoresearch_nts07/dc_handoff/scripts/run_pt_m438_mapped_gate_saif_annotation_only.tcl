set design_name $::env(DESIGN_NAME)
set lib_db [file normalize $::env(LIB_DB)]
set mapped_netlist [file normalize $::env(MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(MAPPED_SDC)]
set gate_saif [file normalize $::env(GATE_SAIF_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $lib_db]]
set_app_var link_path [list "*" $lib_db]
read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
set_operating_conditions $::env(OPERATING_CONDITION)
read_sdc $mapped_sdc
set power_enable_analysis true
set_app_var power_analysis_mode averaged

# M438 is an annotation-coverage gate only.  update_power/report_power are
# deliberately absent; a separate independently reviewed milestone owns PTPX.
read_saif -strip_path $::env(SAIF_INSTANCE) \
    -report_inconsistent_annotation \
    "$output_dir/reports/inconsistent_annotation.rpt" $gate_saif
report_switching_activity -coverage -include_mapping_types \
    > "$output_dir/reports/switching_coverage.rpt"
report_switching_activity -list_annotated -include_mapping_types \
    > "$output_dir/reports/switching_annotated.rpt"
report_switching_activity -list_not_annotated -include_mapping_types \
    > "$output_dir/reports/switching_unannotated.rpt"
report_switching_activity > "$output_dir/reports/switching_summary.rpt"

set marker [open "$output_dir/PT_GATE_SAIF_ANNOTATION_INTERNAL_COMPLETE.txt" w]
puts $marker "M438_GATE_SAIF_ANNOTATION_INTERNAL_COMPLETE=PASS"
close $marker
quit
