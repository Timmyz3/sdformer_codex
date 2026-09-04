set design_name qfit_atlif_csd_reconstruct_t10
set lib_db [file normalize $::env(LIB_DB)]
set min_lib_db [file normalize $::env(MIN_LIB_DB)]
set ddc_file [file normalize $::env(DDC_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

set_app_var search_path [list [file dirname $lib_db] [file dirname $min_lib_db]]
set_app_var target_library [list $lib_db]
set_app_var link_library [list "*" $lib_db $min_lib_db]
read_ddc $ddc_file
current_design $design_name
link
set_min_library $lib_db -min_version $min_lib_db
set_operating_conditions $::env(OPERATING_CONDITION)
update_timing

report_qor > "$output_dir/reports/sta_qor.rpt"
report_area -hierarchy > "$output_dir/reports/sta_area.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 \
    -significant_digits 4 > "$output_dir/reports/sta_setup.rpt"
report_timing -delay_type min -max_paths 100 -nworst 10 \
    -significant_digits 4 > "$output_dir/reports/sta_hold.rpt"
check_timing > "$output_dir/reports/sta_check_timing.rpt"

set marker [open "$output_dir/STA_INTERNAL_COMPLETE.txt" w]
puts $marker "M37_R13_STA_INTERNAL_COMPLETE=PASS"
puts $marker "design=$design_name"
close $marker
quit
