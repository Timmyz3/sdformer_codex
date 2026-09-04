set design_name $::env(DESIGN_NAME)
set lib_db [file normalize $::env(LIB_DB)]
set min_lib_db [file normalize $::env(MIN_LIB_DB)]
set ddc_file [file normalize $::env(DDC_FILE)]
set binary_map [file normalize $::env(BINARY_SAIF_MAP)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir "$output_dir/netlist"
set_app_var search_path [list [file dirname $lib_db] [file dirname $min_lib_db]]
set_app_var target_library [list $lib_db]
set_app_var link_library [list "*" $lib_db $min_lib_db]
read_ddc $ddc_file
current_design $design_name
link
saif_map -read_map $binary_map
saif_map -write_map \
    "$output_dir/netlist/${design_name}.ptpx_saif_map.full.tcl" -type ptpx
saif_map -write_map \
    "$output_dir/netlist/${design_name}.ptpx_saif_map.essential.tcl" \
    -type ptpx -essential
quit
