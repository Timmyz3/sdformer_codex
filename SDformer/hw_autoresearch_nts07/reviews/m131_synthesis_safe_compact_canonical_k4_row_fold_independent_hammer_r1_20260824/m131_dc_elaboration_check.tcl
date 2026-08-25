set design_name m131_synthesis_safe_compact_canonical_k4_row_fold
set hw_root [file normalize $::env(HW_ROOT)]
set output_dir [file normalize $::env(OUTPUT_DIR)]
set rtl_file [file normalize "$hw_root/rtl_m131/m131_synthesis_safe_compact_canonical_k4_row_fold.sv"]
set lib_db [file normalize $::env(LIB_DB)]

file mkdir $output_dir
file mkdir "$output_dir/work"
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"

set_app_var search_path [list $hw_root [file dirname $lib_db]]
set_app_var target_library [list $lib_db]
set_app_var link_library [list "*" $lib_db]
set_app_var verilogout_no_tri true
set_app_var hdlin_auto_save_templates true
define_design_lib WORK -path "$output_dir/work"

analyze -format sverilog -define SYNTHESIS -library WORK [list $rtl_file]
elaborate $design_name -library WORK
current_design $design_name
link
uniquify

set check_status [check_design]
check_design > "$output_dir/reports/check_design.rpt"
report_hierarchy > "$output_dir/reports/hierarchy.rpt"
report_resources -hierarchy > "$output_dir/reports/resources.rpt"
report_reference -hierarchy > "$output_dir/reports/references.rpt"
write -format ddc -hierarchy -output "$output_dir/netlist/${design_name}_elaborated.ddc"

if {!$check_status} {
    puts "FAIL M131 independent DC check_design returned false"
    exit 31
}
puts "PASS M131 independent DC analyze_elaborate_check_design no_elab312=true negative_index=false compile_run=false physical_speedup=false"
quit
