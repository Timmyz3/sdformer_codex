set design_name m528_dead_write_only_1rw_product_capture_island_r2
set macro_cell TS1N28HPCPHVTB128X128M4S
set expected_macro_count 9
set hw_root [file normalize $::env(HW_ROOT)]
set rtl_filelist [file normalize $::env(RTL_FILELIST)]
set lib_db [file normalize $::env(LIB_DB)]
set min_lib_db [file normalize $::env(MIN_LIB_DB)]
set macro_db [file normalize $::env(MACRO_DB)]
set sdc_file [file normalize $::env(SDC_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir $output_dir
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_svf "$output_dir/netlist/${design_name}.svf"
set_app_var search_path [list $hw_root [file dirname $lib_db] \
    [file dirname $min_lib_db] [file dirname $macro_db]]
set_app_var target_library [list $lib_db]
set_app_var link_library [list "*" $lib_db $min_lib_db $macro_db]
set_app_var verilogout_no_tri true
set_app_var hdlin_auto_save_templates true

set fp [open $rtl_filelist r]
set rtl_files {}
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line ne "" && ![string match "#*" $line]} {
        lappend rtl_files [file normalize "$hw_root/$line"]
    }
}
close $fp

analyze -format sverilog -define SYNTHESIS $rtl_files
elaborate $design_name
current_design $design_name
link
uniquify
set_min_library $lib_db -min_version $min_lib_db
if {[info exists ::env(OPERATING_CONDITION)] \
        && $::env(OPERATING_CONDITION) ne ""} {
    set_operating_conditions $::env(OPERATING_CONDITION)
}

set macro_cells_pre [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set macro_count_pre [sizeof_collection $macro_cells_pre]
if {$macro_count_pre != $expected_macro_count} {
    error "M750_FAIL macro_count_pre=$macro_count_pre expected=$expected_macro_count"
}
set_dont_touch $macro_cells_pre true

source $sdc_file
set_wire_load_model -name ZeroWireload [current_design]
set_fix_hold [get_clocks core_clk]
check_design > "$output_dir/reports/check_design_precompile.rpt"
check_timing > "$output_dir/reports/check_timing_precompile.rpt"
report_resources -hierarchy > "$output_dir/reports/resources_precompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_precompile.rpt"

compile_ultra -no_autoungroup
compile_ultra -incremental -no_autoungroup
compile -incremental_mapping -only_hold_time
update_timing

set macro_cells_post [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set macro_count_post [sizeof_collection $macro_cells_post]
if {$macro_count_post != $expected_macro_count} {
    error "M750_FAIL macro_count_post=$macro_count_post expected=$expected_macro_count"
}

set audit [open "$output_dir/reports/macro_binding_audit.txt" w]
puts $audit "status=M750_MACRO_BINDING_STRUCTURAL_PASS"
puts $audit "macro_cell=$macro_cell"
puts $audit "macro_count_pre=$macro_count_pre"
puts $audit "macro_count_post=$macro_count_post"
puts $audit "expected_macro_count=$expected_macro_count"
puts $audit "behavioral_macro_verilog_read_by_dc=false"
puts $audit "register_array_fallback_allowed=false"
puts $audit "macro_db=$macro_db"
close $audit

report_hierarchy > "$output_dir/reports/hierarchy_postcompile.rpt"
report_resources -hierarchy > "$output_dir/reports/resources_postcompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_postcompile.rpt"
report_qor > "$output_dir/reports/qor.rpt"
report_area -hierarchy > "$output_dir/reports/area.rpt"
report_clocks > "$output_dir/reports/clocks.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 \
    -significant_digits 4 > "$output_dir/reports/timing_setup.rpt"
report_timing -delay_type min -max_paths 100 -nworst 10 \
    -significant_digits 4 > "$output_dir/reports/timing_hold.rpt"
redirect "$output_dir/reports/constraint_violators.rpt" {
    report_constraint -max_delay -all_violators -significant_digits 4
    report_constraint -min_delay -all_violators -significant_digits 4
    report_constraint -max_capacitance -all_violators -significant_digits 4
    report_constraint -max_transition -all_violators -significant_digits 4
    report_constraint -max_fanout -all_violators -significant_digits 4
}
check_design > "$output_dir/reports/check_design_postcompile.rpt"
check_timing > "$output_dir/reports/check_timing_postcompile.rpt"

change_names -rules verilog -hierarchy
write_file -format verilog -hierarchy \
    -output "$output_dir/netlist/${design_name}_mapped.v"
write_sdc "$output_dir/netlist/${design_name}_mapped.sdc"
write -format ddc -hierarchy -output "$output_dir/netlist/${design_name}.ddc"
set_svf -off
quit
