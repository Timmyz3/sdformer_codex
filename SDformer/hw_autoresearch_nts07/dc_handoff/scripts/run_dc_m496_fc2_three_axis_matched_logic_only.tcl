set design_name $::env(DESIGN_NAME)
set hw_root [file normalize $::env(HW_ROOT)]
set rtl_filelist [file normalize $::env(RTL_FILELIST)]
set lib_db [file normalize $::env(LIB_DB)]
set min_lib_db [file normalize $::env(MIN_LIB_DB)]
set sdc_file [file normalize $::env(SDC_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir $output_dir
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_svf "$output_dir/netlist/${design_name}.svf"
set_app_var search_path [list $hw_root [file dirname $lib_db] \
    [file dirname $min_lib_db]]
set_app_var target_library [list $lib_db]
set_app_var link_library [list "*" $lib_db $min_lib_db]
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
if {[info exists ::env(ELAB_PARAMETERS)]
        && $::env(ELAB_PARAMETERS) ne ""} {
    set dc_parameters [string map {"=" "=>"} $::env(ELAB_PARAMETERS)]
    elaborate $design_name -parameters $dc_parameters
} else {
    elaborate $design_name
}
set elaborated_design [current_design]
if {$elaborated_design eq ""} {
    error "M496 elaboration left no current design for $design_name"
}
current_design $elaborated_design
link
uniquify
set_min_library $lib_db -min_version $min_lib_db
set_operating_conditions $::env(OPERATING_CONDITION)
source $sdc_file
set_wire_load_model -name ZeroWireload [current_design]
set_clock_uncertainty -hold 0.100 [get_clocks core_clk]
set_fix_hold [get_clocks core_clk]

check_design > "$output_dir/reports/check_design_precompile.rpt"
check_timing > "$output_dir/reports/check_timing_precompile.rpt"
report_resources -hierarchy > "$output_dir/reports/resources_precompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_precompile.rpt"

# The comparison is about integrated functional logic/register cost.  Flatten
# all three points before mapping so child-boundary artifacts cannot favor the
# single-service K1/K8 designs over the eight-service K1x8 construction.
ungroup -all -flatten
compile_ultra
compile_ultra -incremental
compile -incremental_mapping -only_hold_time
set_clock_uncertainty -hold 0.090 [get_clocks core_clk]
update_timing

report_hierarchy > "$output_dir/reports/hierarchy_postcompile.rpt"
report_resources -hierarchy > "$output_dir/reports/resources_postcompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_postcompile.rpt"
report_qor > "$output_dir/reports/qor.rpt"
report_area -hierarchy > "$output_dir/reports/area.rpt"
report_clocks > "$output_dir/reports/clocks.rpt"
report_port -verbose > "$output_dir/reports/ports.rpt"
set port_count_fp [open "$output_dir/reports/port_count.txt" w]
puts $port_count_fp [sizeof_collection [get_ports *]]
close $port_count_fp
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
