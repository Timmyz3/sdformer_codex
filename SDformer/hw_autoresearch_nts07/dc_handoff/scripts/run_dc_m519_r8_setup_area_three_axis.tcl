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
    set failure_fp [open "$output_dir/TCL_EXPLICIT_FAILURE.txt" w]
    puts $failure_fp "status=FAIL_ELABORATION_NO_CURRENT_DESIGN"
    close $failure_fp
    exit 35
}
current_design $elaborated_design
link
uniquify
set_min_library $lib_db -min_version $min_lib_db
set_operating_conditions $::env(OPERATING_CONDITION)
source $sdc_file
set_wire_load_model -name ZeroWireload [current_design]

# R8 preserves the independently reviewed R6 setup/area-only synthesis body.
# All R8 changes are bounded to runner identity/resource/PID provenance.  The
# three ARCH_MODE points therefore still share one pre-CTS flow and hold is a
# diagnostic only; placement/CTS is required for hold closure.
set flow_fp [open "$output_dir/reports/flow_contract.rpt" w]
puts $flow_fp "flow=m519_r8_setup_area_only"
puts $flow_fp "compile_ultra_count=1"
puts $flow_fp "incremental_compile_count=0"
puts $flow_fp "hold_fix_command_count=0"
puts $flow_fp "hold_only_optimization_count=0"
puts $flow_fp "hold_not_closed_at_dc=true"
puts $flow_fp "hold_reports_are_diagnostic_only=true"
close $flow_fp

check_design > "$output_dir/reports/check_design_precompile.rpt"
set precompile_timing_report \
    "$output_dir/reports/check_timing_precompile.rpt"
redirect $precompile_timing_report {check_timing}
set precompile_fp [open $precompile_timing_report r]
set precompile_text [read $precompile_fp]
close $precompile_fp
set precompile_tim209_count [regexp -all -- {TIM-209} $precompile_text]
set precompile_opt150_count [regexp -all -- {OPT-150} $precompile_text]
set precompile_gate_fp \
    [open "$output_dir/reports/precompile_loop_gate.rpt" w]
puts $precompile_gate_fp "TIM-209=$precompile_tim209_count"
puts $precompile_gate_fp "OPT-150=$precompile_opt150_count"

if {$precompile_tim209_count != 0 || $precompile_opt150_count != 0} {
    puts $precompile_gate_fp \
        "status=FAIL_PRECOMPILE_LOOP__EXPLICIT_EXIT36__NO_COMPILE"
    close $precompile_gate_fp
    set failure_fp [open "$output_dir/TCL_EXPLICIT_FAILURE.txt" w]
    puts $failure_fp "status=FAIL_PRECOMPILE_LOOP__EXPLICIT_EXIT36"
    puts $failure_fp "TIM-209=$precompile_tim209_count"
    puts $failure_fp "OPT-150=$precompile_opt150_count"
    close $failure_fp
    exit 36
} else {
    puts $precompile_gate_fp "status=PASS_PRECOMPILE_LOOP_GATE"
    close $precompile_gate_fp
    report_resources -hierarchy \
        > "$output_dir/reports/resources_precompile.rpt"
    report_reference -hierarchy \
        > "$output_dir/reports/references_precompile.rpt"

    ungroup -all -flatten
    set compile_start_seconds [clock seconds]
    compile_ultra
    set compile_end_seconds [clock seconds]
    update_timing
    set compile_fp [open "$output_dir/reports/compile_receipt.rpt" w]
    puts $compile_fp "compile_ultra_count=1"
    puts $compile_fp "compile_start_epoch=$compile_start_seconds"
    puts $compile_fp "compile_end_epoch=$compile_end_seconds"
    puts $compile_fp \
        "compile_wall_seconds=[expr {$compile_end_seconds - $compile_start_seconds}]"
    puts $compile_fp "incremental_compile_count=0"
    puts $compile_fp "hold_optimization_count=0"
    close $compile_fp

    report_hierarchy > "$output_dir/reports/hierarchy_postcompile.rpt"
    report_resources -hierarchy \
        > "$output_dir/reports/resources_postcompile.rpt"
    report_reference -hierarchy \
        > "$output_dir/reports/references_postcompile.rpt"
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
        -significant_digits 4 > "$output_dir/reports/timing_hold_diagnostic.rpt"
    redirect "$output_dir/reports/constraint_setup.rpt" {
        report_constraint -max_delay -all_violators -significant_digits 4
    }
    redirect "$output_dir/reports/constraint_hold_diagnostic.rpt" {
        report_constraint -min_delay -all_violators -significant_digits 4
    }
    redirect "$output_dir/reports/constraint_max_capacitance.rpt" {
        report_constraint -max_capacitance -all_violators \
            -significant_digits 4
    }
    redirect "$output_dir/reports/constraint_max_transition.rpt" {
        report_constraint -max_transition -all_violators \
            -significant_digits 4
    }
    redirect "$output_dir/reports/constraint_max_fanout.rpt" {
        report_constraint -max_fanout -all_violators -significant_digits 4
    }
    check_design > "$output_dir/reports/check_design_postcompile.rpt"
    check_timing > "$output_dir/reports/check_timing_postcompile.rpt"

    change_names -rules verilog -hierarchy
    write_file -format verilog -hierarchy \
        -output "$output_dir/netlist/${design_name}_mapped.v"
    write_sdc "$output_dir/netlist/${design_name}_mapped.sdc"
    write -format ddc -hierarchy \
        -output "$output_dir/netlist/${design_name}.ddc"
    set_svf -off

    set terminal_fp [open "$output_dir/TCL_PASS_TERMINAL.txt" w]
    puts $terminal_fp "status=PASS_M519_R8_SETUP_AREA_DC_TCL_TERMINAL"
    puts $terminal_fp "design=$design_name"
    puts $terminal_fp "TIM-209=$precompile_tim209_count"
    puts $terminal_fp "OPT-150=$precompile_opt150_count"
    puts $terminal_fp "compile_ultra_count=1"
    puts $terminal_fp "incremental_compile_count=0"
    puts $terminal_fp "hold_optimization_count=0"
    puts $terminal_fp "hold_not_closed_at_dc=true"
    close $terminal_fp
    quit
}
