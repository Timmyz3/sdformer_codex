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

# Both points analyze the exact same two-file source corpus.  Only the selected
# top changes; the two production tops have an independently checked identical
# port signature.
set precompile_build_report \
    "$output_dir/reports/precompile_build.rpt"
redirect $precompile_build_report {
    analyze -format sverilog -define SYNTHESIS $rtl_files
    elaborate $design_name
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
    set_fix_hold [get_clocks core_clk]
}
set precompile_design_report \
    "$output_dir/reports/check_design_precompile.rpt"
redirect $precompile_design_report {check_design}
set precompile_timing_report \
    "$output_dir/reports/check_timing_precompile.rpt"
redirect $precompile_timing_report {check_timing}
set precompile_build_fp [open $precompile_build_report r]
set precompile_build_text [read $precompile_build_fp]
close $precompile_build_fp
set precompile_design_fp [open $precompile_design_report r]
set precompile_design_text [read $precompile_design_fp]
close $precompile_design_fp
set precompile_timing_fp [open $precompile_timing_report r]
set precompile_timing_text [read $precompile_timing_fp]
close $precompile_timing_fp
set precompile_all_text \
    "$precompile_build_text\n$precompile_design_text\n$precompile_timing_text"
set precompile_tim209_count [regexp -all -- {TIM-209} $precompile_all_text]
set precompile_opt150_count [regexp -all -- {OPT-150} $precompile_all_text]
set precompile_gate_fp \
    [open "$output_dir/reports/precompile_loop_gate.rpt" w]
puts $precompile_gate_fp "TIM-209=$precompile_tim209_count"
puts $precompile_gate_fp "OPT-150=$precompile_opt150_count"
puts $precompile_gate_fp \
    "sources=precompile_build.rpt,check_design_precompile.rpt,check_timing_precompile.rpt"

# Fail before flattening or mapping if either timing-loop diagnostic is found.
# A top-level Tcl error is not used because prior campaigns showed that dc_shell
# may continue after it; the explicit process exit is the hard boundary.
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

    # The mapping sequence is byte-identical for Fixed and rank3.  Both are
    # flattened standard-cell logic with zero SRAM/ROM macro instances.
    ungroup -all -flatten
    set_cost_priority -design_rule
    compile_ultra
    compile_ultra -incremental
    compile -incremental_mapping -only_hold_time
    compile -incremental_mapping
    update_timing

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
        -significant_digits 4 > "$output_dir/reports/timing_hold.rpt"
    redirect "$output_dir/reports/constraint_violators.rpt" {
        report_constraint -max_delay -all_violators -significant_digits 4
        report_constraint -min_delay -all_violators -significant_digits 4
        report_constraint -max_capacitance -all_violators \
            -significant_digits 4
        report_constraint -max_transition -all_violators \
            -significant_digits 4
        report_constraint -max_fanout -all_violators -significant_digits 4
    }
    check_design > "$output_dir/reports/check_design_postcompile.rpt"
    check_timing > "$output_dir/reports/check_timing_postcompile.rpt"

    # V-2023.12 DC has no report_timing -unconstrained switch.  The runner
    # therefore admits unconstrained endpoints only through the explicit
    # check_timing postcompile audit and preserves this routing note.
    set unconstrained_fp \
        [open "$output_dir/reports/unconstrained_audit_route.rpt" w]
    puts $unconstrained_fp \
        "authority=check_timing_postcompile.rpt checking unconstrained_endpoints"
    close $unconstrained_fp

    change_names -rules verilog -hierarchy
    write_file -format verilog -hierarchy \
        -output "$output_dir/netlist/${design_name}_mapped.v"
    write_sdc "$output_dir/netlist/${design_name}_mapped.sdc"
    write -format ddc -hierarchy \
        -output "$output_dir/netlist/${design_name}.ddc"
    set_svf -off

    set terminal_fp [open "$output_dir/TCL_PASS_TERMINAL.txt" w]
    puts $terminal_fp "status=PASS_M518_MATCHED_R2_DC_TCL_TERMINAL"
    puts $terminal_fp "design=$design_name"
    puts $terminal_fp "TIM-209=$precompile_tim209_count"
    puts $terminal_fp "OPT-150=$precompile_opt150_count"
    close $terminal_fp
    quit
}
