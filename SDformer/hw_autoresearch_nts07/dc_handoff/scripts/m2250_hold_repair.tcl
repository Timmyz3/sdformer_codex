# Incremental hold repair of the already mapped ordinary/TSBG designs.
# Clock/I/O constraints are not relaxed. Outputs go to a new directory.
if {[catch {
    set output $::env(M2250_OUTPUT)
    set input $::env(M2250_INPUT)
    set slow $::env(M2250_SLOW)
    set fast $::env(M2250_FAST)
    file mkdir "$output/reports"
    file mkdir "$output/netlist"
    set_app_var target_library [list $slow]
    set_app_var link_library [list * $slow $fast]
    read_ddc "$input/netlist/m2018_axis.ddc"
    link
    set_min_library $slow -min_version $fast
    set_operating_conditions ssg0p9v125c
    set_wire_load_model -name ZeroWireload [current_design]
    set_svf "$output/netlist/hold_repair.svf"
    redirect "$output/reports/setup_before.rpt" {report_timing -delay_type max -max_paths 3 -significant_digits 6}
    redirect "$output/reports/hold_before.rpt" {report_timing -delay_type min -max_paths 3 -significant_digits 6}
    set_fix_hold [all_clocks]
    # First default-priority attempt inserted no cells despite negative min
    # paths. Prioritize hold optimization; final setup still has to pass.
    set_cost_priority -min_delay
    if {$::env(M2250_GATE_CLOCK) eq "1"} {
        # A common implementation axis, never credited as new sparsity.
        # The foundry ICG cells have latch_posedge_precontrol, not the
        # no-test-pin latch_posedge flavor. Functional test enable is tied low.
        set_clock_gating_style -positive_edge_logic integrated -control_point before \
            -minimum_bitwidth 8 -max_fanout 64
        compile_ultra -incremental -gate_clock
        set test_ports [get_ports -quiet *test*]
        set scan_ports [get_ports -quiet *scan_enable*]
        foreach_in_collection port [add_to_collection $test_ports $scan_ports] {
            set_case_analysis 0 $port
        }
        redirect "$output/reports/clock_gating.rpt" {report_clock_gating}
    }
    compile_ultra -incremental -only_design_rule
    update_timing
    redirect "$output/reports/qor.rpt" {report_qor}
    redirect "$output/reports/area.rpt" {report_area -hierarchy}
    redirect "$output/reports/setup_after.rpt" {report_timing -delay_type max -max_paths 10 -significant_digits 6}
    redirect "$output/reports/hold_after.rpt" {report_timing -delay_type min -max_paths 10 -significant_digits 6}
    redirect "$output/reports/constraints_after.rpt" {report_constraint -all_violators}
    redirect "$output/reports/check_design.rpt" {check_design}
    write_file -format ddc -hierarchy -output "$output/netlist/m2018_axis.ddc"
    write_file -format verilog -hierarchy -output "$output/netlist/m2018_axis_mapped.v"
    write_sdc "$output/netlist/m2018_axis_mapped.sdc"
    set_svf -off
} message]} {
    puts stderr "Hold repair stopped: $message"
    exit 1
}
exit
