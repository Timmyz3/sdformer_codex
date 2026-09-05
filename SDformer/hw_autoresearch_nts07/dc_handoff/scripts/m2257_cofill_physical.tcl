# Common masked LRU4 circuit, three causal axes, identical 3ns constraints.
if {[catch {
    set hw $::env(M2257_HW)
    set out $::env(M2257_OUTPUT)
    set slow $::env(M2257_SLOW)
    set fast $::env(M2257_FAST)
    file mkdir "$out/reports"
    file mkdir "$out/netlist"
    set_app_var target_library [list $slow]
    set_app_var link_library [list * $slow $fast]
    set_app_var verilogout_no_tri true
    set_svf "$out/netlist/m2018_axis.svf"
    saif_map -start
    analyze -format sverilog -define SYNTHESIS [list \
        "$hw/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv" \
        "$hw/rtl_m2249/m2249_c2_consumer_scoped_bank_fill_frontend.sv"]
    elaborate m2249_c2_consumer_scoped_bank_fill_frontend -parameters \
        "SCHEDULE_MODE=>$::env(M2257_MODE),UNION_PREFETCH=>$::env(M2257_UNION)"
    link
    uniquify
    set_min_library $slow -min_version $fast
    set_operating_conditions ssg0p9v125c
    source "$hw/dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
    set_wire_load_model -name ZeroWireload [current_design]
    set_clock_gating_style -positive_edge_logic integrated -control_point before \
        -minimum_bitwidth 8 -max_fanout 64
    set_fix_hold [all_clocks]
    ungroup -all -flatten
    compile_ultra -gate_clock
    update_timing
    redirect "$out/reports/area.rpt" {report_area -hierarchy}
    redirect "$out/reports/setup_after.rpt" {report_timing -delay_type max -max_paths 3 -significant_digits 6}
    redirect "$out/reports/hold_after.rpt" {report_timing -delay_type min -max_paths 3 -significant_digits 6}
    redirect "$out/reports/constraints_after.rpt" {report_constraint -all_violators -significant_digits 6}
    redirect "$out/reports/clock_gating.rpt" {report_clock_gating}
    redirect "$out/reports/check_design.rpt" {check_design}
    change_names -rules verilog -hierarchy
    saif_map -write_map "$out/netlist/m2018_axis.ptpx_map.default.tcl" -type ptpx
    # Existing endpoint-ECO runner consumes these filenames; they are not an
    # assertion that this new design is the original M2018 implementation.
    write_file -format ddc -hierarchy -output "$out/netlist/m2018_axis.ddc"
    write_file -format verilog -hierarchy -output "$out/netlist/m2018_axis_mapped.v"
    write_sdc "$out/netlist/m2018_axis_mapped.sdc"
    set_svf -off
} message]} {
    puts stderr "M2257 stopped: $message"
    exit 1
}
exit
