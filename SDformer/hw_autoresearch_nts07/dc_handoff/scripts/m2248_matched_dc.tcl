# Same RTL/constraints as M2242, with an actual design name in the receipt.
if {[catch {
    set root $::env(M2248_REPO)
    set output $::env(M2248_OUTPUT)
    set mode $::env(M2248_MODE)
    file mkdir "$output/reports"
    file mkdir "$output/netlist"
    define_design_lib WORK -path "$output/WORK"
    set slow $::env(M2248_SLOW)
    set fast $::env(M2248_FAST)
    set_app_var target_library [list $slow]
    set_app_var link_library [list * $slow $fast]
    set_app_var search_path [list $root [file dirname $slow]]
    set_app_var verilogout_no_tri true
    set_app_var hdlin_auto_save_templates true
    set_svf "$output/netlist/m2018_axis.svf"
    saif_map -start
    analyze -format sverilog -define SYNTHESIS [list \
        "$root/hw_autoresearch_nts07/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv" \
        "$root/hw_autoresearch_nts07/rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"]
    elaborate m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend -parameters "SCHEDULE_MODE=>$mode"
    set design_name [get_object_name [current_design]]
    link
    uniquify
    set_min_library $slow -min_version $fast
    set_operating_conditions ssg0p9v125c
    source "$root/hw_autoresearch_nts07/dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
    set_wire_load_model -name ZeroWireload [current_design]
    ungroup -all -flatten
    compile_ultra
    update_timing
    redirect "$output/reports/qor.rpt" {report_qor}
    redirect "$output/reports/area.rpt" {report_area -hierarchy}
    redirect "$output/reports/timing_setup.rpt" {report_timing -delay_type max -max_paths 20 -significant_digits 4}
    redirect "$output/reports/timing_hold_diagnostic.rpt" {report_timing -delay_type min -max_paths 100 -significant_digits 4}
    redirect "$output/reports/check_design.rpt" {check_design}
    change_names -rules verilog -hierarchy
    saif_map -write_map "$output/netlist/m2018_axis.ptpx_map.default.tcl" -type ptpx
    write_file -format verilog -hierarchy -output "$output/netlist/m2018_axis_mapped.v"
    write_sdc "$output/netlist/m2018_axis_mapped.sdc"
    write_file -format ddc -hierarchy -output "$output/netlist/m2018_axis.ddc"
    set_svf -off
    set fp [open "$output/reports/identity.rpt" w]
    puts $fp "design=$design_name\nschedule_mode=$mode\nclock_period_ns=3.0\nmacro_count=0"
    close $fp
} message]} {
    puts stderr "M2248 DC stopped: $message"
    exit 1
}
exit
