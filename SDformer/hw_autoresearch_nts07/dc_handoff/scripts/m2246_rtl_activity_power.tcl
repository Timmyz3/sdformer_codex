# RTL-SAIF power analysis: validate RTL boundary/state, then propagate gates.
# See installed PrimeTime read_saif/report_switching_activity manuals and
# doc/pt/tutpx/averaged/ave_saif.tcl. Do not confuse nonzero-toggle coverage
# with annotation completeness or demand direct RTL annotation on every gate.
if {[catch {
    set power_enable_analysis true
    set power_analysis_mode averaged
    set design_name $::env(M2246_DESIGN)
    set lib $::env(M2246_LIB)
    set output $::env(M2246_OUTPUT)
    file mkdir $output
    set_app_var search_path [list [file dirname $lib]]
    read_db $lib
    set_app_var link_path [list * $lib]
    read_verilog $::env(M2246_NETLIST)
    link_design $design_name
    if {[get_object_name [current_design]] ne $design_name} { error "wrong design" }
    read_sdc $::env(M2246_SDC)
    set_operating_conditions tt0p9v25c -library tcbn28hpcplusbwp35p140tt0p9v25c
    set_wire_load_model -name ZeroWireload -library tcbn28hpcplusbwp35p140tt0p9v25c
    set_input_transition 0.100 [remove_from_collection [all_inputs] [get_ports clk_core]]
    set_clock_transition 0.100 [get_clocks core_clk]
    source $::env(M2246_MAP)
    redirect "$output/saif_annotation.rpt" {
        read_saif $::env(M2246_SAIF) \
            -strip_path tb_m2217_m2018_tsbg_matched_native_saif_power/dut_axis \
            -report_inconsistent_annotation "$output/inconsistent_annotation.rpt"
    }
    if {[info exists ::env(M2246_STATE_SAIF)]} {
        redirect "$output/state_annotation.rpt" {
            read_saif $::env(M2246_STATE_SAIF) \
                -strip_path tb_m2217_m2018_tsbg_matched_native_saif_power/state_power_probe \
                -report_inconsistent_annotation "$output/state_inconsistent.rpt"
        }
    }
    foreach group {primary_inputs sequential rtl} {
        redirect "$output/${group}_sources_before.rpt" {
            report_switching_activity -include_only $group -include_mapping_types
        }
        redirect "$output/${group}_unannotated.rpt" {
            report_switching_activity -include_only $group -list_not_annotated -show_pin
        }
    }
    redirect "$output/check_timing.rpt" {check_timing -verbose}
    redirect "$output/disabled_timing.rpt" {report_disable_timing}
    update_timing -full
    redirect "$output/check_power.rpt" {check_power -verbose}
    update_power
    redirect "$output/sources_after.rpt" {report_switching_activity -include_mapping_types}
    foreach source {file propagated default implied no_switching_activity} {
        redirect "$output/activity_${source}.rpt" {
            report_switching_activity -list_by_source $source -show_pin
        }
    }
    set fp [open "$output/alias_nets.rpt" w]
    foreach bit {19 20 21 22} {
        set name [format {adapter_pending_tag_q[%d]} $bit]
        set nets [get_nets -quiet $name]
        set pins [get_pins -quiet -of_objects $nets]
        puts $fp "$name connected_pins=[sizeof_collection $pins]"
    }
    close $fp
    redirect "$output/power.rpt" {report_power -unit mW -nosplit -significant_digits 8}
    redirect "$output/power_hierarchy.rpt" {report_power -hierarchy -area -unit mW -nosplit -significant_digits 8}
    set fp [open "$output/COMPLETE.txt" w]
    puts $fp "PTPX completed; inspect RTL activity sources before citing energy"
    close $fp
} message]} {
    puts stderr "M2246 stopped: $message"
    exit 1
}
exit
