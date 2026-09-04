# M2105 mapped-netlist averaged power driven by transformation-mapped RTL SAIF.
# This is not mapped-gate VCS activity.  The runner supplies axis-specific maps
# exported by the byte-identical DC script and the matching RTL SAIF.
set axis $::env(M2105_AXIS)
set design_name $::env(M2105_DESIGN_NAME)
set tt_lib_db [file normalize $::env(M2105_TT_LIB_DB)]
set mapped_netlist [file normalize $::env(M2105_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M2105_MAPPED_SDC)]
set default_map [file normalize $::env(M2105_DEFAULT_MAP)]
set essential_map [file normalize $::env(M2105_ESSENTIAL_MAP)]
set rtl_saif [file normalize $::env(M2105_RTL_SAIF)]
set output_dir [file normalize $::env(M2105_OUTPUT_DIR)]

if {$axis eq "ordinary_lru4"} {
    set saif_scope tb_m2105_m2018_tsbg_rtl_saifmap_power.core.dut_base.implementation
    set measurement_cycles 20292
    set scalar_weight_reads 14304
} elseif {$axis eq "tsbg_b4"} {
    set saif_scope tb_m2105_m2018_tsbg_rtl_saifmap_power.core.dut_tsbg.implementation
    set measurement_cycles 7569
    set scalar_weight_reads 4608
} else { error "M2105_FAIL_AXIS" }

foreach path [list $tt_lib_db $mapped_netlist $mapped_sdc $default_map \
        $essential_map $rtl_saif] {
    if {![file isfile $path]} { error "M2105_FAIL_MISSING_INPUT_$path" }
}
file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $tt_lib_db]]
read_db $tt_lib_db
set_app_var link_path [list "*" $tt_lib_db]
read_verilog $mapped_netlist
current_design $design_name
link_design $design_name
if {[sizeof_collection [get_cells -hierarchical -filter "is_black_box==true"]] != 0} {
    error "M2105_FAIL_BLACK_BOX"
}
if {[sizeof_collection [get_cells -hierarchical -filter "is_memory_cell==true"]] != 0} {
    error "M2105_FAIL_NONZERO_MACRO_COUNT"
}
read_sdc $mapped_sdc
set_operating_conditions tt0p9v25c \
    -library tcbn28hpcplusbwp35p140tt0p9v25c
set_wire_load_model -name ZeroWireload \
    -library tcbn28hpcplusbwp35p140tt0p9v25c
set nonclock_inputs [remove_from_collection [all_inputs] [get_ports clk_core]]
set_input_transition 0.100 $nonclock_inputs
set power_enable_analysis true
set_app_var power_analysis_mode averaged
set_app_var timing_save_pin_arrival_and_slack true

# The parser independently classifies/intersects/unions both maps and rejects
# conflicts before this script is reached.  Sourcing both preserves the native
# Synopsys commands and avoids heuristic basename matching.
source $default_map
source $essential_map
redirect -file "$output_dir/reports/saif_annotation_summary.rpt" {
    read_saif -strip_path $saif_scope -report_inconsistent_annotation \
        "$output_dir/reports/inconsistent_annotation.rpt" $rtl_saif
}
report_switching_activity -coverage -include_mapping_types \
    > "$output_dir/reports/switching_coverage.rpt"
report_switching_activity -list_annotated -include_mapping_types \
    > "$output_dir/reports/switching_annotated.rpt"
report_switching_activity -list_not_annotated -include_mapping_types \
    > "$output_dir/reports/switching_unannotated.rpt"
report_switching_activity > "$output_dir/reports/switching_summary.rpt"

proc m2105_read_text {path} {
    set fp [open $path r]
    set value [read $fp]
    close $fp
    return $value
}

# Enforce coverage before any report_power command.  The downstream Python
# parser repeats these gates, but this in-tool gate prevents production of a
# selectable low-coverage power number in the first place.
set annotation_text \
    [m2105_read_text "$output_dir/reports/saif_annotation_summary.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} \
        $annotation_text -> total_nets]
        || ![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} \
        $annotation_text -> annotated_nets annotated_percent]
        || ![regexp {Total number of leaf cells = ([0-9]+)} \
        $annotation_text -> total_leaf]
        || ![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} \
        $annotation_text -> annotated_leaf leaf_percent]} {
    error "M2105_FAIL_ANNOTATION_PARSE_BEFORE_POWER"
}
if {$total_nets <= 0 || $total_leaf <= 0 || $annotated_percent < 95.0 \
        || $leaf_percent < 95.0} {
    error "M2105_FAIL_ANNOTATION_GATE_BEFORE_POWER_NET_${annotated_percent}_LEAF_${leaf_percent}"
}
set coverage_text [m2105_read_text "$output_dir/reports/switching_coverage.rpt"]
if {![regexp -line {^m2018[^ 	]*[ 	]+([0-9.]+)[ 	]+([0-9]+)[ 	]+([0-9]+)[ 	]*$} \
        $coverage_text -> toggle_percent toggled_nets toggle_total]
        || $toggle_total <= 0 || $toggled_nets <= 0 || $toggle_percent < 20.0} {
    error "M2105_FAIL_NONZERO_TOGGLE_COVERAGE_BEFORE_POWER"
}
set inconsistent_text \
    [m2105_read_text "$output_dir/reports/inconsistent_annotation.rpt"]
set inconsistent_rows 0
foreach line [split $inconsistent_text "\n"] {
    set trimmed [string trim $line]
    if {$trimmed ne "" && ![string match "obj_name*" $trimmed]
            && ![regexp {^-+$} $trimmed]} {
        incr inconsistent_rows
    }
}
if {$inconsistent_rows != 0} {
    error "M2105_FAIL_INCONSISTENT_ANNOTATION_BEFORE_POWER_$inconsistent_rows"
}

# Public critical-cone anchors survive synthesis as top-level nets.  A missing
# collection is a hard failure; zero activity is checked by the result parser.
foreach cone {mem_req_valid mem_rsp_valid bridge_valid commit_valid \
        mem_req_accept mem_rsp_accept bridge_accept commit_accept} {
    set nets [get_nets -quiet "${cone}*"]
    if {[sizeof_collection $nets] == 0} {
        error "M2105_FAIL_MISSING_CRITICAL_CONE_$cone"
    }
    report_switching_activity $nets \
        > "$output_dir/reports/critical_${cone}_activity.rpt"
    set cone_live 0
    foreach_in_collection net $nets {
        set toggle_rate [get_attribute -quiet $net toggle_rate]
        if {$toggle_rate ne "" && $toggle_rate > 0.0} {
            set cone_live 1
        }
    }
    if {!$cone_live} {
        error "M2105_FAIL_ZERO_CRITICAL_CONE_BEFORE_POWER_$cone"
    }
}

check_timing -verbose > "$output_dir/reports/check_timing.rpt"
update_timing -full
check_power -verbose -significant_digits 8 \
    > "$output_dir/reports/check_power.rpt"
set check_fp [open "$output_dir/reports/check_power.rpt" r]
set check_text [read $check_fp]
close $check_fp
if {![regexp {check_power succeeded\.} $check_text]} {
    error "M2105_FAIL_CHECK_POWER"
}
update_power
report_power -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/power.rpt"
report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/power_hierarchy.rpt"
report_timing -delay_type max -max_paths 20 -nworst 2 \
    > "$output_dir/reports/timing_setup_tt.rpt"

set fp [open "$output_dir/reports/scope_and_boundary.rpt" w]
puts $fp "milestone=M2105"
puts $fp "axis=$axis"
puts $fp "design=$design_name"
puts $fp "activity=mapped_netlist_power_driven_by_transformation_mapped_RTL_SAIF"
puts $fp "mapped_gate_vcs_activity=false"
puts $fp "saif_scope=$saif_scope"
puts $fp "measurement_cycles=$measurement_cycles"
puts $fp "measurement_duration_ns=[expr {$measurement_cycles * 3.0}]"
puts $fp "scalar_weight_reads=$scalar_weight_reads"
puts $fp "weight_sram_capacity_bytes=294912"
puts $fp "weight_sram_dynamic_energy_in_ptpx=false"
puts $fp "weight_sram_area_in_ptpx=false"
puts $fp "analysis=averaged_prelayout_standard_cell_power"
puts $fp "power_corner=tt0p9v25c"
puts $fp "clock_network=ideal_no_cts"
puts $fp "wireload=ZeroWireload"
puts $fp "macro_count=0"
close $fp
set fp [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $fp "PASS_M2105_PTPX_SOURCE_INTERNAL_PENDING_FAIL_CLOSED_RESULT_PARSER"
close $fp
quit
