set design_name m935_m912_three_stage_exact_parent_match_product_capture_island
set macro_cell TS1N28HPCPHVTB128X128M4S
set expected_macro_count 9
set std_tt_db [file normalize $::env(M1782_STD_TT_DB)]
set std_ss_db [file normalize $::env(M1782_STD_SS_DB)]
set macro_slow_db [file normalize $::env(M1782_MACRO_SLOW_DB)]
set mapped_netlist [file normalize $::env(M1782_MAPPED_NETLIST)]
set mapped_sdc [file normalize $::env(M1782_MAPPED_SDC)]
set gate_saif [file normalize $::env(M1782_GATE_SAIF)]
set output_dir [file normalize $::env(M1782_OUTPUT_DIR)]

file mkdir $output_dir
file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $std_tt_db] \
    [file dirname $std_ss_db] [file dirname $macro_slow_db]]
set_app_var link_path [list "*" $std_tt_db $std_ss_db $macro_slow_db]

read_verilog $mapped_netlist
current_design $design_name
link_design $design_name

# PrimeTime classifies a linked hard-memory Liberty leaf as a black box because
# it has no standard-cell implementation below the leaf.  That classification
# is legal only for the exact nine parent SRAM leaves in this component.  This
# is an allow-list, not a disabled black-box check: every missing, extra,
# hierarchical, or wrong-reference object remains fatal.
set black_boxes [get_cells -hierarchical -filter "is_black_box==true"]
set macro_cells [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set black_box_count [sizeof_collection $black_boxes]
set macro_count [sizeof_collection $macro_cells]
array set expected_names {}
for {set index 0} {$index < $expected_macro_count} {incr index} {
    set expected_names([format {u_parent_scratch/g_slice_%d__u_parent_sram} $index]) 1
}
array set observed_names {}
set inventory_fp [open "$output_dir/reports/black_box_inventory_machine.rpt" w]
puts $inventory_fp "black_box_count=$black_box_count"
puts $inventory_fp "expected_macro_count=$expected_macro_count"
foreach_in_collection cell $black_boxes {
    set cell_name [get_object_name $cell]
    set ref_name [get_attribute -quiet $cell ref_name]
    set is_hierarchical [get_attribute -quiet $cell is_hierarchical]
    set is_black_box [get_attribute -quiet $cell is_black_box]
    puts $inventory_fp "name=$cell_name ref=$ref_name is_hierarchical=$is_hierarchical is_black_box=$is_black_box"
    if {[info exists observed_names($cell_name)]} {
        close $inventory_fp
        error "M1782_FAIL_DUPLICATE_BLACK_BOX_NAME_$cell_name"
    }
    set observed_names($cell_name) 1
}
close $inventory_fp
if {$black_box_count != $expected_macro_count} {
    error "M1782_FAIL_BLACK_BOX_COUNT_$black_box_count"
}
if {$macro_count != $expected_macro_count} {
    error "M1782_FAIL_MACRO_COUNT_$macro_count"
}
foreach expected_name [array names expected_names] {
    if {![info exists observed_names($expected_name)]} {
        error "M1782_FAIL_EXPECTED_MACRO_BLACK_BOX_MISSING_$expected_name"
    }
}
foreach_in_collection cell $black_boxes {
    set cell_name [get_object_name $cell]
    set ref_name [get_attribute -quiet $cell ref_name]
    set is_hierarchical [get_attribute -quiet $cell is_hierarchical]
    set is_black_box [get_attribute -quiet $cell is_black_box]
    if {![info exists expected_names($cell_name)]} {
        error "M1782_FAIL_UNEXPECTED_BLACK_BOX_$cell_name"
    }
    if {$ref_name ne $macro_cell} {
        error "M1782_FAIL_BLACK_BOX_WRONG_REF_${cell_name}_${ref_name}"
    }
    if {$is_hierarchical ne "false"} {
        error "M1782_FAIL_BLACK_BOX_NOT_LEAF_$cell_name"
    }
    if {$is_black_box ne "true"} {
        error "M1782_FAIL_EXPECTED_BLACK_BOX_ATTRIBUTE_$cell_name"
    }
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
proc m1782_read_text {path} {
    set fp [open $path r]
    set value [read $fp]
    close $fp
    return $value
}
redirect -file "$output_dir/reports/saif_annotation_summary.rpt" {
    read_saif -strip_path $::env(M1782_SAIF_INSTANCE) \
        -report_inconsistent_annotation \
        "$output_dir/reports/inconsistent_annotation.rpt" $gate_saif
}
redirect -file "$output_dir/reports/switching_coverage.rpt" {
    report_switching_activity -coverage -include_mapping_types
}
report_switching_activity -list_annotated -include_mapping_types \
    > "$output_dir/reports/switching_annotated.rpt"
report_switching_activity -list_not_annotated -include_mapping_types \
    > "$output_dir/reports/switching_unannotated.rpt"

set saif_text [m1782_read_text "$output_dir/reports/saif_annotation_summary.rpt"]
if {![regexp {Total number of nets = ([0-9]+)} $saif_text -> total_nets]} {
    error "M1782_FAIL_CANNOT_PARSE_TOTAL_NETS"
}
if {![regexp {Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_nets annotated_percent]} {
    error "M1782_FAIL_CANNOT_PARSE_ANNOTATED_NETS"
}
if {![regexp {Total number of leaf cells = ([0-9]+)} \
        $saif_text -> total_leaf_cells]} {
    error "M1782_FAIL_CANNOT_PARSE_TOTAL_LEAF_CELLS"
}
if {![regexp {Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)} \
        $saif_text -> annotated_leaf_cells annotated_leaf_percent]} {
    error "M1782_FAIL_CANNOT_PARSE_ANNOTATED_LEAF_CELLS"
}
if {$total_nets <= 0 || $annotated_nets != $total_nets \
        || $annotated_percent != 100.0} {
    error "M1782_FAIL_EXACT_NET_ANNOTATION_GATE"
}
if {$total_leaf_cells <= 0 || $annotated_leaf_cells != $total_leaf_cells \
        || $annotated_leaf_percent != 100.0} {
    error "M1782_FAIL_EXACT_LEAF_ANNOTATION_GATE"
}

check_timing -verbose > "$output_dir/reports/check_timing.rpt"
update_timing -full
check_power -verbose -significant_digits 8 \
    > "$output_dir/reports/check_power.rpt"
update_power

report_power -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_whole_mapped_c1_including_9macro_liberty.rpt"
report_power -hierarchy -area -unit mW -nosplit -significant_digits 8 \
    > "$output_dir/reports/ptpx_hierarchy_diagnostic_including_9macro_liberty.rpt"
report_timing -delay_type max -max_paths 20 -nworst 2 \
    > "$output_dir/reports/timing_setup_mixed_corner_diagnostic.rpt"
report_timing -delay_type min -max_paths 20 -nworst 2 \
    > "$output_dir/reports/timing_hold_mixed_corner_diagnostic.rpt"

set scope_fp [open "$output_dir/reports/scope_and_boundary.rpt" w]
puts $scope_fp "milestone=M1782"
puts $scope_fp "design=$design_name"
puts $scope_fp "analysis=averaged_prelayout_mapped_gate_activity"
puts $scope_fp "corner_classification=mixed_corner_component_estimate"
puts $scope_fp "standard_cell_power_library=TT_0p9V_25C"
puts $scope_fp "parent_sram_macro_liberty=SSG_0p9V_125C"
puts $scope_fp "not_single_corner_signoff=true"
puts $scope_fp "clock_period_ns=3.000"
puts $scope_fp "measurement_cycles=$::env(M1782_MEASUREMENT_CYCLES)"
puts $scope_fp "saif_duration_ns=$::env(M1782_SAIF_DURATION_NS)"
puts $scope_fp "saif_scope=$::env(M1782_SAIF_INSTANCE)"
puts $scope_fp "public_port_only_testbench=true"
puts $scope_fp "testbench_force_or_release=false"
puts $scope_fp "clock_network=ideal_no_cts"
puts $scope_fp "wireload=ZeroWireload"
puts $scope_fp "spef=false"
puts $scope_fp "macro_count=$macro_count"
puts $scope_fp "black_box_count=$black_box_count"
puts $scope_fp "black_box_policy=exact_9_expected_linked_sram_liberty_leaves_only"
puts $scope_fp "unresolved_or_unexpected_black_box_allowed=false"
puts $scope_fp "primary_report=whole_mapped_c1_top_including_9macro_liberty"
puts $scope_fp "top_minus_macro=false"
puts $scope_fp "ptpx_plus_datasheet_sram_combined=false"
puts $scope_fp "parent_sram_datasheet_is_separate_alternative_sensitivity=true"
puts $scope_fp "not_total_c1_schedule=true"
puts $scope_fp "not_energy_per_frame=true"
close $scope_fp

set marker [open "$output_dir/PTPX_INTERNAL_COMPLETE.txt" w]
puts $marker "PASS_M1782_C1_M1701_EXPECTED_MACRO_LEAF_BLACKBOX_PTPX_TOOL_COMPLETE"
puts $marker "macro_count=$macro_count"
puts $marker "black_box_count=$black_box_count"
puts $marker "black_box_policy=exact_9_expected_linked_sram_liberty_leaves_only"
puts $marker "public_port_only=true"
puts $marker "exact_net_annotation=$annotated_nets/$total_nets=100.0%"
puts $marker "exact_leaf_annotation=$annotated_leaf_cells/$total_leaf_cells=100.0%"
puts $marker "claim_boundary=whole_mapped_component_including_9macro_liberty_plus_separate_uncombined_sram_sensitivity_pending_checker"
close $marker
quit
