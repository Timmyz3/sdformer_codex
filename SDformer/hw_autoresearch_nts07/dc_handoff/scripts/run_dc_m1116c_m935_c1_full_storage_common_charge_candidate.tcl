# M1116C source-only DC Tcl.  This file is authored but not launched here.
# The physical top includes frozen M935 plus its nine live parent macros only.
# Psum/weight/residual ledger capacity remains an identical external common
# charge and is never materialized as dummy/tied-off macros.

set design_name m1116c_m935_c1_full_storage_common_charge_boundary
set macro_cell TS1N28HPCPHVTB128X128M4S
set storage_budget_bytes 245760
set required_boundary_bytes 214912

set hw_root [file normalize $::env(HW_ROOT)]
set rtl_filelist [file normalize $::env(RTL_FILELIST)]
set sdc_file [file normalize $::env(SDC_FILE)]
set mapping_manifest [file normalize $::env(STORAGE_MAPPING_MANIFEST)]
set std_slow_db [file normalize $::env(STD_SLOW_DB)]
set std_fast_db [file normalize $::env(STD_FAST_DB)]
set macro_slow_db [file normalize $::env(MACRO_SLOW_DB)]
set macro_fast_db [file normalize $::env(MACRO_FAST_DB)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

# Derive all storage counts from the canonical mapping manifest.  No literal
# 93/105-macro physical target is legal in this flow.
set map_fp [open $mapping_manifest r]
set mapping_rows 0
set expected_start 0
set represented_bytes 0
set internal_macro_count 0
set internal_macro_capacity_bytes 0
set external_common_charge_bytes 0
set external_common_macro_equivalents 0
set mapping_report_rows {}
while {[gets $map_fp line] >= 0} {
    set line [string trim $line]
    if {$line eq "" || [string match "#*" $line]} { continue }
    set fields [split $line "|"]
    if {[llength $fields] != 14} {
        error "M1116C_FAIL mapping field count != 14"
    }
    set class_name [lindex $fields 0]
    set byte_start [lindex $fields 1]
    set byte_end [lindex $fields 2]
    set class_bytes [lindex $fields 3]
    set placement [lindex $fields 4]
    set class_macro_cell [lindex $fields 5]
    set physical_count [lindex $fields 6]
    set physical_capacity [lindex $fields 7]
    set common_equivalents [lindex $fields 8]
    set live_binding [lindex $fields 11]
    set area_in_dc [lindex $fields 13]

    if {$byte_start != $expected_start} {
        error "M1116C_FAIL mapping gap/overlap at $class_name"
    }
    if {[expr {$byte_end - $byte_start + 1}] != $class_bytes} {
        error "M1116C_FAIL byte range mismatch at $class_name"
    }
    if {$live_binding eq ""} {
        error "M1116C_FAIL empty live/common binding at $class_name"
    }
    set expected_start [expr {$byte_end + 1}]
    set represented_bytes [expr {$represented_bytes + $class_bytes}]
    incr mapping_rows

    if {$placement eq "foundry_macro_internal"} {
        if {$class_macro_cell ne $macro_cell || $area_in_dc ne "true"} {
            error "M1116C_FAIL internal macro class mismatch at $class_name"
        }
        set internal_macro_count [expr {$internal_macro_count + $physical_count}]
        set internal_macro_capacity_bytes [expr {$internal_macro_capacity_bytes
            + $physical_count * $physical_capacity}]
        if {[expr {$physical_count * $physical_capacity}] != $class_bytes} {
            error "M1116C_FAIL internal macro capacity mismatch at $class_name"
        }
    } elseif {$placement eq "identical_external_common_charge"} {
        if {$physical_count != 0 || $physical_capacity != 0
                || $class_macro_cell ne "NONE" || $area_in_dc ne "false"} {
            error "M1116C_FAIL external common charge materialized at $class_name"
        }
        set external_common_charge_bytes [expr {$external_common_charge_bytes
            + $class_bytes}]
        set external_common_macro_equivalents [expr {
            $external_common_macro_equivalents + $common_equivalents}]
    } else {
        error "M1116C_FAIL unknown placement at $class_name"
    }
    lappend mapping_report_rows $line
}
close $map_fp

if {$mapping_rows != 4 || $represented_bytes != $required_boundary_bytes
        || $expected_start != $required_boundary_bytes} {
    error "M1116C_FAIL incomplete 214912-byte mapping"
}
if {$internal_macro_count != 9 || $internal_macro_capacity_bytes != 18432
        || $external_common_charge_bytes != 196480} {
    error "M1116C_FAIL boundary split drift"
}
set unallocated_budget_margin_bytes [expr {$storage_budget_bytes
    - $represented_bytes}]
if {$unallocated_budget_margin_bytes != 30848} {
    error "M1116C_FAIL budget margin drift"
}

file mkdir $output_dir
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_svf "$output_dir/netlist/${design_name}.svf"
set_app_var search_path [list $hw_root [file dirname $std_slow_db] \
    [file dirname $std_fast_db] [file dirname $macro_slow_db] \
    [file dirname $macro_fast_db]]
set_app_var target_library [list $std_slow_db]
set_app_var link_library [list "*" $std_slow_db $std_fast_db \
    $macro_slow_db $macro_fast_db]
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
redirect "$output_dir/reports/link.rpt" {link}
uniquify
set_min_library $std_slow_db -min_version $std_fast_db
set_min_library $macro_slow_db -min_version $macro_fast_db
set_operating_conditions ssg0p9v125c

set macro_lib_cells [get_lib_cells -quiet */$macro_cell]
if {[sizeof_collection $macro_lib_cells] < 1} {
    error "M1116C_FAIL unresolved foundry macro cell"
}
set macro_cells_pre [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set macro_count_pre [sizeof_collection $macro_cells_pre]
if {$macro_count_pre != $internal_macro_count} {
    error "M1116C_FAIL macro_count_pre=$macro_count_pre expected=$internal_macro_count"
}
set_dont_touch $macro_cells_pre true

source $sdc_file
set_wire_load_model -name ZeroWireload [current_design]
set flow_fp [open "$output_dir/reports/flow_contract.rpt" w]
puts $flow_fp "flow=m1116c_m935_c1_full_storage_common_charge_candidate"
puts $flow_fp "clock_period_ns=3.000"
puts $flow_fp "ideal_clock=true"
puts $flow_fp "wireload=ZeroWireload"
puts $flow_fp "compile_ultra_count=1"
puts $flow_fp "incremental_compile_count=0"
puts $flow_fp "hold_fix_command_count=0"
puts $flow_fp "false_path_count=0"
puts $flow_fp "multicycle_path_count=0"
puts $flow_fp "disabled_timing_arc_count=0"
puts $flow_fp "case_analysis_count=0"
puts $flow_fp "dummy_or_tied_off_storage_macros=0"
puts $flow_fp "full_214912B_physically_integrated=false"
puts $flow_fp "external_common_charge_area_modeled=false"
close $flow_fp

set map_report [open "$output_dir/reports/storage_boundary_mapping.rpt" w]
puts $map_report "status=PASS_M1116C_SOURCE_MAPPING_PARSED"
puts $map_report "mapping_rows=$mapping_rows"
puts $map_report "represented_boundary_bytes=$represented_bytes"
puts $map_report "storage_budget_bytes=$storage_budget_bytes"
puts $map_report "unallocated_budget_margin_bytes=$unallocated_budget_margin_bytes"
puts $map_report "internal_parent_macro_count=$internal_macro_count"
puts $map_report "internal_parent_macro_capacity_bytes=$internal_macro_capacity_bytes"
puts $map_report "external_common_charge_bytes=$external_common_charge_bytes"
puts $map_report "external_common_macro_equivalents_diagnostic=$external_common_macro_equivalents"
puts $map_report "external_common_physical_macro_count=0"
puts $map_report "full_214912B_physically_integrated=false"
foreach row $mapping_report_rows { puts $map_report "mapping=$row" }
close $map_report

check_design > "$output_dir/reports/check_design_precompile.rpt"
redirect "$output_dir/reports/check_timing_precompile.rpt" {check_timing}
set pre_fp [open "$output_dir/reports/check_timing_precompile.rpt" r]
set pre_text [read $pre_fp]
close $pre_fp
set pre_tim209 [regexp -all -- {TIM-209} $pre_text]
set pre_opt150 [regexp -all -- {OPT-150} $pre_text]
if {$pre_tim209 != 0 || $pre_opt150 != 0} {
    error "M1116C_FAIL precompile TIM-209/OPT-150"
}

report_resources -hierarchy > "$output_dir/reports/resources_precompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_precompile.rpt"
compile_ultra -no_autoungroup
update_timing

set macro_cells_post [get_cells -hierarchical -filter "ref_name == $macro_cell"]
set macro_count_post [sizeof_collection $macro_cells_post]
if {$macro_count_post != $internal_macro_count} {
    error "M1116C_FAIL macro_count_post=$macro_count_post expected=$internal_macro_count"
}

set total_cell_area [get_attribute [current_design] area]
set internal_parent_macro_area 0.0
foreach_in_collection macro_instance $macro_cells_post {
    set internal_parent_macro_area [expr {$internal_parent_macro_area
        + [get_attribute $macro_instance area]}]
}
set standard_cell_logic_area [expr {$total_cell_area
    - $internal_parent_macro_area}]
set area_fp [open "$output_dir/reports/storage_area_breakdown_machine.txt" w]
puts $area_fp "status=PASS_M1116C_PHYSICAL_DC_BOUNDARY_AREA_ONLY"
puts $area_fp [format "standard_cell_logic_area_um2=%.6f" $standard_cell_logic_area]
puts $area_fp [format "internal_parent_macro_area_um2=%.6f" $internal_parent_macro_area]
puts $area_fp [format "physical_dc_total_area_um2=%.6f" $total_cell_area]
puts $area_fp "internal_parent_macro_count=$macro_count_post"
puts $area_fp "internal_parent_capacity_bytes=$internal_macro_capacity_bytes"
puts $area_fp "external_psum_weight_reserve_common_charge_bytes=$external_common_charge_bytes"
puts $area_fp "external_common_charge_area_um2=UNMODELED_EXCLUDED"
puts $area_fp "full_214912B_total_area_um2=NOT_ADMITTED"
close $area_fp

report_hierarchy > "$output_dir/reports/hierarchy_postcompile.rpt"
report_resources -hierarchy > "$output_dir/reports/resources_postcompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_postcompile.rpt"
report_qor > "$output_dir/reports/qor.rpt"
report_area -hierarchy > "$output_dir/reports/area_hierarchy.rpt"
report_clocks > "$output_dir/reports/clocks.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 \
    -significant_digits 4 > "$output_dir/reports/timing_setup_top100.rpt"
redirect "$output_dir/reports/constraint_setup_all.rpt" {
    report_constraint -max_delay -all_violators -significant_digits 4
}
redirect "$output_dir/reports/constraint_max_capacitance.rpt" {
    report_constraint -max_capacitance -all_violators -significant_digits 4
}
redirect "$output_dir/reports/constraint_max_transition.rpt" {
    report_constraint -max_transition -all_violators -significant_digits 4
}
redirect "$output_dir/reports/constraint_max_fanout.rpt" {
    report_constraint -max_fanout -all_violators -significant_digits 4
}
check_design > "$output_dir/reports/check_design_postcompile.rpt"
redirect "$output_dir/reports/check_timing_postcompile.rpt" {check_timing}

set negative_paths [get_timing_paths -delay_type max -max_paths 200000 \
    -slack_lesser_than 0.0]
set violation_count [sizeof_collection $negative_paths]
set setup_wns 0.0
set setup_tns 0.0
if {$violation_count > 0} {
    set setup_wns 1.0e30
    foreach_in_collection path $negative_paths {
        set path_slack [get_attribute $path slack]
        if {$path_slack < $setup_wns} { set setup_wns $path_slack }
        set setup_tns [expr {$setup_tns + $path_slack}]
    }
    set setup_status VIOLATED_CAPTURED
} else {
    set worst_path [get_timing_paths -delay_type max -max_paths 1]
    if {[sizeof_collection $worst_path] > 0} {
        set setup_wns [get_attribute [index_collection $worst_path 0] slack]
    }
    set setup_status MET
}
set sum_fp [open "$output_dir/reports/setup_summary_machine.txt" w]
puts $sum_fp "status=$setup_status"
puts $sum_fp [format "setup_wns_ns=%.6f" $setup_wns]
puts $sum_fp [format "setup_tns_ns=%.6f" $setup_tns]
puts $sum_fp "setup_violating_paths=$violation_count"
puts $sum_fp "clock_period_ns=3.000"
close $sum_fp

change_names -rules verilog -hierarchy
write_file -format verilog -hierarchy \
    -output "$output_dir/netlist/${design_name}_mapped.v"
write_sdc "$output_dir/netlist/${design_name}_mapped.sdc"
write -format ddc -hierarchy -output "$output_dir/netlist/${design_name}.ddc"
set_svf -off

set terminal_fp [open "$output_dir/TCL_PASS_TERMINAL.txt" w]
puts $terminal_fp "status=PASS_M1116C_DC_EXECUTION_AND_REPORT_CLOSURE"
puts $terminal_fp "setup_status=$setup_status"
puts $terminal_fp "TIM-209=$pre_tim209"
puts $terminal_fp "OPT-150=$pre_opt150"
puts $terminal_fp "internal_parent_macro_count_pre=$macro_count_pre"
puts $terminal_fp "internal_parent_macro_count_post=$macro_count_post"
puts $terminal_fp "external_common_charge_physical_macros=0"
puts $terminal_fp "full_214912B_physically_integrated=false"
puts $terminal_fp "hold_signoff=false"
puts $terminal_fp "power_measured=false"
close $terminal_fp
quit
