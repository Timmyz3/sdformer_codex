set design_name qfit_threshold_late_scale_uq0p24_radix20x4
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

set fp [open $rtl_filelist r]
set rtl_files {}
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line ne "" && ![string match "#*" $line]} {
        lappend rtl_files [file normalize "$hw_root/$line"]
    }
}
close $fp

analyze -format sverilog $rtl_files
elaborate $design_name
current_design $design_name
link
uniquify
set_min_library $lib_db -min_version $min_lib_db
if {[info exists ::env(OPERATING_CONDITION)] \
        && $::env(OPERATING_CONDITION) ne ""} {
    set_operating_conditions $::env(OPERATING_CONDITION)
}

proc m33_uq_resource_audit {stage output_dir require_mapping} {
    set pools [get_cells -quiet -hierarchical \
        -filter "ref_name =~ qfit_signed_int8_mul96_pool*"]
    set leaves [get_cells -quiet -hierarchical \
        -filter "ref_name =~ qfit_signed_int8_mul_leaf*"]
    set pool_count [sizeof_collection $pools]
    set leaf_count [sizeof_collection $leaves]
    set expected_leaf_count [expr {$require_mapping ? 80 : 96}]
    if {$pool_count != 1 || $leaf_count != $expected_leaf_count} {
        error "M33b $stage expected one pool/$expected_leaf_count leaves, got $pool_count/$leaf_count"
    }
    set pool_path [get_object_name [index_collection $pools 0]]
    set report_fp [open "$output_dir/reports/m33_uq_resource_${stage}.rpt" w]
    puts $report_fp "stage=$stage"
    puts $report_fp "pool_count=$pool_count"
    puts $report_fp "leaf_count=$leaf_count"
    puts $report_fp "pool_path=$pool_path"
    set outside_count 0
    set empty_count 0
    set nonempty_count 0
    array set mapped_count {}
    array set mapped_area {}
    if {$require_mapping} {
        foreach_in_collection leaf $leaves {
            set leaf_path [get_object_name $leaf]
            set mapped_count($leaf_path) 0
            set mapped_area($leaf_path) 0.0
        }
        set mapped_cells [get_cells -quiet -hierarchical \
            -filter "is_hierarchical == false"]
        foreach_in_collection cell $mapped_cells {
            set cursor [get_object_name $cell]
            while {1} {
                set slash [string last "/" $cursor]
                if {$slash < 0} { break }
                set cursor [string range $cursor 0 [expr {$slash - 1}]]
                if {[info exists mapped_count($cursor)]} {
                    incr mapped_count($cursor)
                    set cell_area [get_attribute -quiet $cell area]
                    if {$cell_area eq "" || ![string is double -strict $cell_area]} {
                        close $report_fp
                        error "M33b invalid mapped area under $cursor"
                    }
                    set mapped_area($cursor) [expr {
                        [set mapped_area($cursor)] + double($cell_area)
                    }]
                    break
                }
            }
        }
    }
    foreach_in_collection leaf $leaves {
        set leaf_path [get_object_name $leaf]
        if {[string first "${pool_path}/" $leaf_path] != 0} {
            incr outside_count
        }
        set count 0
        set area 0.0
        if {$require_mapping} {
            set count [set mapped_count($leaf_path)]
            set area [set mapped_area($leaf_path)]
            if {$count > 0 && $area > 0.0} {
                incr nonempty_count
            } else {
                incr empty_count
            }
        }
        puts $report_fp "leaf=$leaf_path mapped_cells=$count mapped_area=$area"
    }
    puts $report_fp "pool_external_leaf_count=$outside_count"
    puts $report_fp "nonempty_mapped_leaf_count=$nonempty_count"
    puts $report_fp "empty_mapped_leaf_count=$empty_count"
    if {$outside_count != 0} {
        close $report_fp
        error "M33b multiplier leaf outside sole pool"
    }
    if {$require_mapping && ($nonempty_count != 80 || $empty_count != 0)} {
        close $report_fp
        error "M33b expected exact 80 retained mapped leaves, got $nonempty_count/$empty_count"
    }
    puts $report_fp "logical_leaf_count_precompile=96"
    puts $report_fp "constant_spare_leaves_removed_postcompile=[expr {$require_mapping ? 16 : 0}]"
    puts $report_fp "status=PASS_ONE_POOL_96_LOGICAL_80_MAPPED_16_CONSTANT_SPARES_REMOVED"
    close $report_fp
    return [list $pools $leaves]
}

set pre_audit [m33_uq_resource_audit precompile $output_dir 0]
set pool_cells [lindex $pre_audit 0]
set leaf_cells [lindex $pre_audit 1]
set_ungroup $pool_cells false
set_boundary_optimization $pool_cells false
set_ungroup $leaf_cells false
set_boundary_optimization $leaf_cells false

source $sdc_file
set_clock_uncertainty -hold 0.100 [get_clocks core_clk]
set_fix_hold [get_clocks core_clk]
check_design > "$output_dir/reports/check_design_precompile.rpt"
check_timing > "$output_dir/reports/check_timing_precompile.rpt"
report_resources -hierarchy > "$output_dir/reports/resources_precompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_precompile.rpt"

compile_ultra -no_autoungroup
compile_ultra -incremental -no_autoungroup
compile -incremental_mapping -only_hold_time
set_clock_uncertainty -hold 0.090 [get_clocks core_clk]
update_timing

m33_uq_resource_audit postcompile $output_dir 1
report_hierarchy > "$output_dir/reports/hierarchy_postcompile.rpt"
report_resources -hierarchy > "$output_dir/reports/resources_postcompile.rpt"
report_reference -hierarchy > "$output_dir/reports/references_postcompile.rpt"
report_qor > "$output_dir/reports/qor.rpt"
report_area -hierarchy > "$output_dir/reports/area.rpt"
report_clocks > "$output_dir/reports/clocks.rpt"
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
