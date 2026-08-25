set design_name qfit_atlif_unified_t10_t2_stream_core
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

proc m31_resource_audit {stage output_dir require_nonempty} {
    set pools [get_cells -quiet -hierarchical \
        -filter "ref_name =~ qfit_signed_int8_mul96_pool*"]
    set leaves [get_cells -quiet -hierarchical \
        -filter "ref_name =~ qfit_signed_int8_mul_leaf*"]
    set pool_count [sizeof_collection $pools]
    set leaf_count [sizeof_collection $leaves]
    if {$pool_count != 1} {
        error "M31 $stage requires exactly one multiplier pool, got $pool_count"
    }
    if {$leaf_count != 96} {
        error "M31 $stage requires exactly 96 multiplier leaves, got $leaf_count"
    }
    set pool_path [get_object_name [index_collection $pools 0]]
    set audit_fp [open "$output_dir/reports/m31_resource_audit_${stage}.rpt" w]
    puts $audit_fp "stage=$stage"
    puts $audit_fp "pool_count=$pool_count"
    puts $audit_fp "leaf_count=$leaf_count"
    puts $audit_fp "pool_path=$pool_path"
    set outside_count 0
    set empty_count 0
    array set mapped_count {}
    array set mapped_area {}
    if {$require_nonempty} {
        foreach_in_collection leaf $leaves {
            set leaf_path [get_object_name $leaf]
            set mapped_count($leaf_path) 0
            set mapped_area($leaf_path) 0.0
        }
        # Do not query "${leaf_path}/*": generated instance names contain
        # square brackets, which collection globs interpret as character
        # classes.  Instead enumerate mapped cells once and walk their literal
        # slash-delimited ancestors in Tcl.
        set mapped_cells [get_cells -quiet -hierarchical \
            -filter "is_hierarchical == false"]
        foreach_in_collection cell $mapped_cells {
            set cursor [get_object_name $cell]
            while {1} {
                set slash [string last "/" $cursor]
                if {$slash < 0} {
                    break
                }
                set cursor [string range $cursor 0 [expr {$slash - 1}]]
                if {[info exists mapped_count($cursor)]} {
                    incr mapped_count($cursor)
                    set cell_area [get_attribute -quiet $cell area]
                    if {$cell_area eq "" \
                            || ![string is double -strict $cell_area]} {
                        close $audit_fp
                        error "M31 mapped cell has invalid area: [get_object_name $cell]"
                    }
                    set accumulated [expr {
                        [set mapped_area($cursor)] + double($cell_area)
                    }]
                    set mapped_area($cursor) $accumulated
                    break
                }
            }
        }
    }
    foreach_in_collection leaf $leaves {
        set leaf_path [get_object_name $leaf]
        set leaf_ref [get_attribute $leaf ref_name]
        if {[string first "${pool_path}/" $leaf_path] != 0} {
            incr outside_count
        }
        set leaf_mapped_count 0
        set leaf_mapped_area 0.0
        if {$require_nonempty} {
            set leaf_mapped_count [set mapped_count($leaf_path)]
            set leaf_mapped_area [set mapped_area($leaf_path)]
            if {$leaf_mapped_count <= 0 || $leaf_mapped_area <= 0.0} {
                incr empty_count
            }
        }
        puts $audit_fp "leaf=$leaf_path ref=$leaf_ref mapped_cells=$leaf_mapped_count mapped_area=$leaf_mapped_area"
    }
    puts $audit_fp "pool_external_leaf_count=$outside_count"
    puts $audit_fp "empty_mapped_leaf_count=$empty_count"
    if {$outside_count != 0} {
        close $audit_fp
        error "M31 $stage found $outside_count multiplier leaves outside u_mul_pool"
    }
    if {$require_nonempty && $empty_count != 0} {
        close $audit_fp
        error "M31 $stage found $empty_count empty mapped multiplier leaves"
    }
    puts $audit_fp "status=PASS_EXACT_ONE_POOL_96_LEAVES"
    close $audit_fp
    return [list $pools $leaves]
}

set pre_audit {}
if {[catch {
    set pre_audit [m31_resource_audit precompile $output_dir 0]
} audit_error]} {
    puts stderr "M31 precompile audit fatal: $audit_error"
    exit 91
}
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
report_hierarchy > "$output_dir/reports/hierarchy_precompile.rpt"
report_resources -hierarchy \
    > "$output_dir/reports/resources_precompile.rpt"
report_reference -hierarchy \
    > "$output_dir/reports/references_precompile.rpt"

compile_ultra -no_autoungroup
compile_ultra -incremental -no_autoungroup
compile -incremental_mapping -only_hold_time
set_clock_uncertainty -hold 0.090 [get_clocks core_clk]
update_timing

if {[catch {
    m31_resource_audit postcompile $output_dir 1
} audit_error]} {
    puts stderr "M31 postcompile audit fatal: $audit_error"
    exit 92
}
report_hierarchy > "$output_dir/reports/hierarchy_postcompile.rpt"
report_resources -hierarchy \
    > "$output_dir/reports/resources_postcompile.rpt"
report_reference -hierarchy \
    > "$output_dir/reports/references_postcompile.rpt"
report_qor > "$output_dir/reports/qor.rpt"
report_area -hierarchy > "$output_dir/reports/area.rpt"
report_clocks > "$output_dir/reports/clocks.rpt"
report_port -verbose > "$output_dir/reports/ports.rpt"
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
    report_constraint -min_pulse_width -all_violators -significant_digits 4
    report_constraint -min_period -all_violators -significant_digits 4
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
quit
