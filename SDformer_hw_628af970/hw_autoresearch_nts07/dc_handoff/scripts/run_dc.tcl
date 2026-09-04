set design_name $::env(DESIGN_NAME)
set hw_root [file normalize $::env(HW_ROOT)]
set rtl_filelist [file normalize $::env(RTL_FILELIST)]
set lib_db [file normalize $::env(LIB_DB)]
set min_lib_db ""
if {[info exists ::env(MIN_LIB_DB)] && $::env(MIN_LIB_DB) ne ""} {
    set min_lib_db [file normalize $::env(MIN_LIB_DB)]
}
set sdc_file [file normalize $::env(SDC_FILE)]
set output_dir [file normalize $::env(OUTPUT_DIR)]

file mkdir $output_dir
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_svf "$output_dir/netlist/${design_name}.svf"

set search_paths [list $hw_root [file dirname $lib_db]]
set_app_var target_library [list $lib_db]
set macro_dbs {}
if {[info exists ::env(MACRO_DBS)] && $::env(MACRO_DBS) ne ""} {
    foreach macro_db [split $::env(MACRO_DBS) ":"] {
        set normalized_macro_db [file normalize $macro_db]
        lappend macro_dbs $normalized_macro_db
        lappend search_paths [file dirname $normalized_macro_db]
    }
}
set_app_var search_path $search_paths
set link_libraries [list "*" $lib_db]
if {$min_lib_db ne ""} {
    lappend link_libraries $min_lib_db
}
set_app_var link_library [concat $link_libraries $macro_dbs]
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
if {[info exists ::env(ELAB_PARAMETERS)] && $::env(ELAB_PARAMETERS) ne ""} {
    # Keep the shell-facing contract neutral (NAME=VALUE) and translate to
    # Design Compiler's named-parameter syntax.
    set dc_parameters [string map {"=" "=>"} $::env(ELAB_PARAMETERS)]
    elaborate $design_name -parameters $dc_parameters
    # Parameterized elaboration creates a derived WORK design whose name is
    # tool-generated (for example, TOP_PARAMETER1).  Keep that derived design
    # current; selecting the unparameterized source name here silently loses
    # the requested A/B identity and emits UID-109 while DC keeps running.
    set elaborated_design [current_design]
} else {
    elaborate $design_name
    set elaborated_design [current_design]
}
if {$elaborated_design eq ""} {
    error "elaboration did not leave a current design for $design_name"
}
current_design $elaborated_design
link
uniquify

# A pre-macro run may intentionally compile a zero-area/timing RTL shell for
# an SRAM whose characterized .db is not available yet.  Keep that boundary
# explicit and fail closed on the expected instance count; otherwise DC can
# silently flatten or optimize through an empty behavioral shell and produce
# a misleading logic-area comparison.  These logical shells are not library
# macros and therefore do not make a run paper-PPA admissible.
if {[info exists ::env(PREMACRO_LOGICAL_SHELL_PATTERN)] &&
        $::env(PREMACRO_LOGICAL_SHELL_PATTERN) ne ""} {
    set shell_pattern $::env(PREMACRO_LOGICAL_SHELL_PATTERN)
    set shell_cells [get_cells -hierarchical -filter "ref_name =~ $shell_pattern"]
    set shell_count [sizeof_collection $shell_cells]
    if {![info exists ::env(PREMACRO_LOGICAL_SHELL_COUNT)] ||
            $::env(PREMACRO_LOGICAL_SHELL_COUNT) eq ""} {
        error "PREMACRO_LOGICAL_SHELL_PATTERN requires PREMACRO_LOGICAL_SHELL_COUNT"
    }
    set expected_shell_count $::env(PREMACRO_LOGICAL_SHELL_COUNT)
    if {$shell_count != $expected_shell_count} {
        error "pre-macro logical shell count mismatch: expected $expected_shell_count, got $shell_count"
    }
    set_dont_touch $shell_cells true
    set shell_fp [open "$output_dir/reports/premacro_logical_shells.rpt" w]
    puts $shell_fp "scope=PREMACRO_LOGIC_ONLY"
    puts $shell_fp "paper_ppa_ready=false"
    puts $shell_fp "pattern=$shell_pattern"
    puts $shell_fp "expected_count=$expected_shell_count"
    puts $shell_fp "observed_count=$shell_count"
    foreach_in_collection shell_cell $shell_cells {
        puts $shell_fp "[get_object_name $shell_cell] [get_attribute $shell_cell ref_name]"
    }
    close $shell_fp
}

if {$min_lib_db ne ""} {
    set_min_library $lib_db -min_version $min_lib_db
}

if {[info exists ::env(OPERATING_CONDITION)] && $::env(OPERATING_CONDITION) ne ""} {
    set_operating_conditions $::env(OPERATING_CONDITION)
}

source $sdc_file
if {$min_lib_db ne ""} {
    # Close DC against a deliberately stronger min-delay guard than the
    # 50 ps PrimeTime hold uncertainty.  The default 100 ps guard leaves
    # measurable fast-corner margin instead of a rounded 0.0000 ns pass.
    if {[info exists ::env(DC_HOLD_UNCERTAINTY_NS)]
            && $::env(DC_HOLD_UNCERTAINTY_NS) ne ""} {
        set_clock_uncertainty -hold $::env(DC_HOLD_UNCERTAINTY_NS) \
            [get_clocks core_clk]
    }
    set_fix_hold [get_clocks core_clk]
}
if {[info exists ::env(SAIF_FILE)] && $::env(SAIF_FILE) ne ""} {
    set saif_file [file normalize $::env(SAIF_FILE)]
    if {![file exists $saif_file]} {
        error "SAIF_FILE不存在: $saif_file"
    }
    if {![info exists ::env(SAIF_INSTANCE)] || $::env(SAIF_INSTANCE) eq ""} {
        error "提供SAIF_FILE时必须同时提供SAIF_INSTANCE"
    }
    read_saif -input $saif_file -instance_name $::env(SAIF_INSTANCE)
}
set power_scope_fp [open "$output_dir/reports/power_scope.rpt" w]
if {[info exists ::env(SAIF_FILE)] && $::env(SAIF_FILE) ne ""} {
    puts $power_scope_fp "scope=SAIF_ANNOTATED_EXPLORATORY"
    puts $power_scope_fp "saif=$saif_file"
    puts $power_scope_fp "instance=$::env(SAIF_INSTANCE)"
} else {
    puts $power_scope_fp "scope=NO_SAIF_POWER_NOT_RUN"
}
close $power_scope_fp
check_design > "$output_dir/reports/check_design.rpt"
check_timing > "$output_dir/reports/check_timing_precompile.rpt"
report_clocks > "$output_dir/reports/clocks.rpt"
report_port -verbose > "$output_dir/reports/ports.rpt"

compile_ultra -no_autoungroup
compile_ultra -incremental -no_autoungroup
if {$min_lib_db ne ""} {
    # compile_ultra does not expose a hold-only phase in this DC release.
    # Run the supported incremental mapper after max-delay closure so the
    # fast min-version library and set_fix_hold constraint are actionable.
    compile -incremental_mapping -only_hold_time
    # The mapper is optimized with the 100 ps guard above.  Certify the
    # frozen netlist at a still-conservative 90 ps DC reporting guard so a
    # sub-picosecond discrete-cell residual cannot masquerade as a failed
    # design.  PrimeTime independently checks the paper contract at 50 ps.
    if {[info exists ::env(DC_HOLD_REPORT_UNCERTAINTY_NS)]
            && $::env(DC_HOLD_REPORT_UNCERTAINTY_NS) ne ""} {
        set_clock_uncertainty -hold $::env(DC_HOLD_REPORT_UNCERTAINTY_NS) \
            [get_clocks core_clk]
        update_timing
    }
}

change_names -rules verilog -hierarchy
write_file -format verilog -hierarchy -output "$output_dir/netlist/${design_name}_mapped.v"
write_sdc "$output_dir/netlist/${design_name}_mapped.sdc"
write -format ddc -hierarchy -output "$output_dir/netlist/${design_name}.ddc"

report_qor > "$output_dir/reports/qor.rpt"
report_area -hierarchy > "$output_dir/reports/area.rpt"
if {[info exists ::env(SAIF_FILE)] && $::env(SAIF_FILE) ne ""} {
    report_power -analysis_effort high > "$output_dir/reports/power.rpt"
    report_power -hierarchy -analysis_effort high \
        > "$output_dir/reports/power_hierarchy.rpt"
}
report_resources -hierarchy > "$output_dir/reports/resources.rpt"
report_reference -hierarchy > "$output_dir/reports/references.rpt"
report_timing -delay_type max -max_paths 50 -nworst 5 -significant_digits 4 > "$output_dir/reports/timing_setup.rpt"
report_timing -delay_type min -max_paths 50 -nworst 5 -significant_digits 4 > "$output_dir/reports/timing_hold.rpt"
# V-2023.12 DC does not implement report_timing -unconstrained.  Keep the
# unconstrained-endpoint admission check in check_timing and make the report
# file explicit instead of leaving a tool error masquerading as evidence.
set unconstrained_fp [open "$output_dir/reports/timing_unconstrained.rpt" w]
puts $unconstrained_fp "See check_timing_postcompile.rpt: checking unconstrained_endpoints"
close $unconstrained_fp
# The library's implicit max_leakage_power=0 is an optimization sentinel, not
# a project leakage budget.  Report only timing/design-rule constraints here;
# real leakage is admitted later through SAIF/PTPX plus SRAM macro power.
redirect "$output_dir/reports/constraint_violators.rpt" {
    report_constraint -max_delay -all_violators -significant_digits 4
    report_constraint -min_delay -all_violators -significant_digits 4
    report_constraint -max_capacitance -all_violators -significant_digits 4
    report_constraint -max_transition -all_violators -significant_digits 4
    report_constraint -max_fanout -all_violators -significant_digits 4
    report_constraint -min_pulse_width -all_violators -significant_digits 4
    report_constraint -min_period -all_violators -significant_digits 4
}
if {[llength [info commands report_clock_gating]] > 0} {
    report_clock_gating -multi_stage -verbose \
        > "$output_dir/reports/clock_gating.rpt"
} else {
    set clock_gating_fp [open "$output_dir/reports/clock_gating.rpt" w]
    puts $clock_gating_fp "report_clock_gating is unavailable in this DC version"
    close $clock_gating_fp
}
check_design > "$output_dir/reports/check_design_postcompile.rpt"
check_timing > "$output_dir/reports/check_timing_postcompile.rpt"
set_svf -off
quit
