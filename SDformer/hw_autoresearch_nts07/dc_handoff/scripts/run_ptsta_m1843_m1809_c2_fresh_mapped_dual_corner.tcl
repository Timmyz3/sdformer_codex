# M1843 sealed-source Tcl.  It becomes executable only after the exact M1844
# source review and double-sealed M1846 one-attempt release are caller-pinned.
# Negative setup/hold slack is reported verbatim; this Tcl adds no exception,
# ECO or hold repair and does not convert a timing violation into tool failure.
foreach required_env {
    M1843_AXIS M1843_STD_SLOW_DB M1843_STD_FAST_DB
    M1843_IMPLEMENTATION_NETLIST M1843_IMPLEMENTATION_SDC
    M1843_IMPLEMENTATION_TOP M1843_PT_OUTPUT_DIR
} {
    if {![info exists ::env($required_env)] || $::env($required_env) eq ""} {
        error "M1843 missing required environment $required_env"
    }
}

set axis $::env(M1843_AXIS)
set std_slow_db [file normalize $::env(M1843_STD_SLOW_DB)]
set std_fast_db [file normalize $::env(M1843_STD_FAST_DB)]
set mapped_netlist [file normalize $::env(M1843_IMPLEMENTATION_NETLIST)]
set mapped_sdc [file normalize $::env(M1843_IMPLEMENTATION_SDC)]
set implementation_top $::env(M1843_IMPLEMENTATION_TOP)
set output_dir [file normalize $::env(M1843_PT_OUTPUT_DIR)]

set base_top m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24
if {$axis eq "K8"} {
    set expected_implementation_top "${base_top}_ARCH_MODE0"
} elseif {$axis eq "K1X8"} {
    set expected_implementation_top "${base_top}_ARCH_MODE1"
} else {
    error "M1843 unsupported axis $axis"
}
if {$implementation_top ne $expected_implementation_top
        || $implementation_top eq $base_top} {
    error "M1843 PrimeTime implementation top is base/swapped/noncanonical"
}

set slow_lib_name tcbn28hpcplusbwp35p140ssg0p9v125c
set fast_lib_name tcbn28hpcplusbwp35p140ffg1p05vm40c
set slow_opcond ssg0p9v125c
set fast_opcond ffg1p05vm40c

file mkdir "$output_dir/reports"
set_app_var search_path [list [file dirname $std_slow_db] [file dirname $std_fast_db]]
set_app_var link_path [list "*" $std_slow_db]

read_verilog $mapped_netlist
current_design $implementation_top
link_design $implementation_top
set_min_library $std_slow_db -min_version $std_fast_db
read_sdc $mapped_sdc
set_operating_conditions -analysis_type on_chip_variation \
    -max $slow_opcond -max_library $slow_lib_name \
    -min $fast_opcond -min_library $fast_lib_name

update_timing -full
check_timing -verbose > "$output_dir/reports/check_timing.rpt"
report_analysis_coverage -status_details untested \
    > "$output_dir/reports/analysis_coverage.rpt"
report_global_timing > "$output_dir/reports/global_timing.rpt"
report_timing -delay_type max -slack_lesser_than 1000000 \
    -max_paths 100 -nworst 10 -path_type full_clock_expanded \
    -significant_digits 9 > "$output_dir/reports/timing_setup_slow.rpt"
report_timing -delay_type min -slack_lesser_than 1000000 \
    -max_paths 100 -nworst 10 -path_type full_clock_expanded \
    -significant_digits 9 > "$output_dir/reports/timing_hold_fast.rpt"
report_constraint -all_violators -verbose -significant_digits 9 \
    > "$output_dir/reports/constraint_violators.rpt"
report_clock > "$output_dir/reports/clock.rpt"
report_exceptions -ignored > "$output_dir/reports/exceptions.rpt"
report_design > "$output_dir/reports/design.rpt"
report_wire_load > "$output_dir/reports/wire_load.rpt"
list_libs > "$output_dir/reports/libraries.rpt"

# Machine-readable semantic counts supplement (never replace) the complete
# human reports.  Counts are published even when nonzero; no exception/ECO is
# added to turn a violation into a pass.
set setup_violators [get_timing_paths -delay_type max -slack_lesser_than 0.0 \
    -max_paths 1000000]
set hold_violators [get_timing_paths -delay_type min -slack_lesser_than 0.0 \
    -max_paths 1000000]
set constraint_fp [open "$output_dir/reports/constraint_semantics_machine.txt" w]
puts $constraint_fp "axis=$axis"
puts $constraint_fp "setup_violating_paths=[sizeof_collection $setup_violators]"
puts $constraint_fp "hold_violating_paths=[sizeof_collection $hold_violators]"
puts $constraint_fp "negative_counts_hidden=false"
close $constraint_fp

set setup_paths [get_timing_paths -delay_type max -nworst 1 -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -nworst 1 -max_paths 1]
if {[sizeof_collection $setup_paths] != 1
        || [sizeof_collection $hold_paths] != 1} {
    error "M1843 missing setup or hold timing path"
}
set setup_slack [get_attribute $setup_paths slack]
set hold_slack [get_attribute $hold_paths slack]

set scope_fp [open "$output_dir/reports/runtime_scope.rpt" w]
puts $scope_fp "milestone=M1843"
puts $scope_fp "axis=$axis"
puts $scope_fp "design=$implementation_top"
puts $scope_fp "setup_corner=slow-max_ssg0p9v125c"
puts $scope_fp "hold_corner=fast-min_ffg1p05vm40c"
puts $scope_fp "analysis_type=on_chip_variation"
puts $scope_fp "mapped_netlist=$mapped_netlist"
puts $scope_fp "mapped_sdc=$mapped_sdc"
puts $scope_fp "parasitics=none_prelayout"
puts $scope_fp "paper_claim=false"
puts $scope_fp "timing_exceptions_added=false"
puts $scope_fp "pt_eco=false"
puts $scope_fp "negative_slack_reported_not_hidden=true"
close $scope_fp

set summary_fp [open "$output_dir/reports/timing_summary_machine.txt" w]
puts $summary_fp "axis=$axis"
puts $summary_fp "setup_wns_ns=$setup_slack"
puts $summary_fp "hold_wns_ns=$hold_slack"
puts $summary_fp "setup_corner=ssg0p9v125c"
puts $summary_fp "hold_corner=ffg1p05vm40c"
puts $summary_fp "setup_closed=[expr {$setup_slack >= 0.0}]"
puts $summary_fp "hold_closed=[expr {$hold_slack >= 0.0}]"
puts $summary_fp "negative_slack_reported_not_hidden=true"
close $summary_fp

set marker [open "$output_dir/PTSTA_INTERNAL_COMPLETE.txt" w]
puts $marker "M1843_C2_FRESH_MAPPED_DUAL_CORNER_PT_INTERNAL_COMPLETE=PASS"
puts $marker "axis=$axis"
puts $marker "meaning=REPORTS_COMPLETE_NOT_RESULT_ADMISSION"
puts $marker "paper_claim=false"
close $marker
quit
