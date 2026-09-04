# M2117 source-only matched synthesis and transformation-map exporter.
# Independent M2118 admission is mandatory before execution.  Ordinary and
# TSBG use this byte-identical script; SCHEDULE_MODE is the only axis variable.
set design_name m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend
set mode $::env(M2117_SCHEDULE_MODE)
set hw_root [file normalize $::env(M2117_HW_ROOT)]
set rtl_filelist [file normalize $::env(M2117_RTL_FILELIST)]
set lib_db [file normalize $::env(M2117_LIB_DB)]
set min_lib_db [file normalize $::env(M2117_MIN_LIB_DB)]
set sdc_file [file normalize $::env(M2117_SDC_FILE)]
set output_dir [file normalize $::env(M2117_OUTPUT_DIR)]

if {$mode ne "0" && $mode ne "1"} { error "M2117_FAIL_SCHEDULE_MODE" }
foreach path [list $rtl_filelist $lib_db $min_lib_db $sdc_file] {
    if {![file isfile $path]} { error "M2117_FAIL_MISSING_INPUT_$path" }
}
file mkdir $output_dir
file mkdir "$output_dir/reports"
file mkdir "$output_dir/netlist"
set_svf "$output_dir/netlist/m2018_axis.svf"
set_app_var search_path [list $hw_root [file dirname $lib_db] \
    [file dirname $min_lib_db]]
set_app_var target_library [list $lib_db]
set_app_var link_library [list "*" $lib_db $min_lib_db]
set_app_var verilogout_no_tri true
set_app_var hdlin_auto_save_templates true

set fp [open $rtl_filelist r]
set rtl_files {}
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line ne "" && ![string match "#*" $line]} {
        set source_path [file normalize "$hw_root/$line"]
        if {![file isfile $source_path]} {
            error "M2117_FAIL_MISSING_RTL_$source_path"
        }
        lappend rtl_files $source_path
    }
}
close $fp
if {[llength $rtl_files] != 2} { error "M2117_FAIL_RTL_FILE_COUNT" }

# Track every RTL-to-gate transformation from analysis through final rename.
saif_map -start
analyze -format sverilog -define SYNTHESIS $rtl_files
elaborate $design_name -parameters "SCHEDULE_MODE=>$mode"
set elaborated_design [current_design]
if {$elaborated_design eq ""} { error "M2117_FAIL_NO_CURRENT_DESIGN" }
current_design $elaborated_design
link
uniquify
set_min_library $lib_db -min_version $min_lib_db
set_operating_conditions $::env(M2117_OPERATING_CONDITION)
source $sdc_file
set_wire_load_model -name ZeroWireload [current_design]

redirect "$output_dir/reports/check_design_precompile.rpt" {check_design}
redirect "$output_dir/reports/check_timing_precompile.rpt" {check_timing}
set pre_fp [open "$output_dir/reports/check_timing_precompile.rpt" r]
set pre_text [read $pre_fp]
close $pre_fp
set tim209 [regexp -all -- {TIM-209} $pre_text]
set opt150 [regexp -all -- {OPT-150} $pre_text]
if {$tim209 != 0 || $opt150 != 0} {
    error "M2117_FAIL_PRECOMPILE_LOOP_TIM209_${tim209}_OPT150_${opt150}"
}

ungroup -all -flatten
compile_ultra
update_timing
report_qor > "$output_dir/reports/qor.rpt"
report_area -hierarchy > "$output_dir/reports/area.rpt"
report_timing -delay_type max -max_paths 100 -nworst 10 \
    -significant_digits 4 > "$output_dir/reports/timing_setup.rpt"
report_timing -delay_type min -max_paths 100 -nworst 10 \
    -significant_digits 4 > "$output_dir/reports/timing_hold_diagnostic.rpt"
redirect "$output_dir/reports/constraint_setup.rpt" {
    report_constraint -max_delay -all_violators -significant_digits 4
}
redirect "$output_dir/reports/constraint_hold_diagnostic.rpt" {
    report_constraint -min_delay -all_violators -significant_digits 4
}
redirect "$output_dir/reports/check_design_postcompile.rpt" {check_design}
redirect "$output_dir/reports/check_timing_postcompile.rpt" {check_timing}

change_names -rules verilog -hierarchy
saif_map -report > "$output_dir/reports/saif_map_report.rpt"
saif_map -write_map "$output_dir/netlist/m2018_axis.saif_map.bin"
saif_map -write_map "$output_dir/netlist/m2018_axis.ptpx_map.default.tcl" \
    -type ptpx
saif_map -write_map "$output_dir/netlist/m2018_axis.ptpx_map.essential.tcl" \
    -type ptpx -essential
write_file -format verilog -hierarchy \
    -output "$output_dir/netlist/m2018_axis_mapped.v"
write_sdc "$output_dir/netlist/m2018_axis_mapped.sdc"
write -format ddc -hierarchy -output "$output_dir/netlist/m2018_axis.ddc"
set_svf -off

set fp [open "$output_dir/reports/identity.rpt" w]
puts $fp "milestone=M2117"
puts $fp "schedule_mode=$mode"
puts $fp "design=$elaborated_design"
puts $fp "logic_only=true"
puts $fp "macro_count=0"
puts $fp "clock_period_ns=3.0"
puts $fp "clock_network=ideal_no_cts"
puts $fp "wireload=ZeroWireload"
puts $fp "hold_closed=false"
puts $fp "activity_mapping=dc_saif_map_transformation_tracking"
close $fp
set fp [open "$output_dir/TCL_INTERNAL_COMPLETE.txt" w]
puts $fp "PASS_M2117_DC_SAIFMAP_SOURCE_INTERNAL_PENDING_PARSER_AND_INDEPENDENT_REVIEW"
close $fp
quit
