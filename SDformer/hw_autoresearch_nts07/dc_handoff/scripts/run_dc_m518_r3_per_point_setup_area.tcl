set design_name $::env(DESIGN_NAME)
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

# Both independently launched points analyze this identical two-file corpus.
# Only DESIGN_NAME differs.  The runner separately freezes the 50 ordered
# source declaration tuples; DC expands those buses to 1175 bit-level ports.
set precompile_build_report "$output_dir/reports/precompile_build.rpt"
redirect $precompile_build_report {
    analyze -format sverilog -define SYNTHESIS $rtl_files
    elaborate $design_name
    set elaborated_design [current_design]
    if {$elaborated_design eq ""} {
        set failure_fp [open "$output_dir/TCL_EXPLICIT_FAILURE.txt" w]
        puts $failure_fp "status=FAIL_ELABORATION_NO_CURRENT_DESIGN"
        close $failure_fp
        exit 35
    }
    current_design $elaborated_design
    link
    uniquify
    set_min_library $lib_db -min_version $min_lib_db
    set_operating_conditions $::env(OPERATING_CONDITION)
    source $sdc_file
    set_wire_load_model -name ZeroWireload [current_design]
}

set precompile_design_ok 0
redirect "$output_dir/reports/check_design_precompile.rpt" {
    set precompile_design_ok [check_design]
}
set precompile_timing_ok 0
redirect "$output_dir/reports/check_timing_precompile.rpt" {
    set precompile_timing_ok [check_timing]
}

set precompile_build_fp [open $precompile_build_report r]
set precompile_build_text [read $precompile_build_fp]
close $precompile_build_fp
set precompile_design_fp \
    [open "$output_dir/reports/check_design_precompile.rpt" r]
set precompile_design_text [read $precompile_design_fp]
close $precompile_design_fp
set precompile_timing_fp \
    [open "$output_dir/reports/check_timing_precompile.rpt" r]
set precompile_timing_text [read $precompile_timing_fp]
close $precompile_timing_fp
set precompile_all_text \
    "$precompile_build_text\n$precompile_design_text\n$precompile_timing_text"
set precompile_tim209_count [regexp -all -- {TIM-209} $precompile_all_text]
set precompile_opt150_count [regexp -all -- {OPT-150} $precompile_all_text]

set precompile_gate_fp [open "$output_dir/reports/precompile_gate.rpt" w]
puts $precompile_gate_fp "check_design_ok=$precompile_design_ok"
puts $precompile_gate_fp "check_timing_ok=$precompile_timing_ok"
puts $precompile_gate_fp "TIM-209=$precompile_tim209_count"
puts $precompile_gate_fp "OPT-150=$precompile_opt150_count"
puts $precompile_gate_fp \
    "sources=precompile_build.rpt,check_design_precompile.rpt,check_timing_precompile.rpt"

if {$precompile_design_ok != 1 || $precompile_timing_ok != 1 ||
        $precompile_tim209_count != 0 || $precompile_opt150_count != 0} {
    puts $precompile_gate_fp \
        "status=FAIL_PRECOMPILE_STRUCTURAL_OR_TIMING_GATE__EXPLICIT_EXIT36__NO_COMPILE"
    close $precompile_gate_fp
    set failure_fp [open "$output_dir/TCL_EXPLICIT_FAILURE.txt" w]
    puts $failure_fp \
        "status=FAIL_PRECOMPILE_STRUCTURAL_OR_TIMING_GATE__EXPLICIT_EXIT36"
    puts $failure_fp "check_design_ok=$precompile_design_ok"
    puts $failure_fp "check_timing_ok=$precompile_timing_ok"
    puts $failure_fp "TIM-209=$precompile_tim209_count"
    puts $failure_fp "OPT-150=$precompile_opt150_count"
    close $failure_fp
    exit 36
} else {
    puts $precompile_gate_fp "status=PASS_PRECOMPILE_GATE"
    close $precompile_gate_fp

    report_resources -hierarchy \
        > "$output_dir/reports/resources_precompile.rpt"
    report_reference -hierarchy \
        > "$output_dir/reports/references_precompile.rpt"

    set flow_fp [open "$output_dir/reports/flow_contract.rpt" w]
    puts $flow_fp "flow=m518_r3_per_point_setup_area_only"
    puts $flow_fp "compile_ultra_count=1"
    puts $flow_fp "incremental_compile_count=0"
    puts $flow_fp "hold_fix_command_count=0"
    puts $flow_fp "hold_only_optimization_count=0"
    puts $flow_fp "hold_not_closed_at_dc=true"
    puts $flow_fp "hold_report_generated=false"
    close $flow_fp

    ungroup -all -flatten
    set compile_start_seconds [clock seconds]
    compile_ultra
    set compile_end_seconds [clock seconds]
    update_timing
    set compile_fp [open "$output_dir/reports/compile_receipt.rpt" w]
    puts $compile_fp "compile_ultra_count=1"
    puts $compile_fp "compile_start_epoch=$compile_start_seconds"
    puts $compile_fp "compile_end_epoch=$compile_end_seconds"
    puts $compile_fp \
        "compile_wall_seconds=[expr {$compile_end_seconds - $compile_start_seconds}]"
    puts $compile_fp "incremental_compile_count=0"
    puts $compile_fp "hold_optimization_count=0"
    close $compile_fp

    report_hierarchy > "$output_dir/reports/hierarchy_postcompile.rpt"
    report_resources -hierarchy \
        > "$output_dir/reports/resources_postcompile.rpt"
    report_reference -hierarchy \
        > "$output_dir/reports/references_postcompile.rpt"
    report_qor > "$output_dir/reports/qor.rpt"
    report_area -hierarchy > "$output_dir/reports/area.rpt"
    report_clocks > "$output_dir/reports/clocks.rpt"
    report_port -verbose > "$output_dir/reports/ports.rpt"
    set dc_bit_port_count [sizeof_collection [get_ports *]]
    set port_count_fp [open "$output_dir/reports/dc_bit_port_count.txt" w]
    puts $port_count_fp $dc_bit_port_count
    close $port_count_fp
    report_timing -delay_type max -max_paths 100 -nworst 10 \
        -significant_digits 4 > "$output_dir/reports/timing_setup.rpt"
    redirect "$output_dir/reports/constraint_setup.rpt" {
        report_constraint -max_delay -all_violators -significant_digits 4
    }
    redirect "$output_dir/reports/constraint_max_capacitance.rpt" {
        report_constraint -max_capacitance -all_violators \
            -significant_digits 4
    }
    redirect "$output_dir/reports/constraint_max_transition.rpt" {
        report_constraint -max_transition -all_violators \
            -significant_digits 4
    }
    redirect "$output_dir/reports/constraint_max_fanout.rpt" {
        report_constraint -max_fanout -all_violators -significant_digits 4
    }

    set postcompile_design_ok 0
    redirect "$output_dir/reports/check_design_postcompile.rpt" {
        set postcompile_design_ok [check_design]
    }
    set postcompile_timing_ok 0
    redirect "$output_dir/reports/check_timing_postcompile.rpt" {
        set postcompile_timing_ok [check_timing]
    }
    set structural_fp \
        [open "$output_dir/reports/structured_postcompile_gate.rpt" w]
    puts $structural_fp "check_design_ok=$postcompile_design_ok"
    puts $structural_fp "check_timing_ok=$postcompile_timing_ok"
    puts $structural_fp "dc_bit_level_port_count=$dc_bit_port_count"
    puts $structural_fp "expected_dc_bit_level_port_count=1175"
    puts $structural_fp \
        "source_declaration_tuple_count_authority=runner_prelaunch_parser"
    puts $structural_fp "expected_source_declaration_tuple_count=50"
    puts $structural_fp \
        "unresolved_reference_authority=check_design_ok"
    puts $structural_fp \
        "black_box_macro_authority=area_report_exact_zero_count"
    close $structural_fp

    if {$postcompile_design_ok != 1 || $postcompile_timing_ok != 1 ||
            $dc_bit_port_count != 1175} {
        set failure_fp [open "$output_dir/TCL_EXPLICIT_FAILURE.txt" w]
        puts $failure_fp \
            "status=FAIL_POSTCOMPILE_STRUCTURED_GATE__EXPLICIT_EXIT37"
        puts $failure_fp "check_design_ok=$postcompile_design_ok"
        puts $failure_fp "check_timing_ok=$postcompile_timing_ok"
        puts $failure_fp "dc_bit_level_port_count=$dc_bit_port_count"
        close $failure_fp
        exit 37
    }

    change_names -rules verilog -hierarchy
    write_file -format verilog -hierarchy \
        -output "$output_dir/netlist/${design_name}_mapped.v"
    write_sdc "$output_dir/netlist/${design_name}_mapped.sdc"
    write -format ddc -hierarchy \
        -output "$output_dir/netlist/${design_name}.ddc"
    set_svf -off

    set terminal_fp [open "$output_dir/TCL_PASS_TERMINAL.txt" w]
    puts $terminal_fp "status=PASS_M518_R3_PER_POINT_SETUP_AREA_DC_TCL_TERMINAL"
    puts $terminal_fp "design=$design_name"
    puts $terminal_fp "check_design_ok=$postcompile_design_ok"
    puts $terminal_fp "check_timing_ok=$postcompile_timing_ok"
    puts $terminal_fp "dc_bit_level_port_count=$dc_bit_port_count"
    puts $terminal_fp "compile_ultra_count=1"
    puts $terminal_fp "incremental_compile_count=0"
    puts $terminal_fp "hold_optimization_count=0"
    puts $terminal_fp "hold_not_closed_at_dc=true"
    close $terminal_fp
    quit
}
