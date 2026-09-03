# M1843 sealed-source Tcl.  It becomes executable only after the exact M1844
# source review and double-sealed M1846 one-attempt release are caller-pinned.
foreach required_env {
    M1843_AXIS M1843_HW_ROOT M1843_REFERENCE_FILELIST
    M1843_REFERENCE_TOP M1843_REF_ELAB_PARAMETERS
    M1843_STD_SLOW_DB M1843_IMPLEMENTATION_NETLIST
    M1843_IMPLEMENTATION_TOP M1843_IMPLEMENTATION_SVF
    M1843_FORMALITY_OUTPUT_DIR
} {
    if {![info exists ::env($required_env)] || $::env($required_env) eq ""} {
        error "M1843 missing required environment $required_env"
    }
}

set axis $::env(M1843_AXIS)
set hw_root [file normalize $::env(M1843_HW_ROOT)]
set reference_filelist [file normalize $::env(M1843_REFERENCE_FILELIST)]
set reference_top $::env(M1843_REFERENCE_TOP)
set reference_elab_parameters $::env(M1843_REF_ELAB_PARAMETERS)
set std_slow_db [file normalize $::env(M1843_STD_SLOW_DB)]
set implementation_netlist [file normalize $::env(M1843_IMPLEMENTATION_NETLIST)]
set implementation_top $::env(M1843_IMPLEMENTATION_TOP)
set implementation_svf [file normalize $::env(M1843_IMPLEMENTATION_SVF)]
set output_dir [file normalize $::env(M1843_FORMALITY_OUTPUT_DIR)]

set base_top m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24
if {$reference_top ne $base_top} {
    error "M1843 reference top must remain frozen M1809 base top"
}
if {$axis eq "K8"} {
    set expected_elab_parameters "ARCH_MODE=0"
    set expected_implementation_top "${base_top}_ARCH_MODE0"
} elseif {$axis eq "K1X8"} {
    set expected_elab_parameters "ARCH_MODE=1"
    set expected_implementation_top "${base_top}_ARCH_MODE1"
} else {
    error "M1843 unsupported axis $axis"
}
if {$reference_elab_parameters ne $expected_elab_parameters} {
    error "M1843 axis/reference ARCH_MODE swap or drift"
}
if {$implementation_top ne $expected_implementation_top
        || $implementation_top eq $base_top} {
    error "M1843 implementation must use the axis-derived canonical top"
}

proc m1843_read_frozen_filelist {filelist hw_root} {
    set fp [open $filelist r]
    set rtl_files {}
    while {[gets $fp line] >= 0} {
        set line [string trim $line]
        if {$line ne "" && ![string match "#*" $line]} {
            lappend rtl_files [file normalize "$hw_root/$line"]
        }
    }
    close $fp
    if {[llength $rtl_files] != 13} {
        error "M1843 frozen M1809 reference filelist must have exactly 13 active rows"
    }
    return $rtl_files
}

file mkdir "$output_dir/reports"
set reference_files [m1843_read_frozen_filelist $reference_filelist $hw_root]
set_svf $implementation_svf
read_db -technology_library $std_slow_db
read_sverilog -r $reference_files
set_top r:/WORK/$reference_top -parameter $reference_elab_parameters
read_verilog -i $implementation_netlist
set_top i:/WORK/$implementation_top

match
report_unmatched_points > "$output_dir/reports/formality_unmatched.rpt"
report_black_boxes > "$output_dir/reports/formality_black_boxes.rpt"
set verification_succeeded [verify]
redirect "$output_dir/reports/formality_status.rpt" {
    echo "verify_return=$verification_succeeded"
    report_status
}
report_failing_points > "$output_dir/reports/formality_failing.rpt"
report_aborted_points > "$output_dir/reports/formality_aborted.rpt"
report_unverified_points > "$output_dir/reports/formality_unverified.rpt"
if {!$verification_succeeded} {
    error "M1843 $axis fresh-mapped RTL-to-gate Formality failed"
}

set marker [open "$output_dir/FORMALITY_INTERNAL_COMPLETE.txt" w]
puts $marker "M1843_C2_FRESH_MAPPED_FORMALITY_INTERNAL_COMPLETE=PASS"
puts $marker "axis=$axis"
puts $marker "reference_top=$reference_top"
puts $marker "reference_elab_parameters=$reference_elab_parameters"
puts $marker "implementation_top=$implementation_top"
puts $marker "meaning=REPORTS_COMPLETE_NOT_RESULT_ADMISSION"
close $marker
quit
