# M2205 LM-shell-only Milkyway FRAM-to-NDM conversion preflight.
# The conversion gate is deliberately the first executable phase. No output,
# option mutation, conversion, or report creation may occur before release.

if {[catch {
    if {![info exists ::env(M2205_CONVERSION_GATE)] || $::env(M2205_CONVERSION_GATE) eq ""} {
        error "M2205 missing environment variable M2205_CONVERSION_GATE"
    }
    set m2205_gate [file normalize $::env(M2205_CONVERSION_GATE)]
    puts "M2205_GATE0_TCL_WAITING actual_pid=[pid] gate=$m2205_gate"
    flush stdout
    set m2205_deadline [expr {[clock milliseconds] + 120000}]
    while {![file exists $m2205_gate]} {
        if {[clock milliseconds] >= $m2205_deadline} {
            error "M2205 conversion gate timeout"
        }
        after 10
    }
    set m2205_gate_channel [open $m2205_gate r]
    set m2205_gate_token [read $m2205_gate_channel]
    close $m2205_gate_channel
    if {$m2205_gate_token ne "M2205_MONITOR_RELEASE_ACTUAL_STABLE\n"} {
        error "M2205 conversion gate token mismatch"
    }
} m2205_gate_message m2205_gate_options]} {
    puts stderr "M2205_FATAL_FAIL_CLOSED: $m2205_gate_message"
    if {[dict exists $m2205_gate_options -errorinfo]} {
        puts stderr [dict get $m2205_gate_options -errorinfo]
    }
    exit 42
}
puts "M2205_GATE0_TCL_RELEASED actual_pid=[pid] gate=$m2205_gate"
flush stdout

proc m2205_env {name} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        error "M2205 missing environment variable $name"
    }
    return $::env($name)
}

proc m2205_tree_stats {root label} {
    if {![file exists $root]} { error "M2205 missing $label: $root" }
    set stack [list $root]
    set regular_files 0
    set regular_bytes 0
    while {[llength $stack] > 0} {
        set node [lindex $stack end]
        set stack [lreplace $stack end end]
        set kind [file type $node]
        if {$kind eq "link"} { error "M2205 symbolic link in $label: $node" }
        if {$kind eq "file"} {
            incr regular_files
            incr regular_bytes [file size $node]
        } elseif {$kind eq "directory"} {
            foreach child [glob -nocomplain -directory $node * .*] {
                if {[file tail $child] ni {. ..}} { lappend stack $child }
            }
        } else {
            error "M2205 unsupported node type $kind in $label: $node"
        }
    }
    if {$regular_files <= 0 || $regular_bytes <= 0} {
        error "M2205 empty $label files=$regular_files bytes=$regular_bytes"
    }
    return [list $regular_files $regular_bytes]
}

proc m2205_main {} {
    set work [file normalize [m2205_env M2205_ISOLATED_CWD]]
    set cache [file normalize [m2205_env M2205_LIBRARY_CACHE]]
    set frame_dir [file normalize [m2205_env M2205_FRAME_DIR]]
    set frame_logs [file normalize [m2205_env M2205_FRAME_LOG_DIR]]
    set reports [file normalize [m2205_env M2205_REPORT_DIR]]
    set mw_ref [file normalize [m2205_env M2205_MW_REF]]
    set milkyway_exec [file normalize [m2205_env M2205_MILKYWAY_EXEC]]
    if {[file normalize [pwd]] ne $work} {
        error "M2205 cwd isolation failed actual=[file normalize [pwd]] expected=$work"
    }
    foreach path [list $cache $frame_dir $frame_logs $reports] {
        if {![string match "${work}/*" $path] || ![file isdirectory $path] || [file type $path] eq "link"} {
            error "M2205 invalid isolated directory: $path"
        }
    }
    if {![file exists $milkyway_exec] || ![file executable $milkyway_exec] || [file type $milkyway_exec] ne "file"} {
        error "M2205 Milkyway executable invalid: $milkyway_exec"
    }
    set frame_name m2205_tcbn28hpcplusbwp35p140_frame.ndm
    set frame_ndm [file join $frame_dir $frame_name]
    if {[file exists $frame_ndm]} { error "M2205 overwrite prohibited: $frame_ndm" }

    set_app_var sh_continue_on_error false
    set_app_options -name lib.configuration.local_output_dir -value $cache
    set queried_cache [file normalize [get_app_option_value -name lib.configuration.local_output_dir]]
    if {$queried_cache ne $cache} {
        error "M2205 local_output_dir mismatch actual=$queried_cache expected=$cache"
    }
    puts "M2205_GATE1_LOCAL_OUTPUT_ROUND_TRIP_PASS cache=$queried_cache"

    set_app_options -name lib.setting.milkyway_exec -value $milkyway_exec
    set queried_milkyway [file normalize [get_app_option_value -name lib.setting.milkyway_exec]]
    if {$queried_milkyway ne $milkyway_exec} {
        error "M2205 milkyway_exec mismatch actual=$queried_milkyway expected=$milkyway_exec"
    }
    puts "M2205_GATE2_MILKYWAY_EXEC_ROUND_TRIP_PASS exec=$queried_milkyway"

    set conversion_status [generate_frame_from_mw $frame_name -mw_lib $mw_ref \
        -log_file_dir $frame_logs -output_directory $frame_dir]
    if {$conversion_status != 1} {
        error "M2205 generate_frame_from_mw returned $conversion_status"
    }
    if {![file exists $frame_ndm] || [file type $frame_ndm] ne "file"} {
        error "M2205 frame NDM missing/not regular: $frame_ndm"
    }
    puts "M2205_GATE3_FRAME_CONVERSION_PASS status=1 frame=$frame_ndm"

    set stats [m2205_tree_stats $frame_ndm "frame NDM"]
    set frame_files [lindex $stats 0]
    set frame_bytes [lindex $stats 1]
    puts "M2205_GATE4_NONEMPTY_FRAME_PASS files=$frame_files bytes=$frame_bytes"
    set facts [open [file join $reports machine_facts.txt] w]
    puts $facts "status=RAW_PASS_M2207_M2205_LM_LIBRARY_CONVERSION_PENDING_M2208"
    puts $facts "shell=lm_shell"
    puts $facts "sampled_process_claim=true"
    puts $facts "exhaustive_short_lived_process_claim=false"
    puts $facts "conversion_gate=$m2205_gate"
    puts $facts "local_output_dir=$queried_cache"
    puts $facts "milkyway_exec=$queried_milkyway"
    puts $facts "conversion_status=1"
    puts $facts "frame_ndm=$frame_ndm"
    puts $facts "frame_regular_files=$frame_files"
    puts $facts "frame_regular_bytes=$frame_bytes"
    puts $facts "design_library_created=false"
    puts $facts "rtl_imported=false"
    puts $facts "pnr_invoked=false"
    close $facts
    puts "RAW_PASS_M2207_M2205_LM_LIBRARY_CONVERSION_PENDING_M2208_INDEPENDENT_RESULT_HAMMER"
}

if {[catch {m2205_main} message options]} {
    puts stderr "M2205_FATAL_FAIL_CLOSED: $message"
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 42
}
exit 0
