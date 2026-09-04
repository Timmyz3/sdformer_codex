# M2189: LM-shell-only Milkyway FRAM to NDM conversion preflight.
# No RTL/design import, synthesis, floorplan, placement, CTS, route,
# extraction, timing, area, power, create_lib, or overwrite operation.

proc m2189_env {name} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        error "M2189 missing environment variable $name"
    }
    return $::env($name)
}

proc m2189_tree_stats {root label} {
    if {![file exists $root]} { error "M2189 missing $label: $root" }
    set stack [list $root]
    set regular_files 0
    set regular_bytes 0
    while {[llength $stack] > 0} {
        set node [lindex $stack end]
        set stack [lreplace $stack end end]
        set kind [file type $node]
        if {$kind eq "link"} { error "M2189 symbolic link in $label: $node" }
        if {$kind eq "file"} {
            incr regular_files
            incr regular_bytes [file size $node]
        } elseif {$kind eq "directory"} {
            foreach child [glob -nocomplain -directory $node * .*] {
                if {[file tail $child] ni {. ..}} { lappend stack $child }
            }
        } else {
            error "M2189 unsupported node type $kind in $label: $node"
        }
    }
    if {$regular_files <= 0 || $regular_bytes <= 0} {
        error "M2189 empty $label files=$regular_files bytes=$regular_bytes"
    }
    return [list $regular_files $regular_bytes]
}

proc m2189_main {} {
    set work [file normalize [m2189_env M2189_ISOLATED_CWD]]
    set cache [file normalize [m2189_env M2189_LIBRARY_CACHE]]
    set frame_dir [file normalize [m2189_env M2189_FRAME_DIR]]
    set frame_logs [file normalize [m2189_env M2189_FRAME_LOG_DIR]]
    set reports [file normalize [m2189_env M2189_REPORT_DIR]]
    set mw_ref [file normalize [m2189_env M2189_MW_REF]]
    set milkyway_exec [file normalize [m2189_env M2189_MILKYWAY_EXEC]]
    if {[file normalize [pwd]] ne $work} {
        error "M2189 cwd isolation failed actual=[file normalize [pwd]] expected=$work"
    }
    foreach path [list $cache $frame_dir $frame_logs $reports] {
        if {![string match "${work}/*" $path] || ![file isdirectory $path] || [file type $path] eq "link"} {
            error "M2189 invalid isolated directory: $path"
        }
    }
    if {![file exists $milkyway_exec] || ![file executable $milkyway_exec] || [file type $milkyway_exec] ne "file"} {
        error "M2189 Milkyway executable invalid: $milkyway_exec"
    }
    set frame_name m2189_tcbn28hpcplusbwp35p140_frame.ndm
    set frame_ndm [file join $frame_dir $frame_name]
    if {[file exists $frame_ndm]} { error "M2189 overwrite prohibited: $frame_ndm" }

    set_app_var sh_continue_on_error false
    set_app_options -name lib.configuration.local_output_dir -value $cache
    set queried_cache [file normalize [get_app_option_value -name lib.configuration.local_output_dir]]
    if {$queried_cache ne $cache} {
        error "M2189 local_output_dir mismatch actual=$queried_cache expected=$cache"
    }
    puts "M2189_GATE1_LOCAL_OUTPUT_ROUND_TRIP_PASS cache=$queried_cache"

    # The installed LM manual requires this option before Milkyway FRAM commands.
    set_app_options -name lib.setting.milkyway_exec -value $milkyway_exec
    set queried_milkyway [file normalize [get_app_option_value -name lib.setting.milkyway_exec]]
    if {$queried_milkyway ne $milkyway_exec} {
        error "M2189 milkyway_exec mismatch actual=$queried_milkyway expected=$milkyway_exec"
    }
    puts "M2189_GATE2_MILKYWAY_EXEC_ROUND_TRIP_PASS exec=$queried_milkyway"

    set conversion_status [generate_frame_from_mw $frame_name -mw_lib $mw_ref \
        -log_file_dir $frame_logs -output_directory $frame_dir]
    if {$conversion_status != 1} {
        error "M2189 generate_frame_from_mw returned $conversion_status"
    }
    if {![file exists $frame_ndm] || [file type $frame_ndm] ne "file"} {
        error "M2189 frame NDM missing/not regular: $frame_ndm"
    }
    puts "M2189_GATE3_FRAME_CONVERSION_PASS status=1 frame=$frame_ndm"

    set stats [m2189_tree_stats $frame_ndm "frame NDM"]
    set frame_files [lindex $stats 0]
    set frame_bytes [lindex $stats 1]
    puts "M2189_GATE4_NONEMPTY_FRAME_PASS files=$frame_files bytes=$frame_bytes"
    set facts [open [file join $reports machine_facts.txt] w]
    puts $facts "status=RAW_PASS_M2191_M2189_LM_LIBRARY_CONVERSION_PENDING_M2192"
    puts $facts "shell=lm_shell"
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
    puts "RAW_PASS_M2191_M2189_LM_LIBRARY_CONVERSION_PENDING_M2192_INDEPENDENT_RESULT_HAMMER"
}

if {[catch {m2189_main} message options]} {
    puts stderr "M2189_FATAL_FAIL_CLOSED: $message"
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 42
}
exit 0
