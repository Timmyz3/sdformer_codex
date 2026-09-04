# M2221: one no-conversion LM command/option discovery under -no_init.
# This script must never call generate_frame_from_mw, create/open a library, or P&R.

proc m2221_env {name} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        error "M2221 missing environment variable $name"
    }
    return $::env($name)
}

proc m2221_hex {value} {
    return [binary encode hex [encoding convertto utf-8 $value]]
}

proc m2221_command_available {name} {
    return [expr {[lsearch -exact [info commands $name] $name] >= 0}]
}

proc m2221_query_option {name getter_available} {
    if {!$getter_available} {
        return [list 0 -1 -1 "" [m2221_hex "get_app_option_value unavailable"]]
    }
    set rc [catch [list get_app_option_value -name $name] value]
    if {$rc == 0} {
        return [list 1 0 1 [m2221_hex $value] ""]
    }
    return [list 1 $rc 0 "" [m2221_hex $value]]
}

proc m2221_tree_counts {root} {
    set stack [list $root]
    set ndm 0
    set nlib 0
    while {[llength $stack] > 0} {
        set node [lindex $stack end]
        set stack [lreplace $stack end end]
        set kind [file type $node]
        if {$kind eq "link"} { error "M2221 symbolic link in isolated tree: $node" }
        if {$kind eq "directory"} {
            foreach child [glob -nocomplain -directory $node * .*] {
                if {[file tail $child] ni {. ..}} { lappend stack $child }
            }
        } elseif {$kind eq "file"} {
            if {[string equal -nocase [file extension $node] ".ndm"]} { incr ndm }
            if {[string equal -nocase [file extension $node] ".nlib"]} { incr nlib }
        } else {
            error "M2221 unsupported isolated node type $kind: $node"
        }
    }
    return [list $ndm $nlib]
}

proc m2221_child_count {root} {
    set count 0
    foreach child [glob -nocomplain -directory $root * .*] {
        if {[file tail $child] ni {. ..}} { incr count }
    }
    return $count
}

proc m2221_main {} {
    set work [file normalize [m2221_env M2221_ISOLATED_CWD]]
    set home [file normalize [m2221_env HOME]]
    set tmp [file normalize [m2221_env TMPDIR]]
    set xdg [file normalize [m2221_env XDG_CACHE_HOME]]
    set cache [file normalize [m2221_env M2221_LIBRARY_CACHE]]
    set frame_dir [file normalize [m2221_env M2221_FRAME_DIR]]
    set milkyway_exec [file normalize [m2221_env M2221_MILKYWAY_EXEC]]
    if {[file normalize [pwd]] ne $work} { error "M2221 cwd isolation mismatch" }
    foreach path [list $home $tmp $xdg $cache $frame_dir] {
        if {![string match "${work}/*" $path] || ![file isdirectory $path] ||
                [file type $path] eq "link"} {
            error "M2221 invalid isolated directory: $path"
        }
    }
    set setup_files 0
    foreach base [list $work $home] {
        foreach pattern [list .synopsys* *.setup .tclshrc] {
            foreach candidate [glob -nocomplain -directory $base $pattern] {
                if {[file isfile $candidate]} { incr setup_files }
            }
        }
    }
    if {$setup_files != 0} { error "M2221 setup file contamination count=$setup_files" }
    if {[m2221_child_count $frame_dir] != 0} {
        error "M2221 frame directory not empty at entry"
    }

    puts "M2221_STARTUP mode=no_init setup_files=0 cwd_hex=[m2221_hex $work] home_hex=[m2221_hex $home]"
    array set available {}
    foreach name [list generate_frame_from_mw set_app_options get_app_option_value report_app_options] {
        set available($name) [m2221_command_available $name]
        puts "M2221_COMMAND name=$name available=$available($name)"
    }

    set local_name lib.configuration.local_output_dir
    set local_query [m2221_query_option $local_name $available(get_app_option_value)]
    puts "M2221_OPTION name=$local_name query_attempted=[lindex $local_query 0] query_rc=[lindex $local_query 1] registered=[lindex $local_query 2] value_hex=[lindex $local_query 3] diagnostic_hex=[lindex $local_query 4]"

    set mw_name lib.setting.milkyway_exec
    set mw_query [m2221_query_option $mw_name $available(get_app_option_value)]
    puts "M2221_OPTION name=$mw_name query_attempted=[lindex $mw_query 0] query_rc=[lindex $mw_query 1] registered=[lindex $mw_query 2] value_hex=[lindex $mw_query 3] diagnostic_hex=[lindex $mw_query 4]"

    set attempted 0
    set set_rc -1
    set readback_attempted 0
    set readback_rc -1
    set exact -1
    set set_diagnostic_hex ""
    set readback_value_hex ""
    set readback_diagnostic_hex ""
    if {$available(set_app_options) && $available(get_app_option_value) &&
            [lindex $mw_query 2] == 1} {
        set attempted 1
        set set_rc [catch [list set_app_options -name $mw_name -value $milkyway_exec] set_message]
        if {$set_rc != 0} {
            set set_diagnostic_hex [m2221_hex $set_message]
        } else {
            set readback_attempted 1
            set readback_rc [catch [list get_app_option_value -name $mw_name] readback]
            if {$readback_rc == 0} {
                set readback_value_hex [m2221_hex $readback]
                set exact [expr {$readback eq $milkyway_exec}]
            } else {
                set readback_diagnostic_hex [m2221_hex $readback]
                set exact 0
            }
        }
    }
    puts "M2221_MILKYWAY_SET attempted=$attempted set_rc=$set_rc readback_attempted=$readback_attempted readback_rc=$readback_rc exact=$exact value_hex=$readback_value_hex set_diagnostic_hex=$set_diagnostic_hex readback_diagnostic_hex=$readback_diagnostic_hex"

    set counts [m2221_tree_counts $work]
    set frame_files [m2221_child_count $frame_dir]
    if {$frame_files != 0 || [lindex $counts 0] != 0 || [lindex $counts 1] != 0} {
        error "M2221 forbidden output frame=$frame_files ndm=[lindex $counts 0] nlib=[lindex $counts 1]"
    }
    puts "M2221_NO_SIDE_EFFECTS frame_files=0 ndm_files=0 nlib_files=0 generate_calls=0 create_lib_calls=0 pnr_calls=0"
    puts "RAW_PASS_M2223_M2221_LM_COMMAND_OPTION_DISCOVERY_PENDING_M2224_RESULT_HAMMER"
}

if {[catch {m2221_main} message options]} {
    puts stderr "M2221_FATAL_FAIL_CLOSED: $message"
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 42
}
exit 0
