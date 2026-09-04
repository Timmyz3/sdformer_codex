# M2153: additive library-import-only ICC2 preflight after M2146 rejected
# M2141.  This source contains no RTL/design import, synthesis, floorplan,
# placement, CTS, route, extraction, timing, area, or power operation.

proc m2153_env {name} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        error "M2153 missing environment variable $name"
    }
    return $::env($name)
}

proc m2153_nonempty {collection label} {
    set count [sizeof_collection $collection]
    if {$count <= 0} { error "M2153 empty collection: $label" }
    return $count
}

proc m2153_one_lib {pattern label} {
    set libs [get_libs -quiet $pattern]
    if {[sizeof_collection $libs] != 1} {
        error "M2153 expected exactly one $label library for $pattern"
    }
    return [lindex [get_object_name $libs] 0]
}

proc m2153_read_lines {path} {
    set fh [open $path r]
    set text [read $fh]
    close $fh
    set result {}
    foreach line [split $text "\n"] {
        set line [string trim $line]
        if {$line ne ""} { lappend result $line }
    }
    return $result
}

proc m2153_tree_stats {root label} {
    if {![file exists $root]} { error "M2153 missing $label tree: $root" }
    set stack [list $root]
    set regular_files 0
    set regular_bytes 0
    while {[llength $stack] > 0} {
        set node [lindex $stack end]
        set stack [lreplace $stack end end]
        set node_type [file type $node]
        if {$node_type eq "link"} { error "M2153 symbolic link in $label tree: $node" }
        if {$node_type eq "file"} {
            incr regular_files
            incr regular_bytes [file size $node]
            continue
        }
        if {$node_type ne "directory"} {
            error "M2153 unsupported node type $node_type in $label tree: $node"
        }
        foreach child [glob -nocomplain -directory $node * .*] {
            set tail [file tail $child]
            if {$tail eq "." || $tail eq ".."} { continue }
            lappend stack $child
        }
    }
    if {$regular_files <= 0 || $regular_bytes <= 0} {
        error "M2153 empty $label tree files=$regular_files bytes=$regular_bytes"
    }
    return [list $regular_files $regular_bytes]
}

proc m2153_main {} {
    set work [file normalize [m2153_env M2153_ISOLATED_CWD]]
    set home [file normalize [m2153_env HOME]]
    set tmp [file normalize [m2153_env TMPDIR]]
    set xdg_cache [file normalize [m2153_env XDG_CACHE_HOME]]
    set cache [file normalize [m2153_env M2153_LIBRARY_CACHE]]
    set frame_dir [file normalize [m2153_env M2153_FRAME_DIR]]
    set frame_logs [file normalize [m2153_env M2153_FRAME_LOG_DIR]]
    set design_lib [file normalize [m2153_env M2153_DESIGN_LIB]]
    set mw_ref [file normalize [m2153_env M2153_MW_REF]]
    set tt_db [file normalize [m2153_env M2153_TT_DB]]
    set ss_db [file normalize [m2153_env M2153_SS_DB]]
    set ff_db [file normalize [m2153_env M2153_FF_DB]]
    set masters_file [file normalize [m2153_env M2153_MASTER_LIST]]
    set nxtgrd [file normalize [m2153_env M2153_NXTGRD]]
    set layer_map [file normalize [m2153_env M2153_LAYER_MAP]]
    set reports [file normalize [m2153_env M2153_REPORT_DIR]]
    set expected_rc_name [m2153_env M2153_EXPECTED_RC_TECH_NAME]

    if {[file normalize [pwd]] ne $work} {
        error "M2153 cwd isolation failed actual=[file normalize [pwd]] expected=$work"
    }
    foreach path [list $home $tmp $xdg_cache $cache $frame_dir $frame_logs $reports] {
        if {![string match "${work}/*" $path]} {
            error "M2153 output path escaped isolated cwd: $path"
        }
        file mkdir $path
    }
    if {![string match "${work}/*" $design_lib]} {
        error "M2153 design library escaped isolated cwd: $design_lib"
    }
    set frame_name m2153_tcbn28hpcplusbwp35p140_frame.ndm
    set frame_ndm [file join $frame_dir $frame_name]
    if {[file exists $frame_ndm] || [file exists $design_lib]} {
        error "M2153 overwrite prohibited: output already exists"
    }

    set_app_var sh_continue_on_error false
    set_app_options -name lib.configuration.local_output_dir -value $cache
    set queried_cache [file normalize [get_app_option_value -name lib.configuration.local_output_dir]]
    if {$queried_cache ne $cache} {
        error "M2153 local_output_dir query mismatch actual=$queried_cache expected=$cache"
    }
    puts "M2153_GATE1_OPTION_ROUND_TRIP_PASS cache=$queried_cache"

    set conversion_status [generate_frame_from_mw $frame_name -mw_lib $mw_ref \
        -log_file_dir $frame_logs -output_directory $frame_dir]
    if {$conversion_status != 1} {
        error "M2153 generate_frame_from_mw returned $conversion_status"
    }
    if {![file exists $frame_ndm] || [file type $frame_ndm] eq "link"} {
        error "M2153 frame NDM missing or symbolic link: $frame_ndm"
    }
    puts "M2153_GATE2_FRAME_CONVERSION_PASS status=1 frame=$frame_ndm"

    set_app_var link_library [list $tt_db $ss_db $ff_db]
    set created [create_lib -ref_libs [list $frame_ndm] $design_lib]
    if {[sizeof_collection $created] != 1} {
        error "M2153 create_lib did not return one design library"
    }
    set current [current_lib]
    if {[sizeof_collection $current] != 1} {
        error "M2153 current_lib is not singular after create_lib"
    }
    set current_name [lindex [get_object_name $current] 0]

    set tt_lib [m2153_one_lib *tt0p9v25c* TT]
    set ss_lib [m2153_one_lib *ssg0p9v125c* SS]
    set ff_lib [m2153_one_lib *ffg1p05vm40c* FF]
    set physical_lib [m2153_one_lib tcbn28hpcplusbwp35p140 physical]
    set masters [m2153_read_lines $masters_file]
    if {[llength $masters] != 94 || [llength [lsort -unique $masters]] != 94 || $masters ne [lsort $masters]} {
        error "M2153 expected 94 unique sorted mapped masters"
    }
    set coverage_fh [open [file join $reports master_coverage.tsv] w]
    puts $coverage_fh "master\ttt\tss\tff\tphysical"
    set tt_count 0
    set ss_count 0
    set ff_count 0
    set physical_count 0
    foreach master $masters {
        set tt_hit [m2153_nonempty [get_lib_cells -quiet "${tt_lib}/$master"] "TT $master"]
        set ss_hit [m2153_nonempty [get_lib_cells -quiet "${ss_lib}/$master"] "SS $master"]
        set ff_hit [m2153_nonempty [get_lib_cells -quiet "${ff_lib}/$master"] "FF $master"]
        set physical_hit [m2153_nonempty [get_lib_cells -quiet "${physical_lib}/$master"] "physical $master"]
        incr tt_count
        incr ss_count
        incr ff_count
        incr physical_count
        puts $coverage_fh "$master\t$tt_hit\t$ss_hit\t$ff_hit\t$physical_hit"
    }
    close $coverage_fh
    if {$tt_count != 94 || $ss_count != 94 || $ff_count != 94 || $physical_count != 94} {
        error "M2153 incomplete mapped-master four-view coverage"
    }
    puts "M2153_GATE3_MASTER_COVERAGE_PASS count=94 views=4"

    set core_sites [get_site_defs -quiet -exact core]
    if {[sizeof_collection $core_sites] != 1} {
        error "M2153 expected exactly one exact core site"
    }
    set core_site_name [lindex [get_object_name $core_sites] 0]
    if {$core_site_name ne "core"} {
        error "M2153 core-site name mismatch actual=$core_site_name expected=core"
    }
    set metal_count 0
    foreach layer {M1 M2 M3 M4 M5 M6 M7 M8 M9} {
        m2153_nonempty [get_layers -quiet -exact $layer] "routing layer $layer"
        incr metal_count
    }
    set via_count 0
    foreach layer {VIA1 VIA2 VIA3 VIA4 VIA5 VIA6 VIA7 VIA8} {
        m2153_nonempty [get_layers -quiet -exact $layer] "via layer $layer"
        incr via_count
    }
    set techs [get_techs -of_objects [current_lib]]
    if {[sizeof_collection $techs] != 1} {
        error "M2153 expected one current-library technology"
    }
    set tech_name [lindex [get_object_name $techs] 0]
    puts "M2153_GATE4_PHYSICAL_TECH_PASS site=core site_count=1 metals=M1,M2,M3,M4,M5,M6,M7,M8,M9 vias=VIA1,VIA2,VIA3,VIA4,VIA5,VIA6,VIA7,VIA8 tech=$tech_name"

    read_parasitic_tech -tlup $nxtgrd -layermap $layer_map \
        -name m2153_1p9m_6x1z1u_typ -sanity_check advanced
    set rc_techs [get_parasitic_techs -quiet m2153_1p9m_6x1z1u_typ]
    if {[sizeof_collection $rc_techs] != 1} {
        error "M2153 expected one admitted parasitic technology"
    }
    set actual_rc_name [get_attribute $rc_techs itf_technology_name]
    if {$actual_rc_name ne $expected_rc_name} {
        error "M2153 RC technology identity mismatch actual=$actual_rc_name expected=$expected_rc_name"
    }
    puts "M2153_GATE5_RC_COMPATIBILITY_PASS name=$actual_rc_name"

    redirect -file [file join $reports reference_libraries.rpt] { report_ref_libs }
    redirect -file [file join $reports design_library.rpt] { report_design -library -nosplit }
    save_lib
    set frame_stats [m2153_tree_stats $frame_ndm "frame NDM"]
    set design_stats [m2153_tree_stats $design_lib "design library"]
    set frame_files [lindex $frame_stats 0]
    set frame_bytes [lindex $frame_stats 1]
    set design_files [lindex $design_stats 0]
    set design_bytes [lindex $design_stats 1]
    puts "M2153_GATE6_NONEMPTY_LIBRARY_OBJECTS_PASS frame_files=$frame_files frame_bytes=$frame_bytes design_files=$design_files design_bytes=$design_bytes"

    set facts [open [file join $reports machine_facts.txt] w]
    puts $facts "status=RAW_PASS_M2153_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2156"
    puts $facts "application_option_value=$queried_cache"
    puts $facts "conversion_status=1"
    puts $facts "frame_ndm=$frame_ndm"
    puts $facts "frame_regular_files=$frame_files"
    puts $facts "frame_regular_bytes=$frame_bytes"
    puts $facts "design_lib=$design_lib"
    puts $facts "design_regular_files=$design_files"
    puts $facts "design_regular_bytes=$design_bytes"
    puts $facts "current_library=$current_name"
    puts $facts "tt_library=$tt_lib"
    puts $facts "ss_library=$ss_lib"
    puts $facts "ff_library=$ff_lib"
    puts $facts "physical_library=$physical_lib"
    puts $facts "mapped_master_union_count=94"
    puts $facts "tt_master_coverage=94"
    puts $facts "ss_master_coverage=94"
    puts $facts "ff_master_coverage=94"
    puts $facts "physical_master_coverage=94"
    puts $facts "core_site_name=core"
    puts $facts "core_site_count=1"
    puts $facts "routing_layers=M1,M2,M3,M4,M5,M6,M7,M8,M9"
    puts $facts "via_layers=VIA1,VIA2,VIA3,VIA4,VIA5,VIA6,VIA7,VIA8"
    puts $facts "current_technology=$tech_name"
    puts $facts "rc_technology_name=$actual_rc_name"
    puts $facts "rtl_imported=false"
    puts $facts "pnr_invoked=false"
    close $facts
    puts "RAW_PASS_M2153_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2156_INDEPENDENT_RESULT_HAMMER"
}

if {[catch {m2153_main} message options]} {
    puts stderr "M2153_FATAL_FAIL_CLOSED: $message"
    if {[dict exists $options -errorinfo]} {
        puts stderr [dict get $options -errorinfo]
    }
    exit 42
}
exit 0
