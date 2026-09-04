# M2141: library-import-only ICC2 preflight after the consumed M2135 failure.
# This source intentionally contains no RTL import, design link, floorplan,
# placement, CTS, routing, extraction, timing, area, or power command.

proc m2141_env {name} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        error "M2141 missing environment variable $name"
    }
    return $::env($name)
}

proc m2141_nonempty {collection label} {
    set count [sizeof_collection $collection]
    if {$count <= 0} { error "M2141 empty collection: $label" }
    return $count
}

proc m2141_one_lib {pattern label} {
    set libs [get_libs -quiet $pattern]
    if {[sizeof_collection $libs] != 1} {
        error "M2141 expected exactly one $label library for $pattern"
    }
    return [lindex [get_object_name $libs] 0]
}

proc m2141_read_lines {path} {
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

proc m2141_main {} {
    set work [file normalize [m2141_env M2141_ISOLATED_CWD]]
    set cache [file normalize [m2141_env M2141_LIBRARY_CACHE]]
    set frame_dir [file normalize [m2141_env M2141_FRAME_DIR]]
    set frame_logs [file normalize [m2141_env M2141_FRAME_LOG_DIR]]
    set design_lib [file normalize [m2141_env M2141_DESIGN_LIB]]
    set mw_ref [file normalize [m2141_env M2141_MW_REF]]
    set tt_db [file normalize [m2141_env M2141_TT_DB]]
    set ss_db [file normalize [m2141_env M2141_SS_DB]]
    set ff_db [file normalize [m2141_env M2141_FF_DB]]
    set masters_file [file normalize [m2141_env M2141_MASTER_LIST]]
    set nxtgrd [file normalize [m2141_env M2141_NXTGRD]]
    set layer_map [file normalize [m2141_env M2141_LAYER_MAP]]
    set reports [file normalize [m2141_env M2141_REPORT_DIR]]
    set expected_rc_name [m2141_env M2141_EXPECTED_RC_TECH_NAME]

    if {[file normalize [pwd]] ne $work} {
        error "M2141 cwd isolation failed actual=[file normalize [pwd]] expected=$work"
    }
    foreach path [list $cache $frame_dir $frame_logs $reports] {
        if {![string match "${work}/*" $path]} {
            error "M2141 output path escaped isolated cwd: $path"
        }
        file mkdir $path
    }
    if {![string match "${work}/*" $design_lib]} {
        error "M2141 design library escaped isolated cwd: $design_lib"
    }
    set frame_name m2141_tcbn28hpcplusbwp35p140_frame.ndm
    set frame_ndm [file join $frame_dir $frame_name]
    if {[file exists $frame_ndm] || [file exists $design_lib]} {
        error "M2141 overwrite prohibited: output already exists"
    }

    set_app_var sh_continue_on_error false
    set_app_options -name lib.configuration.local_output_dir -value $cache
    set queried_cache [file normalize [get_app_option_value -name lib.configuration.local_output_dir]]
    if {$queried_cache ne $cache} {
        error "M2141 local_output_dir query mismatch actual=$queried_cache expected=$cache"
    }
    puts "M2141_GATE1_EXACT_APPLICATION_OPTION_PASS path=$queried_cache"

    set conversion_status [generate_frame_from_mw $frame_name -mw_lib $mw_ref \
        -log_file_dir $frame_logs -output_directory $frame_dir]
    if {$conversion_status != 1} {
        error "M2141 generate_frame_from_mw returned $conversion_status"
    }
    if {![file exists $frame_ndm] || [file type $frame_ndm] eq "link"} {
        error "M2141 frame NDM missing or symbolic link: $frame_ndm"
    }
    puts "M2141_GATE2_FRAME_CONVERSION_PASS status=$conversion_status frame=$frame_ndm"

    set_app_var link_library [list $tt_db $ss_db $ff_db]
    set created [create_lib -ref_libs [list $frame_ndm] $design_lib]
    if {[sizeof_collection $created] != 1} {
        error "M2141 create_lib did not return one design library"
    }
    set current [current_lib]
    if {[sizeof_collection $current] != 1} {
        error "M2141 current_lib is not singular after create_lib"
    }

    set tt_lib [m2141_one_lib *tt0p9v25c* TT]
    set ss_lib [m2141_one_lib *ssg0p9v125c* SS]
    set ff_lib [m2141_one_lib *ffg1p05vm40c* FF]
    set physical_lib [m2141_one_lib tcbn28hpcplusbwp35p140 physical]
    set masters [m2141_read_lines $masters_file]
    if {[llength $masters] != 94 || [llength [lsort -unique $masters]] != 94} {
        error "M2141 expected 94 unique mapped masters"
    }
    set coverage_fh [open [file join $reports master_coverage.tsv] w]
    puts $coverage_fh "master\ttt\tss\tff\tphysical"
    set tt_count 0
    set ss_count 0
    set ff_count 0
    set physical_count 0
    foreach master $masters {
        set tt_hit [m2141_nonempty [get_lib_cells -quiet "${tt_lib}/$master"] "TT $master"]
        set ss_hit [m2141_nonempty [get_lib_cells -quiet "${ss_lib}/$master"] "SS $master"]
        set ff_hit [m2141_nonempty [get_lib_cells -quiet "${ff_lib}/$master"] "FF $master"]
        set physical_hit [m2141_nonempty [get_lib_cells -quiet "${physical_lib}/$master"] "physical $master"]
        incr tt_count
        incr ss_count
        incr ff_count
        incr physical_count
        puts $coverage_fh "$master\t$tt_hit\t$ss_hit\t$ff_hit\t$physical_hit"
    }
    close $coverage_fh
    if {$tt_count != 94 || $ss_count != 94 || $ff_count != 94 || $physical_count != 94} {
        error "M2141 incomplete mapped-master view coverage"
    }
    puts "M2141_GATE3_94_MASTER_FOUR_VIEW_COVERAGE_PASS TT=$tt_count SS=$ss_count FF=$ff_count physical=$physical_count"

    set core_sites [get_site_defs -quiet -exact core]
    if {[sizeof_collection $core_sites] <= 0} {
        set core_sites [get_site_defs -quiet *core*]
    }
    m2141_nonempty $core_sites "core site"
    set metal_count 0
    foreach layer {M1 M2 M3 M4 M5 M6 M7 M8 M9} {
        m2141_nonempty [get_layers -quiet -exact $layer] "routing layer $layer"
        incr metal_count
    }
    set via_count 0
    foreach layer {VIA1 VIA2 VIA3 VIA4 VIA5 VIA6 VIA7 VIA8} {
        m2141_nonempty [get_layers -quiet -exact $layer] "via layer $layer"
        incr via_count
    }
    set techs [get_techs -of_objects [current_lib]]
    if {[sizeof_collection $techs] != 1} {
        error "M2141 expected one current-library technology"
    }
    set tech_name [lindex [get_object_name $techs] 0]
    puts "M2141_GATE4_PHYSICAL_TECH_STRUCTURE_PASS core_sites=[sizeof_collection $core_sites] metals=$metal_count vias=$via_count tech=$tech_name"

    read_parasitic_tech -tlup $nxtgrd -layermap $layer_map \
        -name m2141_1p9m_6x1z1u_typ -sanity_check advanced
    set rc_techs [get_parasitic_techs -quiet m2141_1p9m_6x1z1u_typ]
    if {[sizeof_collection $rc_techs] != 1} {
        error "M2141 expected one admitted parasitic technology"
    }
    set actual_rc_name [get_attribute $rc_techs itf_technology_name]
    if {$actual_rc_name ne $expected_rc_name} {
        error "M2141 RC technology identity mismatch actual=$actual_rc_name expected=$expected_rc_name"
    }
    puts "M2141_GATE5_1P9M_6X1Z1U_RC_COMPATIBILITY_PASS itf_technology_name=$actual_rc_name"

    redirect -file [file join $reports reference_libraries.rpt] { report_ref_libs }
    redirect -file [file join $reports design_library.rpt] { report_design -library -nosplit }
    set facts [open [file join $reports machine_facts.txt] w]
    puts $facts "status=RAW_PASS_M2141_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2148"
    puts $facts "application_option_value=$queried_cache"
    puts $facts "conversion_status=$conversion_status"
    puts $facts "frame_ndm=$frame_ndm"
    puts $facts "design_lib=$design_lib"
    puts $facts "tt_library=$tt_lib"
    puts $facts "ss_library=$ss_lib"
    puts $facts "ff_library=$ff_lib"
    puts $facts "physical_library=$physical_lib"
    puts $facts "mapped_master_union_count=94"
    puts $facts "tt_master_coverage=$tt_count"
    puts $facts "ss_master_coverage=$ss_count"
    puts $facts "ff_master_coverage=$ff_count"
    puts $facts "physical_master_coverage=$physical_count"
    puts $facts "core_site_count=[sizeof_collection $core_sites]"
    puts $facts "routing_layers=M1,M2,M3,M4,M5,M6,M7,M8,M9"
    puts $facts "via_layers=VIA1,VIA2,VIA3,VIA4,VIA5,VIA6,VIA7,VIA8"
    puts $facts "current_technology=$tech_name"
    puts $facts "rc_technology_name=$actual_rc_name"
    puts $facts "rtl_imported=false"
    puts $facts "pnr_invoked=false"
    close $facts
    save_lib
    puts "RAW_PASS_M2141_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2148_INDEPENDENT_RESULT_HAMMER"
}

if {[catch {m2141_main} message options]} {
    puts stderr "M2141_FATAL_FAIL_CLOSED: $message"
    if {[dict exists $options -errorinfo]} {
        puts stderr [dict get $options -errorinfo]
    }
    exit 42
}
exit 0
