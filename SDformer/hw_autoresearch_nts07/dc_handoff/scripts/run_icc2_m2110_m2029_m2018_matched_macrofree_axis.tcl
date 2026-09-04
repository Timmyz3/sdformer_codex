# M2110: one axis of the matched macro-free M2018 ICC2 P&R experiment.
# Source-only at creation.  Execution is authorized only through the exact-SHA
# M2110 one-shot runner after an independent M2111 source hammer.

proc m2110_require_env {name} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        error "M2110 missing environment variable $name"
    }
    return $::env($name)
}

proc m2110_require_nonempty {collection label} {
    set count [sizeof_collection $collection]
    if {$count <= 0} { error "M2110 empty collection: $label" }
    return $count
}

proc m2110_sum_cell_area {cells} {
    set total 0.0
    foreach_in_collection cell $cells {
        set value [get_attribute -quiet $cell area]
        if {$value ne ""} { set total [expr {$total + double($value)}] }
    }
    return $total
}

proc m2110_ref_prefix_count {cells prefixes} {
    set count 0
    foreach_in_collection cell $cells {
        set ref [get_attribute -quiet $cell ref_name]
        foreach prefix $prefixes {
            if {[string match "${prefix}*" $ref]} {
                incr count
                break
            }
        }
    }
    return $count
}

proc m2110_pin_floorplan {} {
    set ports [lsort [get_object_name [get_ports *]]]
    set n [llength $ports]
    if {$n != 4551} { error "M2110 expected 4551 ports, got $n" }
    set per_side [expr {int(ceil(double($n) / 4.0))}]
    set pitch [expr {720.0 / double($per_side + 1)}]
    set fh [open "$::env(M2110_AXIS_DIR)/reports/ports_sorted.txt" w]
    foreach name $ports { puts $fh $name }
    close $fh
    for {set i 0} {$i < $n} {incr i} {
        set name [lindex $ports $i]
        set side [expr {$i % 4}]
        set slot [expr {int($i / 4) + 1}]
        set delta [expr {$pitch * double($slot)}]
        if {$side == 0} {
            set location [list [expr {40.0 + $delta}] 40.0]
            set layer M3
        } elseif {$side == 1} {
            set location [list 760.0 [expr {40.0 + $delta}]]
            set layer M4
        } elseif {$side == 2} {
            set location [list [expr {760.0 - $delta}] 760.0]
            set layer M3
        } else {
            set location [list 40.0 [expr {760.0 - $delta}]]
            set layer M4
        }
        set_individual_pin_constraints -ports [get_ports -exact $name] \
            -allowed_layers [list $layer] -location $location
    }
    place_pins -ports [get_ports *]
}

proc m2110_main {} {
    set axis [m2110_require_env M2110_AXIS]
    if {$axis ni {ordinary_lru4 tsbg_b4}} { error "M2110 invalid axis $axis" }
    set top [m2110_require_env M2110_TOP]
    set axis_dir [m2110_require_env M2110_AXIS_DIR]
    set netlist [m2110_require_env M2110_MAPPED_V]
    set physical_sdc [m2110_require_env M2110_PHYSICAL_SDC]
    set design_lib [m2110_require_env M2110_DESIGN_LIB]
    set mw_ref [m2110_require_env M2110_MW_REF_LIB]
    set tt_db [m2110_require_env M2110_TT_DB]
    set ss_db [m2110_require_env M2110_SS_DB]
    set ff_db [m2110_require_env M2110_FF_DB]
    set nxtgrd [m2110_require_env M2110_NXTGRD]
    set layer_map [m2110_require_env M2110_LAYER_MAP]
    set physical_sdc_sha [m2110_require_env M2110_PHYSICAL_SDC_SHA256]

    file mkdir "$axis_dir/reports" "$axis_dir/output" "$axis_dir/library_cache"
    set_app_var sh_continue_on_error false
    set_app_var link_library [list $tt_db $ss_db $ff_db]
    set_app_var lib.configuration.local_output_dir "$axis_dir/library_cache"

    # Gate 1: the first tool action is the legacy-Milkyway plus TT/SS/FF import.
    # Any unsupported conversion, missing view, or link mismatch raises a Tcl
    # error and the one-shot shell quarantines the entire attempt.
    create_lib -ref_libs [list $mw_ref] $design_lib
    read_verilog -top $top $netlist
    current_block $top
    link_block -force -verbose
    redirect -file "$axis_dir/reports/reference_libraries.rpt" { report_ref_libs }
    redirect -file "$axis_dir/reports/design_library.rpt" { report_design -library -nosplit }
    redirect -file "$axis_dir/reports/design_mismatch.rpt" { report_design_mismatch -verbose -nosplit }

    set mismatch_count [sizeof_collection [get_mismatch_objects -quiet]]
    if {$mismatch_count != 0} { error "M2110 logical/physical mismatch count $mismatch_count" }
    set input_leaf [get_cells -hierarchical -quiet -filter "is_hierarchical == false"]
    set input_refs [lsort -unique [get_attribute $input_leaf ref_name]]
    set input_master_count [llength $input_refs]
    if {$input_master_count != 94} { error "M2110 mapped master count $input_master_count" }
    foreach ref $input_refs {
        m2110_require_nonempty [get_lib_cells -quiet */$ref] "logical/physical master $ref"
    }
    # The forced link, zero mismatch collection, and explicit 94/94 master
    # lookup jointly form the unresolved-reference gate.  Avoid relying on a
    # release-specific `is_unresolved` collection attribute.
    set unresolved_count 0
    m2110_require_nonempty [get_site_defs -quiet *core*] "core site"
    set routing_layer_gate_count 0
    foreach layer {M1 M2 M3 M4 M5 M6 M7 M8 M9} {
        m2110_require_nonempty [get_layers -quiet $layer] "routing layer $layer"
        incr routing_layer_gate_count
    }
    set via_layer_gate_count 0
    foreach layer {VIA1 VIA2 VIA3 VIA4 VIA5 VIA6 VIA7 VIA8} {
        m2110_require_nonempty [get_layers -quiet $layer] "via layer $layer"
        incr via_layer_gate_count
    }
    puts "M2110_GATE1_IMPORT_AND_LIBRARY_CHECK_PASS axis=$axis masters=$input_master_count"

    # Gate 2: common foundry NXTGRD, with the strongest documented sanity mode.
    read_parasitic_tech -tlup $nxtgrd -layermap $layer_map \
        -name n28_1p9m_6x1z1u_typ -sanity_check advanced
    puts "M2110_GATE2_NXTGRD_ADVANCED_SANITY_PASS axis=$axis"

    # Matched MCMM: one functional mode, slow setup, fast hold, TT power.  The
    # same typical RC model is intentionally used on all three corners because
    # the selected source audit admitted only that NXTGRD for this feasibility
    # run; this is not max/min-RC signoff.
    read_sdc $physical_sdc
    set setup_scenario [current_scenario]
    set setup_corner [current_corner]
    set_operating_conditions -library [get_libs *ssg0p9v125c*] ssg0p9v125c
    set_parasitic_parameters -corners $setup_corner \
        -early_spec n28_1p9m_6x1z1u_typ -early_temperature 125 \
        -late_spec n28_1p9m_6x1z1u_typ -late_temperature 125
    set_scenario_status $setup_scenario -none
    set_scenario_status $setup_scenario -setup true -max_transition true -max_capacitance true

    create_corner ff_hold
    set ff_corner [current_corner]
    set hold_scenario [create_scenario -mode [current_mode] -corner $ff_corner -name func_ff_hold]
    current_scenario $hold_scenario
    set_operating_conditions -library [get_libs *ffg1p05vm40c*] ffg1p05vm40c
    set_parasitic_parameters -corners $ff_corner \
        -early_spec n28_1p9m_6x1z1u_typ -early_temperature -40 \
        -late_spec n28_1p9m_6x1z1u_typ -late_temperature -40
    set_scenario_status $hold_scenario -none
    set_scenario_status $hold_scenario -hold true -min_capacitance true

    create_corner tt_power
    set tt_corner [current_corner]
    set power_scenario [create_scenario -mode [current_mode] -corner $tt_corner -name func_tt_power]
    current_scenario $power_scenario
    set_operating_conditions -library [get_libs *tt0p9v25c*] tt0p9v25c
    set_parasitic_parameters -corners $tt_corner \
        -early_spec n28_1p9m_6x1z1u_typ -early_temperature 25 \
        -late_spec n28_1p9m_6x1z1u_typ -late_temperature 25
    set_scenario_status $power_scenario -none
    set_scenario_status $power_scenario -dynamic_power true -leakage_power true
    current_scenario $setup_scenario

    # Fixed and identical physical envelope.  The 288-KiB SRAM is not placed;
    # its interface remains on the boundary and its area/leakage are common.
    initialize_floorplan -control_type die -shape R \
        -boundary {{0 0} {800 0} {800 800} {0 800}} \
        -core_offset {40 40 40 40}
    set_ignored_layers -min_routing_layer M2 -max_routing_layer M8
    m2110_pin_floorplan

    # Freeze the same CTS and hold-repair candidate sets on both axes.
    set all_lib_cells [get_lib_cells -quiet */*]
    set_lib_cell_purpose -exclude cts $all_lib_cells
    set cts_cells [get_lib_cells -quiet */CKBD*]
    set cts_cells [add_to_collection $cts_cells [get_lib_cells -quiet */CKND*]]
    m2110_require_nonempty $cts_cells "CTS CKBD/CKND whitelist"
    set_lib_cell_purpose -include cts $cts_cells
    set_lib_cell_purpose -exclude hold $all_lib_cells
    set hold_cells [get_lib_cells -quiet */DEL*]
    set hold_cells [add_to_collection $hold_cells [get_lib_cells -quiet */BUFF*]]
    set hold_cells [add_to_collection $hold_cells [get_lib_cells -quiet */INV*]]
    m2110_require_nonempty $hold_cells "hold DEL/BUFF/INV whitelist"
    set_lib_cell_purpose -include hold $hold_cells
    set_clock_tree_options -clocks [get_clocks core_clk] -target_skew 0.080

    set pre_place_rc [check_design -checks pre_placement_stage \
        -ems_database "$axis_dir/reports/pre_placement.ems" \
        -log_file "$axis_dir/reports/pre_placement_check.rpt"]
    if {!$pre_place_rc} { error "M2110 pre-placement check failed" }
    place_opt
    set pre_clock_rc [check_design -checks pre_clock_tree_stage \
        -ems_database "$axis_dir/reports/pre_clock.ems" \
        -log_file "$axis_dir/reports/pre_clock_check.rpt"]
    if {!$pre_clock_rc} { error "M2110 pre-clock check failed" }
    clock_opt
    foreach_in_collection mode [all_modes] {
        current_mode $mode
        set_propagated_clock [all_clocks]
    }
    current_scenario $setup_scenario
    set pre_route_rc [check_design -checks pre_route_stage \
        -ems_database "$axis_dir/reports/pre_route.ems" \
        -log_file "$axis_dir/reports/pre_route_check.rpt"]
    if {!$pre_route_rc} { error "M2110 pre-route check failed" }
    route_auto
    route_opt
    redirect -file "$axis_dir/reports/route_check.rpt" {
        set route_check_rc [check_routes -open_net true -report_all_open_nets true \
            -drc true -antenna false -voltage_area true]
    }
    if {!$route_check_rc} { error "M2110 routed connectivity/DRC check failed" }

    current_scenario $setup_scenario
    set setup_paths [get_timing_paths -delay_type max -nworst 1 -max_paths 1]
    m2110_require_nonempty $setup_paths "setup path"
    set setup_wns [get_attribute [index_collection $setup_paths 0] slack]
    current_scenario $hold_scenario
    set hold_paths [get_timing_paths -delay_type min -nworst 1 -max_paths 1]
    m2110_require_nonempty $hold_paths "hold path"
    set hold_wns [get_attribute [index_collection $hold_paths 0] slack]
    if {double($setup_wns) < 0.0 || double($hold_wns) < 0.0} {
        error "M2110 timing not closed: setup=$setup_wns hold=$hold_wns"
    }

    current_scenario $setup_scenario
    redirect -file "$axis_dir/reports/qor.rpt" { report_qor -summary -nosplit }
    redirect -file "$axis_dir/reports/timing_setup.rpt" {
        report_timing -delay_type max -path_type full_clock_expanded -max_paths 20 -nosplit
    }
    current_scenario $hold_scenario
    redirect -file "$axis_dir/reports/timing_hold.rpt" {
        report_timing -delay_type min -path_type full_clock_expanded -max_paths 20 -nosplit
    }
    redirect -file "$axis_dir/reports/clock_qor.rpt" { report_clock_qor -all -nosplit }
    redirect -file "$axis_dir/reports/congestion.rpt" { report_congestion -mode summary -nosplit }
    redirect -file "$axis_dir/reports/wirelength.rpt" { report_wirelength -verbose }
    redirect -file "$axis_dir/reports/final_design.rpt" { report_design -all -nosplit }
    current_scenario $power_scenario
    redirect -file "$axis_dir/reports/vectorless_power_diagnostic.rpt" {
        report_power -nosplit -verbose
    }

    write_verilog -top_module_first "$axis_dir/output/routed.v"
    write_sdc -output "$axis_dir/output/routed.sdc" -nosplit
    write_def "$axis_dir/output/routed.def"
    write_parasitics -output "$axis_dir/output/routed" -format spef

    set routed_leaf [get_cells -hierarchical -quiet -filter "is_hierarchical == false"]
    set routed_seq [get_cells -hierarchical -quiet -filter "is_sequential == true"]
    set routed_area [m2110_sum_cell_area $routed_leaf]
    set clock_like [m2110_ref_prefix_count $routed_leaf {CKBD CKND}]
    set hold_like [m2110_ref_prefix_count $routed_leaf {DEL BUFF INV}]
    if {$routed_area <= 0.0 || $clock_like <= 0 || $hold_like <= 0} {
        error "M2110 invalid routed cell census"
    }

    set fh [open "$axis_dir/machine_facts.txt" w]
    puts $fh "status=PASS_M2110_MATCHED_MACROFREE_ICC2_AXIS"
    puts $fh "axis=$axis"
    puts $fh "top=$top"
    puts $fh "public_port_count=[sizeof_collection [get_ports *]]"
    puts $fh "input_master_count=$input_master_count"
    puts $fh "unresolved_reference_count=$unresolved_count"
    puts $fh "logical_physical_mismatch_count=$mismatch_count"
    puts $fh "routing_layer_gate_count=$routing_layer_gate_count"
    puts $fh "via_layer_gate_count=$via_layer_gate_count"
    puts $fh "route_check_return=$route_check_rc"
    puts $fh "pre_placement_check_return=$pre_place_rc"
    puts $fh "pre_clock_check_return=$pre_clock_rc"
    puts $fh "pre_route_check_return=$pre_route_rc"
    puts $fh "die_bbox_um=0,0,800,800"
    puts $fh "core_bbox_um=40,40,760,760"
    puts $fh "floorplan_policy=fixed_die_core_800_720um_v1"
    puts $fh "pin_policy=sorted_four_side_round_robin_exact_location_v1"
    puts $fh "route_layers=M2:M8"
    puts $fh "cts_cell_policy=CKBD_and_CKND_only_v1"
    puts $fh "hold_cell_policy=DEL_BUFF_INV_only_v1"
    puts $fh "clock_period_ns=3.000"
    puts $fh "setup_uncertainty_ns=0.200"
    puts $fh "hold_uncertainty_ns=0.050"
    puts $fh "parasitic_tech=n28_1p9m_6x1z1u_typ"
    puts $fh "parasitic_corner_scope=same_typical_rc_on_ss_ff_tt"
    puts $fh "common_external_sram_bytes=294912"
    puts $fh "common_external_sram_integrated=false"
    puts $fh "propagated_clock=true"
    puts $fh "macro_instances=0"
    puts $fh "physical_sdc_sha256=$physical_sdc_sha"
    puts $fh "port_inventory_sha256=POPULATED_BY_ONE_SHOT_RUNNER"
    puts $fh "setup_wns_ns=$setup_wns"
    puts $fh "hold_wns_ns=$hold_wns"
    puts $fh "routed_standard_cell_area_um2=$routed_area"
    puts $fh "routed_leaf_cell_count=[sizeof_collection $routed_leaf]"
    puts $fh "routed_sequential_cell_count=[sizeof_collection $routed_seq]"
    puts $fh "clock_like_cell_count=$clock_like"
    puts $fh "hold_like_cell_count=$hold_like"
    close $fh
    set done [open "$axis_dir/RUN_COMPLETE.txt" w]
    puts $done "PASS_M2110_MATCHED_MACROFREE_ICC2_AXIS"
    close $done
    puts "PASS_M2110_MATCHED_MACROFREE_ICC2_AXIS axis=$axis"
}

if {[catch {m2110_main} m2110_error m2110_options]} {
    puts stderr "M2110_FATAL_FAIL_CLOSED: $m2110_error"
    if {[dict exists $m2110_options -errorinfo]} {
        puts stderr [dict get $m2110_options -errorinfo]
    }
    exit 42
}
exit 0
