# Data/reset-path ECO after mapped clock gating. No clock exceptions or
# relaxed constraints; report both setup and hold after each buffer round.
if {[catch {
    set input $::env(M2255_INPUT)
    set output $::env(M2255_OUTPUT)
    set slow $::env(M2255_SLOW)
    set fast $::env(M2255_FAST)
    set_app_var target_library [list $slow]
    set_app_var link_library [list * $slow $fast]
    file mkdir "$output/reports"
    file mkdir "$output/netlist"
    read_ddc "$input/netlist/m2018_axis.ddc"
    link
    set_min_library $slow -min_version $fast
    set_operating_conditions ssg0p9v125c
    set_wire_load_model -name ZeroWireload [current_design]
    set_svf "$output/netlist/hold_repair.svf"
    foreach {cell ref} $::env(M2255_UPSIZE) {
        puts "M2255 targeted sizing $cell -> $ref"
        size_cell [get_cells $cell] $ref
    }
    set buffer [get_lib_cells tcbn28hpcplusbwp35p140ssg0p9v125c/BUFFD1BWP35P140]
    set inserted 0
    for {set round 0} {$round < 5} {incr round} {
        update_timing
        set paths [get_timing_paths -delay_type min -max_paths 100000 -nworst 1 -slack_lesser_than 0.0]
        puts "M2255 round=$round violating_endpoints=[sizeof_collection $paths]"
        redirect "$output/reports/round${round}_setup.rpt" {report_timing -delay_type max -max_paths 1 -significant_digits 6}
        redirect "$output/reports/round${round}_hold.rpt" {report_timing -delay_type min -max_paths 1 -significant_digits 6}
        if {[sizeof_collection $paths] == 0} {break}
        set sink_names [list]
        foreach_in_collection path $paths {
            set pin [get_attribute $path endpoint]
            set name [get_object_name $pin]
            # DC also maps some reset equations to clear/set pins. Their
            # recovery/removal checks remain timed; do not touch clock pins.
            if {![regexp {/(D|CN|SN)$} $name]} {error "Unexpected hold endpoint: $name"}
            lappend sink_names $name
        }
        set sinks [get_pins $sink_names]
        set cells [insert_buffer $sinks $buffer -new_cell_names "m2255_hold_r${round}" -new_net_names "m2255_hold_net_r${round}"]
        incr inserted [sizeof_collection $cells]
    }
    update_timing
    redirect "$output/reports/setup_after.rpt" {report_timing -delay_type max -max_paths 3 -significant_digits 6}
    redirect "$output/reports/hold_after.rpt" {report_timing -delay_type min -max_paths 3 -significant_digits 6}
    redirect "$output/reports/area.rpt" {report_area -hierarchy}
    redirect "$output/reports/constraints_after.rpt" {report_constraint -all_violators -significant_digits 6}
    redirect "$output/reports/check_design.rpt" {check_design}
    puts "M2255 total_inserted_buffers=$inserted"
    write_file -format ddc -hierarchy -output "$output/netlist/m2018_axis.ddc"
    write_file -format verilog -hierarchy -output "$output/netlist/m2018_axis_mapped.v"
    write_sdc "$output/netlist/m2018_axis_mapped.sdc"
    set_svf -off
} message]} {
    puts stderr "M2255 stopped: $message"
    exit 1
}
exit
