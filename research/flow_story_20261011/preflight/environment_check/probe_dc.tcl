# Environment readiness only: no RTL design, synthesis or PPA claim.
set_host_options -max_cores 2
set probe_library /opt/tech/tsmc28_2026/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp7t35p140_180a/tcbn28hpcplusbwp7t35p140tt0p9v25c.db
if {![file readable $probe_library]} { error "FLOW_PROBE_LIBRARY_UNREADABLE" }
read_db $probe_library
set probe_libs [get_libs tcbn28hpcplusbwp7t35p140tt0p9v25c]
if {[sizeof_collection $probe_libs] != 1} { error "FLOW_PROBE_LIBRARY_NOT_LOADED" }
puts "FLOW_DC_LIBRARY_READY [get_object_name $probe_libs]"
quit
