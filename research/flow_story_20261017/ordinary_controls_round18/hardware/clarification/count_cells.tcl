# Read the frozen mapped DDC only; no compile/optimization/netlist writes.
set target_library /opt/tech/tsmc28_2026/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp7t35p140_180a/tcbn28hpcplusbwp7t35p140tt0p9v25c.db
set link_library [concat * $target_library]
read_ddc ../results/dc/mapped.ddc
current_design k1_f64
link
set allc [get_cells -hierarchical *]
set leaf [filter_collection $allc "is_hierarchical == false"]
set hierarchy [filter_collection $allc "is_hierarchical == true"]
set f [open actual_cell_counts.txt w]
puts $f "all_hierarchical_cells=[sizeof_collection $allc]"
puts $f "leaf_cells=[sizeof_collection $leaf]"
puts $f "hierarchy_instances=[sizeof_collection $hierarchy]"
foreach_in_collection c $hierarchy {puts $f "hierarchy_instance=[get_object_name $c]"}
close $f
quit
