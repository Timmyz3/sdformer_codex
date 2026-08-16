set block [ord::get_db_block]
set acc_macro_count 0

foreach inst [$block getInsts] {
  if {[[$inst getMaster] getName] eq "fakeram45_128x256"} {
    if {[$inst getOrient] ne "MY"} {
      set old_bbox [$inst getBBox]
      set old_x_min [$old_bbox xMin]
      set old_y_min [$old_bbox yMin]
      set placement_status [$inst getPlacementStatus]

      $inst setPlacementStatus PLACED
      $inst setOrient MY
      $inst setLocation $old_x_min $old_y_min
      $inst setPlacementStatus $placement_status
    }
    incr acc_macro_count
  }
}

puts "Local5 Acc32 pin-access 朝向约束: MY, 宏数量=$acc_macro_count"
if {$acc_macro_count != 20} {
  error "期望约束 20 个 Acc32 SRAM 宏，实际为 $acc_macro_count"
}
