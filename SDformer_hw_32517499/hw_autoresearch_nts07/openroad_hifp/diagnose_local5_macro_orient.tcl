source $::env(SCRIPTS_DIR)/load.tcl

load_design 2_4_floorplan_macro.odb 1_synth.sdc "加载 Local5 OUT32 宏朝向结果"

set block [ord::get_db_block]
set acc_macros {}
set all_macros {}
set non_my_count 0

foreach inst [$block getInsts] {
  if {[[$inst getMaster] getType] eq "BLOCK"} {
    set bbox [$inst getBBox]
    lappend all_macros [list [$inst getName] \
      [$bbox xMin] [$bbox yMin] [$bbox xMax] [$bbox yMax]]
  }
  if {[[$inst getMaster] getName] eq "fakeram45_128x256"} {
    set bbox [$inst getBBox]
    set entry [list [$inst getName] [$inst getOrient] \
      [$bbox xMin] [$bbox yMin] [$bbox xMax] [$bbox yMax]]
    lappend acc_macros $entry
    if {[$inst getOrient] ne "MY"} {
      incr non_my_count
    }
    puts [format "Acc32 SRAM: %s orient=%s bbox=(%.3f,%.3f)-(%.3f,%.3f) um" \
      [$inst getName] [$inst getOrient] \
      [ord::dbu_to_microns [$bbox xMin]] [ord::dbu_to_microns [$bbox yMin]] \
      [ord::dbu_to_microns [$bbox xMax]] [ord::dbu_to_microns [$bbox yMax]]]
  }
}

set overlap_count 0
for {set i 0} {$i < [llength $all_macros]} {incr i} {
  set a [lindex $all_macros $i]
  for {set j [expr {$i + 1}]} {$j < [llength $all_macros]} {incr j} {
    set b [lindex $all_macros $j]
    set overlap_x [expr {[lindex $a 1] < [lindex $b 3] && [lindex $b 1] < [lindex $a 3]}]
    set overlap_y [expr {[lindex $a 2] < [lindex $b 4] && [lindex $b 2] < [lindex $a 4]}]
    if {$overlap_x && $overlap_y} {
      incr overlap_count
      puts "重叠: [lindex $a 0] <-> [lindex $b 0]"
    }
  }
}

puts "Acc32 SRAM 宏数量: [llength $acc_macros]"
puts "全部 SRAM 宏数量: [llength $all_macros]"
puts "非 MY 宏数量: $non_my_count"
puts "全部 SRAM 宏间重叠数量: $overlap_count"
if {[llength $acc_macros] != 20 || [llength $all_macros] != 32 || \
    $non_my_count != 0 || $overlap_count != 0} {
  error "Acc32 宏朝向或几何审计失败"
}
puts "Acc32 宏朝向与几何审计完成"
