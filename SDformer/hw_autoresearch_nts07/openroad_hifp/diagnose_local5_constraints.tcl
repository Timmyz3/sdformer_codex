source $::env(SCRIPTS_DIR)/load.tcl

load_design 5_1_grt.odb 4_cts.sdc "加载 Local5 OUT32 全宏 Direct 设计"
set_propagated_clock [all_clocks]

set relation_padding_pins {}
foreach pin [get_pins -hierarchical */wd_in*] {
  set pin_name [get_full_name $pin]
  if {[regexp {g_w10\.g_bank\[[01]\]\.u_macro/wd_in\[(10|11|12|13|14|15)\]$} \
      $pin_name]} {
    lappend relation_padding_pins $pin
  }
}
set padding_count [llength $relation_padding_pins]
puts "Local5 OUT32 全宏 Direct 约束审计"
puts "================================="
puts "结构性常量填充 pin 数量: $padding_count"
if {$padding_count != 60} {
  error "期望匹配 60 个关系存储高位填充 pin，实际为 $padding_count"
}
foreach pin $relation_padding_pins {
  set_disable_timing -to [get_name $pin] [get_cells -of_objects $pin]
}

set block [ord::get_db_block]
set acc_macro_count 0
foreach inst [$block getInsts] {
  set inst_name [$inst getName]
  if {[string first {g_slice} $inst_name] >= 0} {
    incr acc_macro_count
    set bbox [$inst getBBox]
    puts [format "Acc32 SRAM: %s master=%s orient=%s bbox=(%.3f,%.3f)-(%.3f,%.3f) um" \
      $inst_name [[$inst getMaster] getName] [$inst getOrient] \
      [ord::dbu_to_microns [$bbox xMin]] [ord::dbu_to_microns [$bbox yMin]] \
      [ord::dbu_to_microns [$bbox xMax]] [ord::dbu_to_microns [$bbox yMax]]]
  }
}
puts "Acc32 SRAM 宏数量: $acc_macro_count"
if {$acc_macro_count != 20} {
  error "期望匹配 20 个 Acc32 SRAM 宏，实际为 $acc_macro_count"
}

check_setup -verbose
report_clock_properties
report_checks -path_delay max -group_count 1 -endpoint_count 1 \
  -format full_clock_expanded
report_checks -path_delay min -group_count 1 -endpoint_count 1 \
  -format full_clock_expanded

puts "约束审计完成"
