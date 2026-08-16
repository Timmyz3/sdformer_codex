foreach variable {STD_LIB SRAM_LIB ODB_FILE SDC_FILE} {
  if {![info exists ::env($variable)]} {
    error "$variable must be set"
  }
}

read_liberty $::env(STD_LIB)
read_liberty $::env(SRAM_LIB)
read_db $::env(ODB_FILE)
read_sdc $::env(SDC_FILE)

puts "Motion ZKQI production constraint audit"
puts "======================================="
check_setup -verbose
report_clock_properties
report_checks -path_delay max -group_count 1 -endpoint_count 1 \
  -format full_clock_expanded
