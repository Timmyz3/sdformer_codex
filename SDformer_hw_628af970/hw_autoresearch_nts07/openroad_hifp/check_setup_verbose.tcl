if {![info exists ::env(ODB_FILE)] || ![info exists ::env(SDC_FILE)] ||
    ![info exists ::env(LIB_FILE)]} {
  error "ODB_FILE, SDC_FILE and LIB_FILE must be set"
}

read_liberty $::env(LIB_FILE)
read_db $::env(ODB_FILE)
read_sdc $::env(SDC_FILE)
check_setup -verbose
