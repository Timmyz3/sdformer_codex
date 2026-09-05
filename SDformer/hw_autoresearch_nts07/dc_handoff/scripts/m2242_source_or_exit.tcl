# Do not continue a long synthesis after a Tcl input/path error.
if {[catch {
    if {[info exists ::env(M2242_DC_WORK)]} {
        file mkdir $::env(M2242_DC_WORK)
        define_design_lib WORK -path $::env(M2242_DC_WORK)
    }
    source $::env(M2242_TOOL_SCRIPT)
} message]} {
    puts stderr "Error: M2242 tool script stopped: $message"
    exit 1
}
