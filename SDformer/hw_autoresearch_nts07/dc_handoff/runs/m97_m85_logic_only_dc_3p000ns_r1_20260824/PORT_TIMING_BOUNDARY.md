# M97 port timing boundary

This run is a standalone M85 logic-island diagnostic, not a complete PWP
frontend timing result.

- `bank_row_addresses[79:0]` terminate at output ports representing the address
  side of an excluded external eight-bank memory.
- `bank_words[255:0]` begin at input ports and receive the same 0.250 ns input
  delay as other synchronous inputs.
- There is no address-to-SRAM-to-data arc, SRAM clock-to-Q, bank decoder,
  physical bank routing, or macro setup/hold check in this run.
- `combinational_external_bank_model=true` describes only this timing port cut;
  it does not admit a realizable combinational SRAM.
- `address_to_bank_to_data_timing_closed=false`
- `synchronous_sram_interface=false`
- `complete_pwp_lookup_timing=false`
- `paper_ppa_ready=false`
- `system_speedup=false`
- `headline=false`

All reported area and timing numbers therefore apply only to the M82+M85 logic
island under the recorded ideal-clock, ZeroWireload, zero-macro convention.
