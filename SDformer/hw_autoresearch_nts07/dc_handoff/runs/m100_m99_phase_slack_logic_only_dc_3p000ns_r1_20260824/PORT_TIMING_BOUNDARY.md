# M100 port timing boundary

M100 uses exactly the M97 external bank port cut. It is a logic-island result,
not complete PWP frontend timing.

- `bank_row_addresses[79:0]` end at output ports before an excluded external
  eight-bank memory.
- `bank_words[255:0]` start at input ports with a uniform 0.250 ns input delay.
- No address-to-SRAM-to-data arc, macro clock-to-Q, decoder, bank routing, SRAM
  setup/hold, macro area, or SRAM energy is present.
- `combinational_external_bank_model=true`
- `address_to_bank_to_data_timing_closed=false`
- `synchronous_sram_interface=false`
- `complete_pwp_lookup_timing=false`
- `paper_ppa_ready=false`
- `system_speedup=false`
- `headline=false`

The reported M97/M100 ratios are admissible only as same-backend, same-library,
same-constraint, same-port-cut, zero-macro standard-cell comparisons.
