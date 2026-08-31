# M473 online subset + live parent scratch

Status: `PASS_M473_CPU_DSE_NO_GO`. Best feasible 128 B/cycle CPU point: row_tile=64, banks=8, CAM=64, cycles=389,974,420, same-coordinate product/bit=1.943581x, vs best same-budget M468 zero=1.949744x. Matching unfused-sync upper=746,979,771 cycles.

CPU DSE only. CAM/scheduler/1R1W scratch are not physicalized; performance, RTL, PPA, energy, system and headline remain false.
