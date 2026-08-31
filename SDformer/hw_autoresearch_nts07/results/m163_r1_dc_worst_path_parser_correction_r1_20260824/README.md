# M163 r1 DC worst-path parser correction

M163 r1 is not a passing DC milestone.  Design Compiler completed and reported
no setup/hold violation, but the exact-SHA runner exited 41 because the real
worst setup slack is only `0.0002 ns`, below the predeclared `0.05 ns` margin.
The run directory is fail-closed and must not be cited as a successful result.

The earlier scratch note of `0.0765 ns` came from the last path visible in a
tailed timing report.  Synopsys reports the worst path first.  The corrected
worst path is:

- startpoint: `tile_data[0][0][3]`
- endpoint: `moment_sum_q_reg[47]/D`
- setup slack: `0.0002 ns`
- hold slack: `0.0000 ns`
- logic levels: `88`

Area (`41,749.973937 um2`), cell count (`47,432`), sequential cells (`6,055`)
and zero macro count are diagnostic observations from the rejected run, not a
paper-PPA admission.  The successor must balance both 32-input moment trees and
repeat VCS/DC without weakening the timing-margin guard.
