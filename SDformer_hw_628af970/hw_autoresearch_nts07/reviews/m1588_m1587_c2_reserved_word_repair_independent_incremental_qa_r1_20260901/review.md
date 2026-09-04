# M1588 — M1587 C2 reserved-word repair incremental QA

Decision: **PASS incremental source QA; authorize exactly one new-identity VCS
compile and one `k8_case0` simulation.** M1588 ran no VCS, `simv`, or EDA tool
and consumed no attempt.

The consumed M1586 compile log contains one and only one compiler error:
`Error-[SE]` at testbench line 350, whose token is the SystemVerilog reserved
word `tri`. It did not reach DUT elaboration and produced no simulation.

The repaired testbench changes the identifier `tri` to `tri_state_char` at 24
locations: 21 declaration/call occurrences and three function assignments.
Replacing the new identifier with the old one reconstructs the frozen M1578
testbench byte for byte. Consequently the DUT pair, ordered filelist, hard-wired
M979 K8 case0 stimulus, two independent memory fabrics, event schedule, and
four-state `0/1/X` reporting are unchanged. The RTL wrapper, mapped netlist and
memory model hashes also remain frozen.

The existing Python suite passes 9/9. The independent conservative source
parser rejects 10/10 robustness mutations under CPython 3.10.18 and 3.6.8.
Verilator was detected but intentionally not invoked because this review is
limited to Python and static source analysis.

The future run must use a new result identity, the frozen filelist, and explicit
`-top tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault`. It may consume exactly
one compile and one `k8_case0` simulation. Reuse of the failed M1586 or M1502
binary, UCLI, initreg, force/release, SAIF, PTPX, a second compile, or a second
simulation is not authorized. No RTL/mapped PASS or paper claim exists until
the future run is independently reviewed.
