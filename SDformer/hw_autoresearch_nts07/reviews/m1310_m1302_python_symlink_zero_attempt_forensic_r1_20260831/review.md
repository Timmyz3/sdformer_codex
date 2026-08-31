# M1310 zero-attempt forensic: M1302 Python path gate

M1302 exited before either attempt was consumed because its generic regular-file
checker rejects `/usr/bin/python3` when the logical path is a symlink.  The host
has a three-link exact chain ending at the regular executable
`/usr/libexec/platform-python3.6`; the resolved entity SHA is the same
`9c9502...` already pinned by M1302.

All M1288 and M1302 canonical, work and attempt namespaces were absent when
observed.  The root handoff reported resources normal and PrimeTime licenses
99 issued / 0 in use; this forensic did not repeat that license query.

The repair must be additive.  It must preserve M1302 timing, resource, license,
freshness, one-shot and claim semantics while replacing only the Python path
predicate with an exact symlink-chain check plus a regular executable resolved
entity and SHA check.  Target swap, dangling link, nonregular entity and SHA
drift must fail before attempt consumption.
