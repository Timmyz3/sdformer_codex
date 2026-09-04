# M1546 independent hammer of final M1543 streaming source

Status: **NO-GO for the single-call non-product pilot**.

The final author bytes were pinned to source `cf9a0938...`, test
`488d4ec9...`, and contract `d442e5df...`. Current Python and CPython 3.6
both compile and pass the author's 13-attack suite. Independent tests rejected
23 attack classes. The full 122-member payload seal, checkpoint identity,
upstream authorities, and resource manifest all verify.

The streaming implementation is otherwise structurally sound. It is distinct
from M1539's source, mmap's rather than materializes a plane, retires dependency
tokens per destination, and keeps one address scheduler and one nine-tile
weight cache across the call. A dynamic retirement test confirms that token
reclamation does not erase bank/port calendars, outstanding returns, cycle
state, counters, or address/commit digests. A sparse 64 MiB mmap touch did not
materialize the file in either interpreter.

## Blocking finding

`stream_actual_call()` selects canonical call 0 / sample 10 / D0 / T10, but
the importable engine below it does not enforce that boundary:

`stream_tensor(config, plane, module, call_ordinal)`

An independent synthetic attack passed `BIT_TYPED_K8`, module 2, call 7, and
a custom plane with no canonical path or SHA. The function scheduled 52
requests and returned address and commit digests. No canonical payload was
opened, and this was not an actual pilot or production run; it is a minimal
scope-bypass witness.

The minimum known repair is to reject every module/call except
`PILOT_MODULE`/`PILOT_CALL_ORDINAL` at the start of `stream_tensor`, and bind
the plane's regular-file path, exact shape, and SHA256 to
`selected_pilot_record()` before the first request. After the author re-seals
those bytes, this hammer must run again from zero. The later launch hammer must
also enforce all three non-product configurations in contract order and a
common commit digest.

Until then, no single-call pilot, production population, product
configuration, transaction/cycle/traffic number, RTL/PPA row, or paper claim
is authorized.
