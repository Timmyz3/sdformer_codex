# M1162 source-only protocol verification plan

This plan is authored but **not executed** in M1162.  A fresh different-author
source hammer is required before VCS is authorized.

## Frozen environment rules

- Each request sink may make `ready` depend on its own `valid`; request
  `valid` must therefore be independent of every `ready`.
- A response may assert no earlier than the cycle after its own request
  handshake.  Once asserted, response `valid` and payload remain stable until
  `ready`.
- Weight and psum services are in order.  At most one wrapper transaction is
  outstanding.  First beats need one weight and one psum request/response;
  later beats need only weight.
- Reset cancels the outstanding transaction.  Services discard canceled work;
  a response after reset release without a new accepted request is spurious.

## Required assertions

1. Weight and psum request payloads hold while `valid && !ready`.
2. Acceptance of either request suppresses only that request; the peer remains
   valid with the same tuple until independently accepted.
3. Neither accepted request can fire twice in one transaction.
4. Core issue-data valid requires all required requests accepted and all
   required responses present.
5. Response consumption is atomic on a first beat; neither response ready may
   consume a lone response.
6. Core backpressure leaves both response-ready outputs low and relies on the
   frozen response-hold rule.
7. Request cancellation or tuple mutation, response-before-own-request,
   psum response on a non-first beat, and post-reset spurious response set the
   sticky boundary error.
8. Reset clears all request/accept state and the sticky error.
9. Zero-stall, one-cycle-response completed-beat II is exactly 2; this is a
   protocol coordinate, not a performance claim.

## Directed covers/attacks

- weight accepted first, psum stalled at least four cycles;
- psum accepted first, weight stalled at least four cycles;
- weight response arrives while the peer request is still stalled;
- psum response arrives first after both requests accept;
- both responses held for at least four cycles by core backpressure;
- non-first beat with no psum request;
- reset in each of request-partial, request-complete and response-skew states;
- same-cycle request/response, unsolicited weight response, unsolicited psum
  response, canceled M935 request, tuple mutation and duplicate-request attack.

The future VCS receipt must report nonzero cover counts for both partial-order
cases, both response-order cases, long request stall, long response
backpressure, reset-pending, spurious response, sticky error and the II=2
zero-stall path.  It must not cite any M1114/M528 speedup.

