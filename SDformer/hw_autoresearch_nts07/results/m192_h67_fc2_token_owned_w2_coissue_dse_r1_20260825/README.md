# M192 H67 FC2 token-owned W2 co-issue

M192 preserves the token owner of all 6,523,707 FC2 windows and corrects the
M191 context ambiguity.  On all 120 frozen H67 ep35 FC2 payloads, sequential
W1 replay takes 79,397,844 cycles.  Combining only adjacent windows belonging
to the same token takes 75,099,527 cycles, an exact replay opportunity of
1.057234941x, without a second Acc24 context.  Ideal cross-token W2 reaches
71,233,088 cycles (1.114620273x) but still needs dual token-owned updates.

There are 3,261,820 full W2 batches: 1,417,458 are same-token and 1,844,362
are cross-token.  The first RTL candidate is therefore a bounded same-token
pair fusion frontend; cross-token pairs fall back to W1.  This result does not
include finite fill/waiting, weight SRAM latency, RTL timing/area, complete
FC2, FFN or system cycles and is not a headline speedup.
