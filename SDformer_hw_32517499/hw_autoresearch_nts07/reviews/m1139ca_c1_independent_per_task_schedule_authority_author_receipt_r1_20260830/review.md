# M1139CA independent per-task schedule authority — author review

Status: `PASS_M1139CA_INDEPENDENT_PER_TASK_SCHEDULE_AUTHORITY_AUTHOR__BOUNDED_ONLY_PRODUCTION_STOP`.

`requested_cycle_first` can be independently reconstructed, but only from the frozen M410 task bytes together with the frozen M1016 per-design preprocess/work derivation and recurrence. M1102 alone is insufficient because it retains only one shared maximum preprocess value.

The independently checked recurrence is: first `start=preprocess`; later `start=previous_start+max(previous_work, preprocess)+2`; `requested_cycle_first=sample_global_offset+start-preprocess`; the sample offset advances by `last_start+last_work+2+96000`.

The bounded two-task, three-axis oracle emitted six records and matched candidate `[0,22]`, strongest-zero `[0,12]`, and same-coordinate-bit `[0,14]`. Missing, duplicate and reordered tasks; cycle regression; wrong coordinates, provenance and axis order; first/middle sink failures; and absent production release were rejected. The authority retains O(axes) state and resumes the failed axis without replaying prior committed axes.

Production remains fail-closed before opening M410: zero production rows and records, no digest compiler, real driver, canonical/full run, EDA, GPU, or remote work. Only a different-author bounded hammer is authorized next.
