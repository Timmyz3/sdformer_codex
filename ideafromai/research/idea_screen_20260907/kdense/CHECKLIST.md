# K-Dense scientific-brainstorming — completion checklist

Source: `~/.grok/skills/scientific-brainstorming/SKILL.md` (v1.2) + `references/idea_evaluation.md` + `references/facilitation_workflows.md`.

| Step | Skill requirement | Where | Done |
|---|---|---|---|
| 1 Scope | focal question; purpose/audience/owner/horizon; in/out; constraints classified; knowledge + unresolved; implicated classes | `WORKFLOW.md` §1, `session.json` scope | yes |
| 2 Perspectives | represented, missing, conflicts; facilitator not offering preferred answer first | `WORKFLOW.md` §2; **deviation:** facilitator later drafted D001 | recorded |
| 3 Independent generation | private parallel; ID, statement, contributor, stage, origin, assumptions, predictions, uncertainties, sources | `session.json` ideas I001–I013 | **deviation:** single agent, examples existed |
| 4 Share without evaluation | missing / contradicts / less obvious | `../orchestra/SHARE.md` | yes (no second human) |
| 5 Cluster | explicit relation; keep IDs; merges/splits | `session.json` clusters | yes |
| 6 Criteria before scores | name, direction, anchors, weight, setter, evidence, conflicts; gates outside the sum | `criteria.json`, `WORKFLOW.md` §6 | yes |
| 6 Matrix | `evaluate_matrix.py`; decision null; intervals; weight sensitivity | `matrix.json` | after CLI |
| 7 Adversarial | full template; non-originator; response/owner/status | `ADVERSARIAL_REVIEWS.md` | yes |
| 8 Literature | query/date/sources/for/against/limits/status; verify citations | `LITERATURE_REOPEN.md` | bounded, not exhaustive |
| 8 Reopen | second private round after packet | I009–I013 | yes |
| 9 Gates | humans/animals/clinical/dual-use/unpublished/PDK | `session.json` feasibility_and_ethics_reviews | yes |
| 10 Decision log | ratings, dissent, gates, next_action vocabulary, revisit, unsigned owner | `DECISION_LOG.md` D001 | yes |
| CLI scaffold | `session_scaffold.py` | **deviation:** `build_register.py` because ideas already existed | recorded |
| CLI validate | `validate_register.py` | `validation.json` valid, 0 errors, 0 warnings, 13 ideas, 11 assumptions | yes |
| CLI matrix | `evaluate_matrix.py --weight-delta 0.10` | `matrix.json` `decision: null`; I001 87.5, I012 77.5, I002 67.5 | yes |
| Claim labels | idea / assumption / prediction / located evidence / decision | throughout | yes |
| No auto-winner | matrix decision null | `matrix.json` | yes |
| AI disclosure | responsible_ai template | `AI_DISCLOSURE.md` | yes |
| Cite skill paper | arXiv:2609.00065 fetched 2026-09-07 (v2 2 Sep 2026) | `WORKFLOW.md`, `session.json` notices | yes |

Operating-rule violations that remain visible (not laundered):

1. Independent round was not human-first and saw examples.
2. P03 is the same model as P02.
3. P01 has not signed D001.
4. Orchestra Phase 3 names a process winner; K-Dense still leaves matrix `decision` null.
