#!/usr/bin/env python3.12
"""Build K-Dense session register + score matrix inputs (schema 1.1).

This script is the session record generator. It does not call a network or LLM.
The bundled skill CLIs remain the validators.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path("/home/zhumd/work/sdformer_codex/ideafromai/research/idea_screen_20260907/kdense")

SKILL_CITATION = (
    "Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). "
    "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. "
    "arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065"
)


def idea(
    iid,
    statement,
    stage,
    origin,
    assumptions,
    evid,
    status,
    predictions,
    uncertainties,
    source_refs=None,
    disconfirming=None,
):
    rec = {
        "id": iid,
        "statement": statement,
        "provenance": {
            "origin": origin,
            "contributor_ids": ["P02"],
            "recorded_stage": stage,
            "source_refs": list(source_refs or []),
            "ai_tool": "Grok-4.6 in Grok Build" if origin in {"ai-assisted", "mixed"} else None,
        },
        "assumption_ids": assumptions,
        "predicted_observations": predictions,
        "uncertainties": uncertainties,
        "disconfirming_evidence": list(disconfirming or []),
        "evidence_status": evid,
        "status": status,
    }
    if origin not in {"ai-assisted", "mixed"}:
        rec["provenance"]["ai_tool"] = None
    return rec


def assumption(aid, statement, category, status, test, owner, refs):
    return {
        "id": aid,
        "statement": statement,
        "category": category,
        "status": status,
        "test_or_check": test,
        "owner_id": owner,
        "evidence_refs": refs,
    }


def review(
    idea_id,
    strongest,
    against,
    alternatives,
    measurement_failure,
    sampling_failure,
    prior_challenges,
    harm,
    mitigation,
    residual,
    disposition,
    response,
):
    return {
        "idea_id": idea_id,
        "reviewer_id": "P03",
        "strongest_version": strongest,
        "observation_against": against,
        "alternatives": alternatives,
        "measurement_failure": measurement_failure,
        "sampling_or_generalizability_failure": sampling_failure,
        "prior_evidence_that_challenges": prior_challenges,
        "potential_harm_inequity_or_misuse": harm,
        "mitigation": mitigation,
        "residual_uncertainty": residual,
        "disposition": disposition,
        "response_owner_status": response,
    }


def lit(idea_id, queries, sources, support, challenges, limits, status):
    return {
        "idea_id": idea_id,
        "checked_on": "2026-09-07",
        "queries": queries,
        "sources_screened": sources,
        "support": support,
        "challenges": challenges,
        "search_limits": limits,
        "status": status,
        "reviewer_id": "P02",
    }


COMMON_LIMITS = [
    "English",
    "new-candidates-from-2024-01",
    "arxiv-abs-or-local-notes",
    "not-scopus-wos",
    "full-texts-not-all-read",
    "absence-is-not-novelty",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    ideas = [
        idea(
            "I001",
            "Motion-XOR 32-lane triple-popcount score leaf with K_peer shadow.",
            "independent",
            "mixed",
            ["A001", "A002"],
            "challenge-located",
            "advanced-to-review",
            [
                "Equal-resource MX3P versus FireFly-T-style AND-PopCount on the same 32-lane issue width will spend extra cycles on the temporal XOR popcount.",
                "If ep34 T0 score-leaf share is under about 2 percent, whole-net FPS delta from replacing the software Q7 with the leaf is consistent with zero at the reported precision.",
            ],
            [
                "ep34 attention score-leaf share T0 is unmeasured",
                "whether the XOR term is accuracy-necessary is untested (I003)",
            ],
            disconfirming=[
                "T0 leaf share <2 percent and I003 one-block swap holds subset AEE",
            ],
        ),
        idea(
            "I002",
            "Token-level skip of that leaf when K is zero both timesteps or Q/K are clean versus t0.",
            "independent",
            "mixed",
            ["A003"],
            "challenge-located",
            "revised",
            [
                "On packed Q/K with the ep35 schema, token leaf_needed stays in 0.4-0.6 and 15x15 window fully-clean stays under 0.15.",
                "If the Shiftmax denominator grain is the 15x15 window, lossless skip fraction equals window-clean, not token-skip.",
            ],
            [
                "RTL Shiftmax row is not yet aligned to token or 15x15",
                "ep34 packed Q/K is not on disk",
            ],
            disconfirming=[
                "M2 row dirty approximately 100 percent on ep34",
            ],
        ),
        idea(
            "I003",
            "Replace one stage-2 block Motion-XOR with published SDSA or QKFormer linear Q-K.",
            "independent",
            "literature-inspired",
            ["A004"],
            "support-located",
            "advanced-to-review",
            [
                "Frozen-weight one-block swap moves subset AEE toward public SDformerFlow 1.58 unless a 10-frame finetune pulls it back.",
                "If subset AEE stays at or below ep34 plus a small delta after 10 frames, the XOR term is not necessary for this probe.",
            ],
            [
                "Finetune budget needed to hold AEE 1.259 is unknown",
                "QKFormer is a classification kernel, not an OF kernel",
            ],
            source_refs=[
                "arxiv:2403.16552 QKFormer NeurIPS 2024",
                "arxiv:2407.15801 SDformerFlow SDSA",
            ],
            disconfirming=[
                "10-frame AEE after one-block swap exceeds 1.259 directionally toward 1.58",
            ],
        ),
        idea(
            "I004",
            "Mixed-horizon T=2/T=10 FC join: LoAS FTP as baseline, bounded temporal shared-sum as candidate.",
            "independent",
            "mixed",
            ["A005"],
            "support-located",
            "advanced-to-review",
            [
                "Equal-resource timeline versus direct FTP, including persist-Y and back-pressure, will not match the previously reported add reduction of 20.8/42.2 percent.",
                "A uniform-T=10 LoAS pack of this net is illegal or occupancy-wasteful because attention fibers are T=2.",
            ],
            [
                "Whether mixed-T packing is a large enough delta for a TCAS-II title",
                "Chen/Chang 28nm mux T=4/2/1 is a mixed-T prior",
            ],
            disconfirming=[
                "M4 equal-resource cycle ratio versus FTP at most 1.00",
            ],
        ),
        idea(
            "I005",
            "TSBG B8 same-IO weight-row broadcast on frozen ep34 FC traces.",
            "independent",
            "ai-assisted",
            ["A006"],
            "challenge-located",
            "advanced-to-review",
            [
                "Equal-resource RTL versus an LRU row buffer will not reproduce the CPU serialized 3.89x.",
                "If cycle speedup is under 1.15x and weight-byte reduction is under 30 percent, the M3 gate fails.",
            ],
            [
                "RTL versus CPU gap size",
                "owner previously judged the broadcast family unoriginal",
            ],
            disconfirming=[
                "M3 cycle ratio under 1.15 with byte reduction under 30 percent",
            ],
        ),
        idea(
            "I006",
            "Copy Prosperity product forest onto H67 conv with optional ep34-PAFT.",
            "independent",
            "mixed",
            ["A007"],
            "challenge-located",
            "deferred",
            [
                "Official-model product/bit on the H67 conv layer stays near the already measured 2.33x.",
                "Dual-port SRAM will not raise throughput if memory stall remains approximately 0.",
            ],
            [
                "Title novelty after naming Prosperity HPCA 2025",
            ],
            disconfirming=[
                "Independent novelty check after naming Prosperity fails",
            ],
        ),
        idea(
            "I007",
            "Keep ATLIF membrane inside a mixed-T island; late-BN theta packet at the boundary.",
            "independent",
            "mixed",
            ["A001"],
            "search-incomplete",
            "advanced-to-review",
            [
                "On real BN/PSN intervals, the existing A8 packet recovers membrane state with a measurable residual error.",
                "Synthetic testbench intervals overstate recovery relative to online BN.",
            ],
            [
                "Real BN interval traces are not yet attached",
                "first-mixed-T-ASIC claim is false given Chen/Chang",
            ],
            disconfirming=[
                "Recovery error on real intervals moves AEE past 1.259",
            ],
        ),
        idea(
            "I008",
            "Sparse ConvTranspose decoder island with a complete Table-A.",
            "independent",
            "ai-assisted",
            ["A002"],
            "search-incomplete",
            "deferred",
            [
                "A closed D0/D2/D3 Table-A will show ConvTranspose share comparable to the historical ~22 percent envelope.",
                "Without Table-A, any decoder speedup is not a co-design object.",
            ],
            [
                "m1681 shards are not Table-A",
            ],
            disconfirming=[
                "Closed table shows decoder share too small to justify an island",
            ],
        ),
        idea(
            "I009",
            "Byte-level SRAM permute for T x 15 x 15 attention layout, FireFly-T style.",
            "post-check",
            "literature-inspired",
            ["A008"],
            "support-located",
            "candidate",
            [
                "Mapping FireFly-T byte-write permute onto TS1N28 1RW or 2RW macros is either legal or rejected by the foundry compiler.",
            ],
            [
                "Foundry compiler legalization unknown",
            ],
            source_refs=["arxiv:2505.12771 FireFly-T IEEE TC 2026"],
            disconfirming=[
                "Compiler rejects byte-write on the chosen macro",
            ],
        ),
        idea(
            "I010",
            "Lateral-inhibition training on Motion-XOR co-silence, SpiLiFormer-style.",
            "post-check",
            "literature-inspired",
            ["A004"],
            "search-incomplete",
            "candidate",
            [
                "A same_zero regularizer changes the co-silence histogram; if AEE and fire rate do not move, the training-only face is empty for this brief.",
            ],
            [
                "SpiLiFormer is classification, not optical flow",
                "no circuit face for a 5-page hardware brief",
            ],
            source_refs=["arxiv:2503.15986 SpiLiFormer ICCV 2025"],
            disconfirming=[
                "AEE and same_zero histogram unchanged after the regularizer",
            ],
        ),
        idea(
            "I011",
            "Hardware-foldable running-BN / normalization-free inference path.",
            "post-check",
            "mixed",
            ["A009"],
            "mixed",
            "candidate",
            [
                "Running-BN inference on ep34 either meets AEE 1.259 or reproduces the PAFT-ep4 ~1.47-class failure.",
            ],
            [
                "Prior 1.47 number is PAFT-ep4 identity, not ep34",
            ],
            disconfirming=[
                "ep34 running-BN valid825 exceeds 1.259",
            ],
        ),
        idea(
            "I012",
            "Independent clock-gate of overlap, same_zero, and motion_xor popcounts when that term's support is empty.",
            "post-check",
            "mixed",
            ["A010"],
            "no-direct-evidence-located",
            "candidate",
            [
                "On the same packed Q/K, the fraction of tokens with overlap popcount exactly 0 is much larger than the fraction with same_zero popcount exactly 0 (ep35 overlap mean 0.013).",
                "If the three terms share one fused popcount tree in SV, independent enables will not reduce leaf cycles.",
            ],
            [
                "ep34 term-zero histogram does not exist",
                "datapath fusion in h67_motionxor_score_q7.sv uninspected for this claim",
            ],
            disconfirming=[
                "Software Q7 and the SV already skip empty terms, so RTL enable bits add no cycle delta",
            ],
        ),
        idea(
            "I013",
            "Encoder feature-map temporal skip (CICC DLSS analog), explicitly not Motion-XOR score skip.",
            "post-check",
            "literature-inspired",
            ["A011"],
            "support-located",
            "deferred",
            [
                "Skip rate on encoder activations will not equal attention token skip rate on the same frames.",
            ],
            [
                "Feature-map traces for H67 encoder are not in the QK pack",
            ],
            source_refs=["Zhang CICC 2026 28nm event-OF DLSS (local SOURCE_LEDGER)"],
            disconfirming=[
                "Activation skip rate equals token skip rate, collapsing the claimed split from I001",
            ],
        ),
    ]

    assumptions = [
        assumption(
            "A001",
            "Captured ATLIF outputs consumed by attention are binary {0,theta}.",
            "measurement",
            "partially-supported",
            "T5 on ep34 live capture",
            "P01",
            ["grok46-03-binary-capture-note"],
        ),
        assumption(
            "A002",
            "Attention score leaf is a small fraction of ep34 cycles.",
            "feasibility",
            "untested",
            "M0 T0 envelope on ep34",
            "P01",
            [],
        ),
        assumption(
            "A003",
            "Shiftmax denominator is over a 15x15 window so token skip is not row skip.",
            "mechanistic",
            "partially-supported",
            "M2 align RTL row; ep35 preview window dirty 90.1 percent",
            "P02",
            ["T1_T4_ep35.json"],
        ),
        assumption(
            "A004",
            "Swapping one stage-2 block can be finetuned from ep34 without collapsing AEE past 1.259.",
            "operational",
            "untested",
            "10-frame overlay M5",
            "P01",
            [],
        ),
        assumption(
            "A005",
            "T=2 attention fibers and T=10 neuron fibers cannot share a uniform-T LoAS pack.",
            "mechanistic",
            "partially-supported",
            "Compare mixed-T versus uniform T=10 FTP on same FC2",
            "P02",
            ["LoAS-MICRO-2024-uniform-T"],
        ),
        assumption(
            "A006",
            "CPU TSBG 3.89x serialized speedup will survive equal-resource RTL.",
            "feasibility",
            "challenged",
            "M3 RTL same IO",
            "P01",
            ["tsbg_ep34_same_io_result.json"],
        ),
        assumption(
            "A007",
            "Prosperity copy can be the title contribution after local modifications.",
            "value",
            "challenged",
            "Independent novelty after naming Prosperity",
            "P03",
            ["Prosperity-HPCA-2025"],
        ),
        assumption(
            "A008",
            "Byte-write SRAM helps T x 15 x 15 Motion-XOR layout on foundry 1RW/2RW macros.",
            "feasibility",
            "untested",
            "Map FireFly-T permute onto TS1N28 macros",
            "P02",
            [],
        ),
        assumption(
            "A009",
            "Running-BN inference can meet AEE 1.259 on ep34.",
            "statistical",
            "challenged",
            "Prior PAFT running-BN on a different checkpoint was 1.47",
            "P01",
            ["m247_paft_valid825"],
        ),
        assumption(
            "A010",
            "A term with popcount exactly zero can be clock-gated losslessly and independently of the other two Motion-XOR terms.",
            "mechanistic",
            "untested",
            "Histogram exact-zero overlap/same_zero/motion on ep34; inspect SV fusion",
            "P02",
            ["T1_T4_ep35.json"],
        ),
        assumption(
            "A011",
            "Encoder feature-map temporal skip is a different co-design object from Motion-XOR token skip.",
            "causal",
            "untested",
            "Compare skip rates on the same frames for activations versus Q/K tokens",
            "P03",
            ["SOURCE_LEDGER Zhang CICC 2026"],
        ),
    ]

    session = {
        "schema_version": "1.1",
        "session": {
            "id": "tcasii-of-snn-20260907",
            "title": "TCAS-II co-design object for H67 event optical-flow SNN transformer",
            "question": (
                "Which hardware-software co-design object can be a TCAS-II brief: "
                "spike-driven Q-K optical flow, AEE at most 1.259, circuit face measurable on TSMC 28HPC+?"
            ),
            "date": "2026-09-07",
            "decision_owner_id": "P01",
            "facilitator_id": "P02",
            "participant_ids": ["P01", "P02", "P03"],
            "represented_perspectives": [
                "domain-event-optical-flow",
                "implementation-rtl-28nm",
                "algorithm-snn-transformer",
                "statistics-aee-gates",
                "adversarial-review",
            ],
            "missing_perspectives": [
                "external-circuits-editor",
                "lived-experience-driver",
            ],
            "conflicts_or_power_dynamics": [
                "P02 originated most ideas; P03 is a same-model non-originator role, not an independent lab.",
                "P01 forbade picking a paper title before screening; D001 selects a measurement next action only.",
                "Facilitator drafted D001; owner has not signed.",
            ],
            "examples_shown_before_independent_generation": [
                "grill-me contract",
                "32 mechanism cards",
                "named 2024-2026 priors",
                "ep35 QK census",
            ],
            "skill_citation": SKILL_CITATION,
        },
        "scope": {
            "purpose": "Choose the next measurement, not a paper title.",
            "audience": "P01 and the Codex implementation agent",
            "time_horizon": "2-week measurement pilot; TCAS-II date unknown",
            "in_scope": [
                "SNN transformer event optical flow",
                "spike-driven Q-K attention",
                "TSMC 28HPC+ digital islands",
                "C1/C2 retry, new islands, network swap",
            ],
            "out_of_scope": [
                "analog CIM as PPA headline",
                "replacing spike attention with pooling",
                "ISCAS parallel submission",
                "implementation in this agent",
            ],
            "constraints": [
                {"statement": "valid825 AEE <= 1.259 and better than 1.5848", "classification": "real"},
                {"statement": "must remain SNN Transformer plus spike-driven Q-K", "classification": "real"},
                {"statement": "TSMC 28HPC+ is the PPA node", "classification": "real"},
                {"statement": "one A800 for a full valid825", "classification": "real"},
                {"statement": "1RW friendliness is not required", "classification": "real"},
                {"statement": "no kill-list auto-drop of old C1/C2", "classification": "real"},
                {"statement": "one TCAS-II object with algorithm and circuit faces", "classification": "real"},
                {"statement": "ZCU102 is bonus not blocking", "classification": "negotiable"},
                {"statement": "score-leaf share under about 2 percent means I001 is not the circuit face", "classification": "assumed"},
                {"statement": "ep35 QK census approximates ep34", "classification": "assumed"},
                {"statement": "historical attention share ~0.59 percent still holds on ep34", "classification": "unknown"},
            ],
            "current_knowledge": [
                "ep34 AEE 1.199514 SHA 4bbaf7fc",
                "Motion-XOR formula in h67_motionxor_score_q7.sv",
                "ep35 QK census token skip ~56 percent, window-clean ~9.9 percent",
                "Prosperity outer product/bit 2.33x, mem stall ~0",
                "TSBG CPU B8 3.89x serialized",
                "PAFT-ep4 AEE ~1.47",
            ],
            "unresolved_observations": [
                "ep34 operator share T0",
                "ep34 packed Q/K missing",
                "Shiftmax row grain",
                "whether XOR term is required for AEE",
            ],
            "implicated_classes": {
                "human_subjects": False,
                "animals": False,
                "clinical_care": False,
                "pathogens": False,
                "environmental_release": False,
                "unpublished_traces": True,
                "foundry_pdk": "institutional-not-this-session",
            },
            "prohibited_outputs": [
                "multiplied component speedups",
                "ep35 census pasted as ep34 tables",
                "docs/359 edits",
            ],
        },
        "information_governance": {
            "classification": "unpublished-research-notes",
            "approved_record_location": str(OUT),
            "contains_personal_or_sensitive_data": False,
            "contains_unpublished_or_proprietary_data": True,
            "contains_controlled_or_security_sensitive_data": False,
            "external_ai_use_permitted": "session-already-used-grok-abstracted-mechanisms-only",
        },
        "workflow": [
            {
                "order": 1,
                "stage": "scope",
                "status": "completed",
                "method_or_deviation": "Grill-me contract copied; constraints classified real/assumed/negotiable/unknown.",
            },
            {
                "order": 2,
                "stage": "independent-generation",
                "status": "completed",
                "method_or_deviation": "Single-agent stand-in; examples existed (grill, 32 cards, priors). Origin labeled. Predictions are idea-specific in this rebuild.",
            },
            {
                "order": 3,
                "stage": "structured-sharing",
                "status": "completed",
                "method_or_deviation": "No second human. SHARE.md records missing/contradiction/less-obvious.",
            },
            {
                "order": 4,
                "stage": "clustering",
                "status": "completed",
                "method_or_deviation": "Declared relations; I002 merged into I001; I013 split from I001.",
            },
            {
                "order": 5,
                "stage": "criteria-definition",
                "status": "completed",
                "method_or_deviation": "criteria.json frozen before scores.csv; weights proposed by P02, unsigned by P01.",
            },
            {
                "order": 6,
                "stage": "independent-evaluation",
                "status": "completed",
                "method_or_deviation": "evaluate_matrix.py; decision left null. Single rater P02.",
            },
            {
                "order": 7,
                "stage": "adversarial-review",
                "status": "completed",
                "method_or_deviation": "P03 same-model role; full templates for I001-I005, I007, I012.",
            },
            {
                "order": 8,
                "stage": "literature-check",
                "status": "completed",
                "method_or_deviation": "Bounded 2024+ search then reopen I009-I013. Not Web of Science.",
            },
            {
                "order": 9,
                "stage": "feasibility-and-ethics-gate",
                "status": "completed",
                "method_or_deviation": "Digital 28nm OF accelerator; unpublished traces local; no human subjects.",
            },
            {
                "order": 10,
                "stage": "decision-log",
                "status": "completed",
                "method_or_deviation": "D001 proposed by P02, unsigned by P01. next_action simulation M0-M2.",
            },
        ],
        "ideas": ideas,
        "assumptions": assumptions,
        "clusters": [
            {
                "id": "C-attn",
                "relation": "shared-mechanism spike-score-leaf",
                "idea_ids": ["I001", "I002", "I003", "I009", "I010", "I012"],
                "merges": ["I002-revised-into-I001-as-enable"],
            },
            {
                "id": "C-fc",
                "relation": "shared-outcome FC-join-or-broadcast",
                "idea_ids": ["I004", "I005"],
            },
            {
                "id": "C-conv",
                "relation": "shared-outcome conv-product-reuse-or-feature-map-skip",
                "idea_ids": ["I006", "I013"],
                "splits": ["I013-split-from-I001-so-CICC-analog-cannot-hide"],
            },
            {
                "id": "C-neuron",
                "relation": "shared-scale mixed-T-neuron-boundary",
                "idea_ids": ["I007", "I011"],
            },
            {
                "id": "C-dec",
                "relation": "shared-outcome decoder-table",
                "idea_ids": ["I008"],
            },
        ],
        "criteria": [
            {
                "name": "information_gain",
                "weight": 3,
                "direction": "higher",
                "minimum": 1,
                "maximum": 5,
                "set_by": "P02-unsigned-by-P01",
            },
            {
                "name": "originality_vs_search",
                "weight": 3,
                "direction": "higher",
                "minimum": 1,
                "maximum": 5,
                "set_by": "P02-unsigned-by-P01",
            },
            {
                "name": "feasibility",
                "weight": 2,
                "direction": "higher",
                "minimum": 1,
                "maximum": 5,
                "set_by": "P02-unsigned-by-P01",
            },
            {
                "name": "aee_risk",
                "weight": 2,
                "direction": "lower",
                "minimum": 1,
                "maximum": 5,
                "set_by": "P02-unsigned-by-P01",
            },
        ],
        "adversarial_reviews": [
            review(
                "I001",
                "Keep H67 Q7 formula; 32-lane leaf; K_peer t0 shadow; equal-resource vs AND-PopCount; no system FPS until T0.",
                "T0 score-leaf share under about 2 percent and I003 one-block swap holds subset AEE.",
                [
                    "AND-PopCount plus a cheap extra XOR",
                    "QKFormer linear Q-K matches AEE",
                    "editors read any popcount leaf as FireFly-T",
                ],
                "ep35 census labeled ep34; leaf Hz as network FPS; omitting Q/K projection from attention share.",
                "100-sample ep35 path; S3 vs S1 occupancy differs by ~15x.",
                [
                    "FireFly-T AND-PopCount arXiv:2505.12771",
                    "alpha-XNOR CVPR 2025",
                    "Bishop AAC ISCA 2025",
                    "historical attention ~0.59 percent",
                ],
                "unpublished traces if leaked; no clinical harm",
                "T0 first; name priors; isolate leaf benches; traces local",
                "T0 unknown; XOR necessity unknown",
                "retain",
                "Accept T0 kill-switch. Owner P01. Open until M0.",
            ),
            review(
                "I002",
                "Lossless per-token enable when K is zero both T or Q/K equal t0; no 15x15 memo.",
                "M2 denominator grain ~100 percent dirty (ep35 window dirty 90.1 percent, S3 clean 0).",
                ["energy gating only", "no skip", "term-wise overlap skip I012"],
                "token skip written as Shiftmax skip; 96 percent single-sample skip",
                "ep35 is not ep34; S1 empty would overstate skip without per-stage tables",
                ["DeltaCNN/CBinfer pixel skip", "Zhang CICC 2026 feature-map skip"],
                "silent AEE regression if skip is approximate",
                "merge into I001 as enable; lossless-only unless AEE gated",
                "true RTL row definition",
                "revise",
                "Merged into I001. Owner P02. Closed as title; open as M1/M2.",
            ),
            review(
                "I003",
                "Frozen-weight swap of one stage-2 block; 10-frame finetune only if AEE explodes.",
                "Subset AEE moves toward 1.58 and stays after 10 frames.",
                ["keep Motion-XOR", "swap all 12 blocks (forbidden first)", "SpikePool out of scope"],
                "12-block swap; quoting 1.259 on a 10-frame subset",
                "one sequence is not valid825; ImageNet is not DSEC",
                ["arxiv:2403.16552", "arxiv:2407.15801", "SDT V2"],
                "wasted A800 if before T0",
                "one block; subset first; stop if direction is up",
                "finetune budget unknown",
                "retain",
                "Accept as cheapest algo probe. Owner P01. Optional parallel with M0.",
            ),
            review(
                "I004",
                "LoAS FTP is baseline; mixed-horizon packing plus bounded shared-sum; equal Y-port.",
                "Equal-resource timeline vs FTP at most 1.00x, or reviewer writes LoAS plus ATLIF theta.",
                ["direct FTP", "RSR++", "uniform T=10 LoAS"],
                "add reduction reported as cycle speedup; missing persist-Y",
                "one FC2 capture",
                ["arxiv:2407.14073 LoAS", "Chen/Chang arXiv:2503.19643"],
                "unpublished RTL only",
                "pre-declare mixed-T as the only novelty; M4 equal-resource",
                "whether mixed-T is enough for TCAS-II",
                "retain",
                "Contingent circuit face if T0 fails I001. Owner P01. Dormant until T0.",
            ),
            review(
                "I005",
                "Same-IO B8 vs LRU; cycle >=1.15x or <=5 percent slower with >=30 percent fewer weight bytes.",
                "Equal-resource RTL under 1.15x, or reviewers reuse the unoriginal-broadcast judgment.",
                ["LRU row buffer", "do nothing", "train for more row reuse"],
                "CPU 3.89x as VCS",
                "frozen ep34 traces; new sparsity changes reuse",
                ["Eyeriss", "Gustavson", "TSBG family"],
                "none",
                "comparator-only; M3 gate",
                "RTL vs CPU gap",
                "retain",
                "Comparator not title. Owner P01. M3 after M0 if resources exist.",
            ),
            review(
                "I007",
                "Membrane stays in-island; theta packet at BN boundary on real BN/PSN intervals.",
                "Real intervals show recovery error that moves AEE, or reviewers cite Chen/Chang mixed-T 28nm.",
                ["freeze theta", "move BN earlier", "always wait"],
                "synthetic TB intervals as online BN",
                "one BN group is not all stages",
                ["Gist ISCA 2018", "arXiv:2503.19643"],
                "none",
                "M7 real intervals; do not claim first mixed-T ASIC",
                "real recovery rate",
                "retain",
                "Fifth/contingent. Owner P01. After M0 unless BN is critical path.",
            ),
            review(
                "I012",
                "Clock-gate each of three popcounts when that term's support is empty.",
                "Q7 already short-circuits empty terms, or the three terms share one fused tree.",
                ["only K=0 skip", "only overlap skip", "fuse all three"],
                "mean overlap 0.013 cited as 99 percent skip without a zero histogram",
                "ep35 preview",
                ["ordinary sparse popcount ALU"],
                "none",
                "ep34 exact-zero histogram; then RTL enable bits",
                "SV datapath fusion",
                "retain",
                "Hang on I001 as M1 add-on. Owner P02. Not a separate title.",
            ),
        ],
        "literature_checks": [
            lit(
                "I001",
                ["FireFly-T AND-PopCount", "alpha-XNOR SSA", "Bishop AAC", "Motion-XOR optical flow ASIC"],
                ["arxiv:2505.12771", "CVPR 2025 alpha-XNOR", "arxiv:2505.12281", "h67_motionxor_score_q7.sv"],
                ["No published temporal-peer K XOR popcount ASIC located in this search"],
                ["AND-PopCount and co-silence exist separately", "historical attention ~0.59 percent"],
                COMMON_LIMITS,
                "challenge-located",
            ),
            lit(
                "I002",
                ["DeltaCNN skip", "CBinfer", "event OF DLSS CICC 2026"],
                ["DeltaCNN", "Zhang CICC 2026 local SOURCE_LEDGER"],
                ["Token skip is a known idea"],
                ["Priors skip pixels or feature maps", "ep35 window-clean 9.9 percent"],
                COMMON_LIMITS,
                "challenge-located",
            ),
            lit(
                "I003",
                ["QKFormer", "SDformerFlow SDSA", "Spike-driven Transformer"],
                ["arxiv:2403.16552", "arxiv:2407.15801", "SDT V2 ICLR 2024"],
                ["Public linear and SDSA kernels exist to swap"],
                ["Not OF-native", "AEE may regress toward 1.58"],
                COMMON_LIMITS,
                "support-located",
            ),
            lit(
                "I004",
                ["LoAS FTP MICRO 2024", "RSR++", "mixed-T SNN 28nm"],
                ["arxiv:2407.14073", "arxiv:2503.19643"],
                ["Mixed T=2 vs T=10 is a local delta versus uniform-T LoAS"],
                ["FTP inner-join is a strong prior", "28nm mux T already exists"],
                COMMON_LIMITS,
                "support-located",
            ),
            lit(
                "I005",
                ["TSBG", "Eyeriss", "Gustavson"],
                ["owner prior judgment", "tsbg_ep34_same_io_result.json"],
                ["Broadcast reduces weight bytes on these CPU traces"],
                ["Family is old", "CPU is not RTL"],
                COMMON_LIMITS,
                "challenge-located",
            ),
            lit(
                "I006",
                ["Prosperity HPCA 2025", "SumMerge"],
                ["Prosperity-HPCA-2025", "ICS 2021 SumMerge"],
                ["Local product/bit 2.33x"],
                ["Title novelty challenged by the prior itself"],
                COMMON_LIMITS,
                "challenge-located",
            ),
            lit(
                "I007",
                ["Gist ISCA 2018", "late BN SNN hardware"],
                ["Gist 2018", "A8 RTL notes"],
                ["Packet-at-boundary has local RTL"],
                ["Mixed-T neuron 28nm already published"],
                COMMON_LIMITS,
                "search-incomplete",
            ),
            lit(
                "I008",
                ["sparse ConvTranspose SNN"],
                ["arxiv:2408.15578 FireFly-S", "m1681 shards"],
                ["Decoder share historically large"],
                ["No complete Table-A in this repo"],
                COMMON_LIMITS,
                "search-incomplete",
            ),
            lit(
                "I009",
                ["FireFly-T SRAM permute"],
                ["arxiv:2505.12771"],
                ["Byte-write 3D attention layout described"],
                ["Foundry 1RW/2RW map untested"],
                COMMON_LIMITS,
                "support-located",
            ),
            lit(
                "I010",
                ["SpiLiFormer lateral inhibition"],
                ["arxiv:2503.15986", "ICCV 2025 pp.24539-24548"],
                ["Training analog exists"],
                ["Classification not OF", "not an ALU"],
                COMMON_LIMITS,
                "search-incomplete",
            ),
            lit(
                "I011",
                ["PAFT running-BN", "NF-SpikingVTG"],
                ["m247_paft_valid825", "NF-SpikingVTG name"],
                ["Running versus frozen BN split located locally"],
                ["PAFT-ep4 AEE ~1.47 fails 1.259"],
                COMMON_LIMITS,
                "mixed",
            ),
            lit(
                "I012",
                ["sparse popcount zero-term skip"],
                ["T1_T4_ep35.json"],
                ["ep35 overlap mean 0.013 suggests overlap often empty"],
                ["Ordinary zero-skip ALU", "may already be in SV"],
                COMMON_LIMITS + ["no-ep34-term-histogram"],
                "no-direct-evidence-located",
            ),
            lit(
                "I013",
                ["Zhang CICC 2026 DLSS", "event OF 28nm"],
                ["SOURCE_LEDGER Zhang CICC 2026"],
                ["Feature-map temporal skip exists as a published 28nm OF idea"],
                ["Different object from I001"],
                COMMON_LIMITS,
                "support-located",
            ),
        ],
        "feasibility_and_ethics_reviews": [
            {
                "scope": "all-ideas",
                "human_subjects": "not-applicable",
                "animals": "not-applicable",
                "clinical": "not-applicable",
                "pathogens_or_environmental_release": "not-applicable",
                "biosafety_or_dual_use": "not-applicable",
                "regulatory": "not-assessed",
                "export_or_pdk": "institutional",
                "unpublished_traces": "keep-local",
                "nih_sabv": "not-applicable",
                "feasibility_of_next_action": "pass-for-ideation-M0-M2-are-measurements",
                "status": "review-not-required-for-ideation",
            }
        ],
        "decision_log": [
            {
                "decision_id": "D001",
                "date": "2026-09-07",
                "owner_id": "P01",
                "facilitator_draft": True,
                "owner_signed": False,
                "candidate_ids": ["I001", "I002", "I003", "I004", "I005", "I007"],
                "decision": (
                    "Advance I001 including I002 as one score-leaf object to measurement protocol M0-M2, "
                    "with I012 as a census add-on. If T0 shows attention score leaf under about 2 percent, "
                    "switch circuit-face work to I004. I003 is the cheapest algorithm-face probe. "
                    "I005 remains a comparator not a title default. I007 stays fifth. Owner P01 may override."
                ),
                "rationale": (
                    "Orchestra Phase 3 requires one sharpened plan. I001 is algorithm-native, has RTL, "
                    "and T1 already forbids dishonest skip claims. The numeric matrix is a decision aid only "
                    "and its decision field remains null. Next action is simulation, not protocol development."
                ),
                "dissent": [
                    "P03: I001 numeric lead is conditional on T0 and on XOR being accuracy-necessary.",
                    "P01: not yet recorded.",
                ],
                "uncertainties": [
                    "T0 unmeasured",
                    "no ep34 QK pack",
                    "2 percent threshold is an assumed gate",
                    "single rater",
                    "same-model adversary",
                ],
                "gate_status": {
                    "ethics": "not-applicable",
                    "biosafety_or_dual_use": "not-applicable",
                    "regulatory": "not-assessed",
                },
                "rejected_or_deferred": [
                    "I006 Prosperity is the prior",
                    "I008 no Table-A",
                    "I009 macro unknown",
                    "I010 training-only",
                    "I011 PAFT 1.47 identity",
                    "I013 different object",
                ],
                "next_action": "simulation",
                "revisit_when": (
                    "ep34 T0 envelope available, or P01 files dissent, or I003 10-frame AEE exists, "
                    "or 14 days after this log with no M0"
                ),
            }
        ],
        "notices": [
            "Ideas and scores are not evidence or scientific conclusions.",
            "Clinical, ethics, biosafety, dual-use, regulatory, and institutional review remain separate.",
            "Matrix decision is null; D001 is a facilitator-drafted next action pending P01.",
            SKILL_CITATION,
        ],
    }
    (OUT / "session.json").write_text(json.dumps(session, indent=2) + "\n")

    criteria = {
        "schema_version": "1.0",
        "weight_set_by": "P02 facilitator, unsigned by P01",
        "frozen_before_scores": True,
        "noncompensatory_gates_outside_formula": [
            "AEE <= 1.259 on valid825",
            "spike-driven Q-K identity",
            "ethics/unpublished-trace handling",
        ],
        "criteria": [
            {
                "name": "information_gain",
                "description": "Whether M0-M2 or a 10-frame AEE would discriminate this object from named priors",
                "weight": 3,
                "direction": "higher",
                "minimum": 1,
                "maximum": 5,
                "anchor_min": "no discriminator this month",
                "anchor_mid": "one named measurement splits this object from its prior",
                "anchor_max": "M0-M2 or 10-frame AEE would decide keep/kill",
                "evidence_needed": "named measurement-contract id or overlay",
            },
            {
                "name": "originality_vs_search",
                "description": "Residual novelty after named 2024-2026 priors",
                "weight": 3,
                "direction": "higher",
                "minimum": 1,
                "maximum": 5,
                "anchor_min": "named prior is the idea",
                "anchor_mid": "local modification of a named prior",
                "anchor_max": "no direct mechanism located in this bounded search",
                "evidence_needed": "SOURCE_LEDGER row",
            },
            {
                "name": "feasibility",
                "description": "Can start from existing RTL, traces, or a 10-frame overlay",
                "weight": 2,
                "direction": "higher",
                "minimum": 1,
                "maximum": 5,
                "anchor_min": "needs new traces and A800 and RTL",
                "anchor_mid": "existing capture or 10-frame overlay",
                "anchor_max": "existing SV or census script",
                "evidence_needed": "path on disk",
            },
            {
                "name": "aee_risk",
                "description": "Risk of valid825 AEE exceeding 1.259",
                "weight": 2,
                "direction": "lower",
                "minimum": 1,
                "maximum": 5,
                "anchor_min": "lossless on frozen net",
                "anchor_mid": "one-block finetune",
                "anchor_max": "prior run already missed 1.259",
                "evidence_needed": "identity of any cited AEE number",
            },
        ],
    }
    (OUT / "criteria.json").write_text(json.dumps(criteria, indent=2) + "\n")

    rows = [
        ("I001", 5, 4, 5, 4, 3, 5, 5, 4, 5, 2, 1, 3, "Algorithm-native leaf; T0 unknown", "ep34 share unknown", "challenge-located", "not-applicable"),
        ("I002", 4, 3, 5, 3, 2, 4, 4, 3, 5, 2, 1, 3, "Token skip real; window skip not", "row grain unaligned", "challenge-located", "not-applicable"),
        ("I003", 4, 3, 5, 3, 2, 4, 3, 2, 4, 4, 3, 5, "Cheapest algo face", "AEE regression", "support-located", "not-applicable"),
        ("I004", 4, 3, 5, 2, 2, 3, 4, 3, 5, 2, 1, 3, "Big-share circuit face", "LoAS prior", "support-located", "not-applicable"),
        ("I005", 3, 2, 4, 1, 1, 2, 5, 4, 5, 2, 1, 3, "Largest CPU number", "unoriginal broadcast", "challenge-located", "not-applicable"),
        ("I006", 2, 1, 3, 1, 1, 2, 4, 3, 5, 2, 1, 4, "Prosperity copy", "title novelty", "challenge-located", "not-applicable"),
        ("I007", 3, 2, 4, 3, 2, 4, 4, 3, 5, 2, 1, 3, "Neuron island", "not first mixed-T", "search-incomplete", "not-applicable"),
        ("I008", 3, 2, 4, 2, 1, 3, 2, 1, 3, 2, 1, 3, "Decoder share", "no Table-A", "search-incomplete", "not-applicable"),
        ("I009", 3, 2, 4, 2, 2, 3, 2, 1, 3, 2, 1, 3, "Layout steal", "macro mapping", "support-located", "not-applicable"),
        ("I010", 3, 2, 4, 3, 2, 4, 2, 1, 3, 3, 2, 4, "Training-only face", "not a circuit", "search-incomplete", "not-applicable"),
        ("I011", 3, 2, 4, 2, 1, 3, 3, 2, 4, 4, 3, 5, "BN identity", "1.47 prior", "mixed", "not-applicable"),
        ("I012", 4, 3, 5, 3, 2, 4, 5, 4, 5, 1, 1, 2, "Term-wise skip hangs on I001", "SV fusion unknown", "no-direct-evidence-located", "not-applicable"),
        ("I013", 3, 2, 4, 2, 1, 3, 2, 1, 3, 3, 2, 4, "CICC analog; different object", "no activation traces", "support-located", "not-applicable"),
    ]
    with (OUT / "scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idea_id",
                "information_gain",
                "information_gain_low",
                "information_gain_high",
                "originality_vs_search",
                "originality_vs_search_low",
                "originality_vs_search_high",
                "feasibility",
                "feasibility_low",
                "feasibility_high",
                "aee_risk",
                "aee_risk_low",
                "aee_risk_high",
                "qualitative_review",
                "uncertainties",
                "evidence_status",
                "ethics_status",
            ]
        )
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
