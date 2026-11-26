#!/bin/bash
cd ~/work/sdformer_codex/research/flow_story_20261026/c1_prosperity_denominator
export CUDA_VISIBLE_DEVICES=
P=/opt/anaconda3/envs/pytorch310_cpu/bin/python
echo "start $(date)" > launch.log
$P run_c1_prosperity.py verify > verify.log 2>&1 || { echo "verify failed $(date)" >> launch.log; exit 1; }
$P run_c1_prosperity.py run --workers 32 > run.log 2>&1
echo "end rc=$? $(date)" >> launch.log
