#!/usr/bin/env bash
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
cd "$REPO"

exec bash neuron_experiments/H9_bipolar_self_attention/entrypoints/launch_mvsec_eval_direct.sh