#!/usr/bin/env bash
# T11e: 逐篇拉 arXiv 全摘要（带重试退避，绕共享代理限流）
set -u
OUT=t11_triage/abstracts
mkdir -p "$OUT"

declare -A Q=(
[W0477]='ti:%22The+Sparsity+Ceiling%22'
[W0501]='ti:%22Sigma-Delta+Neural+Network+Conversion%22'
[W0557]='ti:%22Aggressive+SRAM+Voltage+Scaling%22'
[W0538]='ti:%22Provably+Lossless+Acceleration+of+DNN+Mutation+Testing%22'
[W0563]='ti:%22Partition+the+Support,+Reconstruct+the+Residual%22'
[W0495]='ti:%22Elastic+Spiking+Transformers%22'
[W0558]='ti:%22Dynamic+Group+Convolution%22'
[W0507]='ti:%22Training+for+temporal+sparsity+in+deep+neural+networks%22'
[W0622]='ti:%22UniSpike%22'
[W0478]='ti:%22Unequal+Error+Protection+for+DNN+Inference+Memory%22'
)

for id in W0477 W0501 W0557 W0538 W0563 W0495 W0558 W0507 W0622 W0478; do
  ok=0
  for try in $(seq 1 12); do
    out=$(curl -s -m 25 "https://export.arxiv.org/api/query?search_query=${Q[$id]}&max_results=3")
    if [[ -n "$out" && "$out" != *"Rate exceeded"* && "$out" != *"<title>Error</title>"* ]]; then
      echo "$out" > "$OUT/${id}.xml"
      echo "OK $id (try $try)"
      ok=1
      break
    fi
    sleep 25
  done
  [[ $ok -eq 0 ]] && echo "FAIL $id"
  sleep 20
done
echo DONE
