#!/usr/bin/env bash
set -euo pipefail
metadata_here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
metadata_build="$(mktemp -d /tmp/metadata_primitive_sim.XXXXXX)"
trap 'rm -rf -- "$metadata_build"' EXIT
verilator --version > "$metadata_here/simulator_version.txt"
verilator --cc --exe -Wall --top-module metadata_primitive \
    --Mdir "$metadata_build" -CFLAGS '-std=c++14 -O2' \
    "$metadata_here/metadata_primitive.sv" "$metadata_here/tb_metadata.cpp" \
    > "$metadata_here/build.log" 2>&1
make -C "$metadata_build" -f Vmetadata_primitive.mk -j 2 >> "$metadata_here/build.log" 2>&1
"$metadata_build/Vmetadata_primitive" | tee "$metadata_here/results.json"
