#!/usr/bin/env bash
set -euo pipefail

[[ "$#" -eq 12 ]] || exit 2
m1311_logical=$1
m1311_target1=$2
m1311_link2=$3
m1311_target2=$4
m1311_link3=$5
m1311_target3=$6
m1311_entity=$7
m1311_dev=$8
m1311_ino=$9
m1311_mode=${10}
m1311_size=${11}
m1311_sha=${12}

[[ -L "${m1311_logical}" && "$(readlink "${m1311_logical}")" == "${m1311_target1}" ]] || exit 3
[[ "${m1311_target1}" == "${m1311_link2}" ]] || exit 3
[[ -L "${m1311_link2}" && "$(readlink "${m1311_link2}")" == "${m1311_target2}" ]] || exit 3
[[ "${m1311_target2}" == "${m1311_link3}" ]] || exit 3
[[ -L "${m1311_link3}" && "$(readlink "${m1311_link3}")" == "${m1311_target3}" ]] || exit 3
[[ "${m1311_target3}" == "${m1311_entity}" ]] || exit 3
[[ -f "${m1311_entity}" && ! -L "${m1311_entity}" && -x "${m1311_entity}" ]] || exit 4
[[ "$(realpath -e "${m1311_logical}")" == "${m1311_entity}" ]] || exit 4
IFS=: read -r m1311_actual_dev m1311_actual_ino m1311_actual_mode m1311_actual_size \
    < <(stat -Lc '%d:%i:%a:%s' "${m1311_entity}")
[[ "${m1311_actual_dev}" == "${m1311_dev}" && \
   "${m1311_actual_ino}" == "${m1311_ino}" && \
   "${m1311_actual_mode}" == "${m1311_mode}" && \
   "${m1311_actual_size}" == "${m1311_size}" ]] || exit 5
[[ "$(sha256sum "${m1311_entity}" | awk '{print $1}')" == "${m1311_sha}" ]] || exit 6
printf 'PASS_M1311_EXACT_PYTHON_SYMLINK_CHAIN_AND_ENTITY\n'
