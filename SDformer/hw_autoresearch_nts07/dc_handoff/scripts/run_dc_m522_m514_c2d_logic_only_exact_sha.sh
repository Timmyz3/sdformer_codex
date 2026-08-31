#!/usr/bin/env bash
set -euo pipefail

m522_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m522_hw="$(cd "${m522_dc_root}/.." && pwd)"
m522_run="${m522_dc_root}/runs/m522_m514_c2d_logic_only_dc_3p000ns_r4_20260827"
m522_attempt="${m522_dc_root}/runs/.m522_m514_c2d_logic_only_dc_r4.one_shot_attempt"
m522_dc_link="/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
m522_dc="/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell"
m522_slow="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
m522_fast="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
m522_rtl="rtl_m514/m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv"
m522_filelist="dc_handoff/filelists/date_m522_m514_c2d_logic_only_dc.f"
m522_sdc="dc_handoff/constraints/date_m522_m514_c2d_3ns.sdc"
m522_tcl="dc_handoff/scripts/run_dc_m522_m514_c2d_logic_only.tcl"
m522_contract="contracts/m522_m514_c2d_logic_only_dc_contract_r6_20260827.json"
m522_vcs_dir="results/m514_c2d_directed_vcs_r1_20260827"
m522_review_dir="reviews/m514_c2d_directed_vcs_receipt_blind_hammer_r1_20260827"
m522_failure_review_dir="reviews/m522_m514_dc_pretool_failure_hammer_r1_20260827"
m522_tool_failure_review_dir="reviews/m522_m514_dc_tool_invocation_failure_hammer_r1_20260827"
m522_static_review_dir="reviews/m522_m514_dc_static_hammer_r6_20260827"

m522_sha() { sha256sum "$1" | awk '{print $1}'; }
m522_expect() {
    local m522_path=$1
    local m522_expected=$2
    [[ -f "${m522_path}" && ! -L "${m522_path}" ]] || {
        echo "M522 missing/symlinked ${m522_path}" >&2
        exit 10
    }
    [[ "$(m522_sha "${m522_path}")" == "${m522_expected}" ]] || {
        echo "M522 SHA mismatch ${m522_path}" >&2
        exit 10
    }
}

m522_verify_sealed_dir() {
    python3 - "$1" "$2" "$3" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
root = Path(sys.argv[1])
profile = sys.argv[2]
inventory_path = None if sys.argv[3] == '-' else Path(sys.argv[3])

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def under_root(path, anchor):
    try:
        return path.relative_to(anchor)
    except ValueError:
        return None

legacy_allowed = {
    'csrc/_1351757_archive_1.so': {
        'link_text': './/../simv.daidir//_1351757_archive_1.so',
        'target': 'simv.daidir/_1351757_archive_1.so',
        'target_sha256': 'be4c425a88be6d5cc24581c8bc8746e855d66ffbc9f5e7f842ef5adee9bd522d',
    },
    'simv.vdb/snps/coverage/db/testdata/test/assert.verilog.shape.xml': {
        'link_text': '../../common/assert.verilog.shape.xml',
        'target': 'simv.vdb/snps/coverage/db/common/assert.verilog.shape.xml',
        'target_sha256': '7f9d032a25fef79765e43e9ec60afd7ec8255af2a658702bbb9a57bdef3f8781',
    },
}
require(profile in {'zero_symlink', 'historical_vcs_exact2'},
        f'unknown symlink profile {profile}')
allowed = legacy_allowed if profile == 'historical_vcs_exact2' else {}
inventory = {
    'schema': 'm522_verified_root_inventory_v1',
    'root': root.as_posix(),
    'symlink_profile': profile,
    'status': 'OBSERVED_UNVERIFIED',
    'expected_manifest_files': [],
    'actual_regular_files': [],
    'directories': [],
    'symlinks': [],
}
error = None
try:
    require(root.is_dir() and not root.is_symlink(),
            f'{root}: root missing, non-directory, or symlink')
    root_resolved = root.resolve(strict=True)
    manifest = root / 'SHA256SUMS'
    outer = root / 'SHA256SUMS.seal.sha256'
    require(manifest.is_file() and not manifest.is_symlink(),
            f'{root}: manifest missing or symlink')
    require(outer.is_file() and not outer.is_symlink(),
            f'{root}: outer seal missing or symlink')
    expected = {}
    raw_members = {}
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        digest, name = line.split('  ', 1)
        require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                f'{root}: malformed digest at manifest line {line_number}')
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts,
                f'{root}: unsafe manifest path {name!r}')
        normalized_name = path.as_posix()
        require(normalized_name not in expected,
                f'{root}: duplicate normalized manifest member {normalized_name}')
        item = root / path
        require(item.is_file() and not item.is_symlink(),
                f'{root}: manifest member missing/nonregular/symlink {normalized_name}')
        observed_digest = hashlib.sha256(item.read_bytes()).hexdigest()
        require(observed_digest == digest,
                f'{root}: manifest member digest mismatch {normalized_name}')
        expected[normalized_name] = digest
        raw_members[normalized_name] = name
    outer_digest, outer_name = outer.read_text().strip().split('  ', 1)
    require(outer_name == 'SHA256SUMS', f'{root}: malformed outer seal member')
    require(hashlib.sha256(manifest.read_bytes()).hexdigest() == outer_digest,
            f'{root}: outer seal digest mismatch')

    actual_regular = []
    directories = []
    observed_links = {}
    for item in root.rglob('*'):
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            link_text = os.readlink(str(item))
            try:
                resolved = item.resolve(strict=True)
                target_rel_path = under_root(resolved, root_resolved)
                target_rel = None if target_rel_path is None else target_rel_path.as_posix()
                target_regular = resolved.is_file() and not resolved.is_symlink()
                target_sha = hashlib.sha256(resolved.read_bytes()).hexdigest() \
                    if target_regular else None
            except (FileNotFoundError, RuntimeError, OSError):
                target_rel = None
                target_regular = False
                target_sha = None
            observed_links[relative] = {
                'link_path': relative,
                'link_text': link_text,
                'resolved_target': target_rel,
                'target_regular_non_symlink': target_regular,
                'target_sha256': target_sha,
            }
        elif item.is_file():
            if relative not in {'SHA256SUMS', 'SHA256SUMS.seal.sha256'}:
                actual_regular.append(relative)
        elif item.is_dir():
            directories.append(relative)

    inventory['expected_manifest_files'] = [
        {'path': name, 'manifest_spelling': raw_members[name], 'sha256': expected[name]}
        for name in sorted(expected)
    ]
    inventory['actual_regular_files'] = sorted(actual_regular)
    inventory['directories'] = sorted(directories)
    inventory['symlinks'] = [observed_links[name] for name in sorted(observed_links)]
    require(set(actual_regular) == set(expected),
            f'{root}: regular-file topology differs from normalized manifest; '
            f'missing={sorted(set(expected) - set(actual_regular))} '
            f'extra={sorted(set(actual_regular) - set(expected))}')
    require(set(observed_links) == set(allowed),
            f'{root}: symlink topology differs from profile {profile}; '
            f'missing={sorted(set(allowed) - set(observed_links))} '
            f'extra={sorted(set(observed_links) - set(allowed))}')
    for link_path, specification in allowed.items():
        observed = observed_links[link_path]
        require(observed['link_text'] == specification['link_text'],
                f'{root}: link text drift {link_path}')
        require(observed['resolved_target'] == specification['target'],
                f'{root}: resolved target drift/escape {link_path}')
        require(observed['target_regular_non_symlink'],
                f'{root}: link target not regular non-symlink {link_path}')
        require(specification['target'] in expected,
                f'{root}: link target not sealed in manifest {link_path}')
        require(expected[specification['target']] == specification['target_sha256'],
                f'{root}: sealed target expected SHA drift {link_path}')
        require(observed['target_sha256'] == specification['target_sha256'],
                f'{root}: link target observed SHA drift {link_path}')
    inventory['status'] = 'PASS_EXACT_SEALED_ROOT_INVENTORY'
except Exception as exc:
    error = f'{type(exc).__name__}: {exc}'
    inventory['status'] = 'FAIL_EXACT_SEALED_ROOT_INVENTORY'
    inventory['error'] = error
finally:
    rendered = json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + '\n'
    print(f'M522_ROOT_INVENTORY_BEGIN root={root} profile={profile}')
    print(rendered, end='')
    print(f'M522_ROOT_INVENTORY_END root={root} status={inventory["status"]}')
    if inventory_path is not None:
        require(inventory_path.parent.is_dir() and not inventory_path.parent.is_symlink(),
                f'{root}: inventory parent missing or symlink {inventory_path.parent}')
        require(not inventory_path.exists() and not inventory_path.is_symlink(),
                f'{root}: inventory output already exists {inventory_path}')
        inventory_path.write_text(rendered)
if error is not None:
    raise AssertionError(error)
PY
}

m522_verify_output_package() {
    m522_verify_sealed_dir "$1" zero_symlink -
    python3 - "$1" <<'PY'
import json
import math
from pathlib import Path
import sys

def reject_constant(value):
    raise ValueError(f'non-finite JSON constant: {value}')

def assert_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite(child)

root = Path(sys.argv[1])
topology = json.loads((root / 'TOPOLOGY.json').read_text(),
                      parse_constant=reject_constant)
assert topology['schema'] == 'm522_exact_output_topology_v2'
excluded = {'TOPOLOGY.json', 'SHA256SUMS', 'SHA256SUMS.seal.sha256'}
actual_files = {
    item.relative_to(root).as_posix()
    for item in root.rglob('*') if item.is_file() and
    item.relative_to(root).as_posix() not in excluded
}
actual_dirs = {
    item.relative_to(root).as_posix()
    for item in root.rglob('*') if item.is_dir()
}
assert actual_files == set(topology['files_excluding_topology_and_seals'])
assert actual_dirs == set(topology['directories'])
assert not any(item.is_symlink() for item in root.rglob('*'))
receipt = json.loads(
    (root / 'm522_m514_c2d_logic_only_dc_receipt_r4.json').read_text(),
    parse_constant=reject_constant)
assert_finite(receipt)
assert receipt['schema'] == 'm522_m514_c2d_logic_only_dc_receipt_v4'
assert receipt['status'] == 'PASS_M522_M514_C2D_LOGIC_ONLY_DC_3NS'
assert receipt['admission']['cycle_speedup'] is False
assert receipt['admission']['system_speedup'] is False
assert receipt['admission']['paper_ppa_ready'] is False
assert receipt['gates']['negative_wrong_runner_sha_exit_code'] == 10
assert receipt['gates']['precompile_tim209_count'] == 0
assert receipt['gates']['precompile_opt150_count'] == 0
assert receipt['identity']['historical_vcs_symlink_profile'] == 'historical_vcs_exact2'
assert receipt['identity']['sealed_input_root_inventory_count'] == 5
assert receipt['identity']['dc_shell_launcher']['invoked_path'] == \
    '/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell'
inventory_root = root / 'input_root_inventories'
expected_inventories = {
    'm514_vcs.json': ('historical_vcs_exact2', 2),
    'm514_receipt_blind_review.json': ('zero_symlink', 0),
    'm522_second_pretool_failure_review.json': ('zero_symlink', 0),
    'm522_r5_tool_invocation_failure_review.json': ('zero_symlink', 0),
    'm522_r6_static_review.json': ('zero_symlink', 0),
}
assert {path.name for path in inventory_root.iterdir()} == set(expected_inventories)
for name, (profile, symlink_count) in expected_inventories.items():
    inventory = json.loads((inventory_root / name).read_text(),
                           parse_constant=reject_constant)
    assert_finite(inventory)
    assert inventory['schema'] == 'm522_verified_root_inventory_v1'
    assert inventory['status'] == 'PASS_EXACT_SEALED_ROOT_INVENTORY'
    assert inventory['symlink_profile'] == profile
    assert len(inventory['symlinks']) == symlink_count
PY
}

cd "${m522_hw}"
m522_expected_self="${M522_EXPECTED_RUNNER_SHA256:-}"
[[ "${m522_expected_self}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "M522 requires literal M522_EXPECTED_RUNNER_SHA256" >&2
    exit 10
}
m522_expect "dc_handoff/scripts/run_dc_m522_m514_c2d_logic_only_exact_sha.sh" \
    "${m522_expected_self}"
if [[ "${M522_NEGATIVE_PREFLIGHT_TEST:-0}" == 1 ]]; then
    echo "M522 negative preflight unexpectedly passed exact runner SHA" >&2
    exit 11
fi

[[ ! -e "${m522_run}" && ! -L "${m522_run}" ]] || {
    echo "M522 canonical output exists ${m522_run}" >&2
    exit 12
}
[[ ! -e "${m522_attempt}" && ! -L "${m522_attempt}" ]] || {
    echo "M522 r4 one-shot attempt already consumed ${m522_attempt}" >&2
    exit 17
}
if pgrep -x dc_shell >/dev/null || pgrep -x dc_shell-t >/dev/null || \
        pgrep -x snps_shell >/dev/null || \
        pgrep -x fm_shell >/dev/null || pgrep -x pt_shell >/dev/null; then
    echo "M522 refuses to collide with DC/Formality/PT" >&2
    exit 13
fi

[[ -L "${m522_dc_link}" && "$(readlink "${m522_dc_link}")" == "snps_shell" && \
   "$(readlink -f "${m522_dc_link}")" == "${m522_dc}" ]] || {
    echo "M522 dc_shell launcher link identity drift" >&2
    exit 10
}
m522_expect "${m522_dc}" 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2
m522_expect "${m522_slow}" 79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af
m522_expect "${m522_fast}" a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a
m522_expect "${m522_rtl}" 90c44fc9bde839c3cf325ccc8f45c153bf5d30e18de7f39b26d7a4456b017a9a
m522_expect "${m522_filelist}" fc0d31ec1869120528abfbf61736df7ac6828095f6c58f0a4c31edcd892660c7
m522_expect "${m522_sdc}" 9516a8f775ac7e688b9d7813ad613362fd6c03e1548323a2efdce30fdddf3bec
m522_expect "${m522_tcl}" bb749419b25ba91a17cd445a76ee7bc703eabf289fa4769e97c40c71ca8687e8
m522_expect "${m522_contract}" 2b450b9fc32436da9c67c820debe6247169a725ef62bfcfcda1ca0b6a18a7215
m522_expect contracts/m514_c2d_directed_vcs_contract_r1_20260827.json 60e4fe5921a374f399bef82fd1902718428bb8f9d6f3d86dc5d03bda7953ab5b
m522_expect "${m522_vcs_dir}/m514_c2d_directed_vcs_receipt_r1.json" aa6fb4d68c0ec43147481ec3355d8bfdd84777a151cf6f985f20dd763e24d8ee
m522_expect "${m522_vcs_dir}/SHA256SUMS" 4a77e9d980715cfa6ed2c672b1b9f13f4b8e5c3cc23c95cffa2700ccdb210eaf
m522_expect "${m522_vcs_dir}/SHA256SUMS.seal.sha256" 98fdb9c3c74f2e27e8ed6094267d3610caad82c6bc7f3eb1cfe7bedadb6609d3
m522_expect "${m522_review_dir}/m514_c2d_directed_vcs_receipt_blind_hammer_r1.json" 3242e900a60347e28966bf6fd53e6d3e8518bd424b6b287d6d3e402eb79e9e8d
m522_expect "${m522_review_dir}/SHA256SUMS" fcad13f386d2369ccdeb602ec0eb744cbc217f68366ca0fa15df12cdf9cf16cd
m522_expect "${m522_review_dir}/SHA256SUMS.seal.sha256" 64e69c452a841c726105d6cc3c5e441e2a8f21afecbb8205cf3f1aa944284674
m522_expect "${m522_failure_review_dir}/m522_m514_dc_pretool_failure_hammer_r1.json" f18b7a5467793db4dc3ff67475e2f2fd048426ea8a323281cf74143bd0625918
m522_expect "${m522_failure_review_dir}/SHA256SUMS" b836e84c279ddaf0b7d670ccb31441b6c79c5d216303fc9e46383edf382b1ca9
m522_expect "${m522_failure_review_dir}/SHA256SUMS.seal.sha256" c3947bce6d55257e4585b11fbb1ea724d03cf750abec7b182a6ec72a6174af85
m522_expect reviews/m522_m514_dc_static_hammer_r4_20260827/m522_m514_dc_static_hammer_r4.json 2566b06d47cc3a37fd74b8eaaa40c3408e940a93c2a63637c7c3a420b846933d
m522_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
m522_expect "${m522_tool_failure_review_dir}/m522_m514_dc_tool_invocation_failure_hammer_r1.json" a4dd356e29681181cd5eb78b795394ed075f12e213e61c862a8585525f27f746
m522_expect "${m522_tool_failure_review_dir}/SHA256SUMS" 9cb2a122ee9d758c92bf9508bdc01ef042f82e7cba44aa7e59a2791be589c480
m522_expect "${m522_tool_failure_review_dir}/SHA256SUMS.seal.sha256" bdead599b2b692c6a67dd0dd096badeb181f9c4feeb41f0edaa5a1025343f3f0

[[ -d "${m522_static_review_dir}" ]] || {
    echo "M522 independent r6 static launch admission missing" >&2
    exit 15
}
m522_staging=$(mktemp -d "${m522_dc_root}/runs/.m522_m514_c2d_dc_r4.staging.XXXXXX")
m522_complete=0
m522_quarantine_incomplete() {
    local m522_exit_rc=$1
    local m522_failed_path=""
    trap - EXIT
    if [[ ${m522_complete} -ne 1 ]]; then
        if [[ -d "${m522_run}" ]]; then
            m522_failed_path="${m522_run}"
        elif [[ -d "${m522_staging}" ]]; then
            m522_failed_path="${m522_staging}"
        fi
        if [[ -n "${m522_failed_path}" ]]; then
            local m522_quarantine="${m522_dc_root}/runs/m522_m514_c2d_logic_only_dc_r4.failed_or_incomplete.$$.quarantine"
            if [[ -e "${m522_quarantine}" || -L "${m522_quarantine}" ]]; then
                echo "M522 quarantine collision ${m522_quarantine}" >&2
                exit 99
            fi
            python3 - "${m522_failed_path}" "${m522_exit_rc}" <<'PY'
import json
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
runner_exit_code = int(sys.argv[2])
root_stat = os.lstat(root)
assert stat.S_ISDIR(root_stat.st_mode) and not stat.S_ISLNK(root_stat.st_mode)

def scan_no_follow(directory):
    found = []
    with os.scandir(directory) as entries:
        for entry in entries:
            entry_stat = entry.stat(follow_symlinks=False)
            path = Path(entry.path)
            if stat.S_ISLNK(entry_stat.st_mode):
                found.append({
                    'path': path.relative_to(root).as_posix(),
                    'raw_link_text': os.readlink(path),
                })
            elif stat.S_ISDIR(entry_stat.st_mode):
                found.extend(scan_no_follow(path))
    return found

observed = sorted(scan_no_follow(root), key=lambda item: item['path'])
for item in observed:
    path = root / item['path']
    current = os.lstat(path)
    assert stat.S_ISLNK(current.st_mode)
    assert os.readlink(path) == item['raw_link_text']
    os.unlink(path)
assert scan_no_follow(root) == []

inventory_path = root / 'FAILED_SYMLINK_INVENTORY.json'
marker_path = root / 'RUN_FAILED_OR_INCOMPLETE.txt'
assert not inventory_path.exists() and not inventory_path.is_symlink()
assert not marker_path.exists() and not marker_path.is_symlink()
inventory = {
    'schema': 'm522_failed_symlink_quarantine_inventory_v1',
    'status': 'PASS_NOFOLLOW_INVENTORY_UNLINK_ZERO_SYMLINK',
    'runner_exit_code': runner_exit_code,
    'source_root': root.as_posix(),
    'symlink_count': len(observed),
    'symlinks': observed,
}
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
if hasattr(os, 'O_NOFOLLOW'):
    flags |= os.O_NOFOLLOW
fd = os.open(inventory_path, flags, 0o600)
with os.fdopen(fd, 'w') as handle:
    json.dump(inventory, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write('\n')
fd = os.open(marker_path, flags, 0o600)
with os.fdopen(fd, 'w') as handle:
    handle.write('status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\n')
    handle.write(f'runner_exit_code={runner_exit_code}\n')
assert scan_no_follow(root) == []
PY
            mv "${m522_failed_path}" "${m522_quarantine}"
            python3 - "${m522_quarantine}" "${m522_exit_rc}" <<'PY'
import json
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
runner_exit_code = int(sys.argv[2])
root_stat = os.lstat(root)
assert stat.S_ISDIR(root_stat.st_mode) and not stat.S_ISLNK(root_stat.st_mode)
for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
    for name in dirnames + filenames:
        assert not stat.S_ISLNK(os.lstat(Path(directory) / name).st_mode)
inventory_path = root / 'FAILED_SYMLINK_INVENTORY.json'
marker_path = root / 'RUN_FAILED_OR_INCOMPLETE.txt'
assert inventory_path.is_file() and not inventory_path.is_symlink()
assert marker_path.is_file() and not marker_path.is_symlink()
inventory = json.loads(inventory_path.read_text(),
                       parse_constant=lambda value: (_ for _ in ()).throw(
                           ValueError(f'non-finite JSON constant: {value}')))
assert inventory['schema'] == 'm522_failed_symlink_quarantine_inventory_v1'
assert inventory['status'] == 'PASS_NOFOLLOW_INVENTORY_UNLINK_ZERO_SYMLINK'
assert inventory['runner_exit_code'] == runner_exit_code
assert inventory['symlink_count'] == len(inventory['symlinks'])
PY
        fi
    fi
    exit "${m522_exit_rc}"
}
trap 'm522_quarantine_incomplete $?' EXIT

mkdir -p "${m522_staging}/input_root_inventories"
(cd "${m522_vcs_dir}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256)
(cd "${m522_review_dir}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256)
(cd "${m522_failure_review_dir}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256)
(cd "${m522_tool_failure_review_dir}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256)
(cd "${m522_static_review_dir}" && sha256sum -c SHA256SUMS && \
    sha256sum -c SHA256SUMS.seal.sha256)
m522_verify_sealed_dir "${m522_vcs_dir}" historical_vcs_exact2 \
    "${m522_staging}/input_root_inventories/m514_vcs.json"
m522_verify_sealed_dir "${m522_review_dir}" zero_symlink \
    "${m522_staging}/input_root_inventories/m514_receipt_blind_review.json"
m522_verify_sealed_dir "${m522_failure_review_dir}" zero_symlink \
    "${m522_staging}/input_root_inventories/m522_second_pretool_failure_review.json"
m522_verify_sealed_dir "${m522_tool_failure_review_dir}" zero_symlink \
    "${m522_staging}/input_root_inventories/m522_r5_tool_invocation_failure_review.json"
m522_verify_sealed_dir "${m522_static_review_dir}" zero_symlink \
    "${m522_staging}/input_root_inventories/m522_r6_static_review.json"
python3 - "${m522_expected_self}" "${m522_staging}" <<'PY'
import json
from pathlib import Path
import sys
vcs = json.loads(Path('results/m514_c2d_directed_vcs_r1_20260827/m514_c2d_directed_vcs_receipt_r1.json').read_text())
review = json.loads(Path('reviews/m514_c2d_directed_vcs_receipt_blind_hammer_r1_20260827/m514_c2d_directed_vcs_receipt_blind_hammer_r1.json').read_text())
failure = json.loads(Path('reviews/m522_m514_dc_pretool_failure_hammer_r1_20260827/m522_m514_dc_pretool_failure_hammer_r1.json').read_text())
tool_failure = json.loads(Path('reviews/m522_m514_dc_tool_invocation_failure_hammer_r1_20260827/m522_m514_dc_tool_invocation_failure_hammer_r1.json').read_text())
static = json.loads(Path('reviews/m522_m514_dc_static_hammer_r6_20260827/m522_m514_dc_static_hammer_r6.json').read_text())
assert vcs['status'] == 'PASS_M514_C2D_DIRECTED_FUNCTIONAL_COMPLETENESS'
assert vcs['claim_boundary'] == {
    'area': False, 'cycle_speedup': False, 'date_headline': False,
    'directed_functional_completeness': True, 'energy': False,
    'formality': False, 'full_decoder_trace': False,
    'paper_ppa_ready': False, 'system_speedup': False, 'timing': False,
}
assert review['status'] == 'PASS_DIRECTED_FUNCTIONAL_COMPLETENESS_ONLY__NO_PERFORMANCE_OR_PPA_ADMISSION'
assert review['decision']['verdict'] == 'PASS_M514_DIRECTED_FUNCTIONAL_COMPLETENESS_ONLY'
assert failure['status'] == 'PASS_PRETOOL_FAILURE__POSITIVE_DC_AUTHORIZATION_NOT_CONSUMED__ONE_RETRY_AUTHORIZED'
assert failure['decision']['on_repeat_same_pretool_assertion'].startswith('Stop. Do not retry again. Create an r4 runner')
assert tool_failure['status'] == 'PASS_ROOT_CAUSE_DIRECT_SNPS_SHELL_BASENAME_DISPATCH_FAILURE__NO_DC_RESULT__R6_REQUIRED'
assert tool_failure['authorization']['r5_positive_authorization_consumed'] is True
assert tool_failure['root_cause']['direct_snps_shell_dash_shell_is_valid_repair'] is False
assert static['schema'] == 'm522_m514_dc_static_hammer_r6'
assert static['status'] == 'STATIC_GO__EXACT_SHA_ONE_SHOT_DC_AUTHORIZED'
assert static['p0_count'] == 0
assert static['decision']['execution_authorized'] is True
assert static['decision']['authorized_runner_sha256'] == sys.argv[1]
inventory_root = Path(sys.argv[2]) / 'input_root_inventories'
expected_inventories = {
    'm514_vcs.json': ('historical_vcs_exact2', 2),
    'm514_receipt_blind_review.json': ('zero_symlink', 0),
    'm522_second_pretool_failure_review.json': ('zero_symlink', 0),
    'm522_r5_tool_invocation_failure_review.json': ('zero_symlink', 0),
    'm522_r6_static_review.json': ('zero_symlink', 0),
}
assert {path.name for path in inventory_root.iterdir()} == set(expected_inventories)
for name, (profile, symlink_count) in expected_inventories.items():
    inventory = json.loads((inventory_root / name).read_text())
    assert inventory['schema'] == 'm522_verified_root_inventory_v1'
    assert inventory['status'] == 'PASS_EXACT_SEALED_ROOT_INVENTORY'
    assert inventory['symlink_profile'] == profile
    assert len(inventory['symlinks']) == symlink_count
PY

m522_commit_headroom_kib=$(awk '/CommitLimit:/ {limit=$2} /Committed_AS:/ {used=$2} END {print limit-used}' /proc/meminfo)
m522_mem_available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
[[ "${m522_commit_headroom_kib}" -ge 33554432 && \
   "${m522_mem_available_kib}" -ge 67108864 ]] || {
    echo "M522 resource gate failed commit_headroom_kib=${m522_commit_headroom_kib} mem_available_kib=${m522_mem_available_kib}" >&2
    exit 14
}

set +e
M522_EXPECTED_RUNNER_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
M522_NEGATIVE_PREFLIGHT_TEST=1 \
    bash "dc_handoff/scripts/run_dc_m522_m514_c2d_logic_only_exact_sha.sh" \
    >"${m522_staging}/negative_wrong_runner_sha_preflight.log" 2>&1
m522_negative_rc=$?
set -e
[[ "${m522_negative_rc}" -eq 10 ]] || {
    echo "M522 wrong-runner-SHA negative preflight returned ${m522_negative_rc}, expected 10" >&2
    exit 16
}
printf 'test=wrong_runner_sha\nexpected_rc=10\nobserved_rc=%s\nstatus=PASS_NEGATIVE_PREFLIGHT\n' \
    "${m522_negative_rc}" >"${m522_staging}/negative_preflight_receipt.txt"
sha256sum "${m522_rtl}" "${m522_filelist}" "${m522_sdc}" \
    "${m522_tcl}" "${m522_contract}" \
    dc_handoff/scripts/run_dc_m522_m514_c2d_logic_only_exact_sha.sh \
    contracts/m514_c2d_directed_vcs_contract_r1_20260827.json \
    "${m522_vcs_dir}/m514_c2d_directed_vcs_receipt_r1.json" \
    "${m522_vcs_dir}/SHA256SUMS" \
    "${m522_vcs_dir}/SHA256SUMS.seal.sha256" \
    "${m522_review_dir}/m514_c2d_directed_vcs_receipt_blind_hammer_r1.json" \
    "${m522_review_dir}/SHA256SUMS" \
    "${m522_review_dir}/SHA256SUMS.seal.sha256" \
    "${m522_failure_review_dir}/m522_m514_dc_pretool_failure_hammer_r1.json" \
    "${m522_failure_review_dir}/SHA256SUMS" \
    "${m522_failure_review_dir}/SHA256SUMS.seal.sha256" \
    reviews/m522_m514_dc_static_hammer_r4_20260827/m522_m514_dc_static_hammer_r4.json \
    docs/359_DATE终局冻结_20260813.md \
    "${m522_tool_failure_review_dir}/m522_m514_dc_tool_invocation_failure_hammer_r1.json" \
    "${m522_tool_failure_review_dir}/SHA256SUMS" \
    "${m522_tool_failure_review_dir}/SHA256SUMS.seal.sha256" \
    "${m522_static_review_dir}/m522_m514_dc_static_hammer_r6.json" \
    "${m522_static_review_dir}/SHA256SUMS" \
    "${m522_static_review_dir}/SHA256SUMS.seal.sha256" \
    "${m522_dc_link}" "${m522_dc}" "${m522_slow}" "${m522_fast}" \
    >"${m522_staging}/input_sha256.txt"
cp "${m522_contract}" "${m522_staging}/contract.json"
printf 'commit_headroom_kib=%s\nmem_available_kib=%s\n' \
    "${m522_commit_headroom_kib}" "${m522_mem_available_kib}" \
    >"${m522_staging}/resource_preflight.txt"
printf 'runner_expected_sha256=%s\nrunner_observed_sha256=%s\nnegative_preflight_rc=%s\n' \
    "${m522_expected_self}" "$(m522_sha dc_handoff/scripts/run_dc_m522_m514_c2d_logic_only_exact_sha.sh)" \
    "${m522_negative_rc}" >"${m522_staging}/runner_identity.txt"

export DESIGN_NAME=m514_c2_convtranspose_k3s2_polyphase_address_mapper
export HW_ROOT="${m522_hw}"
export RTL_FILELIST="${m522_hw}/${m522_filelist}"
export LIB_DB="${m522_slow}"
export MIN_LIB_DB="${m522_fast}"
export SDC_FILE="${m522_hw}/${m522_sdc}"
export OUTPUT_DIR="${m522_staging}"
export CLOCK_PERIOD_NS=3.000
export OPERATING_CONDITION=ssg0p9v125c

mkdir "${m522_attempt}" || {
    echo "M522 failed to atomically consume r4 one-shot attempt ${m522_attempt}" >&2
    exit 17
}
printf 'status=ONE_SHOT_ATTEMPT_CONSUMED_DO_NOT_RETRY\nrunner_sha256=%s\nlauncher_path=%s\n' \
    "${m522_expected_self}" "${m522_dc_link}" >"${m522_attempt}/ATTEMPT_CONSUMED.txt"

set +e
"${m522_dc_link}" -f "${m522_hw}/${m522_tcl}" \
    >"${m522_staging}/dc.log" 2>&1
m522_rc=$?
set -e
echo "${m522_rc}" >"${m522_staging}/dc.rc"
[[ "${m522_rc}" -eq 0 ]]
! grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' \
    "${m522_staging}/dc.log"
grep -Fq 'Thank you...' "${m522_staging}/dc.log"
grep -Fxq 'TIM-209=0' "${m522_staging}/reports/precompile_loop_gate.rpt"
grep -Fxq 'OPT-150=0' "${m522_staging}/reports/precompile_loop_gate.rpt"
grep -Fxq 'status=PASS_PRECOMPILE_LOOP_GATE' \
    "${m522_staging}/reports/precompile_loop_gate.rpt"
grep -Fxq 'status=PASS_EXPLICIT_IDEAL_CLOCK_NETWORK' \
    "${m522_staging}/reports/ideal_clock_network.rpt"
for m522_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt resources_postcompile.rpt \
        references_postcompile.rpt precompile_build.rpt \
        check_design_precompile.rpt check_timing_precompile.rpt \
        ideal_clock_network.rpt; do
    [[ -s "${m522_staging}/reports/${m522_report}" ]] || exit 30
done
[[ -s "${m522_staging}/netlist/${DESIGN_NAME}_mapped.v" &&
   -s "${m522_staging}/netlist/${DESIGN_NAME}_mapped.sdc" &&
   -s "${m522_staging}/netlist/${DESIGN_NAME}.ddc" &&
   -s "${m522_staging}/netlist/${DESIGN_NAME}.svf" ]] || exit 31
grep -Fq 'slack (VIOLATED)' "${m522_staging}/reports/timing_setup.rpt" \
    "${m522_staging}/reports/timing_hold.rpt" && exit 32 || true
[[ "$(grep -Fc 'This design has no violated constraints.' \
    "${m522_staging}/reports/constraint_violators.rpt")" -eq 5 ]] || exit 33

m522_area=$(awk '/Total cell area:/ {print $4; exit}' "${m522_staging}/reports/area.rpt")
m522_cells=$(awk '/Number of cells:/ {print $4; exit}' "${m522_staging}/reports/area.rpt")
m522_seq=$(awk '/Number of sequential cells:/ {print $5; exit}' "${m522_staging}/reports/area.rpt")
m522_combo=$(awk '/Number of combinational cells:/ {print $5; exit}' "${m522_staging}/reports/area.rpt")
m522_levels=$(awk '/Levels of Logic:/ {print $4; exit}' "${m522_staging}/reports/qor.rpt")
m522_setup=$(awk '/slack \(MET\)/ {print $3; exit}' "${m522_staging}/reports/timing_setup.rpt")
m522_hold=$(awk '/slack \(MET\)/ {print $3; exit}' "${m522_staging}/reports/timing_hold.rpt")
for m522_value in "${m522_area}" "${m522_cells}" "${m522_seq}" \
        "${m522_combo}" "${m522_levels}" "${m522_setup}" "${m522_hold}"; do
    [[ -n "${m522_value}" ]] || exit 34
done
awk -v x="${m522_area}" 'BEGIN {exit !(x > 0)}'
awk -v x="${m522_setup}" 'BEGIN {exit !(x >= 0.0)}'
awk -v x="${m522_hold}" 'BEGIN {exit !(x >= 0.0)}'

python3 - "${m522_staging}" "${m522_area}" "${m522_cells}" \
    "${m522_seq}" "${m522_combo}" "${m522_levels}" \
    "${m522_setup}" "${m522_hold}" "${m522_expected_self}" \
    "${m522_negative_rc}" "$(readlink "${m522_dc_link}")" \
    "$(readlink -f "${m522_dc_link}")" "${m522_dc_link}" \
    "${m522_attempt}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys
run = Path(sys.argv[1])
area, cells, seq, combo, levels, setup, hold = (
    float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]))
runner_sha, negative_rc, launcher_link_text, resolved_dc, invoked_path, attempt_path = (
    sys.argv[9], int(sys.argv[10]), sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14])
for value in (area, levels, setup, hold):
    assert math.isfinite(value)
identity = {}
for line in (run / 'input_sha256.txt').read_text().splitlines():
    digest, path = line.split('  ', 1)
    assert len(digest) == 64 and path not in identity
    identity[path] = digest
precompile_lines = (run / 'reports/precompile_loop_gate.rpt').read_text().splitlines()
tim209_lines = [line for line in precompile_lines if line.startswith('TIM-209=')]
opt150_lines = [line for line in precompile_lines if line.startswith('OPT-150=')]
assert len(tim209_lines) == 1 and len(opt150_lines) == 1
precompile_tim209_count = int(tim209_lines[0].split('=', 1)[1])
precompile_opt150_count = int(opt150_lines[0].split('=', 1)[1])
constraint_clean_count = (run / 'reports/constraint_violators.rpt').read_text().count(
    'This design has no violated constraints.')
assert precompile_tim209_count == 0
assert precompile_opt150_count == 0
assert constraint_clean_count == 5
receipt = {
    'schema': 'm522_m514_c2d_logic_only_dc_receipt_v4',
    'status': 'PASS_M522_M514_C2D_LOGIC_ONLY_DC_3NS',
    'tool': 'Synopsys Design Compiler V-2023.12-SP3',
    'technology': 'TSMC28 HPC+ standard cells',
    'operating_condition': 'ssg0p9v125c',
    'clock_period_ns': 3.0,
    'wireload': 'ZeroWireload',
    'clock_network': 'explicitly ideal at clk_core before compile',
    'analysis_define': 'SYNTHESIS',
    'flatten_before_mapping': True,
    'cell_area_um2': area,
    'cell_count': cells,
    'sequential_cells': seq,
    'combinational_cells': combo,
    'logic_levels': levels,
    'setup_worst_slack_ns': setup,
    'hold_worst_slack_ns': hold,
    'macro_count': 0,
    'identity': {
        'runner_sha256': runner_sha,
        'input_sha256': identity,
        'dc_shell_launcher': {
            'kind': 'symbolic_link',
            'link_text': launcher_link_text,
            'resolved_path': resolved_dc,
            'invoked_path': invoked_path,
            'direct_resolved_target_invocation': False,
        },
        'resolved_dc_target_sha256': identity['/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell'],
        'one_shot_attempt_path': attempt_path,
        'static_review_schema': 'm522_m514_dc_static_hammer_r6',
        'static_review_status': 'STATIC_GO__EXACT_SHA_ONE_SHOT_DC_AUTHORIZED',
        'historical_vcs_symlink_profile': 'historical_vcs_exact2',
        'sealed_input_root_inventory_count': 5,
    },
    'gates': {
        'negative_wrong_runner_sha_exit_code': negative_rc,
        'precompile_sources': [
            'precompile_build.rpt',
            'check_design_precompile.rpt',
            'check_timing_precompile.rpt',
        ],
        'precompile_tim209_count': precompile_tim209_count,
        'precompile_opt150_count': precompile_opt150_count,
        'constraint_classes_clean': constraint_clean_count,
        'setup_met': setup >= 0,
        'hold_met_after_hold_fix': hold >= 0,
    },
    'admission': {
        'm514_logic_only_dc_sta': True,
        'decoder_support_additive_logic_cost': True,
        'three_ns_premacro_timing_met': setup >= 0 and hold >= 0,
        'full_decoder_trace': False,
        'cycle_speedup': False,
        'energy': False,
        'physical_sram': False,
        'formality': False,
        'paper_ppa_ready': False,
        'system_speedup': False,
        'date_headline': False,
    },
    'required_next_gate': 'Independent receipt-blind DC hammer; optional Formality/SAIF only as decoder-support completeness cost.',
}
(run / 'm522_m514_c2d_logic_only_dc_receipt_r4.json').write_text(
    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
(run / 'RUN_COMPLETE.txt').write_text(
    'PASS_M522_M514_C2D_LOGIC_ONLY_DC_3NS\n'
    f'cell_area_um2={area}\nsetup_worst_slack_ns={setup}\n'
    f'hold_worst_slack_ns={hold}\nmacro_count=0\n'
    'cycle_speedup=false\nsystem_speedup=false\npaper_ppa_ready=false\n')

def reject_constant(value):
    raise ValueError(f'non-finite JSON constant: {value}')

def assert_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite(child)

readback = json.loads(
    (run / 'm522_m514_c2d_logic_only_dc_receipt_r4.json').read_text(),
    parse_constant=reject_constant)
assert_finite(readback)
assert readback == receipt
excluded = {'TOPOLOGY.json', 'SHA256SUMS', 'SHA256SUMS.seal.sha256'}
directories = sorted(
    p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_dir())
files = sorted(
    p.relative_to(run).as_posix() for p in run.rglob('*')
    if p.is_file() and p.relative_to(run).as_posix() not in excluded)
assert not any(p.is_symlink() for p in run.rglob('*'))
topology = {
    'schema': 'm522_exact_output_topology_v2',
    'directories': directories,
    'files_excluding_topology_and_seals': files,
}
(run / 'TOPOLOGY.json').write_text(
    json.dumps(topology, indent=2, sort_keys=True, allow_nan=False) + '\n')
members = [p for p in sorted(run.rglob('*')) if p.is_file() and
           p.relative_to(run).as_posix() not in
           {'SHA256SUMS', 'SHA256SUMS.seal.sha256'}]
(run / 'SHA256SUMS').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(run)}\n'
    for p in members))
(run / 'SHA256SUMS.seal.sha256').write_text(
    f"{hashlib.sha256((run / 'SHA256SUMS').read_bytes()).hexdigest()}  SHA256SUMS\n")
PY

m522_verify_output_package "${m522_staging}"
[[ ! -e "${m522_run}" && ! -L "${m522_run}" ]] || exit 35
mv "${m522_staging}" "${m522_run}"
m522_verify_output_package "${m522_run}"
m522_complete=1
echo "PASS_M522_M514_C2D_LOGIC_ONLY_DC run=${m522_run}"
