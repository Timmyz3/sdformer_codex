#!/opt/anaconda3/bin/python3.12
"""Read-only M2210 hammer for the M2209 selective-bank VCS runner repair."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m2211_m2210_m2209_selective_bank_fill_directed_vcs_one_shot.sh"
OLD_RUNNER = HW / "dc_handoff/scripts/run_m2199_m2198_m2197_selective_bank_fill_directed_vcs_one_shot.sh"
TEST = HW / "tests/test_m2209_selective_bank_fill_vcs_runner_repair_source.py"
CONTRACT = HW / "contracts/m2209_m2200_selective_bank_fill_vcs_runner_repair_source_contract_r1_20260904.json"
AUTHOR = HW / "reviews/m2209_m2200_selective_bank_fill_vcs_runner_repair_source_author_receipt_r1_20260904"
M2200 = HW / "reviews/m2200_m2199_m2197_selective_bank_fill_directed_vcs_failure_result_hammer_r1_20260904"
OLD_ATTEMPT = HW / "results/.m2199_m2197_selective_bank_fill_vcs_attempt_consumed"
OLD_QUARANTINE = HW / "results/m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904.failed_or_incomplete.3417494.quarantine"
PYTHON = Path("/opt/anaconda3/bin/python3.12")
PARSER = HW / "system_simulator/scripts/parse_m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def verify_sealed_dir(root: Path) -> int:
    need(root.is_dir() and not root.is_symlink(), f"sealed root invalid: {root}")
    need(not any(path.is_symlink() for path in root.rglob("*")), f"symlink in {root}")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and outer.is_file(), f"missing seal: {root}")
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        listed[rel.lstrip("*")] = digest
    actual = {
        str(path.relative_to(root)): sha(path)
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    need(listed == actual, f"nonexhaustive or mismatched manifest: {root}")
    outer_digest, outer_name = outer.read_text().split(maxsplit=1)
    need(outer_name.strip().lstrip("*") == "SHA256SUMS" and outer_digest == sha(manifest),
         f"outer seal mismatch: {root}")
    return len(actual)


def validate_runner(text: str) -> None:
    required = {
        "python_path": "PYTHON=/opt/anaconda3/bin/python3.12",
        "python_pin": "sha_mode_exact 873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161 755 yes \"${PYTHON}\"",
        "parser_pin": "sha_mode_exact fde65c8372c9eab82ae49caea03137cdd93d0bd996fe65e9549220869a743571 664 no \"${PARSER}\"",
        "parser_launch": "\"${PYTHON}\" -B \"${PARSER}\" --sim-log \"${WORK}/simv.log\"",
        "cleanup_files": "rm -f -- \"${WORK}/simv\" \"${WORK}/vc_hdrs.h\"",
        "cleanup_dirs": "rm -rf -- \"${WORK}/csrc\" \"${WORK}/simv.daidir\" \"${WORK}/simv.vdb\"",
        "absence": "for build_only in simv vc_hdrs.h csrc simv.daidir simv.vdb; do",
        "symlink_gate": "[[ -z \"$(find -P \"${WORK}\" -type l -print -quit)\" ]] || exit 5",
        "result": "m2211_m2209_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904",
        "attempt": ".m2211_m2209_selective_bank_fill_vcs_attempt_consumed",
        "lock": ".m2211_m2209_selective_bank_fill_vcs_launch_lock",
        "no_retry": "retry=false",
        "no_reuse": "reuse_old_artifacts=false",
    }
    for name, token in required.items():
        need(token in text, f"missing {name}")
    need(text.count(required["python_path"]) == 1, "ambiguous Python path")
    need(text.count(required["parser_launch"]) == 1, "ambiguous parser launch")
    need(not re.search(r'(?m)^\s*"\$\{PARSER\}"\s+--sim-log', text), "direct parser execution")
    need(not re.search(r'(?m)^\s*(chmod|cp|install)\b[^\n]*\$\{PARSER\}', text),
         "parser chmod/copy/install")
    need("m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904.failed" not in text,
         "old raw artifact reuse")
    parser_i = text.index(required["parser_launch"])
    token_i = text.index("grep -Fxq 'RAW_PASS_M2199_M2197_DIRECTED_VCS_PENDING_M2200_RESULT_HAMMER'")
    cleanup_i = text.index(required["cleanup_files"])
    absence_i = text.index(required["absence"])
    symlink_i = text.index(required["symlink_gate"])
    complete_i = text.index("printf 'RAW_PASS_M2211_M2209_DIRECTED_VCS_PENDING_M2212_RESULT_HAMMER\\n' >\"${WORK}/RUN_COMPLETE.txt\"")
    seal_i = text.index('seal_dir "${WORK}"', complete_i)
    publish_i = text.index('mv -T -- "${WORK}" "${RESULT}"', seal_i)
    need(parser_i < token_i < cleanup_i < absence_i < symlink_i < complete_i < seal_i < publish_i,
         "successful cleanup/seal ordering")
    for retained in ("license_preflight.log", "vcs_compile.log", "simv.log", "simv.rc",
                     "parser.log", "receipt.json"):
        need(not re.search(r'(?m)^\s*rm\s[^\n]*' + re.escape(retained), text),
             f"success removes retained evidence: {retained}")


def mutation_rejections(source: str) -> dict[str, bool]:
    python_sha = "873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161"
    parser_sha = "fde65c8372c9eab82ae49caea03137cdd93d0bd996fe65e9549220869a743571"
    variants = {
        "direct_parser": source.replace('"${PYTHON}" -B "${PARSER}"', '"${PARSER}"', 1),
        "wrong_python_path": source.replace("PYTHON=/opt/anaconda3/bin/python3.12", "PYTHON=/usr/bin/python3.12", 1),
        "wrong_python_sha": source.replace(python_sha, "0" * 64, 1),
        "wrong_python_mode": source.replace(f"sha_mode_exact {python_sha} 755 yes", f"sha_mode_exact {python_sha} 775 yes", 1),
        "parser_sha_drift": source.replace(parser_sha, "1" * 64, 1),
        "parser_mode_drift": source.replace(f"sha_mode_exact {parser_sha} 664 no", f"sha_mode_exact {parser_sha} 755 yes", 1),
        "missing_simv_vdb_cleanup": source.replace(' "${WORK}/simv.vdb"', "", 1),
        "old_result_identity": source.replace("m2211_m2209_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904", "m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904", 1),
        "retry_enabled": source.replace("retry=false", "retry=true"),
        "old_artifact_reuse": source.replace("reuse_old_artifacts=false", "reuse_old_artifacts=true"),
    }
    rejected: dict[str, bool] = {}
    for name, variant in variants.items():
        try:
            validate_runner(variant)
        except AssertionError:
            rejected[name] = True
        else:
            rejected[name] = False
    need(len(rejected) == 10 and all(rejected.values()), "not all required mutations rejected")
    return rejected


def command_block(text: str, needle: str, end_needle: str) -> str:
    lines = text.splitlines()
    hit = next(index for index, line in enumerate(lines) if needle in line)
    start = hit
    while start >= 0 and not lines[start].lstrip().startswith("env -i "):
        start -= 1
    need(start >= 0, f"command start missing: {needle}")
    end = hit
    while end < len(lines) and end_needle not in lines[end]:
        end += 1
    need(end < len(lines), f"command end missing: {end_needle}")
    return "\n".join(lines[start:end + 1])


def verify_quarantine_snapshot() -> tuple[int, int]:
    snap = json.loads((M2200 / "quarantine_snapshot.json").read_text())
    files = snap["files"]
    links = snap["symlinks"]
    need(len(files) == snap["regular_file_count"] == 92, "M2199 file-count drift")
    need(len(links) == snap["symlink_count"] == 2, "M2199 symlink-count drift")
    for entry in files:
        path = OLD_QUARANTINE / entry["path"]
        need(path.is_file() and not path.is_symlink(), f"M2199 file missing/drift: {path}")
        need(sha(path) == entry["sha256"] and path.stat().st_size == entry["size_bytes"],
             f"M2199 file bytes drift: {path}")
        need(stat.S_IMODE(path.stat().st_mode) == int(entry["mode"], 8),
             f"M2199 file mode drift: {path}")
    for entry in links:
        path = OLD_QUARANTINE / entry["path"]
        need(path.is_symlink() and os.readlink(path) == entry["target"], f"M2199 symlink drift: {path}")
    need(not (OLD_QUARANTINE / "SHA256SUMS").exists() and
         not (OLD_QUARANTINE / "SHA256SUMS.seal.sha256").exists(),
         "M2199 quarantine was retrofitted")
    return len(files), len(links)


def main() -> None:
    need(verify_sealed_dir(M2200) == 7, "M2200 member count")
    need(verify_sealed_dir(AUTHOR) == 5, "M2209 author member count")
    need(verify_sealed_dir(OLD_ATTEMPT) == 1, "M2199 attempt seal")
    contract_sidecar = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
    contract_outer = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256.seal.sha256")
    need(contract_sidecar.read_text().split()[0] == sha(CONTRACT), "contract sidecar")
    need(contract_outer.read_text().split()[0] == sha(contract_sidecar), "contract outer seal")

    identities = {
        "runner": (RUNNER, "19958f56f230ee1ced6c42c54201b79379db803d17f1629b7ecbaa549a477a45"),
        "test": (TEST, "be964ade312e792c38e8240f03a4f49abb644b4e424d418331f10c9d8655b549"),
        "contract": (CONTRACT, "4f44a95b2e22d31afc520a0b62d194a7fbbd175101caa816162773cb6a1247bb"),
        "old_runner": (OLD_RUNNER, "745da777421e5601776f1caf158f4905fdbe8c82f6c0095c118d7b2d98ceb3fb"),
        "python": (PYTHON, "873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161"),
        "parser": (PARSER, "fde65c8372c9eab82ae49caea03137cdd93d0bd996fe65e9549220869a743571"),
        "rtl": (HW / "rtl_m2193/m2193_c2_tsbg_b4_selective_bank_fill_frontend.sv", "f651ea3a3b4dfab04d021a1e44797e7ab72c244cb7edf7496e18ac1ac033339e"),
        "m803": (HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv", "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156"),
        "m2018": (HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv", "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21"),
        "sva": (HW / "verif_m2197/m2197_c2_tsbg_selective_bank_fill_assertions.sv", "8003115edb919e9c5c6c9c36ce4ba75dfb37d9ec9f23e7c4cf59e2aed3b461b4"),
        "tb": (HW / "tb_m2197/tb_m2197_c2_tsbg_selective_bank_fill_directed.sv", "a8a954826324aa20443e7b2acbbc6a0b1b2a92f83ebdd84bfdbb0879920526e3"),
        "filelist": (HW / "dc_handoff/filelists/tcasii_m2197_c2_tsbg_selective_bank_fill_directed_vcs.f", "5beddf477b6938b599cfab962eba60f6d79dceeb825380f2e5cdc6f22b49dc13"),
        "m2197_test": (HW / "tests/test_m2197_c2_tsbg_selective_bank_fill_validation_repair_source.py", "81d4cb93e7534e5ebb6cf68c02ded17db862479ab646deccc9ef9eb60e50dd5d"),
        "docs359": (DOCS359, "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"),
    }
    for name, (path, expected) in identities.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == expected, f"identity drift: {name}")
    need(stat.S_IMODE(PYTHON.stat().st_mode) == 0o755 and os.access(PYTHON, os.X_OK), "Python mode")
    need(stat.S_IMODE(PARSER.stat().st_mode) == 0o664 and not os.access(PARSER, os.X_OK), "parser mode")
    need(stat.S_IMODE(RUNNER.stat().st_mode) == 0o664 and not os.access(RUNNER, os.X_OK), "runner mode")

    source = RUNNER.read_text()
    validate_runner(source)
    rejected = mutation_rejections(source)
    subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)], check=True, timeout=30)

    old = OLD_RUNNER.read_text()
    need(command_block(source, '"${VCS}" -full64', 'vcs_compile.log') ==
         command_block(old, '"${VCS}" -full64', 'vcs_compile.log'), "VCS compile command drift")
    need(command_block(source, '300s "${WORK}/simv"', 'simv.log') ==
         command_block(old, '300s "${WORK}/simv"', 'simv.log'), "simv command drift")

    regular_count, symlink_count = verify_quarantine_snapshot()
    result_paths = [
        HW / "results/m2211_m2209_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904",
        HW / "results/.m2211_m2209_selective_bank_fill_vcs_attempt_consumed",
        HW / "results/.m2211_m2209_selective_bank_fill_vcs_launch_lock",
    ]
    need(not any(path.exists() or path.is_symlink() for path in result_paths), "M2211 identity not virgin")
    need((OLD_ATTEMPT / "ATTEMPT_CONSUMED.txt").read_text().startswith("status=M2199_ATTEMPT_CONSUMED"),
         "M2199 attempt state")

    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "pt_shell", "fm_shell",
               "icc2_shell", "icc2_exec", "lm_shell", "lm_shell_exec"}
    collisions = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            names = {(proc / "comm").read_text().strip(), Path(os.readlink(proc / "exe")).name}
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if names & blocked:
            collisions.append([proc.name, sorted(names)])
    need(not collisions, f"same-UID EDA process active: {collisions}")

    print(json.dumps({
        "status": "PASS_M2210_INDEPENDENT_READ_ONLY_MECHANICAL_HAMMER",
        "sealed_inputs": {"m2200_members": 7, "m2209_author_members": 5, "m2199_attempt_members": 1},
        "identity_count": len(identities),
        "mutation_rejections": rejected,
        "parser_launch": "fixed-python-3.12-dash-B",
        "success_cleanup_items": 5,
        "m2199_snapshot": {"regular_files": regular_count, "symlinks": symlink_count, "modified": False},
        "m2211_virgin": True,
        "vcs_compile_command_unchanged": True,
        "simv_command_unchanged": True,
        "same_uid_eda_collisions": [],
        "vcs_runs": 0,
        "license_queries": 0,
        "eda_runs": 0,
        "gpu_runs": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
