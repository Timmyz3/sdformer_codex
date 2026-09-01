#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1715 blocking-queue-order successor for the exact M1710 C2 campaign.

M1710 failed before consuming its attempt because its pre-lock collision scan
observed another cooperating C1 dc_shell.  M1715 permanently binds that sealed
failure and never retries M1710.  It first blocks on the shared flock, then
performs the same-UID collision scan and the exact six-source runtime/force
rebind, and only then may consume its own fresh attempt.  Workloads, budgets,
force/lexists gates and per-launch rescans remain unchanged.  This source is
inert until an exact different-author M1716 hammer and M1717 release exist.
"""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CONTRACT = HW / "contracts/m1715_m1710_m1684_c2_queue_order_repair_production_energy_source_contract_r1_20260901.json"
M1716 = HW / "reviews/m1716_m1715_m1710_m1684_c2_queue_order_repair_production_energy_source_hammer_r1_20260901"
M1717 = HW / "contracts/m1717_m1716_m1715_m1710_m1684_c2_queue_order_repair_production_energy_launch_release_r1_20260901.json"
CHECKER = HW / "system_simulator/scripts/check_m1715_m1710_m1684_c2_queue_order_repair_production_energy_source.py"
TEST = HW / "system_simulator/tests/test_m1715_m1710_m1684_c2_queue_order_repair_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1715_checker", CHECKER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1715 checker import unavailable")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

BASE = HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901"
M1677 = HW / "reviews/m1677_m1661_m1652_c2_resource_gate_successor_three_axis_dc_result_hammer_r1_20260901"
M1627 = HW / "reviews/m1627_m1613_c2_registered_fault_directed_vcs_result_independent_hammer_r1_20260901"
M1568 = HW / "reviews/m1568_m1502_c2_mapped_first_fault_forensic_r1_20260901"
M1502_FAILURE = HW / "results/m1502_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine"
M1609 = HW / "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"
M1684_RUNNER = HW / "dc_handoff/scripts/run_m1684_m1661_c2_m1609_fresh_mapped_production_energy_one_shot.py"
M1684_CONTRACT = HW / "contracts/m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_contract_r1_20260901.json"
M1684_AUTHOR = HW / "reviews/m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_author_receipt_r1_20260901"
M1685_FAILED = HW / "reviews/m1685_m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_hammer_r1_20260901"
M1698_RUNNER = HW / "dc_handoff/scripts/run_m1698_m1684_m1661_c2_shared_eda_queue_production_energy_one_shot.py"
M1698_CONTRACT = HW / "contracts/m1698_m1684_c2_shared_eda_queue_production_energy_source_contract_r1_20260901.json"
M1698_AUTHOR = HW / "reviews/m1698_m1684_c2_shared_eda_queue_production_energy_source_author_receipt_r1_20260901"
M1699_FAILED = HW / "reviews/m1699_m1698_m1684_c2_shared_eda_queue_production_energy_source_hammer_r1_20260901"
M1686_FORBIDDEN = HW / "contracts/m1686_m1685_m1684_m1661_c2_m1609_fresh_mapped_production_energy_launch_release_r1_20260901.json"
M1700_FORBIDDEN = HW / "contracts/m1700_m1699_m1698_m1684_c2_shared_eda_queue_production_energy_launch_release_r1_20260901.json"
M1710_RUNNER = HW / "dc_handoff/scripts/run_m1710_m1684_m1661_c2_runtime_bound_shared_eda_queue_production_energy_one_shot.py"
M1710_CHECKER = HW / "system_simulator/scripts/check_m1710_m1684_c2_runtime_bound_shared_eda_queue_production_energy_source.py"
M1710_TEST = HW / "system_simulator/tests/test_m1710_m1684_c2_runtime_bound_shared_eda_queue_production_energy_source.py"
M1710_CONTRACT = HW / "contracts/m1710_m1684_c2_runtime_bound_shared_eda_queue_production_energy_source_contract_r1_20260901.json"
M1710_AUTHOR = HW / "reviews/m1710_m1684_c2_runtime_bound_shared_eda_queue_production_energy_source_author_receipt_r1_20260901"
M1711_REVIEW = HW / "reviews/m1711_m1710_m1684_c2_runtime_bound_shared_eda_queue_production_energy_source_hammer_r1_20260901"
M1712_RELEASE = HW / "contracts/m1712_m1711_m1710_m1684_c2_runtime_bound_shared_eda_queue_production_energy_launch_release_r1_20260901.json"
M1710_FAILURE = HW / "results/m1710_c2_shared_eda_queue_production_energy_r1_20260901.failed_or_incomplete.quarantine"
M1710_ATTEMPT = HW / "results/.m1710_c2_shared_eda_queue_production_energy_attempt_consumed"
M1710_RESULT = HW / "results/m1710_c2_shared_eda_queue_production_energy_r1_20260901"
M1710_PRIVATE = HW / "results/m1710_c2_shared_eda_queue_production_energy_r1_20260901.private_build.unsealed_do_not_cite"
FILELISTS = {
    "k8": HW / "dc_handoff/filelists/date_m1684_c2_m1609_k8_fresh_mapped_production_energy.f",
    "k1x8": HW / "dc_handoff/filelists/date_m1684_c2_m1609_k1x8_fresh_mapped_production_energy.f",
}
UCLI = HW / "dc_handoff/scripts/m1684_c2_m1609_fresh_mapped_production_energy.ucli.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1684_c2_m1609_fresh_mapped_production_energy_tt0p9v25c.tcl"
DIRECT_EXECUTION_PATHS = {
    "dc_handoff/tb/m1684_c2_m1609_production_binary_fault_assertions.sv",
    "dc_handoff/tb/tb_m1684_c2_m1609_fresh_mapped_production_energy.sv",
    "dc_handoff/scripts/m1684_c2_m1609_fresh_mapped_production_energy.ucli.tcl",
    "dc_handoff/scripts/run_ptpx_m1684_c2_m1609_fresh_mapped_production_energy_tt0p9v25c.tcl",
    "dc_handoff/filelists/date_m1684_c2_m1609_k8_fresh_mapped_production_energy.f",
    "dc_handoff/filelists/date_m1684_c2_m1609_k1x8_fresh_mapped_production_energy.f",
}
DESIGN = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24"
NET_REL = "netlist/" + DESIGN + "_mapped.v"
SDC_REL = "netlist/" + DESIGN + "_mapped.sdc"
TOP = "tb_m1684_c2_m1609_fresh_mapped_production_energy"
SAIF_SCOPE = TOP + ".core.dut"

CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
TT_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db")
SS_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")

ATTEMPT = HW / "results/.m1715_c2_queue_order_repair_production_energy_attempt_consumed"
RESULT = HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901"
FAILURE = HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901.private_build.unsealed_do_not_cite"
WORK = HW / ("results/.m1715_c2_queue_order_repair_production_energy_work." + str(os.getpid()))
STAGE = HW / ("results/.m1715_c2_queue_order_repair_production_energy_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m1715_c2_queue_order_repair_production_energy_failure_stage." + str(os.getpid()))
LOCK = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")

CYCLES = {"k8": [51, 131, 486, 1231, 14],
          "k1x8": [53, 133, 499, 1246, 14]}
EVENTS = [20, 41, 90, 110, 0]
COUNTS = {"vcs_compiles": 2, "simv_runs": 10,
          "saif_files": 10, "ptpx_runs": 10}
SOURCE_CLAIMS = dict((key, False) for key in (
    "vcs", "mapped_functionality", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))

STATIC_SHA = {
    "m1684_runner": "1c7acc502c010809d56dacd78d857dfb5a44cca74e12025134424c6b9c80b77f",
    "m1684_contract": "7fa827aca2ee236a06010d037ca03dac80fc1491abc59a3162c0092bc84e1683",
    "m1684_author_manifest": "5142d91321007cffdcd9f35ab3707f7901a80a91d2d11347224c224ed86b6d30",
    "m1684_author_outer": "dba513a1e26cb9fd18e9a2f8532b2f5f07893c6d3f174f5f2acc523a83afab2a",
    "m1684_author_receipt": "2f1bb4c7ce1a1355c488b75a059e6efe34a305ab01012365a32d0c7370b1724b",
    "m1685_failed_manifest": "141a432b28bb18b52c06c31bf58479bafe5b074b9770d6c8d123bd03c8a4195f",
    "m1685_failed_outer": "7a6fbaec942122be117e3ea9161f4727689cae67360f48b24490c0f08d0ece33",
    "m1685_failed_review": "d0ee4de559e65eb77053cedd79d95f6d724984d5fe876a8428886675a0bac98f",
    "m1698_runner": "60bf8751d6d8eecf3aa469b1bda113b69172f05295bdd712413c9b8c543049be",
    "m1698_contract": "2b1ec47117ac3bdc71be8bd6eecff454f84a5ee0c4ee16d6095440a00a81e9de",
    "m1698_author_manifest": "97091f7e9b6d548412b93676a15dd369e1f5169041aa5d0aaca72ef82c48cf54",
    "m1698_author_outer": "cc675964c73b6e33e1470143f77b6ce1c95a26abf589ee8d5a1931c4c4bdb76c",
    "m1698_author_receipt": "61420ee154335682885e574d3e72a0eb56c932183386cc3a9e5642398a1438f4",
    "m1699_failed_manifest": "38cdfedb6ff7fef2d95d3e7b5f418613fa43d83f2474cb847b784b3b2bb2272c",
    "m1699_failed_outer": "67410e64966845cd8f40e0c829591e9e4adc012b3aecdcdf37536cf9679e6a56",
    "m1699_failed_review": "3b775ecfb3e1278f3571fa8488808a630351e82e24face0af424a01fb244f3f2",
    "m1710_runner": "bf6acc942274fd11bd7731a4120dda833e04ae2cfa0322cd45ac00fbdcafc01b",
    "m1710_checker": "2a93e84b01e53f174bd97f8c7ae8e46dfa1c71bd34ac62e9a4e6ede2a31f74c5",
    "m1710_test": "c4e6be1d08f1d6a64b40ae977579627e999312f2779b1e3107d51f96dfb0cc96",
    "m1710_contract": "9086a6c306b3150e325a6a98b10ada135811fa91d029f100f3d4095bee1da1e5",
    "m1710_author_receipt": "7885281396f3686ca9c076f9cf53e2abdedc164bb421fd0a3dc9051afcb01468",
    "m1710_author_manifest": "2a817b2dd73d574a1d40ad596c64be5d35793ef417c55206d5e6c214be92be0d",
    "m1710_author_outer": "bf6493dec24b4250c6fd4a9264ab6238f03eac88b8bfbfe3de58687799136254",
    "m1711_review": "f9b867a2f272d6d00f55b3ebc463713f2aa2ac152e44a8d99f761fdbcf898b0e",
    "m1711_manifest": "d33dac5f8d39709c33964b164d011f6944c7a0530147fe1da3955b8abcd76d71",
    "m1711_outer": "a04a59fc0311c72fb29d530d9c07f58d6f9e484fab2dc7bee872239a75679fcb",
    "m1712_release": "e5c5371897333962fd372370d4c13a942f56c399b1341d302ece5818fc423a50",
    "m1712_release_sum": "354efd6661f9633fd6948087329702b8e36657bda1733a0580ad099ed7812d65",
    "m1712_release_outer": "dba7b74ca9985fc7c4bfc8e2e1e1a51513e324b7d30c45ef43fa43848a576aa8",
    "m1710_failure_json": "7a334d6a40f7f25ac4152b65a414020a890b1e41a83f62c479589dfd6c8c77ac",
    "m1710_failure_manifest": "c3f67b8ae21f828e0fb007b5933fe20a749950652dfa4c514889c9c47810c73e",
    "m1710_failure_outer": "e8e26336bc59ad54fb781b127e76fb821bc9cfd6e5ee2af5335e86dc4691cf4b",
    "m1609": "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    "cell": "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    "tt_db": "d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070",
    "ss_db": "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    "vcs": "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    "pt": "afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef",
    "lmutil": "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    "m1661_manifest": "22388b70b68f4b038a464446704bdc37fb9f51d536fc12b656b0e51045f5efac",
    "m1661_outer": "f41253a98d74e7b5087c39f49ddbade856ac825f1286c0c73ccf18bdbc6cd4a2",
    "m1677_review": "b05b551375c244746ce10990f2f4ac0757b6e82e3922fc7db8583bd5d1ffc2f5",
    "m1677_manifest": "760dcc9226414e205b8498cbf5a2b051e272f2c64313ac2942e90982f3c0b83d",
    "m1677_outer": "8966dd938975e183784c9017e9eaaf59c641b14ce832c30f9735a08f73463708",
    "m1627_review": "ab4f2187667301a37fbd5f523687a8971282e642163d42886edcdc138edc43d4",
    "m1627_manifest": "670edd3dbf60d0d6122fd4ee769c623456f9774da9c0960c9ce2a3291276df51",
    "m1627_outer": "7443f9553a22cf9189320cb0f1b9850b839dea16f8bb0d92c94da6659113034c",
    "m1568_review": "b88067a9ef94b24960d9d5ba86973b23c7b10a89386c9c624ffa82d8131081b2",
    "m1568_manifest": "279a60e1aaec03523da21f216ef9bbcc22eaba3daf75feb92a0d4976f2a17d71",
    "m1568_outer": "74a8848d6b082ce954d1182a2438ad7f2be6bce7fadda9c8b324feeee0e3bbc8",
    "m1502_failure": "2bad717f51fa99e2526b4ec8b7b305b4bbbf60b84728d6f799de59aa72bfe7d2",
    "m1502_manifest": "a5f02446e2a687c535b16498d5f3cd5a69bd0c15b5eb8ff43d032103a397081e",
    "m1502_outer": "82dfce2ce39c59fdfd61f9501d9806bef67d865c1896d247f2a70d381d237129",
    "k8_net": "6c62d99b444ba25f8eb3f1e491479b44f5613b0323e032af8150e81c84f393c4",
    "k8_sdc": "852c62c1ed8d4a6c69a8fdd17ac7c3b18f0cdee271fb4aaa25fba6a2f77535eb",
    "k1x8_net": "5316db453f0ca70524ea18091e0924f79d116afd46d5432906f3182d1ccfd704",
    "k1x8_sdc": "17414d50eda57b2ba6f1ff3f376c24d2be6c70e9b625f717202cc72ce53c49f2",
}


class Failure(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path: Path, digest: str) -> None:
    if (not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode) or sha(path) != digest):
        raise Failure("identity drift: " + str(path))


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise Failure("duplicate JSON key: " + key)
            value[key] = item
        return value
    if not path.is_file() or path.is_symlink():
        raise Failure("JSON absent/nonregular: " + str(path))
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise Failure("JSON root is not object")
    return value


def verify_seal(root: Path, manifest_sha: str, outer_sha: str) -> None:
    if not root.is_dir() or root.is_symlink():
        raise Failure("sealed directory invalid: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise Failure("outer seal content drift: " + str(root))
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        if (name in listed or rel.is_absolute() or ".." in rel.parts):
            raise Failure("unsafe/duplicate manifest member")
        exact(root / rel, digest)
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != listed:
        raise Failure("sealed population drift: " + str(root))


def seal_dir(root: Path) -> None:
    rows = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in candidate")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            rows.append((path.relative_to(root).as_posix(), sha(path)))
    rows.sort()
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(digest + "  " + name + "\n"
                                for name, digest in rows))
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n")


def publish_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100,
                 os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def verify_contract_sources(contract: dict[str, Any]) -> None:
    if (contract.get("schema") !=
            "m1715_m1710_m1684_c2_queue_order_repair_production_energy_source_contract_r1_v1"
            or contract.get("status") !=
            "SOURCE_ONLY__M1716_REVIEW_AND_M1717_RELEASE_REQUIRED__NO_EDA"
            or contract.get("claim_boundary") != SOURCE_CLAIMS):
        raise Failure("source contract semantic drift")
    rows = contract.get("source_files")
    if not isinstance(rows, list):
        raise Failure("source file inventory absent")
    seen = set()
    for row in rows:
        if set(row) != {"path", "sha256"} or row["path"] in seen:
            raise Failure("source inventory malformed")
        path = HW / row["path"]
        exact(path, row["sha256"])
        seen.add(row["path"])
    required = {RUNNER.relative_to(HW).as_posix(), CHECKER.relative_to(HW).as_posix(),
                TEST.relative_to(HW).as_posix()}
    if not required.issubset(seen):
        raise Failure("source inventory incomplete")


def verify_authority() -> dict[str, Any]:
    pins = {name: os.environ.get(name, "") for name in (
        "M1715_EXPECTED_RUNNER_SHA256",
        "M1715_EXPECTED_SOURCE_CONTRACT_SHA256",
        "M1715_EXPECTED_M1716_REVIEW_SHA256",
        "M1715_EXPECTED_M1716_MANIFEST_SHA256",
        "M1715_EXPECTED_M1716_OUTER_FILE_SHA256",
        "M1715_EXPECTED_M1717_RELEASE_SHA256")}
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in pins.values()):
        raise Failure("fresh M1716/M1717 exact SHA authority absent")
    exact(RUNNER, pins["M1715_EXPECTED_RUNNER_SHA256"])
    exact(CONTRACT, pins["M1715_EXPECTED_SOURCE_CONTRACT_SHA256"])
    contract = strict_json(CONTRACT)
    verify_contract_sources(contract)
    verify_seal(M1716, pins["M1715_EXPECTED_M1716_MANIFEST_SHA256"],
                pins["M1715_EXPECTED_M1716_OUTER_FILE_SHA256"])
    exact(M1716 / "review.json", pins["M1715_EXPECTED_M1716_REVIEW_SHA256"])
    review = strict_json(M1716 / "review.json")
    exact(M1717, pins["M1715_EXPECTED_M1717_RELEASE_SHA256"])
    release_sum = Path(str(M1717) + ".sha256")
    release_outer = Path(str(M1717) + ".sha256.seal.sha256")
    if (release_sum.read_text() != sha(M1717) + "  " + M1717.name + "\n"
            or release_outer.read_text()
            != sha(release_sum) + "  " + release_sum.name + "\n"):
        raise Failure("M1717 double seal drift")
    release = strict_json(M1717)
    if review.get("status") != (
            "PASS_M1716_M1715_M1710_M1684_C2_QUEUE_ORDER_REPAIR_PRODUCTION_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT"):
        raise Failure("M1716 status drift")
    expected_budget = {"future_m1715_attempts": 1, "automatic_retry": False,
                       "vcs_compiles": 2, "simv_runs": 10,
                       "saif_files": 10, "ptpx_runs": 10}
    if (release.get("status") !=
            "AUTHORIZE_ONE_M1715_C2_QUEUE_ORDER_REPAIR_PRODUCTION_ENERGY_ATTEMPT"
            or release.get("authorization") != expected_budget
            or release.get("identity") != {
                "runner_sha256": sha(RUNNER),
                "source_contract_sha256": sha(CONTRACT),
                "m1716_review_sha256": sha(M1716 / "review.json")}
            or release.get("claim_boundary") != SOURCE_CLAIMS):
        raise Failure("M1717 authority semantic drift")
    return contract


def forbidden_release_namespaces_absent() -> None:
    for payload in (M1686_FORBIDDEN, M1700_FORBIDDEN):
        for path in (payload, Path(str(payload) + ".sha256"),
                     Path(str(payload) + ".sha256.seal.sha256")):
            if os.path.lexists(path):
                raise Failure("permanently forbidden release namespace exists: "
                              + str(path))


def verify_m1710_pre_attempt_failure() -> None:
    """Bind the exhausted M1710 launch identity and forbid any M1710 retry."""
    exact(M1710_RUNNER, STATIC_SHA["m1710_runner"])
    exact(M1710_CHECKER, STATIC_SHA["m1710_checker"])
    exact(M1710_TEST, STATIC_SHA["m1710_test"])
    exact(M1710_CONTRACT, STATIC_SHA["m1710_contract"])
    verify_seal(M1710_AUTHOR, STATIC_SHA["m1710_author_manifest"],
                STATIC_SHA["m1710_author_outer"])
    exact(M1710_AUTHOR / "author_receipt.json",
          STATIC_SHA["m1710_author_receipt"])
    verify_seal(M1711_REVIEW, STATIC_SHA["m1711_manifest"],
                STATIC_SHA["m1711_outer"])
    exact(M1711_REVIEW / "review.json", STATIC_SHA["m1711_review"])
    exact(M1712_RELEASE, STATIC_SHA["m1712_release"])
    release_sum = Path(str(M1712_RELEASE) + ".sha256")
    release_outer = Path(str(M1712_RELEASE) + ".sha256.seal.sha256")
    exact(release_sum, STATIC_SHA["m1712_release_sum"])
    exact(release_outer, STATIC_SHA["m1712_release_outer"])
    if (release_sum.read_text() != STATIC_SHA["m1712_release"] + "  " +
            M1712_RELEASE.name + "\n"
            or release_outer.read_text() != STATIC_SHA["m1712_release_sum"] +
            "  " + release_sum.name + "\n"):
        raise Failure("M1712 predecessor release double-seal drift")
    verify_seal(M1710_FAILURE, STATIC_SHA["m1710_failure_manifest"],
                STATIC_SHA["m1710_failure_outer"])
    exact(M1710_FAILURE / "failure.json", STATIC_SHA["m1710_failure_json"])
    failed = strict_json(M1710_FAILURE / "failure.json")
    expected_counts = {"vcs_compiles": 0, "simv_runs": 0,
                       "saif_files": 0, "ptpx_runs": 0}
    if (failed.get("status") != "FAILED_OR_INCOMPLETE"
            or failed.get("phase") != "SOURCE_CHAIN"
            or failed.get("error") != "Failure"
            or failed.get("attempt_consumed") is not False
            or failed.get("counts") != expected_counts
            or failed.get("automatic_retry") is not False
            or failed.get("canonical_result") is not False
            or failed.get("partial_axis_citable") is not False):
        raise Failure("M1710 pre-attempt failure semantic drift")
    for path in (M1710_ATTEMPT, M1710_RESULT, M1710_PRIVATE):
        if os.path.lexists(path):
            raise Failure("M1710 retry/residue namespace exists: " + str(path))


def runtime_bind_execution_sources() -> None:
    old_contract = strict_json(M1684_CONTRACT)
    rows = old_contract.get("source_files")
    if not isinstance(rows, list):
        raise Failure("M1684 execution inventory absent")
    mapping = {}
    for row in rows:
        if (type(row) is not dict or set(row) != {"path", "sha256"}
                or row["path"] in mapping):
            raise Failure("M1684 execution inventory malformed")
        mapping[row["path"]] = row["sha256"]
    if not DIRECT_EXECUTION_PATHS.issubset(set(mapping)):
        raise Failure("M1684 direct execution inventory incomplete")
    for rel in sorted(DIRECT_EXECUTION_PATHS):
        path = HW / rel
        exact(path, mapping[rel])
        forbidden_init = "init" + "reg"
        if forbidden_init in path.read_text().lower():
            raise Failure("forbidden initialization token in runtime source: " + rel)
        if CHECK.active_force_present(path):
            raise Failure("active force in runtime source: " + rel)


def verify_predecessors_and_inputs() -> None:
    verify_m1710_pre_attempt_failure()
    exact(M1698_RUNNER, STATIC_SHA["m1698_runner"])
    exact(M1698_CONTRACT, STATIC_SHA["m1698_contract"])
    verify_seal(M1698_AUTHOR, STATIC_SHA["m1698_author_manifest"],
                STATIC_SHA["m1698_author_outer"])
    exact(M1698_AUTHOR / "author_receipt.json", STATIC_SHA["m1698_author_receipt"])
    verify_seal(M1699_FAILED, STATIC_SHA["m1699_failed_manifest"],
                STATIC_SHA["m1699_failed_outer"])
    exact(M1699_FAILED / "review.json", STATIC_SHA["m1699_failed_review"])
    m1699 = strict_json(M1699_FAILED / "review.json")
    if (m1699.get("verdict") != "FAIL_CLOSED_NO_M1700_RELEASE"
            or m1699.get("authorization", {}).get("m1700_release_authoring") is not False):
        raise Failure("M1699 failed-review/M1700 denial drift")
    forbidden_release_namespaces_absent()
    exact(M1684_RUNNER, STATIC_SHA["m1684_runner"])
    exact(M1684_CONTRACT, STATIC_SHA["m1684_contract"])
    verify_seal(M1684_AUTHOR, STATIC_SHA["m1684_author_manifest"],
                STATIC_SHA["m1684_author_outer"])
    exact(M1684_AUTHOR / "author_receipt.json", STATIC_SHA["m1684_author_receipt"])
    verify_seal(M1685_FAILED, STATIC_SHA["m1685_failed_manifest"],
                STATIC_SHA["m1685_failed_outer"])
    exact(M1685_FAILED / "review.json", STATIC_SHA["m1685_failed_review"])
    failed_review = strict_json(M1685_FAILED / "review.json")
    if (failed_review.get("verdict") != "FAIL_CLOSED_NO_M1686_RELEASE"
            or failed_review.get("authorization", {}).get("m1686_release_authoring") is not False):
        raise Failure("M1685 failed-review/M1686 denial drift")
    verify_seal(BASE, STATIC_SHA["m1661_manifest"], STATIC_SHA["m1661_outer"])
    verify_seal(M1677, STATIC_SHA["m1677_manifest"], STATIC_SHA["m1677_outer"])
    exact(M1677 / "review.json", STATIC_SHA["m1677_review"])
    verify_seal(M1627, STATIC_SHA["m1627_manifest"], STATIC_SHA["m1627_outer"])
    exact(M1627 / "review.json", STATIC_SHA["m1627_review"])
    verify_seal(M1568, STATIC_SHA["m1568_manifest"], STATIC_SHA["m1568_outer"])
    exact(M1568 / "review.json", STATIC_SHA["m1568_review"])
    verify_seal(M1502_FAILURE, STATIC_SHA["m1502_manifest"], STATIC_SHA["m1502_outer"])
    exact(M1502_FAILURE / "failure.json", STATIC_SHA["m1502_failure"])
    exact(M1609, STATIC_SHA["m1609"])
    for path, digest in ((CELL, STATIC_SHA["cell"]), (TT_DB, STATIC_SHA["tt_db"]),
                         (SS_DB, STATIC_SHA["ss_db"]), (VCS, STATIC_SHA["vcs"]),
                         (PT, STATIC_SHA["pt"]), (LMUTIL, STATIC_SHA["lmutil"]),
                         (PYTHON, STATIC_SHA["python"])):
        exact(path, digest)
    for axis in ("k8", "k1x8"):
        exact(BASE / axis / NET_REL, STATIC_SHA[axis + "_net"])
        exact(BASE / axis / SDC_REL, STATIC_SHA[axis + "_sdc"])
    m1677 = strict_json(M1677 / "review.json")
    m1627 = strict_json(M1627 / "review.json")
    m1568 = strict_json(M1568 / "review.json")
    failure = strict_json(M1502_FAILURE / "failure.json")
    if (m1677.get("status") !=
            "PASS100_M1677_M1661_M1652_C2_RESOURCE_GATE_SUCCESSOR_THREE_AXIS_DC_RESULT_ADMITTED"
            or m1627.get("directed_behavior", {}).get(
                "legal_terminal_no_false_pulse") != 1
            or m1627.get("directed_behavior", {}).get(
                "illegal_raw_latched") != 1
            or m1568.get("sealed_boundary", {}).get("first_fault_ps") != 28500
            or failure.get("phase") != "SIM_k8_0"):
        raise Failure("M1502 root-cause/M1609/fresh-netlist chain drift")
    forbidden_release_namespaces_absent()


def namespaces_fresh() -> None:
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(path):
            raise Failure("namespace residue: " + str(path))
    for pattern in (".m1715_c2_queue_order_repair_production_energy_work.*",
                    ".m1715_c2_queue_order_repair_production_energy_stage.*",
                    ".m1715_c2_queue_order_repair_production_energy_failure_stage.*"):
        if next((HW / "results").glob(pattern), None) is not None:
            raise Failure("stale private namespace: " + pattern)


def _parent_pid(pid: int) -> int | None:
    """Return Linux PPID without being confused by spaces in process names."""
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text()
        tail = text[text.rfind(")") + 1:].split()
        return int(tail[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError,
            ValueError, IndexError):
        return None


def _owned_or_ancestor(pid: int, runner_pid: int | None = None) -> bool:
    """Allow the runner, its ancestors and live descendants; reject peers."""
    runner_pid = os.getpid() if runner_pid is None else runner_pid
    ancestry = set()
    cursor = runner_pid
    while cursor > 1 and cursor not in ancestry:
        ancestry.add(cursor)
        parent = _parent_pid(cursor)
        if parent is None:
            break
        cursor = parent
    if pid in ancestry:
        return True
    seen = set()
    cursor = pid
    while cursor > 1 and cursor not in seen:
        if cursor == runner_pid:
            return True
        seen.add(cursor)
        parent = _parent_pid(cursor)
        if parent is None:
            break
        cursor = parent
    return False


def collision_gate() -> None:
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
               "pt_shell", "fm_shell", "icc2_shell", "common_shell_exec",
               "common_shell_exe"}
    hits = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or _owned_or_ancestor(int(item.name)):
            continue
        try:
            if item.stat().st_uid != os.getuid():
                continue
            comm = (item / "comm").read_text().strip()
            argv = [Path(part.decode(errors="replace")).name
                    for part in (item / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in blocked or blocked.intersection(argv):
            hits.append((item.name, comm, argv[:4]))
    if hits:
        raise Failure("same-UID EDA collision: " + repr(hits))


def resource_gate() -> None:
    values = {}
    for row in Path("/proc/meminfo").read_text().splitlines():
        fields = row.replace(":", "").split()
        if fields and fields[0] in {"MemAvailable", "SwapFree",
                                    "CommitLimit", "Committed_AS"}:
            values[fields[0]] = int(fields[1])
    if values.get("MemAvailable", 0) < 24 * 1024 * 1024:
        raise Failure("MemAvailable below 24 GiB")
    if values.get("SwapFree", 0) < 8 * 1024 * 1024:
        raise Failure("SwapFree below 8 GiB")
    if values.get("CommitLimit", 0) - values.get("Committed_AS", 0) < 24 * 1024 * 1024:
        raise Failure("commit headroom below 24 GiB")
    free_bytes = shutil.disk_usage(HW / "results").free
    if free_bytes < 16 * 1024 * 1024 * 1024:
        raise Failure("result disk free below 16 GiB")


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
             "LM_LICENSE_FILE": str(LICENSE_FILE)}
    value.update(extra)
    return value


def run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int,
        output: Path) -> None:
    # This call is deliberately adjacent to subprocess.run: the common flock
    # prevents cooperating campaigns from entering, while this ancestry-aware
    # rescan rejects a non-cooperating same-UID EDA peer without rejecting a
    # live child owned by this campaign.
    if Path(command[0]).name in {"vcs", "pt_shell"}:
        collision_gate()
    with output.open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise Failure("tool failure: " + " ".join(command[:2]))


def result_identity() -> dict[str, str]:
    return {
        "runner_sha256": sha(RUNNER),
        "source_contract_sha256": sha(CONTRACT),
        "m1716_review_sha256": sha(M1716 / "review.json"),
        "m1717_release_sha256": sha(M1717),
        "shared_eda_queue_path": str(LOCK),
        "m1710_runner_sha256": STATIC_SHA["m1710_runner"],
        "m1710_failure_json_sha256": STATIC_SHA["m1710_failure_json"],
        "m1710_failure_manifest_sha256": STATIC_SHA["m1710_failure_manifest"],
        "m1698_runner_sha256": STATIC_SHA["m1698_runner"],
        "m1699_failed_review_sha256": STATIC_SHA["m1699_failed_review"],
        "m1684_runner_sha256": STATIC_SHA["m1684_runner"],
        "m1685_failed_review_sha256": STATIC_SHA["m1685_failed_review"],
        "m1661_manifest_sha256": STATIC_SHA["m1661_manifest"],
        "m1677_review_sha256": STATIC_SHA["m1677_review"],
        "m1627_review_sha256": STATIC_SHA["m1627_review"],
        "m1609_rtl_sha256": STATIC_SHA["m1609"],
        "k8_mapped_netlist_sha256": STATIC_SHA["k8_net"],
        "k1x8_mapped_netlist_sha256": STATIC_SHA["k1x8_net"],
    }


def main() -> int:
    if len(sys.argv) != 1:
        raise Failure("M1715 accepts no arguments")
    state: dict[str, Any] = {"phase": "SOURCE_CHAIN", "attempt": False,
                             "complete": False, "vcs_compiles": 0,
                             "simv_runs": 0, "saif_files": 0,
                             "ptpx_runs": 0}
    lock_handle = LOCK.open("a+")
    try:
        verify_authority()
        verify_predecessors_and_inputs()
        namespaces_fresh()
        state["phase"] = "QUEUE_WAIT"
        # Blocking queue acquisition is the repair: a cooperating C1/C2 job
        # is waited out rather than being misclassified by a pre-lock scan.
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state["phase"] = "POST_LOCK_COLLISION"
        collision_gate()
        state["phase"] = "POST_LOCK_RUNTIME_REBIND"
        runtime_bind_execution_sources()
        forbidden_release_namespaces_absent()
        resource_gate()
        namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"
        if not LICENSE_FILE.is_file() or LICENSE_FILE.is_symlink():
            raise Failure("license file invalid")
        license_check = subprocess.run(
            [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60, check=False)
        if license_check.returncode != 0:
            raise Failure("license preflight failed")

        state["phase"] = "PRE_ATTEMPT_RUNTIME_REBIND"
        collision_gate()
        runtime_bind_execution_sources()
        forbidden_release_namespaces_absent()
        state["phase"] = "ATTEMPT_CONSUME"
        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1715_C2_QUEUE_ORDER_REPAIR_PRODUCTION_ENERGY_ATTEMPT_CONSUMED",
            "identity": result_identity(), "budget": COUNTS,
            "axes": ["k8", "k1x8"], "cases": [0, 1, 2, 3, 4],
            "automatic_retry": False})
        seal_dir(ATTEMPT)
        WORK.mkdir()
        (WORK / "build").mkdir()
        (WORK / "candidate").mkdir()

        for axis in ("k8", "k1x8"):
            state["phase"] = "COMPILE_" + axis
            axis_dir = WORK / "build" / axis
            axis_dir.mkdir()
            state["vcs_compiles"] += 1
            command = [str(VCS), "-full64", "-sverilog", "+v2k",
                       "-timescale=1ns/1ps", "-assert", "svaext",
                       "-debug_access+r", "-lca", "+vcs+lic+wait",
                       "-Mdir=csrc", "-f", str(FILELISTS[axis]),
                       "-top", TOP, "-o", "simv"]
            run(command, cwd=axis_dir,
                env=clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                               "VCS_ARCH_OVERRIDE": "linux"}),
                timeout=3600, output=axis_dir / "compile.log")
            if not (axis_dir / "simv").is_file():
                raise Failure("fresh simv absent: " + axis)
            for case_id in range(5):
                state["phase"] = "SIM_" + axis + "_" + str(case_id)
                state["simv_runs"] += 1
                candidate = WORK / "candidate"
                saif = candidate / (axis + "_case" + str(case_id) + ".saif")
                log = candidate / (axis + "_case" + str(case_id) + ".log")
                report = candidate / (axis + "_case" + str(case_id) + ".assert.report")
                run(["./simv", "-lca", "+M979_UCLI_SAIF",
                     "+M979_CASE=" + str(case_id), "-no_save", "-assert",
                     "report=" + str(report), "-ucli", "-i", str(UCLI)],
                    cwd=axis_dir,
                    env=clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                                   "VCS_ARCH_OVERRIDE": "linux",
                                   "M1684_SAIF_FILE": str(saif)}),
                    timeout=1200, output=log)
                check = candidate / (axis + "_case" + str(case_id) + ".saif_check.json")
                run([str(PYTHON), "-I", str(CHECKER), "--mode", "saif",
                     "--axis", axis, "--case", str(case_id),
                     "--cycles", str(CYCLES[axis][case_id]),
                     "--saif", str(saif), "--log", str(log)],
                    cwd=HW, env=clean_env({}), timeout=180, output=check)
                checked = strict_json(check)
                if (checked.get("status") !=
                        "PASS_M1684_BINARY_CLEAN_DUT_ONLY_PRODUCTION_SAIF"
                        or checked.get("accepted_sources") != EVENTS[case_id]):
                    raise Failure("mapped production/SAIF check failed")
                state["saif_files"] += 1

        # all ten mapped production SAIF gates must close before any PTPX call.
        if any(state[key] != COUNTS[key]
               for key in ("vcs_compiles", "simv_runs", "saif_files")):
            raise Failure("mapped VCS/SAIF campaign incomplete before PTPX")

        metric_rows = []
        for axis in ("k8", "k1x8"):
            for case_id in range(5):
                state["phase"] = "PTPX_" + axis + "_" + str(case_id)
                state["ptpx_runs"] += 1
                candidate = WORK / "candidate"
                saif = candidate / (axis + "_case" + str(case_id) + ".saif")
                pt_dir = candidate / (axis + "_case" + str(case_id) + ".ptpx")
                pt_dir.mkdir()
                run([str(PT), "-f", str(PT_TCL)], cwd=HW,
                    env=clean_env({
                        "DESIGN_NAME": DESIGN, "TT_LIB_DB": str(TT_DB),
                        "SDC_LIB_DB": str(SS_DB),
                        "MAPPED_NETLIST": str(BASE / axis / NET_REL),
                        "MAPPED_SDC": str(BASE / axis / SDC_REL),
                        "GATE_SAIF_FILE": str(saif),
                        "OUTPUT_DIR": str(pt_dir), "SAIF_INSTANCE": SAIF_SCOPE,
                        "SAIF_DURATION_NS": str(CYCLES[axis][case_id] * 3),
                        "MEASUREMENT_CYCLES": str(CYCLES[axis][case_id]),
                        "ACCEPTED_SOURCES": str(EVENTS[case_id]),
                        "AXIS": axis, "CASE_ID": str(case_id),
                        "FAULT_BINARY_CLEAN": "true",
                        "REGISTERED_FAULT_PUBLIC_ZERO": "true"}),
                    timeout=3600, output=pt_dir / "ptpx.log")
                marker = pt_dir / "PTPX_INTERNAL_COMPLETE.txt"
                if (not marker.is_file()
                        or "PASS_M1684_C2_M1609_FRESH_MAPPED_PRODUCTION_PTPX"
                        not in marker.read_text()):
                    raise Failure("PTPX completion marker absent")
                report = pt_dir / "reports/ptpx_power.rpt"
                power_check = pt_dir / "power_check.json"
                run([str(PYTHON), "-I", str(CHECKER), "--mode", "power",
                     "--power-report", str(report)], cwd=HW,
                    env=clean_env({}), timeout=120, output=power_check)
                power = strict_json(power_check)
                metric_rows.append({"axis": axis, "case": case_id,
                                    "cycles": CYCLES[axis][case_id],
                                    "accepted_sources": EVENTS[case_id],
                                    **power})

        if any(state[key] != value for key, value in COUNTS.items()):
            raise Failure("execution-count drift")
        metrics = CHECK.aggregate_metrics(metric_rows)
        metrics.update({
            "schema": "m1715_m1710_m1684_c2_queue_order_repair_production_energy_metrics_r1_v1",
            "status": "CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "scope": "fresh_M1661_M1609_logic_only_premacro_TT0p9V25C_equal_bandwidth_five_directed_cases",
            "clock_period_ns": 3.0, "axes": metrics["axes"],
            "case_rows": metric_rows})

        state["phase"] = "SUCCESS_STAGE"
        STAGE.mkdir()
        shutil.copytree(WORK / "candidate", STAGE / "candidate")
        for axis in ("k8", "k1x8"):
            shutil.copy2(WORK / "build" / axis / "compile.log",
                         STAGE / (axis + ".compile.log"))
        write_json(STAGE / "metrics.json", metrics)
        write_json(STAGE / "receipt.json", {
            "schema": "m1715_m1710_m1684_c2_queue_order_repair_production_energy_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": result_identity(), "one_shot": {
                "attempt_consumed": True, **COUNTS, "automatic_retry": False},
            "axes": ["k8", "k1x8"], "cases_per_axis": 5,
            "same_workloads_clock_and_measurement_window": True,
            "accepted_sources_per_axis": sum(EVENTS),
            "fault_binary_clean_required": True,
            "registered_fault_public_zero_required": True,
            "shared_eda_queue": {
                "path": str(LOCK), "held_for_entire_campaign": True,
                "blocking_acquire_before_first_collision_scan": True,
                "ancestry_aware_post_lock_and_pre_vcs_ptpx_rescan": True},
            "runtime_binding": {
                "direct_execution_sources": 6,
                "checked_after_blocking_lock_and_before_attempt": True,
                "active_force_rescanned": True,
                "m1686_and_m1700_namespaces_forbidden": True},
            "predecessor_failure": {
                "m1710_attempt_consumed": False,
                "m1710_execution_counts": {"vcs_compiles": 0,
                    "simv_runs": 0, "saif_files": 0, "ptpx_runs": 0},
                "m1710_retry_forbidden": True},
            "claim_boundary": SOURCE_CLAIMS})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1715_M1710_C2_QUEUE_ORDER_REPAIR_PRODUCTION_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER\n")
        seal_dir(STAGE)
        publish_no_replace(WORK, PRIVATE)
        publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1715_M1710_C2_QUEUE_ORDER_REPAIR_PRODUCTION_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if not state["complete"]:
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE", "phase": state["phase"],
                    "error": type(error).__name__,
                    "attempt_consumed": state["attempt"],
                    "counts": {key: state[key] for key in COUNTS},
                    "automatic_retry": False, "canonical_result": False,
                    "partial_axis_citable": False})
                seal_dir(FAIL_STAGE)
                publish_no_replace(FAIL_STAGE, FAILURE)
            except BaseException:
                pass
            if WORK.is_dir() and not PRIVATE.exists():
                try:
                    publish_no_replace(WORK, PRIVATE)
                except BaseException:
                    pass
        raise
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
