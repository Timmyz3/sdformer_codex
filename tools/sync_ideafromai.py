#!/usr/bin/python3.12
"""Copy a bounded research snapshot into this repository; dry-run by default.

The external ideafromai directory remains the editing source. This script never
runs Git, follows symlinks, or removes files it has not previously managed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import filecmp
import json
import os
from pathlib import Path
import re
import shutil
import zipfile


REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "ideafromai"
DEFAULT_SOURCE = Path("/home/zhumd/work/ideafromai")
TEXT_LIMIT = 8 * 1024 * 1024
NPZ_LIMIT = 1024 * 1024
STATE = ".sync-managed-files.json"
RESERVED = {STATE, "README_SYNC.md", ".gitignore"}
TEXT_EXT = set("""
.md .rst .tex .bib .py .pyi .c .cc .cpp .cxx .h .hh .hpp .sv .svh .v .vh
.sh .bash .tcl .sdc .xdc .f .mk .cmake .ys .toml .yaml .yml .ini .cfg
.json .jsonl .csv .tsv .txt .log .rpt .out .dais .dat .mem .hex .xml
.html .css .js .svg .ipynb .wrap
""".split())
TEXT_NAMES = {"Makefile", "CMakeLists.txt", "Dockerfile", "meson.build",
              "meson.options", ".gitignore", ".gitattributes", ".gitmodules",
              ".clang-format", ".editorconfig"}
BINARY_EXT = set("""
.pt .pth .ckpt .npy .onnx .h5 .hdf5 .safetensors .bin .bitpack .tar .gz
.zip .zst .xz .bz2 .7z .o .a .so .dll .dylib .exe .pyc .pyo .d .gch
.vcd .fsdb .saif .vvp .wlf .vpd
""".split())
SMALL_REPORT_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".xlsx"}
DATA_DIR = re.compile(r"(^|_)(capture\d*|cache\d*|caches|dataset|datasets|training_data|train_cache)(_|$)")
CAPTURE_NAME = re.compile(r"(^capture[_\d]|_(source|codes|windows|activity)\.|^[a-z]+_city_|^(gates|accepted|valid_decisions|predictions)\.)")
PARAMETER_KEYS = {"weight_int8", "wq", "weights", "weight", "u_int8", "u",
                  "v", "source_a", "consumer_a", "conv2_u_r16", "w2",
                  "first_factor32", "coefficients", "threshold_int32",
                  "group_low_bit_width", "group_constant", "mask_original",
                  "fixed_level_index", "node_specific_index", "lifting_q12",
                  "as_q16", "u_conv2_theta_q16", "f_q16", "u_ped_q16", "v_ped_q16"}
SECRET_RULES = [
    ("private-key", re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    ("github-token", re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})")),
    ("aws-access-id", re.compile(rb"AKIA[A-Z0-9]{16}")),
    ("service-token", re.compile(rb"(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}|hf_[A-Za-z0-9]{30,})")),
    ("url-embedded-password", re.compile(rb"(?:https?|ssh)://[^\s/:@]{1,80}:[^\s/@]{8,100}@")),
]


def directory_reason(parts):
    for part in parts:
        if part.startswith(".venv") or part in {"venv", "env", "env312", "site-packages"}:
            return "environment"
        if part in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
            return "cache_or_nested_git"
        if (part in {"obj_dir", "obj", "verilator_obj", "CMakeFiles", "build", "dist"}
                or part.endswith("_obj") or part.startswith("obj_")):
            return "build_directory"
    return None


def classify(path, relative, size):
    """Return (included category, exclusion reason), without loading tensors."""
    if reason := directory_reason(relative.parts[:-1]):
        return None, reason
    if relative.as_posix() in RESERVED:
        return None, "snapshot_control_filename"
    name, suffix = path.name, path.suffix.lower()
    if name.lower() in {".env", ".netrc", ".git-credentials", "id_rsa", "id_ed25519"}:
        return None, "credential_filename"
    if suffix in BINARY_EXT:
        return None, "binary_payload_or_build"
    if suffix == ".npz":
        if size > NPZ_LIMIT:
            return None, "npz_over_1MiB"
        if (any(part.lower() in {"codes", "training", "train_data"} or DATA_DIR.search(part.lower())
                for part in relative.parts[:-1]) or CAPTURE_NAME.search(name.lower())):
            return None, "npz_capture_codes_or_training_data"
        with zipfile.ZipFile(path) as archive:
            keys = {Path(entry).stem.lower() for entry in archive.namelist()}
        if keys & PARAMETER_KEYS or any(key.endswith("_dictionary") for key in keys):
            return "small_parameter_or_test_npz", None
        return None, "npz_not_recognized_parameter_or_test"
    if suffix in SMALL_REPORT_EXT:
        if "literature" in relative.parts and suffix == ".pdf":
            return None, "external_paper_pdf"
        return ("small_report_asset", None) if size <= NPZ_LIMIT else (None, "report_asset_over_1MiB")
    policy = name.upper().startswith(("LICENSE", "COPYING", "NOTICE", "COPYRIGHT", "AUTHORS"))
    if suffix not in TEXT_EXT and name not in TEXT_NAMES and not policy:
        return None, "unselected_file_type"
    if size > TEXT_LIMIT:
        return None, "text_over_8MiB"
    return "license" if policy else "text", None


def generated_files(source, npz_paths):
    readme = f"""# ideafromai Git snapshot

唯一编辑源是 `{source}`。本目录是主仓中的受控快照，请在原目录修改研究文件，
再运行 `/usr/bin/python3.12 tools/sync_ideafromai.py --sync`；不在两处分别编辑。
默认不带 `--sync` 时只扫描并输出计数，不写文件。其他机器可显式传 `--source`。

纳入源码、文档、RTL/TB、配置和结果文本，单文件上限 **8 MiB**，包括低于该上限的
完整编译 DAG。小图片、作者自己的报告等附件上限 1 MiB；外部论文 PDF 保留在本地，
来源链接仍在文献笔记中。第三方干净源码及许可证保留，嵌套 `.git` 与环境/构建不复制。

NPZ 上限 **1 MiB**，只纳入含已列明权重/系数/阈值/字典/分组/小测试参数字段的文件。
规则在同步脚本 `PARAMETER_KEYS` 中；只读取 ZIP 成员名，不执行 pickle 或加载张量。
排除 capture/cache/dataset、独立 codes/training 数据目录，以及源门、码流、窗口、预测
等捕获文件。带 train16/recovery 的研究目录不会整体排除：其中代码、结果和模型参数
仍按上述规则选择。必要参数若仅存在于捕获目录或大文件内，本快照不会擅自拆出。

虚拟环境、缓存、仿真构建、波形、执行文件、PT/checkpoint、原始 NPY/bitpack、完整捕获
及压缩训练包不纳入。大于 8 MiB 的逐源轨迹/范围/DAG 也留在原目录；邻近汇总和生成
脚本保留。因此本快照支持审阅与有齐备参数/测试输入的局部重建，**不是完整数据备份**，
也不保证只凭 Git clone 就能重跑训练、整网 AEE 或所有仿真。外部数据、checkpoint、
环境依赖和未随源码取回的第三方子模块仍须按各子目录说明准备。

同步仅单向复制。本地 `{STATE}` 只记录该脚本管理过的相对路径，供下一次清理已退出
选集的文件；不含哈希，不提交 Git。首次同步不删除既有文件；遇到不同内容的未管理
目标文件会报告冲突而不覆盖。疑似凭据检查只输出路径和类型，命中的文件不复制；
它是对所选文本的有限模式扫描，不是对大型载荷的安全认证。同步脚本不执行 Git。
"""
    ignore = """# The sync script selects snapshot content; runtime data stays at the source.
/.sync-managed-files.json
**/.venv*/
**/venv/
**/env/
**/__pycache__/
**/.git/
**/obj_dir/
**/*_obj/
**/build/
*.pyc
*.o
*.a
*.so
*.pt
*.pth
*.ckpt
*.npy
*.npz
*.bin
*.bitpack
*.tar
*.gz
*.zip
*.zst
*.vcd
*.fsdb
*.saif
*.vvp
# Exact exceptions for the small parameter/test archives selected by the script.
""" + "".join(f"!/{name}\n" for name in sorted(npz_paths))
    return {"README_SYNC.md": readme.encode(), ".gitignore": ignore.encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sync", action="store_true", help="apply the one-way snapshot update")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_dir() or source == DEST or source in DEST.parents or DEST in source.parents:
        parser.error("source must be an existing directory separate from the snapshot")
    selected = {}
    included, excluded = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    suspects = []
    for base, directories, files in os.walk(source, followlinks=False):
        directories.sort()
        for name in sorted(files):
            path = Path(base) / name
            relative = path.relative_to(source)
            if path.is_symlink():
                excluded["symlink"][0] += 1
                continue
            size = path.stat().st_size
            kind, reason = classify(path, relative, size)
            if reason == "credential_filename":
                suspects.append({"path": relative.as_posix(), "type": reason})
            if kind in {"text", "license"}:
                data = path.read_bytes()
                if b"\0" in data:
                    kind, reason = None, "binary_disguised_as_text"
                else:
                    found = [label for label, pattern in SECRET_RULES if pattern.search(data)]
                    if found:
                        suspects.extend({"path": relative.as_posix(), "type": label} for label in found)
                        kind, reason = None, "suspected_secret"
            bucket = included[kind] if kind else excluded[reason]
            bucket[0] += 1
            bucket[1] += size
            if kind:
                selected[relative.as_posix()] = path

    generated = generated_files(source, [name for name in selected if name.endswith(".npz")])
    for name, data in generated.items():
        selected[name] = data
        included["snapshot_instructions"][0] += 1
        included["snapshot_instructions"][1] += len(data)
    state_path = DEST / STATE
    previous = set(json.loads(state_path.read_text())["files"]) if state_path.exists() else set()
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in previous):
        parser.error("managed-file list contains a non-relative path")
    create, update, unchanged, conflicts = [], [], [], []
    for name, item in selected.items():
        target = DEST / name
        if not target.exists() and not target.is_symlink():
            create.append(name)
            continue
        same = (target.is_file() and not target.is_symlink() and
                (target.read_bytes() == item if isinstance(item, bytes)
                 else filecmp.cmp(item, target, shallow=False)))
        if same:
            unchanged.append(name)
        elif name in previous:
            update.append(name)
        else:
            conflicts.append(name)
    remove = sorted(name for name in previous - selected.keys() if (DEST / name).is_file())
    report = {"mode": "sync" if args.sync else "dry-run", "source": str(source), "destination": str(DEST),
              "selected_files": len(selected), "selected_bytes": sum(v[1] for v in included.values()),
              "included_counts_bytes": dict(included), "excluded_counts_bytes": dict(excluded),
              "create": len(create), "update": len(update), "unchanged": len(unchanged),
              "remove_previously_managed": len(remove), "unmanaged_conflicts": conflicts,
              "suspected_credentials_paths_types_only": suspects}
    if args.sync and not conflicts:
        for name in create + update:
            target, item = DEST / name, selected[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(item, bytes):
                target.write_bytes(item)
            else:
                shutil.copy2(item, target)
        for name in remove:
            (DEST / name).unlink()
        state_path.write_text(json.dumps({"files": sorted(selected)}, indent=2) + "\n")
        report["applied"] = True
    else:
        report["applied"] = False
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
