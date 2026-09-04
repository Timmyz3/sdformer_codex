#!/usr/bin/env python3
"""Fail-closed custom-function/command closure audit for the M831 R19 runner.

This checker is deliberately independent of ``bash -n``.  Bash accepts an
undefined command word syntactically; M784/R15 demonstrated that this is not a
sufficient launch gate.  The checker removes heredoc payloads, enumerates every
top-level custom function definition, recognizes every reachable custom command
word (including trap handlers), and rejects missing or duplicate definitions.

It also binds every host command used by the runner to a regular, non-symlinked
executable and exact SHA256 from the adjacent whitelist.  The VCS binary,
vcsMsgReport and lmutil are absolute-path assets already bound by the runner;
the generated ``./simv`` is explicitly classified as a post-compile artifact
and is outside the pre-mkdir source closure.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from collections import Counter
from pathlib import Path


DEF_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{")
HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CUSTOM_PREFIXES = (
    "fail", "strict_", "require_", "verify_", "build_", "write_",
    "seal_", "reset_", "cleanup", "signal_", "scan_", "resolve_",
    "read_oom_", "resource_", "runtime_", "finalize_",
)
SHELL_BUILTINS = {
    ":", "[", "[[", "alias", "bg", "bind", "break", "builtin", "caller",
    "cd", "command", "compgen", "complete", "continue", "declare", "dirs",
    "disown", "echo", "enable", "eval", "exec", "exit", "export", "false",
    "fc", "fg", "getopts", "hash", "help", "history", "jobs", "kill", "let",
    "local", "logout", "mapfile", "popd", "printf", "pushd", "pwd", "read",
    "readonly", "return", "set", "shift", "shopt", "source", "suspend", "test",
    "times", "trap", "true", "type", "typeset", "ulimit", "umask", "unalias",
    "unset", "wait",
}
RESERVED = {
    "if", "then", "elif", "else", "fi", "while", "until", "do", "done",
    "case", "esac", "for", "select", "in", "function", "time", "coproc",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_heredocs(text):
    kept = []
    terminator = None
    for line in text.splitlines():
        if terminator is not None:
            if line == terminator or line.lstrip("\t") == terminator:
                terminator = None
            continue
        kept.append(line)
        match = HEREDOC_RE.search(line)
        if match:
            terminator = match.group(1)
    if terminator is not None:
        raise RuntimeError("unterminated heredoc while auditing runner")
    return "\n".join(kept) + "\n"


def remove_comments(line):
    out = []
    single = double = escaped = False
    for char in line:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\" and not single:
            out.append(char)
            escaped = True
            continue
        if char == "'" and not double:
            single = not single
            out.append(char)
            continue
        if char == '"' and not single:
            double = not double
            out.append(char)
            continue
        if char == "#" and not single and not double:
            break
        out.append(char)
    return "".join(out)


def command_words(code):
    """Return conservative shell command words, including trap string heads."""
    found = []
    command_head = re.compile(
        r"(?:^\s*|[;&|]\s*|\(\s*|\$\(\s*|\b(?:if|elif|while|until|then|do|else)\s+)"
        r"(?:!\s*)?(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s;|&]+)\s+)*"
        r"([A-Za-z_][A-Za-z0-9_]*|\./[A-Za-z_][A-Za-z0-9_.-]*)"
    )
    trap_head = re.compile(r"\btrap\s+['\"]([A-Za-z_][A-Za-z0-9_]*)\b")
    for line_no, raw in enumerate(code.splitlines(), 1):
        line = remove_comments(raw)
        if DEF_RE.match(line):
            # The definition name is not a call; commands after the opening
            # brace remain visible to the generic command-head matcher.
            line = DEF_RE.sub("{", line, count=1)
        for match in command_head.finditer(line):
            word = match.group(1)
            if match.end(1) < len(line) and line[match.end(1)] == "=":
                continue
            if word not in RESERVED:
                found.append((line_no, word))
        for match in trap_head.finditer(line):
            found.append((line_no, match.group(1)))
    return found


def audit_text(text, whitelist):
    code = without_heredocs(text)
    defs = Counter()
    for line in code.splitlines():
        match = DEF_RE.match(line)
        if match:
            defs[match.group(1)] += 1
    duplicate_defs = sorted(name for name, count in defs.items() if count != 1)
    words = command_words(code)
    custom_calls = [(line, word) for line, word in words
                    if word in defs or word.startswith(CUSTOM_PREFIXES)]
    undefined = sorted({word for _, word in custom_calls if defs.get(word, 0) == 0})

    external_seen = {word for _, word in words
                            if word not in defs and word not in SHELL_BUILTINS
                            and word not in RESERVED and not word.startswith(CUSTOM_PREFIXES)
                            and word != "./simv"
                            and shutil.which(word, path="/usr/bin:/bin") is not None}
    timeout_literal = "/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save"
    if timeout_literal in code:
        external_seen.add("timeout")
    external_seen = sorted(external_seen)
    allowed = whitelist["commands"]
    unknown_external = sorted(set(external_seen) - set(allowed))
    unused_whitelist = sorted(set(allowed) - set(external_seen))
    regular_sha_errors = []
    for command, identity in sorted(allowed.items()):
        path = Path(identity["path"])
        try:
            st = path.lstat()
        except FileNotFoundError:
            regular_sha_errors.append(f"{command}:missing:{path}")
            continue
        if not stat.S_ISREG(st.st_mode) or path.is_symlink() or not os.access(path, os.X_OK):
            regular_sha_errors.append(f"{command}:not-regular-executable:{path}")
            continue
        actual = sha256(path)
        if actual != identity["sha256"]:
            regular_sha_errors.append(f"{command}:sha:{actual}")
        resolved = shutil.which(command, path="/usr/bin:/bin")
        if resolved is None or Path(resolved).resolve() != path.resolve():
            regular_sha_errors.append(f"{command}:resolved-path:{resolved}")

    stale = "verify_r13_failure_m770_and_author_preflight_prerequisites"
    stale_present = bool(re.search(rf"\b{re.escape(stale)}\b", code))
    return {
        "definitions": dict(sorted(defs.items())),
        "custom_calls": [{"line": line, "name": word} for line, word in custom_calls],
        "duplicate_definitions": duplicate_defs,
        "undefined_custom_calls": undefined,
        "stale_r15_short_name_present": stale_present,
        "external_commands_seen": external_seen,
        "exact_timeout_literal_present": timeout_literal in code,
        "unknown_external_commands": unknown_external,
        "unused_external_whitelist_entries": unused_whitelist,
        "regular_executable_sha_errors": regular_sha_errors,
        "pass": not duplicate_defs and not undefined and not stale_present
                and not unknown_external and not unused_whitelist and not regular_sha_errors,
    }


def mutate(text, mode):
    full = "verify_r13_failure_m770_m782_and_author_preflight_prerequisites"
    stale = "verify_r13_failure_m770_and_author_preflight_prerequisites"
    if mode == "none":
        return text
    if mode == "delete-definition":
        return re.sub(rf"(?m)^{re.escape(full)}\(\)\s*\{{", "deleted_definition() {", text, count=1)
    if mode == "rename-definition":
        return re.sub(rf"(?m)^{re.escape(full)}\(\)\s*\{{", f"{full}_renamed() {{", text, count=1)
    if mode == "inject-stale":
        anchor = 'CURRENT_PHASE="pre_mkdir_vcs_full64_identity_and_license_status"'
        if anchor not in text:
            raise RuntimeError("dry-run boundary anchor missing")
        return text.replace(anchor, stale + "\n" + anchor, 1)
    raise RuntimeError("unknown mutation: " + mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", type=Path)
    parser.add_argument("whitelist", type=Path)
    parser.add_argument("--mutation", choices=("none", "delete-definition", "rename-definition", "inject-stale"), default="none")
    parser.add_argument("--expect-fail", action="store_true")
    args = parser.parse_args()
    if not args.runner.is_file() or args.runner.is_symlink():
        raise RuntimeError("runner must be a regular non-symlink file")
    if not args.whitelist.is_file() or args.whitelist.is_symlink():
        raise RuntimeError("whitelist must be a regular non-symlink file")
    whitelist = json.loads(args.whitelist.read_text(encoding="utf-8"))
    result = audit_text(mutate(args.runner.read_text(encoding="utf-8"), args.mutation), whitelist)
    observed_pass = bool(result["pass"])
    expected_pass = not args.expect_fail
    result.update({"schema": "m831_r19_runner_function_closure_v1",
                   "mutation": args.mutation, "expected_pass": expected_pass,
                   "observed_pass": observed_pass})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if observed_pass == expected_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
