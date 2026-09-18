#!/bin/bash
# T54: 判定 `bin_ref`（1.19861）与锚点（1.19951）的 0.075% 差异来自哪里。
#
# 已知（已核）：
#   - checkpoint md5/sha256 与锚点记录**逐位相同**（4bbaf7fc…）；
#   - config sha256 与锚点记录**逐位相同**（630e735c…）；
#   - 同一 overlay 下重评 `bin_ref` **逐位相同**（确定性成立）；
#   - 与锚点相比 80/93 层 spikes 有微小差异、正负号混杂、量级一致（~2e-5 相对）
#     ⇒ 指向"跨环境浮点差"而非"局部语义改动"。
# 唯一剩下的自变量 = **overlay 那几处修复**。
#
# 本脚本把 overlay 换回**改前快照**重评 bin_ref（新目录，不覆盖现有产物），然后
# **无论成功失败都还原改后文件**并校验 md5。
#
# 判读：
#   得到 1.19951 ⇒ 修复确实改了前向（"前向逐位不变"是错的，要查是哪一处）；
#   得到 1.19861 ⇒ 差异来自环境（锚点是 2026-08-30 跨环境产物），修复前向中性成立，
#                  锚点数字应带容差带引用。
set -u
D=/root/private_data/work/sdformer_codex/SDformer
E=$D/neuron_experiments/H9_bipolar_self_attention
O=$E/overlay/models/STSwinNet_SNN
PY=/opt/conda/envs/sdformerflow/bin/python
BK=/root/postfix_overlay_backup_20260918
SRC=/root/prefix_src

echo "=== T54 A/B 开始 $(date -Is) ==="
mkdir -p "$BK/models" "$BK/atlif"

# --- 1. 备份**改后**（当前）文件 ---
cp "$O/bsa_attention.py"                "$BK/models/bsa_attention.py"
cp "$O/h9_load_audit.py"                "$BK/models/h9_load_audit.py"
cp "$O/atlif_ternary_psn/installer.py"  "$BK/atlif/installer.py"
cp "$E/entrypoints/train.py"            "$BK/train.py"
md5sum "$O/bsa_attention.py" "$O/h9_load_audit.py" \
       "$O/atlif_ternary_psn/installer.py" "$E/entrypoints/train.py" | tee "$BK/postfix.md5"

restore() {
    echo "--- 还原改后 overlay ---"
    cp "$BK/models/bsa_attention.py"       "$O/bsa_attention.py"
    cp "$BK/models/h9_load_audit.py"       "$O/h9_load_audit.py"
    cp "$BK/atlif/installer.py"            "$O/atlif_ternary_psn/installer.py"
    cp "$BK/train.py"                      "$E/entrypoints/train.py"
    md5sum "$O/bsa_attention.py" "$O/h9_load_audit.py" \
           "$O/atlif_ternary_psn/installer.py" "$E/entrypoints/train.py"
    echo "--- 校验（应与上面 postfix.md5 一致）---"
    sed 's# .*/# #' "$BK/postfix.md5" | sort > /tmp/expect.txt
    md5sum "$O/bsa_attention.py" "$O/h9_load_audit.py" \
           "$O/atlif_ternary_psn/installer.py" "$E/entrypoints/train.py" \
      | sed 's# .*/# #' | sort > /tmp/got.txt
    diff /tmp/expect.txt /tmp/got.txt && echo "RESTORE_OK" || echo "RESTORE_MISMATCH"
}
trap restore EXIT

# --- 2. 装入**改前**快照 ---
cp "$SRC/bsa_attention.py" "$O/bsa_attention.py"
cp "$SRC/h9_load_audit.py" "$O/h9_load_audit.py"
cp "$SRC/installer.py"     "$O/atlif_ternary_psn/installer.py"
cp "$SRC/train.py"         "$E/entrypoints/train.py"
echo "--- 改前 overlay 已装入 ---"
md5sum "$O/bsa_attention.py" "$O/h9_load_audit.py" \
       "$O/atlif_ternary_psn/installer.py" "$E/entrypoints/train.py"

# --- 3. 重评 bin_ref（覆盖 spike_profile.json；改后那份已存为 run1.json） ---
"$PY" -u /root/t53_polarity_zeroshot.py bin_ref
rc=$?
echo "--- 评测 rc=$rc ---"

# trap 负责还原
echo "=== T54 A/B 结束 $(date -Is) ==="
echo "=== T54_DONE rc=$rc ==="
