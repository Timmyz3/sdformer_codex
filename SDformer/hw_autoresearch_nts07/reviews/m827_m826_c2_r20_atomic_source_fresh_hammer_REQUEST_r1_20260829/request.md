# M827 fresh independent source-hammer request for M826/C2 R20

请由未参与 M826 authoring 的 receipt-blind reviewer 审查。不得信任作者 PASS，也不得运行 VCS、simv、license 查询或任何 EDA。

M826 唯一允许的改动是关闭 M823-P1-01：future final-hammer 的 authorization 必须在键、值与 Python 类型上完全等于闭合的一次 VCS/simv/license 集合。必须实际构造合法 future chain 和至少以下五类非法 final hammer：`run_vcs=false`、`run_simv=false`、`query_license=false`、`max_attempts=0`、extra key；五类必须全部被拒绝。还需穷举缺键、bool/int 类型混淆，并继续攻击 duplicate key 和 nonfinite JSON。

M803 RTL/SVA/TB/filelists、五档 exact 周期以及 M822 的四类 attempt receipt `false/false/true/true` 必须保持冻结。reviewer 应独立生成四份 CLI 双封 receipt，证明 exact collision 双侧 no-clobber、rename 后损坏仍保守 consumed。

PASS100 也只允许之后另行 author true release 与 final-hammer request；本请求不授权立即启动 VCS、simv、license 或正式 attempt。
