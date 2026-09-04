# M622 fresh r5 true-launch hammer request

请对唯一 M621 production admission 与 true release 做 fresh、independent、read-only launch hammer。允许内部 authorization validate-only、旧 M614 admission 拒绝测试和 lineage preflight；严禁 runner `--execute`、formal analyzer、正式 result/attempt/consumed、GPU、EDA 或 remote。

必须验证 M621 admission 的 exact schema、SHA/双封、`launch_now=true`、`release=true`、`max_attempts=1`，以及 M620 PASS98 review SHA `30e3027a…3d5b`、manifest `08d09f01…e9e3`、outer `49597ca2…5a45`。还须核验冻结 r5 runner、contract/candidate、M615 true release 与 M616 FAIL evidence 的完整身份。

必须证明旧 M614 admission 因完整 authorization path 不同而被 M617 fail-close；用 `lexists/lstat` 确认所有 r5 result/attempt/consumed/result-runtime-adapter staging/qraw/qstage/qfinal 坐标不存在。重新采集三次、间隔 2 秒的资源/cgroup/collision 样本。

PASS gate 为 score≥95、P0=0、P1=0。PASS 仅表示 root 在唯一 invocation 紧前 fresh live recheck 后可调用一次 component analyzer；release 本身不是执行，raw result 仍必须经过 fresh independent result hammer。
