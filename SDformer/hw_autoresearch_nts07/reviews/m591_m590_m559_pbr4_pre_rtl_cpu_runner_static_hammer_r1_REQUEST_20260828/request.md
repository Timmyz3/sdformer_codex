# M591｜M590 / M559 PBR4 r6 repaired immutable CPU source fresh static hammer request

请由未参与 M590 authoring 的 reviewer 做 fresh independent、read-only source hammer。完整读取 M588 FAIL、
M559 r4/M562、递归 imports/goldens、M590 source contract、M591 handoff/future schema与 r6 analyzer/runner。

不得运行正式 analyzer/runner，不得读取真实 M511/weight payload，不得创建 launch candidate、authorization、
wrapper、result或attempt，不得运行 RTL/VCS/DC/PT/PTPX/Formality、训练/GPU/远端。

必须用 AST、shell semantics与 synthetic small golden逐项打击：

1. ready必须是 frozen `low3!=000`；四个 production architecture traces逐 event对上18/18/22/21；
   terminal必须有 state/owner/bitmap/clear-index/count/hash/directory ownership并逐 event对上两个 golden；
2. candidate descriptor显式 typed value/sign；candidate signed INT8(+1)/Acc24 wrap；direct reference必须独立
   mmap/offset/signed-decode/kernel geometry，禁止调用 `event_taps` 或 `WeightSet.get`；输出 owner/address/beat/
   hold/retire与 mismatch必须计算；
3. 每行 mandatory source/descriptor/group/weight/psum/backing/output/directory/occupancy/capacity账本；
   696.24M/926.88M/11.04M/1600必须由 production执行计数断言；四点 common ledger/hash、traffic/
   conservation/hidden resource gate必须控制GO与support；
4. N0--N9无反向边；shell在exec前rehash analyzer；analyzer精确验证 execution/source/schema、四stage review
   schema/status/100/0/0与md/json/manifest/outer、wrapper path/PID/starttime/cmdline和输入双封；
5. candidate/reference mmap在publication前close；任一post-attempt失败（含post-rename final verify）必须重封
   attempt并把staging或canonical output移到唯一quarantine，最终canonical output absent；
6. runner/analyzer/source/schema/request/handoff exact SHA；canonical M590 result/attempt与所有N3--N7对象仍 absent。

P0/P1必须0/0才可PASS。PASS仍不授权执行，只允许另开 launch-candidate review authoring。

