# M778 请求：M777 decoder address-timed additive repair fresh hammer

请先完整阅读 M773 的 FAIL62，再对 M777 做独立、只读、fail-closed 的 source hammer。本请求不授权 M686/M699 production replay，不允许生成 decoder cycle/speedup/result，不允许 RTL/VCS/EDA/GPU/remote。

M777 只修一套旧模型，不开新 matcher。必须把配置标签剥掉后再攻击三条执行路径：A1 是 source-tile-local OSG header + 独立 K1 descriptor/weight service；equal-service K1x8 是 bank-unique group 内的独立 descriptor/weight service；typed K8 是同一 contributor group 的 bundled descriptor + multi-bank weight request。三者必须保留相同 96 lane、Acc24、245760 B、3 ns、192 B/cycle、resource hash、contributor 数、dense commit hash，且只准比较 K8/K1x8。

重点攻击 221184 B psum 物理边界、D3 100 个有限 stripe、dirty eviction/restore 的外部收费、K24/K25 bank-local golden、source/descriptor/refill 三类外部流量、M712/M722-r2 可执行 oracle 和 D1 完整 shape/density fallback。通过也只能申请下一份 additive one-shot release，不能形成论文数字。
