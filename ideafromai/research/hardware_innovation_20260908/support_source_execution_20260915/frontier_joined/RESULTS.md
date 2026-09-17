# 四槽源执行的完整子链结果

真实 X→源 PSN→最近码/响应类→RTL D 展开→FC1 H384→后级完整 T10 PSN。只报实际 joined RTL 的最后 gate 周期，不相加源叶和后端叶。

主分母为 31 个未参与 pair 选择的训练帧，各固定 32 个抽样位置；仍属于学生训练缓存，非 valid825。先前 frame0 保留训练选择角色，另外两例为诊断。后端固定 τ/增益函数，未闭动态 BN、FC2、shortcut、整网 AEE。

资源：106 个分别实例化的乘法单元（源10+后端96，阶段串行）、256KiB共同参数池、8×128bit每bank一笔在途、3840B门桥、192B D。各臂共同给128B X holding和128B图cache；相对最初源核增加96B X。根均读物理6174，消除了仅根bank位置不同的对照偏差。

## 旧最近响应 W′

| 同权重函数执行臂 | ready周期 | BP周期 | 源实际标量MAC | ready/BP总字节 |
|---|---:|---:|---:|---:|
| static64 + next-X PF | 1989440 | 2179410 | 6348800 | 4508160/4508160 |
| one code | 1879645 | 2232632 | 5543800 | 6258768/6258272 |
| resident frontier code | 1767487 | 2124910 | 4224030 | 6214848/6214224 |
| one class | 1875922 | 2232336 | 5524400 | 6227040/6226336 |
| resident frontier class | 1763758 | 2122178 | 4213500 | 6177216/6176704 |

| 比较，正数表示周期减少 | ready | BP |
|---|---:|---:|
| frontier_code_vs_one_code | 5.9670% | 4.8249% |
| frontier_code_vs_static | 11.1566% | 2.5007% |
| frontier_class_vs_frontier_code | 0.2110% | 0.1286% |
| frontier_class_vs_one_class | 5.9791% | 4.9347% |

## 源费用选择 W″

| 同权重函数执行臂 | ready周期 | BP周期 | 源实际标量MAC | ready/BP总字节 |
|---|---:|---:|---:|---:|
| static64 + next-X PF | 1989440 | 2179175 | 6348800 | 4504576/4504576 |
| one code | 1879645 | 2232487 | 5543800 | 6255184/6254688 |
| resident frontier code | 1767487 | 2124840 | 4224030 | 6211264/6210640 |
| one class | 1860905 | 2213912 | 5447400 | 6242112/6241808 |
| resident frontier class | 1761921 | 2118445 | 4188830 | 6194544/6194144 |

| 比较，正数表示周期减少 | ready | BP |
|---|---:|---:|
| frontier_code_vs_one_code | 5.9670% | 4.8218% |
| frontier_code_vs_static | 11.1566% | 2.4934% |
| frontier_class_vs_frontier_code | 0.3149% | 0.3010% |
| frontier_class_vs_one_class | 5.3191% | 4.3121% |

## 判断

普通 code 已获得前沿并行与驻留增量；class 必须只取相对同权限 code 的额外收益。W″与W′是不同近似权重，跨表不能声称无损加速或继承AEE。前沿/决策图/缓存归借入底座；目前测试用于确定按需生产接口是否值得保留，尚不构成新颖性或PPA证明。

全部 840 个完整 H384 命令通过，Y/U/gate 各 103,219,200 次核对；完整源输出不喂给DUT，仅逐 active(t,c)检查实际结果。无 VCS/DC/PT/Formality、PPA、动态BN或整网测量。

启动时首次脚本在编译期间被修改导致shell偏移执行失败，未产生可用结果；本表仅使用稳定脚本重启后的PASS运行。

运行：`bash run.sh`；再导出 `prepare.py --expanded [--adapt]`，按 `run_expanded.sh` 运行并执行本汇总。
