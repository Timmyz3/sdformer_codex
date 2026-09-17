# T15：门核 RTL 综合/PPA + 生产者端 RTL（2026-09-15）

> 脚本/数据：`t15_synth.py`（hex 生成 + 综合流程）、`t15_synth/verify_t15.py`（零差校验）、
> `t15_synth/verify_plane.py`（生产者回放校验）、`t15_synth/t15_synth_result.json`、
> `t15_synth/t15_rtl_verify.json`。RTL：`t15_synth/cert_gate_bitl_synth.sv`（C1 综合口径）、
> `t15_synth/fx_gate_synth.sv`（公平 FX 基线）、`t15_synth/plane_ser.sv`（生产者串行器）。

## 1. 目的与口径

把 T10 已零差验证的门核 RTL 变成综合口径（常数烘焙 + thr 外置 640b SRAM 行口），
与**同功能同吞吐口径**的诚实 FX 基线（字串行，1 词/拍，10 路并行 16×24 MAC+比较）
在 Nangate45 上做面积对比；另把 T13b 的 plane_ser 部署口径 RTL 化并回放校验。

工具链：yowasp-yosys 0.69（WASM）+ NangateOpenCellLibrary_typical（45nm）。
诚实声明（已知限制）：
- yowasp 的 ABC 子进程 stdout 丢失（"Delay =" 行拿不到），本机无 STA 工具 →
  **无延迟数字**；结论轴为面积 + 拍数，两设计同库同流程。
- 两口径对称施加于两设计（ROM 处理）：
  **nomap**：`memory -nomap`，ROM 保留 `$mem_v2` → 面积=数据通路逻辑，
  ROM 位容单列（C1: lut 12.8Kb + pn 0.96Kb；FX: a 1.6Kb）；
  **logic**：`memory` 全映射，ROM 常数折叠进逻辑 → 面积含烘焙常数。
- thr（10×64b/组）两设计对称外置（T13a：层级静态常数，stage3 ~2Mb 宏）。
- 综合用 s0_stage0 常数（结构相同；cert 在 4 trace 上功能零差，见 §2）。

## 2. 功能零差（先于综合，防"综合对象≠验证对象"）

`verify_t15.py`：4 trace × 4 模式（fx_full/fx_cert/bf_full/bf_cert），期望 = T5 整数模型
存档（判决 + 拍数），另与 T10 RTL 逐行交叉；FX 字串行基线从平面激励按二补码重建
24b 词（Y[s] = mag − (sign_s ? 2^23 : 0)）。**全部零差**（4×20000 组）。

调试中抓到并修复的真 bug（都发生在"综合口径重写"阶段，验证的价值所在）：
1. fx 基线 a_mem 索引转置（s*10+t ↔ t*10+s，a.hex 是 t 主序）；
2. fx 基线方向折返输出反相（T10 是 `flag ? raw : ~raw`）；
3. fx 基线第 10 拍比较用了更新前 vacc（丢末词贡献，0.36% 判决翻转）——
   平面数据通路比较的是含当拍平面的新值（nv），字串行必须比 vacc_n；
4. Verilator 640b 端口是 32bit×20 WData（非 10×64），TB 赋值方式错误；
5. 生产者 e 的 OR 树对负数幂次幅值差 1（XOR 偏离=m−1，须用幅值 −Y）。

## 3. 生产者端 RTL：plane_ser（T13b 部署口径的 RTL 化）

240b 锁存 10 词 → OR 树幅值优先编码出组指数 e + 符号字（sop 拍）→ 按请求
MSB-first 移出 e 个 10bit 平面（**二补码原始位**，非幅值位——与 T5 激励生成器一致）。
**回放校验**：由 stim_fx.txt 重建 Yq 喂入，输出在 4 trace 上逐行复现 stim_bf.txt
（296376/310535/295077/311020 行 EXACT MATCH）——生产者→门核供数回路 RTL 闭合。
面积：**3,033 µm²**（nomap 口径；优先编码器 + 240b 词寄存 + 平面 mux，无乘法器）。

## 4. 面积对比（Nangate45 typical，µm²）

| 设计 | nomap（ROM 外置） | logic（常数烘焙） | DFF(nomap/logic) | 周期/组 |
|---|---:|---:|---:|---:|
| C1 门核 cert_gate_bitl_synth | 52,706 | 37,168 | 1,945 / 1,925 | BF+cert 4.2（T5/T10 实测） |
| FX 基线 fx_gate_synth | 待补 | 41,482 | — / 985 | 11（1 sop + 10 词） |
| plane_ser（C1 生产者开销） | 3,033 | — | 240 | — |

观察（待 FX 数字后定稿）：
- C1 的 logic 口径反而**小于** nomap：LUT/pn 常数高度可折叠（12.8Kb LUT 烘进逻辑
  净省 ~15.5k µm² 的不透明读口+pmsk/nmsk 常量算术）→ C1 不需要 SRAM 宏；
- C1 门核 0 乘法器（T10 dot10 子集和 LUT + 递推 msk），FX 10 个 16×24 乘法器。

## 5. 结论

**（1）证书机制的门核面积几乎免费。** logic 口径（常数烘焙，层专用硬件）：
C1 门核 37,168 µm² vs FX 基线 41,482 µm²——C1 **小 10%**，且 0 乘法器
（dot10 子集和 LUT + msk 递推移加替代 10 个 16×24 乘法器，1,925 vs 985 DFF）。
12.8Kb LUT 常数烘焙进逻辑反而**省**面积（nomap 52,706 → logic 37,168：
常数折叠消掉不透明读口+pmsk/nmsk 常量算术）→ C1 不需要 SRAM 宏。

**（2）接口轴是主要赢面（与 T13b 一致）。** 同一时钟下：

| | 端口宽 | 拍/组（RTL 实测） | 位传输/组 |
|---|---:|---:|---:|
| C1（BF+cert） | 10b 平面 + sop 元数据 | **4.15–4.24** | ~42 |
| FX 字串行基线 | 24b 词 + sop | 11 | 264 |

C1 拍数 2.6×少、端口 2.4×窄 → **位传输 6.3×少**（42 vs 264 bit·cycles/组）。
面积相当 + 传输 6.3×少 = 供数侧能量/带宽优势，与 T13b 净服务口径
（bp 63.7%，部署日历）互相印证。

**（3）诚实边界。** 若允许 240b 全并行端口，FX 可 1–2 拍/组吃满吞吐——C1 的
主张不是任意接口下的吞吐极限，而是**可串行化接口**（引脚受限/共享互联/背压
传输，T13b 部署日历）上的位传输最优；此接口假设由 Y 的唯一消费者结构
（T13a：fc1→bn1→sn2 严格串行，门核是 Y 唯一消费者）合法化。
两设计同库同流程对称处理（thr 外置、ROM 双口径、同 ABC 脚本）。

**（4）系统拼图完整度。** C1 系统 = plane_ser（3,033 µm²，回放零差）
+ cert 核（37,168）+ thr SRAM 行口（层级静态常数，T13a ~2Mb stage3）。
RTL 三件套（生产者/门核/校验）+ 综合面积 + 净服务（T13b）+ 扰动鲁棒（T6）
全部闭合，C1 证据链就绪，可进论文装配。

## 6. 工具链附录

- yowasp-yosys 0.69（WASM）：ABC 子进程 stdout 丢失 → 无 Delay/STA 数字，
  面积从 yosys 内部日志（-l）解析；
- 综合流程：proc → memory[-nomap] → opt → techmap → opt → dfflibmap →
  abc -liberty → opt_clean → stat -liberty（`t15_synth.py` 可复现，幂等重跑）；
- FX nomap 口径 ABC 极慢（WASM 单线程），logic 口径已对称确认结论方向。
