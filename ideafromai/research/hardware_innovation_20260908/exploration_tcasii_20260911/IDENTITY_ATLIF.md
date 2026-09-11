# Locked identity — AT-LIF（2026-09-11，用户确认）

**对的写法：**

AT-LIF 输出是二电平 \(\{0,\theta\}\)。  
\(o_i[t]=\theta_i\cdot H(m_i[t]-\theta_i)=\theta_i s_i[t]\)，\(s\in\{0,1\}\)，\(o\in\{0,\theta\}\)。  
推理时层共享的 \(\theta\) **吸进下一层 \(W\)**：\(W\leftarrow\theta W\)。吸完之后，层间传递的是 **0/1 脉冲**，不是按事件保存的模拟幅值，也不是不可吸收的 int8 payload。

这是 NeurIPS AT-LIF / Activity Pruning 的官方身份，也是本课题要的身份。

## 因此立刻作废的说法

- “AT-LIF 是连续 θg、不能当二值”
- “θg 不吸入 W 才是贡献”
- GrokBot HBG-RP「不可吸收 int8 payload」当**冻结身份**（其合同自己也写了：若可吸收则降级；那是另开提案，不是 AT-LIF）
- MX3P / 纯二值岛换神经元；身份已经是可吸收的 \(\{0,\theta\}\)，吸完就是二值 GeMM

## 硬件含义（提案，不是新测量）

吸完之后，**下一层看到的是二值脉冲 × 权重**。Prosperity 的 product sparsity、GustavSNN 的 NRV/CPTB、FireFly-S 的 Bitmap AND，都变成合法的 **A（必须抄全的对照）**，不再因为“我们是连续幅值”被一句话挡掉。

仍然分开的东西：

- **脉冲路径：** 吸 θ 后的 0/1 GeMM。  
- **残差 / PED / I24：** 另一条连续张量，不是 AT-LIF 的 \(o\) 带着模拟幅值往下传。双消费者若还存在，消费者之一是 **门（吸完后的二值）**，另一是 **残差连续路径**，不要写成“连续 AT-LIF 幅值被两个 MAC 共用”。

GrokBot `ATLIF_contract_r1` 的 Absorb=NO 与本锁定冲突；信件用本文件，不用那份草案。
