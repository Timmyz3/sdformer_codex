# 空间R16及三种剪枝接口的完整825质量

| 模型/接口 | 逐帧平均AEE | 像素加权AEE |
|---|---:|---:|
| NB0 | 1.447936665574 | 1.380873963807 |
| flat_R8 | 1.327635022608 | 1.266917603446 |
| spatial_q13 | 1.259814783312 | 1.191656419628 |
| spatial_q11 | 1.254430537270 | 1.193013541332 |
| moment | 1.296504089440 | 1.228066895051 |
| native_tap | 1.285952568387 | 1.221726543851 |
| unconstrained | 1.258343102157 | 1.191656389748 |

人口完全相同：825帧、48,152,523有效像素、18序列。独立diverse10与完整825子集逐帧一致。

同一A80080GB/env312（Torch2.2.2+cu121）评估，无训练。NB0是本地原SDformerFlow PSN/SDSA ep29完整flow[-1]；候选保留已部署matched-dense学生与粗头。此比较满足用户的同任务baseline质量门，不能把整个差距归于R16或Winograd。

Q13与Q11分别独立评估，没有将旧浮点R16或Q13精度借给Q11。Q11仅一次由Winograd系数和的13bit硬件上限推导的选择；量化有损，固定Q11后的普通卷积与Winograd重排则整数精确相等。

三种剪枝均不训练、不使用验证集活动来选择系数：moment在原Q11整数域投影到一条变换项为零的约束；native_tap按相同BN/输出尺度权重删除一个物理抽头组；unconstrained直接删同一变换项而不投影原卷积。最后一种是独立两相位算子，保存p2并将输出尺度减半，重新RNE构造a_q40，不能称为普通共享1×3卷积。

moment和unconstrained已经使用同一个三项RTL，所有记录周期完全一致；其差异必须由完整网络质量判断。native_tap用其更快的普通执行臂作性能对照，不能强迫它走已测更慢的通用Winograd。十帧为完整825的子集，只用于排错，最终取舍以本表为准。

三种剪枝均在实际网络重捕18序列首帧的固定边缘/内部36tile：source和原identity与Q11母体相同，Z/raw/J/wide/I24与各自独立gold逐值一致。它连接网络函数与组件RTL输入，不表示36tile代表完整网络耗时。

平均改善不代表每帧改善：Q11相对NB0有729帧更好、96帧变差，单帧最大AEE增加4.444663。Q11逐帧平均略好于Q13，但像素加权平均略差，两项都保留。

源为AT-LIF {0,θ}，θ已吸入系数。网络实际注入新I24/16384，下一位级读取者每帧检查完全相等；首diverse帧135tile的原source/identity/Z/p/J/wide/I24与本地独立gold相等。整网仍有浮点算子，不是整网RTL/bittrue证明。

首次cuDNN FP64空间卷积出现约2.73e-12非整数微扰并被停止。当前只在整数oracle内部禁用cuDNN算法变换，直接FP64整数点积；未对错误中间值就地round，未改其它网络的TF32设置。

[逐序列误差](by_sequence.csv)、[逐帧配对汇总](comparison.json)、[Q13原始结果](deployed_valid.json)、[Q11原始结果](q11/deployed_valid.json)、[moment](moment/deployed_valid.json)、[native_tap](native_tap/deployed_valid.json)、[unconstrained](unconstrained/deployed_valid.json)。

AEE与网络运行时间不产生任何硬件倍率。真实周期、端口与消费者以相应RTL目录为准；当前无物理PPA。
