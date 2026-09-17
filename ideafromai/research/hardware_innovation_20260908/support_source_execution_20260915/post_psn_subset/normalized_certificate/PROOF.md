# 把移位移出前缀关键路径

仍是同一个精确证书，不是新增数学，也不预报周期/面积收益。原GROUP后第m平面的完整区间为

`L = nv*2^m + N*(2^m-1)`，`H = nv*2^m + P*(2^m-1)`，其中N<=0、P>=0。

令δ=1用于正BN增益（门为U>=τ），δ=0用于负BN增益（门为U<=τ）。GROUP实际计算并保存signed49的`qN=τ+N−δ`、`qP=τ+P−δ`，随后

```
lower_hit = (nv+N) > floor(qN/2^m)
upper_hit = (nv+P) <= floor(qP/2^m)
```

lower_hit使门等于positive_gain，upper_hit使门等于!positive_gain；常量通道仍优先。因为对整数z,x，`z*2^m>x`等价`z>floor(x/2^m)`，上式对正gain分别等价L>=τ和H<τ，对负gain分别等价L>τ和H<=τ。算术右移必须在有符号49位上进行，不能逻辑右移。

τ为signed48端点而P/N来自十个signed16系数时，τ+P/N−δ可能越过signed48；预加和与比较采用signed49。nv+P/N也显式扩展，不依赖表达式隐式宽度。m=0仍保留原等号归属，不允许统一改成一个非严格比较。`check_identity.py`以直接整数L/H为独立参照，覆盖τ端点、每个界的±1及m=0..23。

硬件差分：原prefix→左移→tail加法→比较，改为prefix→P/N加法→比较；τ右移在另一条输入支路。GROUP复用PLANE的两个bound加法位置构建qN/qP，原symbol prefix独立完成。去掉tail递推/存储，增加160×49bit阈值坐标寄存器。旧tail共20×48bit，所以净数据状态预计增加860B，具体以实际RTL为准。仍有LUT读、两级prefix加法和全锁定归约，不能只数去掉的移位就宣布Fmax改善。

完整Y、真实GROUP、H96打包和参数加载保持相同。若仿真周期不变，只报逻辑/时序待测；只有实际标准单元映射或后续布局测量才能验证关键路径假说。
