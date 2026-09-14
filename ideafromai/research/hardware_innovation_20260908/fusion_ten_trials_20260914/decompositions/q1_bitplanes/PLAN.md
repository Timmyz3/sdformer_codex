# D1 Q1位平面归约：实现前合同

B：完整K864的二值源与signed3系数可写为 `z_r=popcount(g&Q0_r)+2popcount(g&Q1_r)-4popcount(g&Q2_r)`。原生source按P/T窗口构成bitmap必须在RTL完成，不由TB送g。一次16K，8rank并行popcount，三系数位逐拍；source bitmap全40×54×16bit，逐k写40个合法源位，包含边界及死k置0。每次有效16bit源字才请求对应128bit系数平面；最后仍完整Q2 cachedOS与所有rawp输出。

完整A：bitserial/popcount及静态位平面是普通精确算术。强对照mode14是最新双P packed/native窗/cachedOS，mode15使用位平面；8×32bit共享加法链、8×19×13乘法表达式、8×16bit popcount树、额外bitmap和系数两格式、所有端口共同分配。位平面128bit权重口宽于原Q1 24bit，基线拥有同权限；不跨旧资源点报等面积。系数的普通格式与位平面由同cfg并行写入，真实存储与写bit权限明确收费。

必须支付native源窗装入、每k bitmap生成、源字扫描、三平面权重请求/许可背压、popcount、符号项、最终z写及scan，再完整Q2/psum/drain。额外源bitmap可全覆盖重写而无需预清；不是免费动态预处理。共享ALU在普通packed更新断carry13，在popcount累加及Q2为完整32bit。无中间舍入。

失败数：若很多16bit字非空但popcount小，bitmap构造与三平面请求可能超过事件更新；实际完整RTL决定，不用统计停家族。14旧fixture×两mode×双背压×无reset重启112runs，无训练/消费者/PPA。
