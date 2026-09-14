# 起步 CPU-only：性能与技术选择

2026-09-14 用户明确决定：“改成 cpu only，一开始不要太复杂。”本页是当前设备与复杂度边界；替代早期把 CPU/GPU 协同作为默认起步方案的建议，不改写历史 benchmark 或冻结科学值。

## 1. 当前选择

- 当前恢复与 S1 用 **CPU** 完成 Actor/Critic、autograd、采集、回放与审计。GPU、CUDA 可用性、GPU broker、AMP 不作为该范围开工或通过的必要条件。
- 保持 Python + 已验证 PyTorch 接口。已有安装带 CUDA 不代表实际使用 GPU；不为改执行设备自动重装环境。CPU-only 必须由实际模型/tensor device 和执行配置证实。
- 先复用当前代码、SQLite/文件持久化和 Shanxi Docker。此次不引入新的数据库服务、Rust/Go 改写、通用控制平台或参数扩容。
- CPU-only 不等于单 worker；独立任务可有界并行，同账户时间顺序仍串行。worker 数与算子线程数共同预算，不能让每个进程都占满全机线程。
- 宿主已有 GPU 无需因此拆除；容器是否取消 GPU 挂载由 Sol 按具体执行 profile 落实。移除 GPU 挂载属于实际配置变更，不由文档发布假装完成。

## 2. 怎样落实到当前 S1

Sol 先核实 accepted runtime 实际 device、依赖、preflight 与 runner profile。复用 CPU 路径，明确哪些 GPU 必需检查对当前 CPU 阶段不适用；不能全局放松其他阶段的检查。

冻结模型、奖励、优化器、种子、预算和判据不变。CPU profile 或数值执行身份改变时，记录新绑定并验证相关已知答案、恢复和来源一致性；如果修改代码，按实际 diff 重新审核，不能用旧 SHA 的接受结论背书。

现有 RC2 性能目标也不能因切到 CPU 就事后改小；若不满足，报告瓶颈与影响，通过显式后继工程合同处理。所有 runtime 证据仍走 GitHub Actions → Shanxi Docker。本页不解除 S1 冻结，不代替 R8 reauthorization。

## 3. 硬件与优先顺序

3700X、16 GB RAM、SSD 热区、单 HDD 是当前资源边界。线程、内存与磁盘按实际 profile 核验，不能把 swap 当高速训练内存。

优化顺序：减少重复计算/复制 → 合规批量读写 → 有界并行与背压 → 实测仍存在的热点。SSD 承担活跃随机读写与恢复状态，HDD 适合封存顺序访问；具体 hot/cold 落位按 RC2 资格。shm 用于 IPC，不承担跨故障的唯一持久事实。

“合规批量写”不能改变 replay 可见性、学习更新顺序、checkpoint 生效时点、失败保留或提交语义。逻辑事件、物理写入、学习 batch 是不同层次。

## 4. 后续真实器官与 GPU

S1 的合成市场层不构成外部器官资格。S2 先对实际授权行情和真实冻结权重做 CPU 加载、因果输入、内存与耗时 canary；可以复用按数据/预处理/权重身份绑定的纯市场表示，不能错误共享账户相关状态。

若 CPU 成本成为已测阻塞，再提出单独 GPU 方案并由用户确认阶段范围；不要擅自换小器官、解冻权重或拿合成特征替代真实接入。GPU 容量估计见 [未来容量规划](BRAIN_CAPACITY_ROADMAP_3700X_1060.md)，不是当前实施清单。

## 5. 历史证据与停止条件

CC 当时的性能选择 `PERFORMANCE FIRST AMONG SEMANTICALLY QUALIFIED IMPLEMENTATIONS` 和 fast topology 保留为历史资格事实，见 [integration receipt](../authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json)。其合成 throughput 不外推为 S1/真实器官吞吐。

当前 RC2 按已确认方向：达到冻结工程目标并留有明确余量后，优先维护简单的合格方案；挑战者由实际瓶颈触发。记录墙钟、合规更新/决策吞吐、worker/thread、内存/交换、I/O 与恢复结果即可服务当前选择，GPU 指标在 CPU-only 运行中标不适用。

主线仍是得到有效 S1 结果。没有证据指向语言、数据库或计算设备的瓶颈时，不为它们单开重构。
