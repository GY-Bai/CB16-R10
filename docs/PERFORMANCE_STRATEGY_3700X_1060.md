# 3700X / GTX 1060 主机的性能与语言迁移策略

日期：2026-09-12。状态：**原设计策略已由 CC Thread D + Integration 实施、测量并 hard cutover；本文现在同时保留设计原则与已验证结果。**

当前 canonical authority：

- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`

Canonical topology：

`CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER`

## 1. 最终结论

早期性能设计的优先级仍成立：

**消除重复计算和复制 → 批量化及连续存储 → 跨账户并行 → 有界队列/大块 IO → 编译热点 → 只有真实热点仍存在时再考虑 Rust/Go。**

但当前状态已经从“待测假设”推进为实际 CC qualification：

- Thread A 保持 runtime/account/execution correctness oracle。
- Thread D 负责 same-account scheduler、bounded fact transport、durable chunk writer 和 performance implementation。
- 旧 `gpu_inference_broker.py`、`multiprocess_trajectory_farm.py`、`vectorized_physics.py` 不属于 canonical CC runtime dependency，也没有 compatibility fallback。
- fast path 失败时 fail closed。

## 2. Shanxi 已测结果

首个冻结资格 benchmark 由 integration receipt 记录：

- 主机：Ryzen 7 3700X / GTX 1060 6GB / 16GB RAM 环境；runner `shanxi-docker-r11`。
- workload：16 accounts × 64 market steps = 1024 transitions/run。
- reference 与 fast 各 7 次，交替顺序。
- Reference median：`7.535366 transitions/s`，median wall time `135.892535 s`。
- Integrated fast median：`638.765768 transitions/s`，median wall time `1.603092 s`。
- Median speedup：`84.769×`。
- reference 与 fast semantic checksum 相同：`c864052eab1c107ab73a88d96ddb527bfc60819a41165b1495ecc74d56644988`。
- final-account checksum 相同：`29d33e7639029f40c6edfb1c7fe9fff26e4af3f418cf5109a066a55de13e282e`。

最新 PR handoff head `5d236e9d2368a79e830bec1971b949de88c032df` 又重新跑完完整 qualification，仍选择相同 topology；重复 run 的性能数字只作为稳定性确认，不覆盖 receipt 中首次冻结的 benchmark。

选择规则始终是：

`PERFORMANCE FIRST AMONG SEMANTICALLY QUALIFIED IMPLEMENTATIONS`

即先通过科学语义资格，再在 PASS 候选中选择端到端吞吐最高者。

## 3. 性能优化不能改变的科学含义

性能实现无权改变：

- same logical account 的时序连续性；
- signed account economics、negative equity、liability；
- nominal action、permission、target sizing、execution 的分离；
- stochastic behavior 的真实 `log_mu` 与 policy/RNG/generation provenance；
- failure/terminal raw facts；
- frozen organ / trainable Brain ownership；
- 算术期望收益方向；
- FINAL/fresh-data firewall；
- 不把固定 SL/TP/max-hold/cooldown 偷渡成自主策略；
- 不通过跳步、免费补资、缩短后果窗口或删失败来换吞吐。

如果更快实现改变这些含义，它不是性能优化，而是新的 science version，必须重新资格认定。

## 4. 最有价值的结构性优化

### 4.1 冻结市场表示复用

价格接受者假设下，同一市场时点的冻结市场表示不因账户动作改变。多个账户可以共享同一纯市场表示，账户观察和路径后果仍独立。

缓存 key 必须绑定数据、预处理、器官权重和归一化版本；可训练 Brain 或依赖账户/策略记忆的状态不能按市场时间错误共享。

### 4.2 跨账户并行，同账户串行

可以并行不同账户、不同独立分叉和不同种子；同一逻辑账户必须保持：

`action_t -> execution_t -> account_(t+1) -> next action`

不能为了 worker 利用率提前执行尚未看到真实后果的下一步。

### 4.3 有界传输和 durable writer

高吞吐 collector 必须使用按字节有界的 queue/backpressure，不能靠无限内存积压或丢失败记录获得吞吐。

大块连续写入、durable chunk、确认后提交是当前正确方向；事务完整性边界不能因为性能工作被静默取消。

### 4.4 Storage tiering

原则仍是：

- RAM：有界 active working set；
- SSD：热分片、活跃索引、近期恢复状态；
- HDD：冷历史和封存轨迹，尽量顺序访问。

选择样本与物理读盘顺序分开：可优化 IO 排序，但不能因此改变 replay sampling distribution。

## 5. GTX 1060 / Pascal 边界

GTX 1060 属于 Pascal compute capability 6.1。不要默认 Tensor Core、BF16 或现代编译后端在该卡上具有新 GPU 同样收益。

当前已经验证的 PyTorch/CUDA 组合应优先保持；工具链升级必须单独做兼容性与端到端收益测试。低精度、AMP、`torch.compile` 等不能只凭理论支持就进入 canonical runtime。

NVIDIA Pascal guidance：

- https://developer.nvidia.com/cuda-legacy-gpus
- https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html

## 6. Python、Numba、Rust、Go 的当前规则

| 区域 | 当前默认 | 何时升级 |
|---|---|---|
| PyTorch Actor/Critic/autograd | Python + PyTorch | 不因外层语言换成 Rust/Go 就预期矩阵计算自动加速 |
| 数值账户/执行热点 | 先结构化数组、向量化、Numba/编译循环 | 只有端到端 profile 证明仍为主要瓶颈时考虑 Rust coarse-grained kernel |
| 序列化、扫描、压缩 | 先减少对象构造、固定布局、批量 IO | Python 对象开销仍占主要关键路径时再考虑 Rust extension |
| 调度/网络服务 | 先现有 Python orchestrator/队列 | 只有并发 service/GC/运维开销有实测瓶颈时才考虑 Go |
| GPU/IO 等待 | 改 batching、数据布局、阶段重叠 | 换语言本身不能增加显存或消除硬件等待 |

Rust 若进入，优先做粗粒度 batch extension，不做每 bar/每字段 Python↔Rust 高频往返。Go 不作为 Actor/Critic 数值实现默认路线。

Numba performance guidance：

- https://numba.readthedocs.io/en/stable/user/performance-tips.html

PyTorch multiprocessing/tuning guidance：

- https://docs.pytorch.org/docs/2.13/notes/multiprocessing.html
- https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

## 7. 后续性能工作的采用门槛

未来任何性能改动必须同时报告：

- exact code/science identity；
- workload、seed、账户数、market steps、policy generation；
- compliant transitions/s 与 wall clock；
- CPU process/thread 数及每核负载；
- PSS/cgroup working set、major faults、swap；
- SSD/HDD throughput 与 IO wait；
- GPU peak VRAM、active/transfer/kernel time（若可测）；
- queue depth bytes、oldest-item age、batch distribution；
- semantic checksum / final-account checksum 或等价 correctness oracle；
- FINAL/fresh-data firewall。

只报 microkernel 倍数、GPU utilization、loss/s 或单一 batch latency 都不足以替换 canonical topology。

## 8. Amdahl 纪律仍有效

若热点占总时间 `p`，局部提速 `s`，理想总加速上限：

`1 / ((1-p) + p/s)`

语言迁移或原生 kernel 的价值必须按**端到端关键路径**衡量，不能把局部 10× 宣称成全系统 10×。

## 9. 文档地位

本文早期版本是 CC 之前的性能设计研究；当前实现事实以 integration receipt/spec、`CURRENT_STATE.md` 和 `CC_INTEGRATION_HANDOFF.md` 为准。

历史设计建议仍可用于未来扩展，但“BC 是当前入口”“尚未在 Shanxi 实测”“性能工作尚未重开实施体系”等旧措辞已经被 CC 四线程和 hard cutover 实际结果替代。

当前性能路线不是继续重写全仓库，而是在 canonical CC runtime 上对新 workload 做版本化、可测、可回退到上一**CC qualified version**的优化；不得回退到已退休的 legacy performance runtime。
