# 3700X / GTX 1060 主机的性能与语言迁移策略

日期：2026-09-12。硬件由用户提供：Ryzen 7 3700X、GTX 1060 6GB、16GB RAM、小容量 SSD 与 1TB HDD。SSD/HDD 型号、剩余空间、实时负载未测量。本文是设计与测量建议，不是主机 benchmark 或代码实施任务，不修改科学 gate、数据范围和执行语义。

## 1. 核心判断

优先顺序：**消除重复计算和复制 → 批量化及连续存储 → 跨账户并行 → 编译热点 → 仍有证据时局部 Rust；Go 只在调度/网络服务确实成为瓶颈时考虑。** 没有任何一步仅因为“使用 Python”就必须重写。

主要依据是当前代码的结构和官方性能资料。以下热点是待测假设，不能称为 Shanxi 已实测瓶颈。当前实施入口以 [BC Round 2 TODO](R11_BC_ROUND2_CODE_ALIGNMENT_TODO.md) 为准；性能工作不重开另一套并行实施体系，不绕过语义资格。

## 2. 当前代码中已有的积木

核对代码快照：`1b9e3a3ba75d5911aa7995020eac54c182a8cf03`。已阅读 AGENTS、下列模块及 Actor R1；没有启动 runner 或科学训练。

| 文件 | 可参考/复用的设计 | 必须保留的版本边界 |
|---|---|---|
| [gpu_inference_broker.py](../cb16_local_opt/gpu_inference_broker.py) | 单 CUDA owner、共享内存、批量请求 | 旧请求为 Market64 / Account6，不能直接当作 BC 新观察和随机政策接口；须增加适用的概率/RNG/政策身份传递 |
| [multiprocess_trajectory_farm.py](../cb16_local_opt/multiprocess_trajectory_farm.py) | spawn CPU worker、每 worker 线程限制、memmap、限制在途任务 | 旧 replay job 不是新政策每步交互 Collector，调度积木复用不等于科学语义兼容 |
| [market_runtime_cache_r11.py](../cb16_local_opt/market_runtime_cache_r11.py) | campaign 内只读市场缓存 | 对象内只加载一次，不意味着多个进程共用一份；现有 hourly npz 也不是通用分钟器官缓存 |
| [vectorized_physics.py](../cb16_local_opt/vectorized_physics.py) | 数组化账户更新思路 | 模块明确不是历史标量 kernel 的字节等价实现，并带 max-hold 等旧语义；不得当作 BC 的直接性能替代品 |
| [actor_critic_physics_adapter_r1.py](../cb16_local_opt/actor_critic_physics_adapter_r1.py) | 当前 BC 仓位执行与有符号账本连接 | `_state_sha` 使用 asdict / JSON / SHA；replace、重复验证与状态构造值得采样分析，但不是已证明热点 |
| [actor_policy_r1.py](../cb16_local_opt/actor_policy_r1.py) | 正在形成随机方向和条件风险接口 | 当前 BC-037 为非 FLAT 内部连续风险、端点无点质量；性能实现不得恢复旧 Beta 端点候选或破坏行为概率 |

以上文件名不是性能 PASS。旧高吞吐实现若改变成交、退出、抽样、信息边界或失败记账，就不能以加速名义接入新链。

## 3. 哪些地方可以并行、向量化或少算

| 环节/可能症状 | 首选方法 | 并行维度和限制 | 是否先换语言 |
|---|---|---|---|
| 冻结器官反复处理相同市场窗口 | 按市场来源、时点、器官权重及预处理版本缓存输出；批量生成、分块读取 | 同一时点多账户共享相同市场表示；不同市场窗口可批处理，但保持原有因果窗口与归一化 | 否 |
| 大量账户 Physics 推进占 CPU | 持久 worker，每 worker 处理一批账户；结构化数组或 Numba 编译循环 | 多账户/独立分叉并行；同一账户的 action→account→下一 action 保持时序 | 先优化，必要时 Rust |
| 小网络每个账户单独调用 GPU | 依据政策版本聚合 ready 账户，批量 Actor；或比较 CPU 小批推理 | 批处理只改变执行组织；保存每个账户 RNG 与 log_mu，等待上限不改变环境决策时钟 | 否 |
| 大量 Python 对象构造、JSON、IPC 复制 | 固定数组布局、引用共享市场、复用缓冲区、批量传输；减少不必要的重复序列化 | 权威 receipt/提交边界保留；不得直接关掉校验或删除轨迹 | 只有剩余 CPU 热点才考虑 Rust |
| replay 随机读使 HDD 忙、CPU/GPU 空闲 | HDD 顺序分片读入 SSD/RAM 活跃集，按预选样本索引合并 IO | 先按既定抽样分布选样本，再优化读盘次序；不能只取最近/方便读取的样本而改分布 | 否 |
| Critic/Actor 学习 GPU 占满 | PyTorch 批量前后向、减少逐元素 Python 调用和 CPU/GPU 往返 | 一个逻辑 learner 维护更新顺序；在一张 GPU 上多 learner 不会复制算力 | 通常否 |
| V-trace 目标计算慢 | 向量化 batch 维；时序递推先保留简洁 tensor/编译循环 | 倒序依赖仍存在；较短序列不默认实现复杂并行 scan；浮点归约次序需验证 | 很低优先级 |
| 写盘/压缩阻塞 Collector | 有界写队列、连续 chunk、有限预取和压缩 worker | 确认持久化后才确认事务；writer 跟不上时背压，不能丢失败或无限积压 | 先 IO 布局，后评估 Rust/Go |
| 多账户比赛或独立种子 | 并行 rollout，GPU 请求合批 | 相同环境未来不因此变成独立市场证据；保留完整比较分母 | 否 |

Numba 可把适合的数值循环编译为原生代码，不必先改 Rust；其官方性能指南也要求先针对真实负载 profile。普通 CPython 的纯 Python CPU 循环不能靠增加线程自动并行，原生库释放 GIL 的操作和 IO 则可能适合线程。[Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)、[PyTorch multiprocessing](https://docs.pytorch.org/docs/2.13/notes/multiprocessing.html)

### 最有价值的可复用计算

价格接受者假设下，市场器官输出不随账户动作改变。如果同一市场时点有 N 个账户，市场表示原则上可生成一次，然后与 N 份不同账户观察结合；每个账户的策略与后果仍需独立计算。这消除重复器官调用，不意味着整条管线自动加速 N 倍。

仅缓存真正冻结、纯市场、确定性的输出。可训练 Brain stems 或依赖账户/策略历史的隐藏状态不能跨 checkpoint 任意缓存；缓存 key 必须包含数据与预处理版本，且不得使用未来数据拟合归一化器。不要为省计算把滚动窗口改为另一种上下文定义。

## 4. 这台主机建议从什么规模开始

以下全部是待测起点，不是最优值、容量保证或强制 gate。

| 资源 | 建议起点 | 增大前看什么 |
|---|---|---|
| 3700X 的 8 核 / 16 线程 | Collector/Physics 从 2、4 个持久进程比较；必要时试 6，8 留作有余量的候选 | 端到端合规 transitions/s、单进程 PSS、上下文切换、IO 等待；16 逻辑线程不等于16份完整物理核算力 |
| 数值库线程 | 每个 CPU worker 先限制 OMP/MKL/OpenBLAS/PyTorch 为 1 线程 | 把 worker、learner、压缩和库内部线程一起计预算，避免各自开满核 |
| 每 worker 的活跃账户 | 先比较总活跃账户 16、32、64；与进程数分开调 | batch 填充、账户状态大小、推理等待和环境吞吐；不等于新增独立市场样本 |
| GPU | 一个 CUDA owner；Collector 不各自加载一份模型和 CUDA context | 模型、激活、临时 workspace、分配器及 pinned buffers 的实际峰值 |
| learner / 收集 | 先分时：固定政策收集、后台或下一阶段学习；有余量再有限重叠 | learner 吞吐、Collector 等待、政策陈旧程度；不让正在行动的模型被更新 |
| 16GB RAM | 首轮将总常驻/活跃工作集控制在约 10–12GB 内，给 OS 和突发留余量；已有主机负载须扣除 | 实际容器 memory limit、PSS/cgroup memory、page cache、major faults、swap-in/out；不能把各组件上限同时加满 |
| 队列 | 按字节限制，先容纳每 worker 约 1–2 个生产 chunk | writer/learner 能否持续消费、最老样本年龄、峰值内存；不是只按消息条数限制 |

PyTorch 官方明确指出过量进程叠加进程内线程会降低效率，且 CUDA 子进程须采用兼容的 spawn/forkserver 路径。共享内存能减少大 payload 复制，但仍有同步、生命周期和首次搬运成本。[Multiprocessing best practices](https://docs.pytorch.org/docs/2.13/notes/multiprocessing.html)

对小 Central Brain，batch=1 的 CPU 推理可能比“进程间请求＋传输＋GPU 启动”更合算；这是必须在本机测量的候选，而非断言 CPU 一定更快。重型冻结器官若已缓存，在线决策负载与原始 encoder 推理负载差别很大。PyTorch 数值算子本来就在原生后端执行，换外层语言不会让同一个矩阵乘法自动变快。[PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## 5. GTX 1060 的特定边界

NVIDIA 将 GTX 1060 列为 compute capability 6.1 的 Pascal GPU。不要默认现代 Tensor Core、BF16/AMP 或编译后端会给它同样收益。Pascal 官方指南区分不同芯片的半精度吞吐，明确 GP104 的原生 FP16 吞吐远低于 FP32；该比率不是本文对 GTX 1060 所有算子的实测。建议 FP32 基线，低精度只在兼容性、概率数值和端到端收益得到验证后使用。[NVIDIA GPU 表](https://developer.nvidia.com/cuda-legacy-gpus)、[Pascal tuning](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html)

CUDA Toolkit 13.0 已移除 Maxwell/Pascal/Volta 的离线编译及相关库支持。工具链与驱动是不同层；旧兼容二进制并非因此立即失效。继续使用已在 Shanxi 通过的 PyTorch/CUDA/驱动组合，检查 wheel 所含架构与实际小算子；不要把“升级到最新 CUDA/torch.compile”列为默认优化。Numba CPU JIT 不依赖 CUDA；不能由它可用推断 GPU JIT 可用。[CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/)、[NVIDIA compatibility guidance](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)

冻结器官预计算可与 Brain 训练分时使用显存，避免所有器官、全部 replay 和 learner 同时驻留。模型未能同时装入不意味着需要更换训练算法；首先拆分阶段。微批/梯度累积只缓解显存，不能承诺速度增加或偷偷改变有效 batch 和更新规则。

## 6. SSD 与 HDD 如何分工

**HDD 存冷历史和已封存轨迹，SSD 存当前活跃分片、索引和近期恢复数据，RAM 保持有界工作集。** SSD 空间未知，因此不要求把全部历史搬上 SSD；先定义可用空间和缓存淘汰规则。冷归档要保存并校验成功再回收临时副本，不因冷热分层删除失败经历。

- 尽量顺序读取和追加较大 chunk，不要多 worker 各自从 HDD 随机读取海量小文件。
- 使用只读 `.npy` / memmap 等可切片布局，或将压缩分片顺序解压到活跃缓存。`.npz` 是 ZIP 容器；对压缩 npz 设置 mmap 参数不能视为获得了底层未压缩数组的按页随机读取。
- memmap 节省全量载入，不消除 HDD 寻道和缺页；随机冷访问依然可能慢。Python 索引后形成的复制也应计入内存。
- raw Market 与冻结器官表示可按时点引用；不在每个账户 transition 重复存整段 K 线和相同 Market96。账户观察、动作、执行、奖励和来源仍完整记录。
- 选择样本和读盘次序分开：可合并读取已选样本所在 chunk 后按既定顺序组 batch，不能因 IO 方便而改变学习人群、代际比例和选择概率。
- 对完整性边界做批量写入/校验设计，不直接取消 fsync、hash、receipt；持久化确认点改变需要明确恢复合同。

NumPy 官方将 memmap 定义为无需全量加载而访问文件片段的工具；`.npy` 支持内存映射，`.npz` 为多个 `.npy` 的 ZIP 包。[NumPy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)、[NumPy format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)

容量算例仅展示量级：十资产、每年365天、每分钟一条、96维 float32，为 `10×365×1440×96×4 = 2,018,304,000` 字节，约1.88GiB/年；五年约9.40GiB，尚未含时间戳、账本、版本和训练张量。这个数不是实际语料大小，也不规定决策频率。若每条经验平均1KiB、每秒生成1000条，每天约88.5GB；不限制存储重复和生产速率，1TB也并非无限经验池。此速率是假设，不是主机性能预测。

## 7. 什么才值得改 Rust，什么才值得改 Go

| 部分 | Python/原生库优先 | 迁移条件与目标语言 |
|---|---|---|
| 分支密集、长时间重复的账户/执行状态机 | 先剥离对象编排，尝试结构化数组、Numba；保留当前标量参考 | 若仍占端到端主要时间且语义稳定，首选 Rust 批量内核；金融守恒和事件顺序需与参考逐项一致 |
| 大批经验编码、解码、扫描、压缩或哈希外围逻辑 | 先固定二进制布局、复用现有原生库、减少重复工作 | Python 对象处理仍是热点才考虑 Rust 扩展；hash 算法本身已有原生实现，改外壳未必有收益 |
| 队列、网络连接、独立长驻调度服务 | 先复用已有 Python orchestrator、线程/async IO、有界队列 | 仅当并发服务、GC/对象开销或运维需求有实测依据时评估 Go；不为一台主机预建微服务体系 |
| PyTorch Actor/Critic、autograd、优化器 | 保持 Python + PyTorch 的原生 CPU/CUDA 后端 | 不建议为性能整体迁移 Go/Rust，科研迭代与概率正确性更重要 |
| replay 元数据、实验配置、receipt、分析 | 保持清楚的 Python 实现 | 通常不是首要热点，先测量；正确性检查不可因低利用率猜测而移除 |
| HDD 随机访问、PCIe 往返、GPU 已饱和 | 改数据布局、批量、阶段和资源使用 | 换语言不能消除硬件等待或增加显存 |

若选 Rust，优先作为 Python 的粗粒度扩展：一次传一批连续数组和状态，而非每根 bar 逐字段往返；Rust 内完成一批账户的确定性计算后返回。涉及下一步必须调用 Actor 的场景，不能跳过决策来扩大 Rust 调用粒度。PyO3 可在原生计算期间 detach，配合 Rust 线程；这不自动让 Python 对象访问无锁或让数组传递零复制，须说明内存所有权与生命周期。[PyO3 parallelism](https://pyo3.rs/main/parallelism)

Rust 和 Go 都能写数值代码或服务，以上是本项目的集成成本判断，不是语言能力定律。选择 Go 服务也需要先 profile CPU、heap、阻塞和锁竞争；Go 自带这些诊断工具，但不会因为使用 goroutine 就让 HDD 或 GPU 更快。[Go diagnostics](https://go.dev/doc/diagnostics)

## 8. 给实施者的测量和采用规则

在已授权执行范围内，选择一个真实可运行且语义已固定的最小片段。先测冷启动，再测预热后的稳定阶段；没有 runner 权限时只交测量方案，不用别的硬件数据冒充 Shanxi。

必须同时记录：

- 代码/科学版本、环境和数据身份、政策/种子、工作量与结果校验。
- 市场加载/编码、Actor、Physics、序列化、队列等待、读写、learner 的时间；并行流水线用关键路径/等待解释，不把重叠时间直接相加。
- 合规 transitions/s、learner updates/s、固定已知答案门槛的总墙钟时间；仅 loss/s 或 GPU 利用率不足以决定方案。
- CPU每核负载、线程数、PSS或cgroup工作集、page faults/swap、磁盘吞吐和等待、GPU峰值显存及传输/算子时间。GPU异步计时应在测量边界用事件或适当同步，不把 enqueue 时间误作实际完成时间。
- 固定物理/经济语义的对照结果：余额和权益守恒、费用、反转顺序、终态、动作概率与账户接续。浮点容差需预先定义，尤其清算/最小订单边界不能只比较平均误差。

先逐项比较缓存/批量/worker/IO，保留同一工作量；再对已证 CPU 热点做 Numba/Rust 小原型。原型须包含实际语言边界、序列化和复制成本，不能只报脱离管线的内核倍数。编译时间与冷启动单列，不混进或藏出总成本。

用 Amdahl 上限做取舍：若某热点占总时间 p、局部提速 s，理想总加速为 `1 / ((1-p)+p/s)`，还未扣新增开销。仅占10%的代码即使快10倍，全程也只有约1.10倍；占60%的热点快5倍，理想约1.92倍。性能预测据此给范围，不许承诺“换 Rust 就快十倍”。

增加 worker 或异步深度可能改变到达顺序、随机数消耗、政策滞后及数据分布；监控 log_mu、policy version、经验年龄和抽样权重，不能只看 transitions/s。同一账户不乱序，不靠免费重置、跳过费用/失败、缩短后果窗口、换数据范围来获得加速。

每个候选有足够证据判定后停止扩展测试。继续增加 worker 已不提高端到端吞吐，或导致 swap/积压时，回退并处理真正瓶颈，不再扩大并行。只有 GPU 已持续吃满且 IO/CPU 供应充足时，才能把该阶段称为 GPU 算力受限；稀有尾部和独立市场证据不足则是统计问题，换语言不能解决。

## 9. 与文档分工的关系

本页提供性能设计和审阅原则；GPT-5.6 sol 在当前 BC 任务与授权内拆解实现。它不要求现在开展全仓库重写、吞吐工程或硬件采购。首次测量暴露具体阻塞后，再将对应优化挂到受影响组件；结果记录精确版本与正确性/速度证据。

资料方法：使用 Exa 的8次定向搜索共返回40条候选结果位置（含重复），按并行/存储、GPU兼容、编译与语言迁移几个方向筛选，再核读上述官方页面。文中算例由公式计算，代码热点和资源配置是本项目的设计推断，均不是引用他人 benchmark 作为本机结果。
