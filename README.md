# Central Brain / CB16

CB16 的目标是在沙盒中训练一个单资产 Trader：读取市场和账户状态，发出交易动作，让执行后的账户状态影响后续决策，并通过反复练习和分代学习改进策略。

人负责资产选择、资金配置和宏观判断；模型学习该账户的交易与风险取舍。训练经验由市场、账户、动作、执行及后续结果共同产生。具体目标见 [VISION](docs/VISION.md)。

本仓库保存源代码、协议、文档和脱敏证据，不保存原始市场数据、模型权重、checkpoint 或密钥。

## 从这里接手

- **Agent 必读**：[AGENTS.md](AGENTS.md)。
- **文档导航**：[docs/README.md](docs/README.md)。
- **文档组织与维护职责**：[DOCUMENTATION_MAP](docs/DOCUMENTATION_MAP.md)。
- **当前做到哪里**：[CURRENT_STATE](docs/CURRENT_STATE.md)，包含 main 与未合并分支的区别。
- **理念与实施对齐**：[PRINCIPLE_ALIGNMENT](docs/PRINCIPLE_ALIGNMENT.md) / [COMPONENT_REQUIREMENTS](docs/COMPONENT_REQUIREMENTS.md)；Sol 按 [角色原则](docs/SOL_ROLE_PRINCIPLES.md) 和 CURRENT_STATE 指定的当前任务工作，AC TODO 保留历史身份。
- **如何学习**：[LEARNING_CONTRACT](docs/LEARNING_CONTRACT.md)。
- **具体算法设计**：[TRAINING_ALGORITHM_R0](docs/TRAINING_ALGORITHM_R0.md) / [资格计划](docs/TRAINING_QUALIFICATION_R0.md)；实际接线、学习和经济证据分别查对应阶段的 receipt，不由设计文档推定。
- **如何判断策略**：[EVALUATION_PRINCIPLES](docs/EVALUATION_PRINCIPLES.md)。
- **组件分别负责什么**：[ARCHITECTURE_MAP](docs/ARCHITECTURE_MAP.md)。
- **哪些已决定、哪些待定**：[DECISIONS](docs/DECISIONS.md) / [OPEN_QUESTIONS](docs/OPEN_QUESTIONS.md)。
- **双基准与沙盒晋升决定**：[ECONOMIC_ORDERING_AND_PROMOTION](docs/ECONOMIC_ORDERING_AND_PROMOTION.md)。
- **后 CC 科学执行路线**：[POST_CC_SCIENTIFIC_PROGRAM_R0](docs/POST_CC_SCIENTIFIC_PROGRAM_R0.md)：契约 → 可学习性 → 历史执行/学习 → 条件性容量 → 经济确认。

当前工作 PR、已合格基础、候选能力和精确版本统一维护在 [CURRENT_STATE](docs/CURRENT_STATE.md)。本页不再复制旧 AC 阶段的实现断点；文档与基础设施的通过不构成交易盈利证明。

## TODO / 技术债

- **仓库级长期 TODO**：[MAIN_TODO](docs/MAIN_TODO.md)。这里记录具有长期收益、但不应被临时混入当前 scientific stage 的工程与 Infra 技术债。
- **Shanxi Docker Runner 技术债**：当前重点包括专用 `cb16-scratch`、`CB16_CI_WORKER_ROOT` / Provisioner ownership 漂移、`UV_CACHE_DIR` 可写性、preflight writable-root 覆盖、版本化 runner build/start contract、资源与 cleanup 策略。详细拆分见 [MAIN_TODO](docs/MAIN_TODO.md#td-shanxi--shanxi-docker-runner--provisioning-技术债)。
- 技术债条目本身**不授权**修改冻结科学语义；当前 stage 的执行权限仍以 CURRENT_STATE、stage TODO 和 authority/receipt 为准。

## 代码与执行入口

| 路径 | 内容 |
|---|---|
| [cb16_local_opt/](cb16_local_opt/) | Brain、反馈、训练及 R11 运行组件 |
| [authority/](authority/) | 版本化语义、执行范围、receipt 和历史证据 |
| [scripts/](scripts/) | 各实验与阶段的执行入口；先查对应协议再运行 |
| [.github/workflows/](.github/workflows/) | GitHub Actions 与 Shanxi 执行入口 |
| [ci/](ci/) | 仓库检查、运行环境与远程 relay 支持 |

主线原有科学冻结仍由 [SEMANTIC_FREEZE_V1](authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json) 记录。新目标与旧冻结的差异见 [决策记录](docs/DECISIONS.md)；本轮文档不会自动改变运行代码、奖励或实验权限。

## 运维资料

已有资料按各自版本适用：[依赖与资产](docs/DEPENDENCY_AND_ASSET_PROVISIONING.md)、[恢复](docs/DISASTER_RECOVERY.md)、[安全边界](docs/SECURITY_BOUNDARY.md)、[UV](docs/UV_INSTALLATION.md)、[Remote Relay 运维](docs/OPERATIONS.md)。历史 relay 的结果可位于 `ci-results/runs/<commit_sha>/`；R11 Actions 的具体产物与持久化位置以该 run 的协议和 receipt 为准。
