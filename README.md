# Central Brain / CB16

训练一个单资产 Trader：读取市场与账户，采取动作，承担账户后果，再从成功和失败经历中反复学习。人负责资产选择、资本分配与宏观方向；模型学习交易和风险取舍。

**起步 CPU-only。** 先完成受控学习闭环，再按科学路线接入真实历史分钟行情和冻结预训练器官。设备拥有 GTX 1060 不代表当前阶段需要 GPU。

## 最短阅读路线

1. [最高理念](docs/VISION.md)：要学什么、怎样比较胜负。
2. [科学与技术地图](docs/ARCHITECTURE_MAP.md)：每个目标落在哪个组件，哪些已实现、哪些仍需验证。
3. [当前断点](docs/CURRENT_STATE.md)：main、工作分支、部署与证据的区别，接下来执行什么。

Agent 从 [AGENTS.md](AGENTS.md) 确定角色与执行边界；专题资料见 [文档入口](docs/README.md)。当前进度只在 CURRENT_STATE 维护，不从旧 TODO 的 ACTIVE 标签推定状态。

## 仓库内容

| 路径 | 作用 |
|---|---|
| [cb16_local_opt](cb16_local_opt/) | 策略、账户、学习与持久化实现 |
| [authority](authority/) | 版本化合同、receipt 与历史证据 |
| [scripts](scripts/) / [.github/workflows](.github/workflows/) | 阶段入口与 GitHub Actions → Shanxi Docker 执行 |
| [docs](docs/) | 理念、当前导航、任务与历史解释 |
| [ci](ci/) | 环境预检、静态检查与运行支持 |

仓库存代码、合同与脱敏证据；行情、权重、checkpoint 和 secret 按各自存储合同管理，不写入 Git。已存在文件不代表当前 canonical 依赖；历史合同不因文档整理而改写。

[长期技术债](docs/MAIN_TODO.md) 是待办来源，不自动成为 S1 的前置条件。组件 PASS、执行 success、受控可学习性、经济改善和迁移分别需要对应证据。
