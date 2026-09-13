# 三层对齐、监督审核与代码交付协议

日期：2026-09-13。来源：用户本轮明确指定三种角色关系、Shanxi Docker CI 和合并前逻辑审查要求。本文规定后续协作方式，不追溯修改已关闭任务、冻结科学合同或 receipt。当前任务仍从 [S0/S1 总控](R11_POST_CC_S0_S1_TODO.md) 接手。

## 1. 产物对齐与审核关系

| 角色 | 向谁/什么对齐 | 主要产物 | 由谁监督审核 | 自己监督审核什么 |
|---|---|---|---|---|
| 用户 | 自己的目标、资源与取舍 | 最高理念、授权和目标修订 | 用户保留最终决定权 | Astra 的理念与原则文档 |
| Astra | 用户的最高理念与已确认决定 | 理念、原则、研究方向和文档审阅 | 用户 | Sol 的 TODO 是否忠于文档、具体且可执行 |
| Sol | 经用户授权的理念/原则文档与适用 authority | TODO、科学具体化、验收设计、代码审核记录 | Astra | DS Flash 的代码、测试和 CI 证据是否满足 TODO |
| DS Flash | 经 Astra 审阅的 Sol TODO 与冻结合同 | 分支代码、测试实现、GitHub Actions 运行与交付证据 | Sol | 自检自己的实现；自检不替代 Sol 审核 |

产物对齐方向：`用户理念 → Astra 文档 → Sol TODO → DS Flash 代码`。监督方向：`用户审核 Astra；Astra 审核 Sol；Sol 审核 DS Flash`。

这些模型名称表示用户指定的当前分工，不是模型能力证明，也不自动授予数据、运行或合并权限。替换执行模型不能取消对应的上游审查职责；执行者不能自己署名为 Sol/Astra 来补审核。

Astra 本角色不承担业务代码实现；Sol 对拆解、数学/判定具体化和代码正确性审查负责；DS Flash 在已授权范围内持续实现、修复并运行 CI。Sol 不得把代码审核退化为转述 DS 的报告，Astra 也不能仅凭 TODO 排版完整就认定任务可执行。

下游发现上游矛盾时，提交具体条款、最小反例和影响，交给相应上游修订。DS 不得为通过测试改目标或门槛；Sol 不得在 TODO 中偷换理念；Astra 改变用户取舍时须回到用户。已经明确且授权的其他工作继续，不要求用户逐个批准普通工程动作。

## 2. 每层交付都要能追溯

| 交接 | 至少提供 |
|---|---|
| Astra → 用户 / Sol | 文档路径与版本、哪些来自用户明确决定、哪些是推导/建议、适用边界；本轮用户明确要求可直接落实，不能伪造其他尚未发生的用户审核 |
| Sol → Astra / DS | 上游条款映射、沿用的 task ID、输入与实现范围、依赖、公式/控制/判据、CI 入口和预算、交付与停止条件 |
| DS → Sol | 实际代码 SHA、净差异、完成的 task、测试与运行来源、失败/未覆盖项、最早未完成项 |
| Sol → 合并执行者 | 被审查 SHA、相关文件/符号、公式和分支审查结果、Shanxi CI 链接与实际测试 SHA、未解决问题以及明确的代码接受/退回结论 |

审核可以作为可追溯的 PR review 或仓库审阅文档；文档路径可用 `docs/reviews/<task>-code-review-<date>.md`。不要求为形式新增 receipt schema，也不把计划中的审核写成已完成。

## 3. 运行证据必须来自 GitHub Actions → Shanxi Docker

**涉及运行行为的代码、测试或运行配置，在合并 main 前，必须由 GitHub Actions CI 调度 Shanxi Docker 执行覆盖该变更的有界冒烟/专项测试。** 正式 scientific qualification 同样通过这条执行路径，且继续满足各阶段原有资格要求。冒烟不代替完整科学资格。

ChatGPT 临时沙盒只用于读取、编辑、静态检查和准备可审阅产物。不得在那里运行业务训练、冒烟或专项测试，也不得为此临时装环境或另造模拟 runtime；不得用推理猜测依赖、GPU、吞吐或运行结果。临时沙盒缺 CUDA/数据不证明 Shanxi 不可执行。现有 repo-guard 在 GitHub-hosted runner 上做静态检查是正常的，但不是 Shanxi runtime evidence。

当前 main 快照 `7ed1e337ffe05761dd5f179c8a8e5d1092f57118` 的实际入口：

| 用途 | 已有入口与限制 |
|---|---|
| 共享 CI 前置检查 | `.github/workflows/_cb16-shanxi-preflight.yml`；先做 repo、Shanxi workflow topology、Node action policy |
| 有界 R11 authority 冒烟 | `.github/workflows/cb16-r11-main-smoke.yml`；当前只有 `workflow_dispatch`，不会因 push main 自动运行；业务 job 为 `runs-on: [self-hosted, shanxi-docker-r11]`，依赖前置检查，timeout 15 分钟 |
| 已完成 S0 的资格入口 | `.github/workflows/cb16-r11-post-cc-s0-qualification.yml`；用于理解已验证接线，不要求因每个新 TODO 重跑已关闭 S0 |
| S1 新行为覆盖 | 由现有 S1-021 及 S1-D 实现/绑定对应 workflow 和专项测试；上述 Main Smoke 只列旧 Stage-4 authority 测试，不能据此宣称新 durable replay / joint learner 已通过 |

任务作者必须指明使用哪个入口、测试集合与资源上限；缺少覆盖时列为本任务的 CI 实现工作，DS 继续补齐，不能拿旧绿色 run 顶替。不得为凑测试恢复 legacy runtime、绕过 preflight、访问 FINAL/fresh 或同时占用 canonical runner 运行冲突任务。

选择工作分支触发已有 workflow 后，核对实际 checkout SHA，不只看 dispatch 时的分支名。代码审核和相关 CI 证据必须覆盖拟合并代码及其集成状态；PR head 与 test-merge SHA 分别记录。此后代码/测试/配置改变，重新审查受影响部分并运行相应 CI；不得以祖先提交的成功代替后续代码验证。

交付至少记录：workflow 路径与版本、run URL/ID 和 attempt、job/runner 身份、实际 checkout SHA、Docker preflight、verified canonical Python、命令/测试集合、执行数量和失败/跳过项。按任务合同保存必要结果/artifact ID/hash；现有 workflow 未导出的文件不得被报告为已上传。

`queued`、`skipped`、`cancelled`、仅 repo-guard 通过或零相关测试均不构成业务冒烟 PASS。执行者要读取相关 job 的步骤和结果，而非只读 workflow 的绿色状态。runner 离线或 CI 权限不足时报告真实阻塞及缺失证据，继续可做的文档/代码准备，不能改用临时沙盒制造资格。

纯说明性文档变更可做文档静态检查与 repo-guard，无需为无运行差异反复占用 Shanxi；必须明确“本次未运行 runtime smoke”。文档若被程序读取或改变实际执行配置，按运行变更处理。这一区分不豁免任何代码的合并前审查。

## 4. DS Flash 代码的重点审查义务

用户已多次观察到 DS Flash 在高密度逻辑、复杂公式转代码、比较方向和直接 `if/elif/else` 分支上出现错误；项目据此要求重点审查。这里记录的是用户的项目经验和防错政策，不是对某个模型版本作未经验证的普遍能力排名。同样标准适用于其他执行模型。

Sol 写 TODO 时必须标出高风险公式/分支及其独立验收方法；合并 main 前必须逐项审查实际代码，不能仅检查有没有测试、类型是否正确或 CI 是否绿色。

| 审查对象 | 必须检查的具体内容 | 可接受的验证方式 |
|---|---|---|
| 公式 → 代码 | 符号/下标/时间对齐、单位与维度、正负号、概率与 log 概率、归一化分母、裁剪位置、梯度归属和 stop-gradient | 逐项映射公式与符号；独立手算短序列/解析答案；必要时与简单标量参考实现对照 |
| 比较与直接条件分支 | `<`/`<=`/`>`/`>=`、等号归属、`and`/`or`/`not`、分支优先级、互斥/穷尽、fall-through、提前 return | 条件/决策表；阈值以下/恰等/以上；相反符号输入和冲突条件 |
| mask、空值与数值边界 | 空集合、零分母、NaN/Inf、缺失值、零/负权益与负债、shape/broadcast；fallback 是否扩大样本或抹掉失败 | 独立边界例；验证按合同拒绝或处理，不能用静默 clamp/全行 fallback 改义 |
| 动作、状态与信用 | nominal/executed action、true joint log_mu/log_pi、terminal/truncation/bootstrap、reject 后推进、换代与恢复 | 短轨迹逐步账本及信用核算；实际链路/恢复测试，检查反例能失败 |
| gate 与最终 verdict | 比较方向、分母、全 seed 聚合、负控、缺失 gate、空结果、异常路径是否误判 PASS | 预构造 PASS/FAIL/边界结果；证明错号/错比较/缺字段会被拒绝，而非复用被测代码生成“期望答案” |

复杂表达式可以拆成有名中间量、明确辅助函数和决策表；不要求全项目改成某种语言，也不禁止必要分支。代码更短或“看起来优雅”不能成为接受理由。测试 oracle 不应复制同一套复杂公式，使实现和测试共享同一错误。

审查覆盖变更及直接受影响的调用关系，不无限重审整个历史仓库。发现具体错误，DS 修复并在 Shanxi CI 验证，Sol 复审；未解决的公式/控制流错误、缺少必要 CI 或缺少 Sol 审核，均不能将该代码合并 main。有效 `SCIENTIFIC_FAIL` 则按原任务证据与收尾规则处理，不能为了让科学结果变绿而改目标。

## 5. 文档约定与平台强制分开

本轮建立的是执行与审核纪律。文档发布不等于 GitHub branch protection / required checks 已配置，也不等于 Sol 已完成任何未来代码审查。当前权限、规则和 workflow 是否自动阻止合并，须由执行者查实；不得依赖不存在的平台拦截。

GitHub 官方说明：`runs-on` 决定执行机器；`container` 决定显式 job 容器，未设置时步骤在 runner 环境执行。本仓库使用部署在 Docker 内的 runner 和既有 preflight，不因 YAML 没有 `container:` 就擅自新增嵌套容器或改部署。见 [Running jobs in a container](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/run-jobs-in-a-container)。

GitHub 的 required checks 关联实际提交，且某些 skipped/neutral 状态可能满足平台条件；本项目因此另要求确认相关业务测试真正执行。见 [Troubleshooting required status checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/troubleshooting-required-status-checks)。官方文档支持 CI 机制说明；三层角色与 DS Flash 审查强度来自用户决定。
