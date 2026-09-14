# 文档职责与最短接手规则

2026-09-14 精炼。目标是让科学要求、代码与证据对得上，并减少重复阅读。本页不增加科学门槛或另建文档框架。

## 一种信息只维护一个主入口

| 信息 | 主入口 | 维护责任 |
|---|---|---|
| 用户目标、已确认取舍 | [VISION](VISION.md)，详细决定见 DECISIONS | Astra 向用户对齐 |
| 科学要求到实际技术的对应 | [ARCHITECTURE_MAP](ARCHITECTURE_MAP.md) | Astra 审边界；Sol 在变更交付中指出映射变化 |
| 当前阶段、PR、阻塞与下一次运行 | [CURRENT_STATE](CURRENT_STATE.md) | Sol 随阶段审阅/集成同步，Astra 检查组织一致性 |
| 角色与审核义务 | [ROLES_AND_REVIEW_PROTOCOL](ROLES_AND_REVIEW_PROTOCOL.md) | Astra；Sol/DS principle 引用并细化 |
| 本次具体参数、范围、门槛 | CURRENT_STATE 指向的 stage TODO / authority / manifest | Sol 版本化冻结并审阅 |
| 实现与运行结果 | 实际代码 SHA、Actions、artifact、review、receipt | DS 交付，Sol 独立核验 |
| 历史推导与失败 | 原版本文件/提交、既有 review | 保留追溯，不当作当前待办 |

README 和 AGENTS 只路由当前状态，避免复制进度。已合入的资格框架只定义证明纪律；具体 stage 的科学证明由 adapter 和 reviewer 完成，不成为第二份最高理念。

## 阅读与交付

默认读取 VISION → ARCHITECTURE_MAP → CURRENT_STATE，然后按角色读取当前任务及直接依赖；不要求掌握全部历史。历史数字、旧算法候选、合成测试和真实数据接入必须有明确身份。

Sol 的任务映射应能回答：保护哪个科学要求、复用哪个实际组件、缺什么、用什么证据关闭。可在已有 TODO/PR 中写清，不为普通改动新增整套 schema 或平行规范。代码存在、测试通过、Sol 接受、合入 main、宿主部署分别记录。

收尾时同步当前导航和直接受影响的映射；仍未完成的事项列具体阻塞。仅文档改动不要求重跑历史业务测试。发现需要运行修复的缺口，转给 Sol，不用本轮文字改写伪造已完成。

## 历史与后继

旧文件的 ACTIVE、待实现、owner-unresolved 按原 SHA 理解，不覆盖新证据。新目标也不追溯改变旧 receipt。冲突处理需说明条款、身份和替代范围，不能简单以“文件较新”决定科学权限。

保留原算法/资格 R0 的推导；实际网络、分布、优化器、预算以当前阶段合同为准。CPU-only 是用户本次明确后继设备决定，按性能策略落地；不是删除历史 GPU 证据或修改科学参数的理由。
