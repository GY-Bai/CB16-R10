# Qualification TODO Authoring Addendum

本文件补充 `SOL_ROLE_PRINCIPLES.md` 与历史 `S_SERIES_TODO_AUTHORING_PRINCIPLES.md`。

复杂 S-series TODO 发布前，Sol 必须先完成 Stage Qualification Profile，并逐项确认：

1. 每个 capability claim 已拆成 machine-checkable proof obligations；
2. 每个 mandatory obligation 有 producer、consumer、gate、artifact proof、hostile counterexample；
3. behavior/target/evaluation、runtime/review/record/merge、parent/child checkpoint 等 identity 不混用；
4. RNG owner、seed derivation、isolation、construction order 已定义；
5. 合法 edge case 的 `PROCESS / ZERO_CONTRIBUTION / FAIL_CLOSED / SCIENTIFIC_FAIL` 已预先决定；
6. formal-only runner/gate 使用 production-shape canary；
7. durable/restart-safe claim 能在删除 scratch 后 artifact-only 复核并对中间 corruption fail closed；
8. TODO 没有把尚未定义的科学选择留给 DS Flash；
9. Fix-impact Matrix 能区分普通 bugfix 与需要 versioned authority 的 science change；
10. exact-SHA GitHub Actions -> Shanxi Docker 证据路径已经指定。

这十项未闭合时，TODO 不应以“可执行”状态交给 implementation agent。
