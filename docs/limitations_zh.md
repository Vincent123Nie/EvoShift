# 限制与非目标

英文原文：[limitations.md](limitations.md)

这份文档明确区分系统已经控制的风险、只测量但尚未解决的风险，以及仍属于开放研究的问题。

## 研究有效性

### Replay 不是独立测试集

候选由已观察到的 stream failure 产生，又在 recent/protected replay buffer 上验证。配对 replay 可以降方差和发现直接回归，但反复测试会过拟合该 buffer。

Probation future audit 用之后的相关交互补充候选级 memory-on/off 证据；`evoshift audit` 在不同、冻结、不可变异的 stream 上测试完整演化状态。但系统仍没有：

- 在线保留完全独立的 promotion buffer；
- 保证每个 probation 在 stream 结束前得到足够后续样本；
- 在 untouched public held-out 上为每张卡做反事实；
- 为反复 sequential looks 提供 anytime-valid confidence。

因此可测的是流内候选归因和完整状态 held-out transfer，而不是候选级 held-out 因果泛化。

### 统计功效和依赖性

Promotion gate 对配对样本使用 percentile bootstrap。小窗口的 CI 粗糙，stream 样本可能自相关，重复测试多个候选还存在多重比较。当前尚未实现 cluster/block bootstrap、序贯检验、Holm correction 或完整 exact McNemar 分析。

### 基线范围

Static、Self-Refine 和 Reflexion 是仓库内受控概念基线，不是原项目官方复现。它们适合在相同 solver/provider 下消融验证机制，但外部论文数字只能作为背景。

### Provider 非确定性

远程模型 alias 可能在不通知的情况下改变。相同 prompt 也可能受采样、批处理、安全系统或部署影响。幂等和缓存减少重复工作，但不能把远程模型变成不可变实验对象。

## 算法限制

### 词法检索

BM25 和 token-Jaccard MMR 透明、CPU-only，但会漏掉 paraphrase、多语言语义匹配和深层冲突。Dense/hybrid retrieval 应在同一成本和 promotion 协议下比较，而不能默认更优。

Provenance-domain gate 可阻止无关高效用记忆污染上下文，但依赖稳定 domain metadata，也可能抑制相关任务间的有益迁移。

### 记忆 Credit 不完美

Solver 上报的 `applied_memory_ids` 只允许来自 retrieved set；若为空，runner 给整个 retrieved set 记账。这无法识别多张记忆的交互，也无法完全区分 base model 能力和记忆贡献。Leave-one-out、influence 或 Shapley 更因果，但成本更高。

### 漂移检测是启发式

Page-Hinkley + novelty EWMA 为常量内存且可审计，但可能对困难样本簇误报警、漏掉 conditional/covariate shift 或反应太慢。阈值依赖 benchmark calibration。ADWIN、CUSUM、learned detector 和显式 change-point posterior 都应作为消融。

### 策略搜索范围窄

慢环只确定性修改白名单 retrieval/write 超参数，易审计但搜索空间小。GEPA 风格文本演化、contextual bandit、Bayesian optimization 或更复杂 memory operator 可能提升能力，同时也扩大评估成本和安全面。

### 记忆投毒与语义安全

Schema、Shadow、Replay、protected examples 和 solver instruction 能降低投毒风险，但不能证明一条看似有益的 directive 在所有上下文都安全，也不能完全抵御隐藏 prompt injection 或语义 paraphrase 攻击。

Shadow recurrence e-process 的 anytime-valid 性质只在声明的 exact-signature recurrence probability bound 下成立。Exact signature 会拆散语义等价假设；semantic clustering 又会引入错误合并攻击面。持续协调的恶意来源仍可能满足 alternative recurrence model。生产自动提升必须增加领域策略检查和对抗评估。

### Dynamic Trust 可能慢或被劫持

孤立矛盾先被视为可疑，只有同 context 重复一致才确认变化。这降低 premature update，但在 context 稀疏时延迟真实适应。持续攻击者可成为 committed majority，因为该模型测的是时序一致性，不是语义真值或签名真实性。

### Future Audit 尚非序贯校准

非对称早停可减少负 utility 暴露，但固定窗口 bootstrap 被反复查看到 `future_audit_max_observations` 并不 anytime-valid，普通噪声可能增加错误 rollback。

### Active Audit 是选择性的

Active auditor 只在可信失败、且 solver 明确上报 applied ID 时运行，成本可控但不会审计 silent harm、成功但无必要的记忆或多卡交互。Ledger 跨版本累计负证据，并非 context-windowed 或 anytime-valid test。

合成 PolicyShift 的 valid/stale 标签只用于 oracle metrics。真实部署仍需要可验证 policy provenance、可执行反事实或人工复核。

## 数据集限制

### BBH 只是分布变化代理

有序 BBH task family 提供清晰 domain shift 和确定性 scoring，但不直接代表长对话记忆、工具调用、企业政策遵循或生产状态。Frontier model 还可能在预训练中接触过 BBH。

Held-out family 对 EvoShift run 未见，但未必对 base model 未见；它测量外部状态迁移，不是底模零样本新颖性。

### 记忆基准需要专用 Reader

LongMemEval_S adapter 已固定并验证 500 条 cleaned 数据、470/30 scored/abstention 划分和官方 retrieval metrics，但尚未实现 downstream reader QA、temporal grader 或 LongMemEval-V2。LoCoMo 仍需专用 adapter/scorer。将这些数据扁平化进通用 Hugging Face adapter 会破坏其记忆任务结构。

Tau3 retail policy drift 使用公开来源和 EvoShift 版本 overlay，不是官方 tau3-bench 任务，不能报告为官方 tau3 leaderboard 分数。

## 系统限制

- SQLite 适合单机单 writer 研究 runtime，不适合高 QPS、多 writer 或多 region；
- cache namespace 不是租户隔离边界；
- artifact 含 prompt、reference、model output，未静态加密；
- 没有完整 dependency lockfile，跨日期解析可能不同；
- manifest 记录配置 model ID，但没有 provider-side checksum/deployment version；
- 未配置真实价格时只能用 token 作为成本代理；
- reverse proxy 可能忽略 idempotency、漏报 usage 或拒绝可选字段；
- 下载数据和大部分 run artifact 被 Git ignore，正式实验需要独立保留策略。

## 非目标

EvoShift 当前不尝试：

- 训练或微调模型参数；
- 生成并执行任意工具或源码 patch；
- 提供企业多租户 Agent 平台；
- 官方复现所有公开 baseline；
- 用合成 demo 或 smoke run 宣称 leaderboard SOTA；
- 保证所有 distribution shift 下单调提升。

## 优先路线图

1. 用冻结公开 API 模型和真实 policy/tool 环境验证 active-memory 因果治理；
2. 用 context-windowed、序贯有效的退休检验替代累计 failure-only evidence；
3. 在 untouched policy stream 上执行每张卡的 counterfactual audit；
4. 增加 exact McNemar 和 seed-cluster/block bootstrap；
5. 增加 LongMemEval-V2、temporal reader 和 same-reader QA transfer；
6. 增加 dense/hybrid retrieval 与 contradiction-aware admission；
7. 增加 BFCL structured tool state 和官方 executable scoring；
8. 增加 tenant-aware storage/cache isolation 和加密 artifact export。

在这些工作完成前，最强的诚实表述是：EvoShift 是一个经过测试、可审计的 API-only 测试时适应研究框架，而不是已证明的生产服务或 SOTA 算法。
