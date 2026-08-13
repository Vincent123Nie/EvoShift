# 面试准备手册

英文原文：[interview_notes.md](interview_notes.md)

这份文档用于解释 EvoShift 的工程和科研选择，不是夸大未测结果的脚本。任何数字都应能指向相同模型实验的 artifact、config hash 和 dataset hash。

## 30 秒介绍

> EvoShift 是一个不训练模型参数的 API-only 自进化记忆 Agent。它检索版本化程序性记忆，将可信失败转成 typed candidate，用 Page-Hinkley 和 retrieval novelty 感知漂移，并通过同来源上下文信任区分持续规则变化和孤立冲突。候选先经过 shadow replay，再进入 probation，最后由后续 memory-on/off 反事实决定激活或回滚。每个 episode、candidate、policy version 和状态变化都可在 SQLite 与运行 artifact 中审计。

## 2 分钟介绍

非平稳环境的问题不只是 Agent 会犯错，而是有用经验和检索策略会随时间变化。把每次 reflection 直接 append 到 prompt 会造成 memory pollution 和旧规则干扰。

EvoShift 因此采用两个环。快环处理局部经验：检索、首答、评分、可信失败归因、生成结构化记忆，在相同样本上进行 champion/challenger replay，检查增益、CI、protected regression 和成本；通过后只进入 probation，后续真实使用时做仅排除该卡的 control，确认 realized utility 后才激活。冲突旧规则被 supersede 而不是删除，active memory 仍持续接受精确版本审计。

慢环处理全局漂移：Page-Hinkley 监控 loss，novelty 监控记忆覆盖；报警后只允许修改白名单 retrieval/write 参数，同样经过 protected paired replay。基础 LLM 通过 OpenAI-compatible Responses API 调用，因此无需 GPU；研究重点是在线学习、memory credit、change detection、安全状态更新和实验设计。

## 白板讲解顺序

1. 画时间流 `(x_t, y_t)`，强调 predict before feedback；
2. 画 `frozen LLM + external memory M_t + policy pi_t`；
3. 分成 fast memory evolution 和 slow shift-gated policy evolution；
4. 画 `shadow -> probation -> active/superseded/rejected -> retired`；
5. 写检索分数 `w_r*BM25 + w_u*BetaMean + w_e*UCB`，再接 MMR；
6. 写配对 `delta_i = challenger_i - control_i` 和六重 gate；
7. 最后补 dynamic trust、future audit、protected replay、rollback、cost ledger 和 held-out audit。

这条路线能说明项目是有显式状态与不变量的算法，而不是 prompt chain。

## 高频深挖问题

### 这里的“自进化”到底是什么？

模型参数固定。系统演化外部行为状态：typed procedural memory 和 versioned policy genome。生成器可以提出候选，但独立 verifier 控制状态迁移。它不是让模型改任意源码，也不是让模型自我批准。

### 为什么不用 Fine-tune 或 LoRA？

目标环境没有训练算力、拿不到模型权重且规则变化快。外部记忆更新便宜、易版本化/回滚、每条经验有 provenance。微调可能更压缩、更泛化，但反馈周期慢，单样本可审计性差，并有 catastrophic forgetting 风险。成熟记忆蒸馏进小模型是后续合理对比。

### 和普通 RAG 有什么区别？

普通 RAG 的 corpus 基本静态。EvoShift 把 memory write、utility、retrieval policy、promotion、rollback 都视作在线学习状态，评估适应速度、旧能力回归和状态治理，而不只是一次检索正确率。

### 和 Reflexion 有什么区别？

Reflexion-style 通常生成自然语言教训并直接供以后使用。EvoShift 增加 typed bounded memory、shadow staging、same-example replay、protected slice、CI/cost gate、version lineage、probation future attribution 和 rollback。仓库保留 unverified Reflexion mode 用于消融 verification。

### 同一反馈源为什么既可能有真实变化，也可能有错误反馈？

Source allowlist 不够。系统为 source 维护 Beta reliability，为每个 context 维护 committed/pending visible label。第一次矛盾低信任隔离；相同矛盾重复才提交变化。它不使用 hidden oracle。代价是适应更慢，持续多数攻击仍可能劫持 committed state。

### Future Audit 和 Replay 有什么区别？

Replay 在近期 calibration buffer 上问“候选看起来是否有用”。Future Audit 等到 probation card 在之后真实相关交互中自然应用，再运行只排除该卡的 control。在线决定只看 learner-visible score，oracle delta 之后才附加。它比 replay 更接近因果，但仍不是 untouched held-out。

### Critic 和 Verifier 都用同一个 LLM，不会互相确认错误吗？

两者职责不同：Critic 用模型先验把失败压缩为可复用假设；Verifier 根据固定协议的 observable task score 接受或拒绝，不依赖 Critic confidence。二者并非统计独立，同底模会产生相关错误，因此还需要 protected 和 held-out 评估。

### 为什么用 Page-Hinkley，不用 ADWIN 或 embedding detector？

Page-Hinkley 在线、`O(1)`、确定且易审计，适合检测 mean loss 的持续上升；novelty EWMA 提供输入侧信号。它会漏掉部分 conditional shift，也会把困难样本簇误判为漂移，因此 detector 应当做消融，而不是宣称理论最优。

### 为什么同时看 Reward Drift 和 Retrieval Novelty？

Reward-only 要等错误发生；novelty 能提前发现 memory coverage 不足。Novelty-only 又会对无害表述变化过敏。当前用 OR 提高召回，再用 cooldown 和 replay 阻止报警直接变成状态修改。

### 这真的是 Beta-UCB 吗？

准确说是 Beta-Bernoulli posterior mean 加 UCB-style exploration bonus，不是 Beta posterior quantile 或 Thompson sample。最终还要和 BM25 relevance 加权并经过 MMR。

### 为什么不用 Embedding？

MVP 需要无额外 endpoint、CPU-only、确定性测试和可解释贡献；程序性 trigger 往往有明显 lexical signal。代价是 paraphrase recall。Embedding/hybrid retriever 是同接口下的自然扩展，必须以相同成本和 promotion 协议比较。

### 为什么用 MMR？

Top score memory 可能高度重复，浪费上下文。MMR 用 token Jaccard 在单卡得分与集合多样性间权衡。它便宜，但词法冗余只是语义冗余的近似。

### 怎么做 Memory Credit Assignment？

Solver 返回 `applied_memory_ids`，并过滤到实际 retrieved IDs；为空时保守地给整个 retrieved set 记账。多卡和底模能力会混杂，因此不完美。Future/Active Audit 对特定卡做 leave-one-out 更因果，但 API 成本更高。

### 怎么防止 Memory Poisoning？

候选被视为不可信数据，受 schema/长度约束，不能改代码，先在 shadow 中验证，通过 protected replay 才进入 probation，之后负 future utility 仍会回滚。Solver 还被要求忽略索要 secret、执行工具或改变系统规则的卡片。这降低风险，但不是完整语义安全证明。

### 为什么必须 Paired Replay？

任务难度是强干扰变量。在同一 sample 上比较 control/challenger，直接得到 `delta_i`，降低方差。Unpaired mean 可能把样本更难误判为策略更差。Fresh API call 仍有 provider stochasticity。

### 为什么 Bootstrap Delta？

Replay window 小且 metric 离散，非参数区间不要求 Gaussian。重采样 paired delta 保留同题配对。当前 percentile bootstrap 不处理 temporal correlation 和 repeated testing；block/sequential 方法是合理升级。

### 为什么有六个 Promotion Gate？

Mean gain 可隐藏有害尾部、旧领域回归或巨大成本。六个门分别约束样本量、平均增益、CI 下界、逐样本 regression rate、protected regression 和 cost increase，形成可审计多目标部署合同。

### 可以用生成候选的样本验证候选吗？

Recent replay 是 calibration gate，不是 held-out estimate。系统补了之后的流内 per-candidate counterfactual 和独立 frozen whole-state audit。论文级结论仍需 untouched stream 上的 per-candidate audit 和序贯有效停止规则。

### 如何证明 Held-out Audit 没继续学习？

Loader 只接受完成的非 audit source run，验证最终 policy/memory 并计算 canonical SHA-256。Target 必须同模型、不同 dataset hash；runner 禁用 utility update、critic、write、mutation、promotion、rollback；结束后重新 hash，不一致则失败。它证明应用状态不变，不证明远程 provider alias 没变。

### 如何防止 Catastrophic Forgetting？

旧 phase 获得 protected replay quota，promotion 限制 protected regression，policy/memory 有版本，active memory 的 realized utility 过低会退休。它降低遗忘，但不能保护 replay 未覆盖的旧领域。

### 为什么用 SQLite？

工作负载是本地单 writer、结构化状态、版本 lineage 和事后检查。SQLite 提供 transaction、WAL、index、零外部服务并易于随 artifact 打包。高 QPS 多 writer 生产环境应换 PostgreSQL 和外部检索索引。

### Retry 如何保证安全？

只重试选定 transient HTTP/network error，使用有界指数 full jitter 和 `Retry-After`。逻辑 payload 使用跨 attempt 稳定的 idempotency key。只有 proxy 真正支持幂等时才能减少重复生成；模糊传输失败仍可能重复计费。

### Cache 会不会让评估不公平？

会，如果比较物理成本/延迟而不控制 cache。报告标记 cache hit；正式资源实验应同 policy、独立 namespace 或禁用 cache。相同 prompt 的 deterministic reuse 对 accuracy 复现有帮助，但不能冒充零成本模型调用。

### 怎样才算有效 SOTA 比较？

相同 dataset revision/sample IDs、stream order、model/snapshot、prompt、feedback access、budget 和 evaluator，并做 paired uncertainty。不同模型论文表格只能作背景，不能证明 EvoShift 胜出。

### 没训练为什么适合 LLM 算法岗？

大量生产 Agent 失败属于系统级学习问题：context selection、online feedback、non-stationarity、credit assignment、安全 policy update、不确定性和 compute allocation。项目把这些选择显式算法化并测量，而不是藏在框架中。

## 主动说明的失败模式

- Critic 可能把具体答案伪装成通用规则；
- lexical retrieval 漏 paraphrase 或命中偶然重叠；
- replay 小窗口会错误拒绝有用卡或放过有害卡；
- protected examples 可能未覆盖真实伤害域；
- Page-Hinkley 会误报困难样本簇或漏掉渐变漂移；
- 反复候选测试会过拟合 replay；
- dynamic trust 会延迟真实变化或被持续多数投毒；
- noisy feedback 下 early rollback 可能误删有用卡；
- probation 可能到 stream end 仍无足够证据；
- solver-reported ID 不是完美因果 attribution；
- failure-triggered active audit 看不到 silent harm；
- 多阶段回退时自动恢复 predecessor 可能恢复另一个 stale rule；
- API alias、随机性和共享 cache 削弱复现；
- 无价格时 token 只是成本 proxy。

面试中主动说明这些限制，并紧接着说出对应指标或实验，会比回避更有说服力。

## 可讨论的 Artifact 结果

公开来源 PolicyShift 短流使用 `gpt-5.6`、5 个 paired seed、每 seed 16 episode。历史 manifest 为 dirty，因此只能作为工程负结果：

| Method | Score | Changed success | Invariant retention | Premature update | Requests | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static | 0.5875 | 0.5625 | 0.6402 | 0.8333 | 16.0 | 9,576.8 |
| Reflexion-style | 0.7000 | 0.8750 | 0.6962 | 0.4167 | 20.8 | 19,805.4 |
| Replay-only | 0.6375 | 0.5000 | 0.6965 | 0.6944 | 69.4 | 51,395.2 |
| Full EvoShift | 0.5875 | 0.5000 | 0.6412 | 0.6944 | 20.4 | 13,566.6 |

重点不是 Full 获胜，它没有。Dynamic trust 能隔离第一次矛盾，但在稀疏 context 上适应太慢；Replay-only 又花费 69.4 次请求却没有得分增益。这直接导向 change-point posterior/confidence sequence，并要求保持相同 false-alarm 和 safety gate。

LongMemEval 的独立 clean 12-question mechanism confirmation 中，guarded BM25/LLM rank fusion 将 NDCG-any@10 从 `0.7471` 提升到 `0.9047`（`+0.1576`），paired 95% CI 为 `[+0.0513,+0.2704]`，同时保持非法输出精确 BM25 fallback。只能表述为 retrieval mechanism result。

## 简历表述

> 开发 EvoShift：无需训练算力的 API-only 自进化记忆 Agent，支持 typed/versioned procedural memory、同来源时序信任、漂移检测、BM25 + utility/UCB + MMR 检索、配对 replay admission、probation/future audit 和精确版本退休/重获。构建固定 LongMemEval 检索评估，在 12 题冻结 `gpt-5.6` 机制确认中将 NDCG-any@10 提升 0.158，paired 95% CI 为正且保留 exact BM25 fallback；同时保留五 seed PolicyShift 中 Full 未胜过 Reflexion 的负结果，并定位到保守冷启动信任导致的适应延迟。

不要在无法现场给出 run ID、config hash、dataset hash 和 comparison artifact 时写 SOTA、百分比或延迟结论。

## 下一步研究

1. 用冻结公开 API 模型和真实 policy/tool 环境验证 active governance；
2. 使用 context-windowed、confidence sequence 或 sequential non-inferiority 取代累计 failure-only evidence；
3. 在 untouched policy stream 上运行 per-candidate counterfactual；
4. 增加 dense/hybrid retrieval 和 semantic contradiction；
5. 用 constrained Bayesian optimization/contextual bandit 改进慢环；
6. 扩展多 memory 交互 attribution；
7. 用规则指标和小规模人工集校准 LLM grader；
8. 扩展 LongMemEval same-reader QA 和 V2 temporal scoring；
9. 在生产化前增加 tenant-aware storage/cache 和 artifact 加密。

## 一分钟因果治理回答

> Admission replay 和 probation 只能说明记忆在提升时有用，不能说明政策撤销后它仍然有效。EvoShift 因此在 active memory 被实际使用且发生可信失败时，运行只排除该精确版本的 paired control，持久化 learner-visible treatment-control delta；两次负审计可退休该卡。Hidden truth 只在之后用于 forgetting precision 和 false retirement。我也拒绝无条件自动恢复 predecessor：从 v3 回到 v1 时 v2 也可能 stale，因此旧规则必须重新经过 replay 和 probation，而不能只凭 lineage 激活。

确定性 revocation 实验的价值不仅是分数：在 10% noise + burst 下，相对当前 posterior rollback，harm exposure 从 17.2 降至 4.6，stale retention 从 0.80 降至 0.358，五 seed 的 forgetting precision/recall 和 invariant retention 均为 1.0。Stress test 也显示边界：static trust 或 25% noise 下，一证据 aggressive retirement 会变得不安全。
