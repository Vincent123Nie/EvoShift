# EvoShift 评估协议

英文原文：[evaluation.md](evaluation.md)

本文件规定如何评估 VERA，同时避免把在线适应变成测试集泄漏。它是实验协议，不是结果报告；其中数值不代表新的实证结论。

## 1. 待验证主张

主张刻意保持狭窄：

> 在冻结相同基础模型、任务 prompt、benchmark stream 和资源预算时，经过验证的跨任务经验，应比静态 prompting、Self-Refine 和未经验证的 Reflexion-style memory 更可靠地提高 domain shift 后的 prequential performance。

Primary endpoint 是完整 stream 上首答得分的配对变化。主要适应指标是 post-shift gain；protected-slice regression、总 provider usage 和 held-out transfer 是必须报告的安全/效率指标。

实验单位是 benchmark sample。API retry 不是新样本；replay 已评分样本是适应计算，不是新的测试观测。

## 2. 数据与任务流

### 2.1 BBH Adapter

BBH adapter 固定上游 revision `9ee07bd481feebf959a6b59d61ea57bdcf30964d`，并按 `src/evoshift/benchmarks/bbh.py` 中 manifest 校验每个文件的 SHA-256 和 byte size。这只能证明评估了哪些公开文件，不能证明 base model 预训练时未见过 BBH。

`configs/experiments/evoshift_bbh_full.yaml` 的主 stream：

| Phase | BBH subset | 角色 | 每 seed 样本数 |
| ---: | --- | --- | ---: |
| 00 | `boolean_expressions` | source；protected | 32 |
| 01 | `date_understanding` | source；protected | 32 |
| 02 | `disambiguation_qa` | source | 32 |
| 03 | `multistep_arithmetic_two` | source | 32 |
| 04 | `causal_judgement` | source | 32 |
| 05 | `navigate` | source | 32 |
| 06 | `tracking_shuffled_objects_five_objects` | source | 32 |
| 07 | `word_sorting` | source | 32 |

每方法每 seed 共 256 个首答 episode。Subset 顺序固定；`shuffle: true` 只在 subset 内以 `evaluation.seed` 确定性打乱。每个 sample 保存 source revision/index、phase、domain、shift boundary 和 protected flag。Smoke config 每 phase 只有四题，只能检查管线。

### 2.2 Meta-development、主测试和 Held-out

BBH 没有专门 continual-learning split，因此按 task family 划分：

| Split | 推荐 subset | 允许用途 |
| --- | --- | --- |
| Meta-development | `logical_deduction_three_objects`、`object_counting`、`sports_understanding`、`temporal_sequences` | Debug、prompt、阈值和 retrieval 超参 |
| Main prequential test | 主配置八个 subset | 锁定后每 seed 一次；首答评分后才能消费反馈 |
| Held-out transfer | `formal_fallacies`、`geometric_shapes`、`hyperbaton`、`web_of_lies` | 冻结 memory evaluation；禁用 critic/write/mutation/reward feedback |

三个 split 使用相同 pinned revision 和 `phase_size: 32`。Task family 若换 split，必须版本化实验并重跑所有方法。

`evoshift audit` 加载 source run 的最终 policy/active memories，要求相同模型、不同 dataset hash、禁用全部状态变化，并在结束时验证状态 SHA-256 不变。`evoshift audit-memories` 还会对每个 active `(memory_id, version)` 做 frozen leave-one-out，报告配对 delta/CI、protected/future slice 和 application coverage。零 coverage 只能写“未测试”，不能写“无效”。

### 2.3 Seed

开发至少 3 个 paired seed；简历或论文风格最终表至少 5 个。所有方法使用完全相同 seed 列表。Seed 改变 subset 内样本选择和顺序，不改变 domain 顺序。必须报告每 seed、均值和考虑 seed cluster 的不确定性，不能只报告最佳 seed。

## 3. Prequential 协议与标签防火墙

Episode `t` 的合法顺序：

1. 只加载 `< t` 产生的 memory/policy/feedback state；
2. Solver 只接收 task、domain 和 retrieved cards，不接收 reference；
3. 在评估前持久化首答；
4. 用隐藏 oracle 计算 `Episode.score`；
5. 独立计算学习器可见 `Episode.feedback_score`；
6. 将两个分数作为不可变记录写入；
7. 之后才把允许的 feedback 暴露给 adaptation；主协议为 `reward_only`；
8. 候选、replay、promotion、rollback 和 policy mutation 可在 `t+1` 前进行。

主结果禁止：未来样本泄漏、首答前展示 reference、将 `reference_upper_bound` 混入公平条件、看过主测试后调参、丢弃失败调用/seed/困难 phase、人工改写最终 run 的候选卡。

`grader_feedback` 只有在 benchmark 环境本来就提供该反馈时才公平。

## 4. 基线与公平性

| 方法 | 首答行为 | 跨任务写入 | Promotion 验证 | Policy 演化 |
| --- | --- | ---: | ---: | ---: |
| Static | 单次 solve，无学习记忆 | 否 | 否 | 否 |
| Self-Refine | draft + 一次同题 refine | 否 | 否 | 否 |
| Reflexion | 单次 solve，失败后 append experience | 是 | 否 | 否 |
| Replay-only | shadow + paired replay，无 dynamic trust/post-audit | 是 | 是 | 是 |
| Full VERA | typed memory、replay、probation、rollback | 是 | 是 | 是 |

必须固定 provider/model、system prompt、reasoning effort、output limit、retry、dataset revision/hash、样本 ID/顺序、scorer、cache policy、预算和反馈模式。

使用 `storage.isolate_runs: true` 防止状态串用。正式成本/延迟比较还必须关闭共享 cache 或为每个 method/seed 使用独立 cache；否则后运行的方法可能零成本命中。`evoshift compare` 按 `sample_id` 对齐，并拒绝不完整配对。

相同 ceiling 不等于相同实际计算。表中必须同时报告自然计算结果和 `costs.json` BudgetLedger 中 uncached 外部调用/token/美元。`metrics.json -> resources` 只覆盖前台任务调用。

当前没有严格 consumed-compute-matched baseline，因此应写“same-backbone with full cost reporting”，不能写“完全同计算预算”。价格配置为零时美元结论无效，使用 token proxy。

## 5. Scorer 与双通道反馈

支持 `exact_match`、`normalized_exact_match`、`token_f1`、`numeric`、`multiple_choice` 和 `binary_choice`。Partial-credit scorer 的 `ScoreBundle.primary` 为连续分数，success 由预先固定的 `metadata.success_threshold` 判定。

`Episode.score` 是报告能力的隐藏 oracle；`Episode.feedback_score` 是 drift、credit、candidate 和 replay 可使用的观测。普通基准二者相同，PolicyShift 可以显式提供 noisy `feedback_reference`。

FeedbackTrust 只能读取 visible source、context key 和 learner-visible feedback。`feedback_kind`、`feedback_corrupted`、oracle、phase、attack annotations 都只能用于事后指标。

报告必须包含 oracle/feedback agreement、FP/FN feedback、noise rate、score gap、mean trust、eligible rate、corrupted quarantine 和 clean quarantine。

### PolicyShift

确定性企业流在同一 refund domain 中改变隐藏政策。支持 version recurrence，例如 `v1 -> v2 -> v3 -> v2 -> v1 -> v2`。Prompt 只暴露 customer tier 和 request day，不暴露 policy version、valid/stale memory tag 或 reversion label。

每个样本属于一个政策切片：

- `transition_case`：本次边界后正确决策发生变化；
- `protected`：所有 policy version 下答案不变；
- `future_change_case`：当前未变、未来才变，过早采用未来规则属于错误。

Noise 可随机翻转 feedback；attack 可强化旧政策；hard same-source 模式可在真实变化前注入短暂、一致的未来政策 burst。干净、噪声和攻击可共享同一 source，因此 source allowlist 不足以解决问题。

指标包括 changed success、old-rule leakage、invariant retention、future-change success、premature update、corrupted-feedback following 和 attack-following。

### Tau3Retail-PolicyDrift

该 adapter 基于 MIT tau3-bench v1.0.1 retail policy/task vocabulary，固定上游 commit、size 和 SHA-256，并在其静态政策上加入 EvoShift 明确版本化 overlay。它不是官方 tau3-bench task，不能报告为官方分数。

重复 seed 报告使用 hierarchical paired interval：先重采样 seed cluster，再在每个 seed 内重采样 paired sample。一 seed mini 只能验证协议。五 seed `gpt-5.6` 负结果记录在 `experiment_policyshift_v1_frozen.md`。

## 6. 指标定义

令首答分数为 `s_t`、二值成功为 `y_t`、stream 长度为 `T`。

### 整体与分阶段表现

\[
\text{MeanScore}=\frac{1}{T}\sum_{t=1}^{T}s_t,
\qquad
\text{SuccessRate}=\frac{1}{T}\sum_{t=1}^{T}\mathbb{1}[y_t].
\]

相同指标按 phase 分开计算。

### Post-shift gain

对每个已知/检测边界后前 `W` 个 episode 的去重并集 `U`：

\[
\text{PostShiftGain}=\frac{1}{|U|}\sum_{t\in U}
(s_t^{\text{VERA}}-s_t^{\text{baseline}}).
\]

默认 `W=20`。已知 phase boundary 和 detector alarm 应分开报告，避免 false alarm 改变评估窗口。

### AUAC、恢复和 Regret

\[
\text{AUAC}=\frac{1}{x_n-x_1}\sum_{i=1}^{n-1}
(x_{i+1}-x_i)\frac{a_i+a_{i+1}}{2}.
\]

Recovery steps 是 shift 后最近完整 `w` 窗口均值首次达到 shift 前表现 `recovery_fraction` 的已消耗 episode 数；stream 结束仍未恢复为 `N/A`，不是 0。

\[
R_T=\sum_{t=1}^{T}(c_t-s_t).
\]

默认 `c_t=1`，即累计错误；与 baseline 比较时用其逐样本分数作为 `c_t`。

### Backward transfer 与遗忘

令 `A_(i,j)` 为适应到 domain `i` 后在 domain `j` 固定 probe 上的表现：

\[
\text{BWT}=\frac{1}{N-1}\sum_{j=0}^{N-2}(A_{N-1,j}-A_{j,j}),
\]

\[
\text{Forgetting}=\frac{1}{N-1}\sum_{j=0}^{N-2}
(\max_{i\in\{j,\ldots,N-1\}}A_{i,j}-A_{N-1,j}).
\]

普通 runner 不自动构建该 matrix。Held-out forward transfer 必须是在 untouched domain 上，完整 frozen state 相对 static 的配对增益。

### Promotion precision 与覆盖率

\[
\text{PromotionPrecision}=
\frac{\sum_i \mathbb{1}[p_i=1 \land g_i>0]}
{\sum_i \mathbb{1}[p_i=1]}.
\]

- `replay_estimated_promotion_precision` 使用 replay mean delta；
- `realized_promotion_precision` 只使用完成 future audit 的候选；
- `realized_promotion_coverage = completed audits / all replay promotions`。

Expired candidate 不进入 realized precision 或 false rollback。未启用 future audit 时 realized precision 为 `N/A`。

### Active governance 指标

包括 harmful active-memory exposure、stale-memory retention、selective-forgetting precision/recall、false retirement、exact-version audit coverage、budget utilization、retirement latency、early-retirement precision、predecessor reactivation 和 reacquisition correctness。

Static 没有 stale exposure 是因为它不学习，不代表它解决了遗忘；必须和 changed-case success 一起看。

### 资源和延迟

前台 usage 报告 request、token、USD、每 episode/成功 token、mean/p50/p95 latency。任何结论都要同时给出 BudgetLedger 总量，使 adaptation overhead 可见。Cache response 保留逻辑 token，但增量 cost/latency 为零；成本或延迟结论应关闭或隔离 cache。优先使用 Pareto plot 或 accuracy-at-budget，而非任意加权总分。

## 7. 统计分析

对相同 sample ID 的 baseline/candidate 形成 `delta_t`，`paired_bootstrap_ci` 对 delta 有放回重采样。CLI 默认 2,000 次、95% CI、seed 42；最终离线分析建议至少 10,000 次。五 seed 时使用 hierarchical/cluster bootstrap，而非把所有 episode 当独立样本。

二值 success 还应报告 exact paired McNemar，以只有 VERA 正确和只有 baseline 正确的 discordant count 构造小样本 exact binomial test。多组预注册比较应做 Holm correction。统计显著但 effect size 小、protected slice 有害或成本不可接受，仍不足以 promotion。

## 8. Promotion Gate 与独立审计

六重 gate 与 [algorithm_zh.md](algorithm_zh.md) 一致。Full BBH 当前值：

| 参数 | 值 |
| --- | ---: |
| Replay window | 16 |
| 最少验证样本 | 8 |
| Bootstrap | 2,000 |
| Confidence | 0.95 |
| 最小 mean gain | 0.03 |
| 最小 CI lower bound | 0.00 |
| 最大 regression rate | 0.10 |
| 最大 protected regression | 0.02 |
| 最大 cost increase | 0.30 |

Protected phase 是已见过的回归 probe，不是 held-out。Whole-state audit 应先运行 static held-out control，再用同一 held-out config 执行 evolved state audit。Candidate-level publication claim 仍需在 untouched dataset 上对每张 confirmed card 做 frozen counterfactual；该流程目前未完全自动化。

## 9. 消融矩阵

| ID | Variant | 隔离的问题 |
| --- | --- | --- |
| B0 | Static | 任何 adaptation 是否有益 |
| B1 | Self-Refine | 只增加同题推理是否足够 |
| B2 | Reflexion | 未验证 append memory 的效果 |
| V0 | Full VERA | 完整方法 |
| A1 | No slow policy evolution | policy mutation 的贡献 |
| A2 | Historical control | fresh paired replay 的贡献 |
| A3 | No protected replay | 对旧能力和遗忘的贡献 |
| A4 | No UCB | exploration 的贡献 |
| A5 | No utility | learned utility 的贡献 |
| A6 | No MMR | 去冗余的贡献 |
| A7 | BM25 only | 完整 hybrid retriever 的贡献 |
| A8 | Static trust | same-source temporal trust 的贡献 |
| A9 | No future audit | replay estimate 与 realized utility 差异 |
| A10 | Symmetric audit | harm exposure 与 false rollback 权衡 |
| A11 | No supersession | 冲突状态清理的贡献 |

Reflexion 同时缺少慢环，因此不是纯粹“去掉 verification”的严格消融。每行只改变命名组件，不能在 test stream 上分别调参。

## 10. LongMemEval 与后续公开基准

LongMemEval_S retrieval 是独立只读 benchmark：固定 cleaned 500-entry 文件，排除 30 个 abstention，评估 470 个 scored question 的官方 Recall-all/NDCG 和补充 MRR。

第一阶段复现 user-turn BM25。可选 `bm25_llm_rerank_fused` 在 top-20 匿名候选上使用固定 `0.4/0.6` BM25/LLM rank sum，tie 回到 BM25。LLM 不可见 source session ID、answer、answer session、question ID/type 或 BM25 score；未知/重复 ID、malformed response、provider/fusion error 精确 fallback BM25。

该 track 只评估 evidence retrieval；完整 memory-system claim 还需要同 reader 的 downstream QA。小 `max_per_type` screen 不能支持 leaderboard/SOTA 结论。LoCoMo、LongMemEval-V2、BFCL 等必须使用各自专用 adapter、license/revision manifest、官方 scorer 和同样标签防火墙，不能和 BBH 合并成一个总分。

## 11. 可复现检查表

每个正式 run 必须保留：

- `manifest.json`：commit/dirty、model、algorithm、seed、config/dataset hash；
- `resolved_config.yaml`：全部 provider/budget/evolution/benchmark 参数；
- `predictions.jsonl`：不可变首答；
- `promotion_decisions.jsonl`：paired delta 和全部 gate；
- `metrics.json`：前台指标；
- `costs.json`：外部调用 ledger；
- `summary.json`：最终 policy、active memory 和 audit note；
- `evoshift compare` 生成的配对 JSON/Markdown。

还要记录模型标识和评估日期。Closed API alias 后续变化时，新运行是 replication，不一定是相同 snapshot。

## 12. SOTA 红线

除非同时满足以下条件，否则不得使用 “SOTA”“state of the art”或“超过 leaderboard”：

- 数据 revision、官方 split、prompt/tool、scorer 和资源规则完全一致；
- VERA 在 official test 前冻结，或双方使用相同 online-adaptation protocol；
- base model 和 test-time compute 匹配，或差异作为比较核心；
- 报告全部预注册 seed/phase 和不确定性；
- 主测试 label/reference 未在首答前进入 adaptation；
- 独立 held-out 证明 replay memory 泛化；
- 报告 adaptation 总成本，不只 foreground solve；
- 明确模型/数据污染限制。

在公开 BBH 答案评分后继续适应，可以形成合法的 **prequential online-learning result**，但不是官方 static BBH leaderboard score。当前可辩护表述是：在明确声明的 EvoShift stream 上做 same-backbone 配对比较，并同时给出 CI、回归、成本和 held-out 证据。
