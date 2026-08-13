# VERA 算法

英文原文：[algorithm.md](algorithm.md)

VERA 全称 **Verified Experience Replay and Adaptation**，是 EvoShift 的双时间尺度测试时学习算法。基础语言模型保持冻结；系统只演化外部程序性记忆和受明确约束的检索/写入策略。

核心假设是可证伪的：

> 在非平稳任务流中，经过验证后才提升的经验，以及由漂移触发的受限策略变异，应当比静态提示或未经验证的记忆更快恢复变化后的性能，同时不对受保护旧领域和资源成本造成不可接受的回归。

当前实现尚未证明公开基准上的算法优势。PolicyShift 真实模型矩阵是已记录的负结果；LongMemEval 只验证了检索机制；合成数据只用于生命周期和机制测试。正式优势结论必须具备同模型基线、配对 seed、held-out 评估、不确定性和完整成本。

## 记号

在 episode `t`：

- `x_t`：任务和元数据；
- `y_t`：环境结果或隐藏参考；
- `r_t in [0,1]`：首答得分；
- `M_t`：当前激活的版本化记忆集合；
- `pi_t`：当前 `PolicyGenome`；
- `R(x_t, M_t, pi_t)`：记忆检索函数；
- `f_theta`：冻结的 API 模型；
- `c_t`：token 或美元成本。

首次预测为：

```text
y_hat_t = f_theta(system_prompt, x_t, R(x_t, M_t, pi_t))
```

只有 `y_hat_t` 被提交并评分后，当前反馈才可影响 `M_(t+1)` 或 `pi_(t+1)`。决策时信任状态只能读取 episode `< t` 已提交的反馈，不能读取本轮 `feedback_reference`。

## 端到端循环

```text
for sample in stream:
    pre_state = feedback_trust.pre_predict(sample)
    memories = retrieve(sample.prompt, active_or_probation_memory, champion_policy)
    prediction = solve(sample, memories)

    oracle_score = hidden_evaluator(sample.reference, prediction.answer)
    feedback_score = observable_evaluator(sample.feedback, prediction.answer)
    trust = feedback_trust.observe_feedback(sample)

    if trust >= memory_update_threshold:
        update_memory_posteriors(prediction.applied_memory_ids, feedback_score)
    if trust >= drift_threshold:
        shift = detector[sample.domain].update(feedback_score, retrieval_novelty)

    if trust >= candidate_threshold and eligible_failure:
        candidate = critic(sample, prediction, feedback_score, shift)
        evidence = candidate_pool.observe(candidate)
        if candidate_pool.ready(evidence):
            shadow = stage(candidate)
            decision = paired_replay(shadow, current_regime, protected_history)
            probation_or_activate_or_reject(decision)

    for probation_card used on a later trusted interaction:
        control = solve_with_exact_card_excluded(sample)
        update_future_audit(feedback_score, control.feedback_score)

    if shift.detected and recent_failures and not fast_loop_resolved:
        patch = bounded_policy_mutation(champion_policy, recent_failures)
        decision = paired_replay(patch, current_regime, protected_history)
        accept_or_reject_policy(decision)
```

Static、Self-Refine、Reflexion-style、Replay-only 和 Full EvoShift 共用同一个 runner，只切换相应能力。这样 scorer、数据顺序、provider、artifact 和成本口径保持一致。

## 检索：领域约束 BM25 + Beta 效用 + UCB + MMR

每张记忆由 trigger、scope、tags、directive 和 anti-pattern 拼成可检索文档。分词器支持小写字母数字 token 和 CJK bigram。Critic 产生记忆时，runner 还会写入确定性的 `source_domains` provenance；该字段不是 LLM 自报。

默认情况下，有 provenance domain 的记忆只有在当前 domain 匹配时才有资格进入排序。`allow_cross_domain_transfer=true` 仅用于预注册消融，防止高效用但无关的卡片成为通用提示词污染。

BM25：

```text
IDF(q) = ln(1 + (N - n_q + 0.5) / (n_q + 0.5))

BM25(d, q) = sum_q IDF(q) * f(q,d) * (k1 + 1)
              / (f(q,d) + k1 * (1 - b + b * |d| / avgdl))

relevance_i = raw_i / max_j(raw_j)
```

每张记忆维护初始化为 `alpha=1, beta=1` 的 Beta-Bernoulli 成功后验：

```text
utility_i = alpha_i / (alpha_i + beta_i)
```

探索项为 UCB 风格奖励：

```text
exploration_i = min(
    1,
    sqrt(2 * ln(total_memory_uses + 2) / (use_count_i + 1)) / 2
)
```

多样化之前的总分：

```text
score_i = w_r * relevance_i
        + w_u * utility_i
        + w_e * exploration_i
```

准确说，这是 **Beta 后验均值 + UCB 风格探索项**，不是 Beta 分位数，也不是 Thompson Sampling。最后使用 MMR 选择最多 `top_k` 张记忆：

```text
MMR(i) = lambda * score_i
       - (1 - lambda) * max_{j in S} Jaccard(document_i, document_j)
```

渲染后的记忆上下文还受 token budget 约束。

### 检索新颖度

当已有记忆库时：

```text
novelty_t = 1 - max_i relevance_i
```

若存在记忆但没有任何 eligible memory，则 novelty 为 `1.0`，使跨域输入能被漂移检测器感知。只有空记忆库冷启动时才抑制该信号。

## 可观测反馈信任

静态模式只读取 `metadata.feedback_source`。动态模式额外读取应用提供的 context key 和学习器可见的 `feedback_reference`，绝不读取隐藏 oracle、`feedback_corrupted`、policy version、phase 或攻击标记。

来源 `s` 的 Beta 初始状态由先验 `p_s` 和强度 `k` 给出：

```text
alpha_s = 1 + p_s * k
beta_s  = 1 + (1 - p_s) * k
trust_s = alpha_s / (alpha_s + beta_s)
```

每个 `(source, context)` 保存 committed label 和 pending contradiction：

- 与 committed 一致：加强来源可靠性；
- 第一次矛盾：低信任隔离，不立即改规则；
- 相同矛盾重复达到阈值：确认 context change，并提交新 label；
- 旧 label 先恢复：把 pending burst 视为短暂噪声。

漂移、记忆效用更新、候选生成和 replay 使用四个独立信任阈值。这是时序一致性模型，不是语义真值验证；持续控制多数反馈的攻击者仍可能占据 committed 状态。

## 在线漂移检测

VERA 对有界 loss 使用 Page-Hinkley，并加入检索 novelty 的 EWMA：

```text
loss_t = 1 - clip(reward_t, 0, 1)
mean_t = mean_(t-1) + (loss_t - mean_(t-1)) / t
cum_t  = cum_(t-1) + loss_t - mean_t - delta
min_t  = min(min_(t-1), cum_t)
PH_t   = cum_t - min_t
```

当 `t >= min_instances`、不在 cooldown 且 `PH_t > threshold` 时报警。Novelty 使用：

```text
novelty_ewma_t = a * novelty_t + (1 - a) * novelty_ewma_(t-1)
```

每个 domain 有独立检测器；报警后重置累计量并进入 cooldown。Page-Hinkley 的优点是每轮 `O(1)`、可解释、可确定性测试；缺点是可能把困难样本簇误判为漂移，也可能漏掉条件漂移或缓慢漂移。

## 失败归因与经验蒸馏

可信失败后，Critic 只能看到受限结构化信息：任务和 domain、Agent 答案/置信度/短理由、学习器可见反馈、少量当前记忆摘要和 shift report。它不能看到 benchmark phase 或隐藏 oracle。

输出必须是受 Pydantic 限制的 JSON，包含 failure type、signature、evidence、confidence 和程序性记忆字段。失败类型覆盖写入缺失、检索缺失、排序错误、冲突错误、推理错误、格式错误、知识缺失、工具错误和 unknown。

反馈模式：

- `reward_only`：只看成功/失败，是主实验默认；
- `grader_feedback`：只在环境确实提供 grader 文本时公平；
- `reference_upper_bound`：oracle 上界，不能混入公平在线结果。

Critic 可由 LLM 生成候选，但 promotion 不由 Critic 自己决定。非法 JSON 会降级为低置信度通用候选。

## 候选证据、暂存与去重

低于 `write_confidence_threshold` 的候选立即拒绝。`CandidateEvidencePool` 按规范化 typed signature 聚合同类候选，控制被拒候选的冷却和重试，并抑制已有 active signature 的重复付费验证。

条件 shadow e-process 可按 `(domain, feedback_source, feedback_context)` 家族更新序列证据：同家族同信号为正证据，同家族冲突信号为负证据，不同家族直接跳过。它只使用学习器可见字段。

通过写入门的候选和已有记忆比较 token Jaccard：高于 `dedup_similarity_threshold` 时复用 `memory_id`、增加 version、合并 provenance；否则使用稳定 SHA-256 ID。所有新候选先进入 `SHADOW`，正常检索只看 `ACTIVE` 和可选的 `PROBATION`。

## 配对 Replay

Replay buffer 同时照顾新分布和旧能力：

- 约三分之二为当前检测 regime 的普通样本，按候选相关性和新近程度选择；
- 最多约三分之一为 protected phase 或显式 protected examples，可跨 regime；
- 剩余槽位用近期 episode 填充。

记忆验证时，control 使用当前 active memory，challenger 强制加入 shadow candidate。启用 `paired_replay` 时两边都重新求解。策略验证始终重新运行 champion 和 challenger。

```text
delta_i = score(challenger_i) - score(control_i)
```

同样本配对降低任务难度方差，但不能消除 API 随机性、顺序效应和时间相关性。没有美元定价时，资源门使用 token，而不是把调用当作免费。

## 配对 Bootstrap 与提升门

系统对配对 delta 向量有放回重采样，计算均值的 percentile CI。必须先形成配对 delta 再 bootstrap，不能分别采样两组结果。

候选只有同时通过六个门才提升：

1. 样本数不少于 `min_validation_examples`；
2. 平均配对增益不少于 `min_mean_gain`；
3. bootstrap 下界不少于 `min_ci_lower_bound`；
4. 负 delta 比例不超过 `max_regression_rate`；
5. protected slice 平均回归不超过 `max_protected_slice_regression`；
6. 相对成本增幅不超过 `max_cost_increase_ratio`。

系统保存每个检查、均值、区间、逐样本 delta、回归率和成本比。平均分为正并不足以 promotion。

当前门的限制是小窗口低统计功效、样本可能自相关、反复测试带来多重比较、replay 不是 held-out、replay gain 可能高估未来效用。Block bootstrap、置信序列、FDR 控制和独立 promotion/audit split 都是合理扩展。

## Probation 与未来反事实审计

Replay 通过的记忆默认进入 `PROBATION` 而非直接 `ACTIVE`。当它之后在相关真实 episode 中被使用时，系统已有 memory-on 的正常答案，再额外运行一次只排除该精确卡片的 control：

```text
future_delta_j = visible_feedback(candidate_on_j)
               - visible_feedback(candidate_off_j)
```

在线决策只使用可信的学习器可见 delta；隐藏 oracle delta 只在决定完成后附加，用于事后评估。

停止规则非对称：

- 达到早期伤害计数且累计均值为负，可立即回滚；
- 正向确认必须达到正常最小证据并通过全部门；
- 证据不足时保持 pending，直到最大观察数；
- 流先结束则标为 `expired`，不伪装成 rollback。

该规则限制了伤害暴露，但不是 anytime-valid 推断；反复查看普通 bootstrap CI 会膨胀误差。

## Active 记忆的持续因果治理

Probation audit 只回答“新卡是否应激活”，不能回答“旧卡在政策撤销后是否仍有效”。当一张 confirmed active card 被明确列入 `applied_memory_ids`、本轮出现可信失败且预算允许时，系统运行精确版本 leave-one-out：

```text
active_delta_t = visible_feedback(memory_on_t)
               - visible_feedback(memory_off_t)
```

只排除指定 `(memory_id, version)`，其余可观测状态固定。`MemoryItem` 持久化审计次数、正/负/中性观察、delta sum 和最后审计 index。普通退休要求两次负证据且均值低于阈值；更严格的 fast path 只有在同轮漂移报警且 delta 足够负时才允许一次证据退休。

Probation memory 不进入 active audit，避免 admission audit 与 active governance 竞争。隐藏 valid/stale 标签和 oracle 只用于事后指标。

## 慢环策略变异

当前 proposer 是确定性的，不让 LLM 生成任意代码：

- 写入/检索缺失：提高 `top_k` 和 relevance 权重；
- 排序错误：提高 utility 权重和 MMR lambda；
- 冲突错误：降低 `top_k` 和 dedup threshold；
- 其他情况：按 novelty 小幅调整 exploration 权重。

所有值由 `PolicyGenome` schema 截断；patch 记录 base version、hypothesis、supporting failure IDs 和 fingerprint。修改非白名单字段或应用 stale patch 会失败。

若同一 episode 的快环已经成功提升一张记忆，默认抑制慢环，避免为已局部解决的问题再付全局 replay 成本。

## 在线效用、冲突和回滚

Solver 返回的 `applied_memory_ids` 必须属于实际 retrieved allowlist；若没有上报，runner 保守地给 retrieved set 记账。

```text
success: alpha <- alpha + 1
failure: beta  <- beta + 1
utility <- alpha / (alpha + beta)
```

至少使用 `rollback_min_uses` 次后，效用低于 `rollback_utility_threshold` 的 active memory 会退休。回滚只作用于有害卡片，不回滚整个 run。

Confirmed candidate 可以声明 `supersedes_memory_ids`。只有确认后旧规则才进入 `SUPERSEDED`，保留 provenance 但退出检索。默认不自动复活 predecessor，因为多阶段回退中 predecessor 也可能过时；旧规则复现时应作为新版本重新经过 replay 和 probation。

## 指标和结论口径

报告覆盖：整体/分阶段得分、AUAC、regret、恢复步数、post-shift gain、遗忘、replay 与 realized promotion precision、audit coverage、信任隔离、future audit 结果、冲突替代、harm exposure、stale retention、选择性遗忘 precision/recall、漂移事件、token/成本和延迟。

三个关键口径：

1. `replay_estimated_promotion_precision` 偏乐观；`realized_promotion_precision` 必须和 `realized_promotion_coverage` 一起看。
2. 流内 future audit 比 replay 更接近因果，但仍不等于 untouched public held-out。
3. episode usage 只覆盖前台解题；`costs.json` 的 BudgetLedger 才覆盖 critic、replay 等全部外部调用。

## 冻结 Held-out 状态审计

`evoshift audit` 加载完成的 prequential run，解析最终 active memories 和 policy，并对规范化 JSON 计算 SHA-256。目标 audit 强制：

```text
target_model == source_model
target_dataset_hash != source_dataset_hash
critic/write/utility/policy mutation/promotion/rollback == disabled
final_state_hash == source_state_hash
```

任何记忆或策略字段变化都会使 audit 失败。它测量完整演化状态的 forward transfer，不自动识别单张记忆贡献；后者由更昂贵的 `audit-memories` leave-one-out 完成。

## 复杂度与成本

设 active memory 数为 `N`、replay window 为 `W`、bootstrap 次数为 `B`：

- 普通 episode：1 次 solver API 调用；
- eligible failure：额外 1 次 Critic；
- memory verification：最多 `2W` 次 solver；
- policy verification：`2W` 次 solver；
- bootstrap：`O(BW)` 本地 CPU；
- memory state：`O(N)` 加 SQLite 中保留的历史版本。

主要成本来自 replay API inference，而不是本地统计。关键控制项是 validation window、候选调度、shift cooldown、cache、总请求/token/美元预算和并发。

## 必要消融

至少比较：

- 无记忆、静态记忆、Self-Refine、未经验证 Reflexion、完整 VERA；
- 去掉 drift gate 或 policy evolution；
- 只有 relevance 与 relevance + utility + exploration；
- 去掉 MMR、protected replay 或 freshly paired control；
- 不同 replay window 和 promotion threshold；
- static trust 与 dynamic same-source trust；
- replay-only 与 probation/future audit；
- 对称与非对称 future stopping；
- 关闭 conflict supersession 或 active causal audit。

所有超参数必须在 calibration split 或更早的 stream 上选定，并在最终测试前冻结。看过测试结果后再调参，会使置信区间失去解释力。
