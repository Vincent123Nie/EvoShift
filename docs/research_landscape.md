# Research landscape and project positioning

This note records the public projects and datasets checked on **2026-08-04**.
It is a positioning document, not a leaderboard comparison. Repository
descriptions and paper claims below belong to their authors; EvoShift has not
reproduced those published numbers.

## Closest self-evolution systems

| Work | Main evolving object | Resource profile | What EvoShift borrows | Important difference |
|---|---|---|---|---|
| [Reflexion](https://github.com/noahshinn/reflexion) ([paper](https://arxiv.org/abs/2303.11366)) | Verbal reflection from execution feedback | API inference; task-specific retry loops | Natural-language experience can improve a frozen model | EvoShift adds typed state, shadow staging, paired promotion gates, protected replay, versioning, and rollback |
| [ExpeL](https://github.com/LeapLabTHU/ExpeL) | Cross-task experiences and distilled natural-language insights | Offline experience gathering over HotpotQA, ALFWorld, WebShop, and FEVER | Experiences should transfer across tasks rather than remain one-episode reflections | EvoShift is prequential and shift-aware; it verifies every state transition instead of accepting an offline insight bank as fixed |
| [Voyager](https://github.com/MineDojo/Voyager) ([paper](https://arxiv.org/abs/2305.16291)) | Executable skill library plus automatic curriculum | API inference plus a Minecraft environment and generated code execution | Lifelong agents need compositional skills, feedback, and self-verification | EvoShift never executes model-generated code and targets generic scored streams rather than one embodied environment |
| [A-MEM](https://github.com/agiresearch/a-mem) ([paper](https://arxiv.org/abs/2502.12110)) | Dynamically linked and refined memory notes inspired by Zettelkasten | LLM calls, embeddings, and a vector store | Memory organization and links are learned state, not static RAG plumbing | A-MEM emphasizes memory topology; EvoShift emphasizes causal utility, promotion evidence, non-stationarity, and rollback |
| [GEPA](https://github.com/gepa-ai/gepa) | Prompts, code, configurations, and other textual parameters | Evaluation-driven reflective evolutionary search; no gradients required | Full traces can guide targeted textual mutation, and optimization is multi-objective | GEPA is primarily an optimizer over an evaluation set; EvoShift is an online two-time-scale state machine with bounded mutations and no test-time code execution |
| [ACE](https://github.com/ace-agent/ace) ([paper](https://arxiv.org/abs/2510.04618)) | Structured context/playbook through generator, reflector, curator, and incremental deltas | API-only offline or online context adaptation | Incremental updates reduce context collapse and preserve useful detail | ACE is the closest context-evolution reference. EvoShift's differentiator is explicit drift triggering, replay-based statistical admission, protected regression, posterior rollback, and a cryptographically fingerprinted frozen audit |
| [AgentEvolver](https://github.com/modelscope/AgentEvolver) ([paper](https://arxiv.org/abs/2511.10395)) | Tasks, exploration experience, credit assignment, and model policy | End-to-end training framework with CUDA/model training | Self-questioning, experience-guided navigation, and attribution are useful decomposition axes | Its core contribution includes policy training; EvoShift deliberately studies what can be achieved with a black-box API and no GPU |
| [MemSkill](https://github.com/ViktorAxelsen/MemSkill) ([paper](https://arxiv.org/abs/2602.02474)) | Meta-memory skills describing what, how, where, and when to remember or forget | Data-driven skill evolution; released controllers and training/evaluation stack | The memory operator itself can evolve, not only remembered content | EvoShift currently evolves experience content and bounded retrieval/write hyperparameters; learnable span-level memory operators are a high-value future extension |

The repository's Static, Self-Refine, and Reflexion modes are **conceptual
controls**, not copied official implementations. They share EvoShift's solver,
provider, scorer, stream, and accounting so that one algorithmic behavior can be
changed at a time.

## The defensible novelty claim

EvoShift should not claim to invent self-reflection, experience memory, textual
evolution, or agentic memory. Its defensible contribution is the integration of
five ideas that are often evaluated separately:

1. a prequential `predict -> score -> evolve` protocol with a label firewall;
2. fast experience evolution and slower shift-triggered policy evolution;
3. typed shadow state and same-sample champion/challenger replay;
4. joint gain, uncertainty, regression, protected-slice, and cost gates;
5. online rollback plus a same-model, different-stream, mutation-free audit
   whose state identity is verified before and after evaluation.

A suitable resume description is therefore “verified test-time self-evolution
under distribution shift,” not “first self-evolving Agent” or “SOTA memory
system.”

## Public datasets and why BBH is the primary benchmark

| Dataset | What it tests | Strength for this project | Limitation / work required |
|---|---|---|---|
| [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) | Diverse difficult reasoning task families with deterministic targets | Public, lightweight, easy to phase by domain, cheap to score, and suitable for controlled distribution shifts and held-out families | It is not a long-conversation memory benchmark and may be present in model pretraining |
| [LongMemEval-V2](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2) and [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | Long-term memory for web, enterprise, and chat assistants | Directly measures retention, temporal reasoning, and cross-session retrieval; EvoShift now has a pinned LongMemEval_S session-retrieval adapter | Downstream reader QA, temporal grading, and LongMemEval-V2 remain unimplemented; long contexts materially increase API cost |
| [LoCoMo](https://github.com/snap-research/locomo) ([paper](https://arxiv.org/abs/2402.17753)) | Very long multi-session conversational memory and QA | Strong memory-specific qualitative error analysis | Only ten source conversations; split construction and leakage control need care, and the generic row adapter is insufficient |
| [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) / [HF data](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | Single/multi-turn function calling, parallel calls, relevance, and tool use | Industrially recognizable, structured execution-based scoring, and natural tool-schema shifts | Requires a first-class tool-call representation and official evaluator rather than treating calls as free-form text |
| [tau-bench](https://github.com/sierra-research/tau-bench) and [tau2-bench](https://github.com/sierra-research/tau2-bench) | Tool-Agent-User interaction in realistic domains | Best industrial extension for policy adherence, stateful tools, and long-horizon recovery | Environment setup, user simulation, database state, and pass^k evaluation would lengthen the first project cycle substantially |

BBH is used first because the research variable is the **adaptation mechanism**,
not environment engineering. It gives exact labels, ordered task-family shifts,
protected old domains, and untouched held-out domains while remaining runnable
with API credits and local CPU only.

## Recommended extension order

1. **LongMemEval-V2**: extend the implemented LongMemEval_S retrieval adapter
   with temporal reader QA and same-reader transfer scoring.
2. **BFCL**: evolve tool-selection and argument-construction memories with the
   official executable evaluator; this is the strongest short industrial
   extension.
3. **tau2-bench**: evaluate policy and tool-use adaptation in a real stateful
   environment after the core statistical protocol is stable.

LoCoMo is valuable for qualitative memory analysis, but its small number of
source conversations makes uncertainty and leakage handling more delicate than
the three extensions above.

## Interview comparison matrix

| Question | EvoShift answer |
|---|---|
| Why not Reflexion? | Append-only reflection has no admission control, regression protection, or utility rollback. |
| Why not ACE? | ACE is the closest context-evolution baseline; EvoShift makes drift, statistical promotion, protected safety, version rollback, and frozen transfer audit first-class. |
| Why not GEPA? | GEPA searches textual artifacts against an evaluation set; EvoShift controls online state transitions in a non-stationary stream. |
| Why not A-MEM? | A-MEM learns memory structure and links; EvoShift asks whether a memory or retrieval-policy change has measurable causal value before activation. |
| Why not AgentEvolver or SkillRL-style training? | The project constraint is black-box API access and no training compute; weight updates are explicitly out of scope. |
| Why not claim SOTA? | The public-source PolicyShift table is a short historical dirty-worktree engineering freeze, LongMemEval is retrieval-only, and there is no official baseline reproduction, per-candidate held-out reader audit, or leaderboard-compatible protocol. |

## Claim boundary

The following statements are currently supportable:

- the provider protocol, deterministic pipeline, benchmark integrity checks,
  frozen audit invariants, and quality gates are tested;
- EvoShift can run same-model prequential and held-out experiments with bounded
  API budgets;
- a deterministic synthetic demo validates orchestration only.

The following statements require future artifact-backed experiments:

- EvoShift improves BBH over the strongest same-model baseline;
- its replay promotions generalize per candidate;
- it reduces forgetting on long-term memory benchmarks;
- it matches or exceeds published ACE, GEPA, A-MEM, MemSkill, or AgentEvolver
  results.
