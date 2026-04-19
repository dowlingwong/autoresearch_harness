### 3.2 参照物：InterDeepResearch（arxiv 2603.12608）

- **作者**：Bo Pan, Lunke Pan, Yitao Zhou, Qi Jiang, Zhen Wen, Minfeng Zhu, Wei Chen（2026-03-13）
- **问题诊断**（formative study）：现有 deep research agent 有三个痛点——
  1. **process observability**（看不到 agent 在想什么）
  2. **real-time steerability**（中途插不进手）
  3. **context navigation**（追溯不方便）
- **方法**：research context management framework，分三层（Information / Action / Session），支持 dynamic context reduction 和 cross-action backtracking。
- **界面**：三个协调视图 + 专用交互控件。
- **评估**：Xbench-DeepSearch-v1 / Seal-0 benchmark + **正式 user study**。
- **场景**：deep research = **信息检索 / 文献综述类**任务。

**关键观察：InterDeepResearch 的整个叙事是 HCI 风格的**（formative study → design goals → system → user study）。这套模版在 NeurIPS workshop、CHI、UIST、IUI 都认。要效仿的就是这个写法模版。

### 3.3 你 vs InterDeepResearch — 差异化空间很大

| 维度 | InterDeepResearch | 你（交互式审计路线） |
|---|---|---|
| 任务类型 | **信息检索 / 综述**（读网页） | **实验驱动的科研**（改代码、跑训练、看指标） |
| Agent 行动空间 | web search、read、summarize | **修改代码 + 执行训练 + 读指标**，副作用更重 |
| "审计"的语义 | evidence provenance（某句话来自哪个网页） | **因果归因**（某次 AUC 变化来自哪个 diff / 哪次 keep 决策） |
| 风险 | 错误信息 | 数据泄漏、代码破坏、实验历史污染 |
| 现成基础设施 | 他们做的就是 framework 本身 | 你已经有 control plane + append-only JSONL + commit hash，**底层记录比他们完整** |

一句话：**他们做"信息型 deep research"的人机协作，你可以做"实验型 deep research（autonomous ML research / AI4Science）"的人机协作。这是一个干净的未占领生态位。**

你已经有的东西刚好就是这条路线**最难的那一半**：一套结构化、可审计、可回放的实验记忆（JSONL + git commit + keep/discard + rationale）。InterDeepResearch 类的信息型系统没有这个，因为他们的 action 没有外部可验证 ground truth，你有（AUC、loss、训练曲线）。

### 3.4 三个可选故事角度（按推荐度排）

#### A. Interactive Provenance for Autonomous ML Research（最推荐）

- **Claim**：现有 autonomous ML agent（AI Scientist、MLAgentBench、AIDE）追求 end-to-end 自主，但研究员没法信任、没法干预、没法复用。我们提出"可交互审计"作为一种新范式。
- **Core contribution**：
  1. 一个"实验版"的 context 分层（对标 InterDeepResearch 三层）：**Decision Level（keep/discard + rationale）/ Run Level（commit + metrics + diff）/ Campaign Level（跨轮策略演化）**。
  2. 一套交互机制：**点任意 AUC 数据点 → 看 diff + rationale + memory state → 可 fork 一个 what-if 分支重跑**。
  3. "洞察卡片"：让研究员标记哪些 manager rationale 是真洞察、哪些是 agent 的幻觉，这些标注反过来塞回 memory 影响后续轮次（**human feedback 变成结构化 memory**，而不是 RLHF 那种隐式标注）。
- **相对于 InterDeepResearch 的 delta**：他们追溯一句话的来源；你追溯一次性能提升的因果链，而且你能让用户 fork & replay（信息检索里做不到 replay）。

#### B. Governed Human-Agent Co-Pilot for Experimental Science

- **Claim**：把原来的 control plane 重新包装成"人和 agent 共享的实验协议"。人和 agent 都通过同一套 `/run /keep /discard /memory` API 操作，审计轨迹同构。
- 好处：直接踩 **trustworthy / safe agent** 的赛道，正好匹配 NeurIPS 常设的 SafeAI / Agentic Safety workshop。
- 缺点：故事稍偏工程，需要补一个"无 control plane 就出事"的失败对照。

#### C. Emergent Explore-Exploit as an Interpretability Handle

- **Claim**：把那个 Round1 广度搜索 → Round2 架构尝试 → Round3 精调的"涌现行为"，变成**可视化/可干预**的对象。研究员可以
