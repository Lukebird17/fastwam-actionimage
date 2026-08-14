# 交接给 Codex 的 Prompt — Action Images @ WAM Scale (ICLR 2027)

> 用法：把下面「═══ PROMPT 开始 ═══」到「═══ PROMPT 结束 ═══」之间的内容整体粘给 Codex。
> 它是自包含的，不需要我们的对话历史。

---

═══ PROMPT 开始 ═══

# 角色与任务

你是一名机器人学习 / 视觉-语言-动作（VLA）/ 世界动作模型（World Action Model, WAM）方向的研究员。
我将一个正在进行中的 ICLR 2027 投稿项目交接给你。你的任务是：

1. **构思 / 打磨创新点**：基于下面给出的现状与证据，判断现有创新点够不够、如何加强，必要时提出新的角度。
2. **设计具体实验**：给出可立即执行、优先级清晰、资源估算明确的实验清单。

请先完整读完下面的背景，再输出：(a) 你对创新点的判断与强化方案；(b) 一份排好优先级、带工程量与风险评估的实验计划。**不要写代码**，先出研究方案，等我确认后再动手。

---

## 一、项目一句话

把机器人的**本体感觉（proprioception）渲染成一张图像（action image），当作一帧条件图注入预训练视频扩散骨干（Wan2.2-TI2V-5B video DiT）**，而不是压成低维向量 token 喂给动作模块。目标是让机器人的动作状态也能吃到视频模型的视觉先验。

目标会议 **ICLR 2027**（California，2027-04；摘要截止预计 2026-09 中下旬，**时间紧**）。

## 二、站在哪些工作的肩膀上（必须先读懂）

- **Action Images v1**（arXiv 2604.06168，我们自己）：把 7-DoF 动作渲染成多视角 RGB Gaussian heatmap，
  把 policy learning 变成多视角视频生成。**v1 里 action image 是"生成目标"（输出端）**。v1 自认
  limitation："not yet been fully developed into a closed-loop policy"。
- **FastWAM**（arXiv 2603.16666，学长基座）：证明 WAM 推理时不需要真的生成未来视频，训练时视频
  协同训练即可；架构是 MoT = Wan2.2-TI2V-5B video DiT + ActionDiT（共享自注意力）。**它把 proprio
  当低维 token 处理，走不进视频骨干的视觉先验——这正是我们的切入点。**
- **本工作 = 把 Action Images 迁进 FastWAM**：action image 从 v1 的"生成目标"变成"**条件接口**"
  （输入端）。这是角色的根本反转，也是新内容所在。

## 三、核心创新点（当前定位，供你评判/强化）

- **C1【主打】Proprioception-as-image 作为条件接口**：渲染 proprio 成像素帧 pin 进视频 latent 序列。
  依据：(1) arXiv 2608.03052 综述枚举了 proprio 注入的 5 种接口，**无一是像素级**——我们补第 6 种；
  (2) 最近的 GeoProp(2607.07101）只在**特征空间**做 FiLM，不是像素条件帧、不进视频骨干。
  (3) 双向使用（v1 生成 ↔ v2 条件）→ 天然闭环 replan，回应 v1 的 limitation。
- **C2【已弱化】干净条件帧注入的实现关键**：真实消融后**只有"pin 位置"站得住**（RoPE 把帧索引编码进
  Q/K，条件帧必须 pin 在训练见过的位置，错位则 0.82→0.10）。t=0 时间标记、注意力隔离在本设定
  **不显著**（见下）。红线：t=0 机制已被 Diffusion Forcing(2407.01392）提出，**绝不抢首次**。
- **C3【支撑】双臂 16D / RoboTwin-50 / Aloha-Agilex 扩展**（v1 是 7-DoF 单臂 RLBench）。
  双臂视频生成已被 CRAFT(2604.03552）等占，只当扩展证据，不当头条。

**明确放弃（红海）**：跳过 test-time imagination（UVA 2503.00200 抢占+FastWAM 子领域爆炸）；
latent-action（DLAM/WALA/LaWAM）；不确定性/何时 replan（2026-08 刚被 GUARD/SAFECAST 占）；
触觉（已 6 篇）/音频（RoboTwin 仿真音频真实性是致命前提）。

## 四、已跑完的真实实验数据（务必基于这些判断，别凭空设计）

**公平性前提**：对照两臂拿到**完全相同的 16D proprio**（同数据管线、同 545 episodes、同归一化、同
文本缓存），唯一区别是实验臂把同一 proprio 额外渲染成图像当条件帧 → 干净的"像素 vs 向量"单变量对比。

| 证据 | 结果 | 统计 | 判定 |
|---|---|---|---|
| **C1 主对比** poker 100集 | 实验 0.85 vs 对照 0.74 | McNemar **p=0.035**(17 vs 6 discordant) | ✅ **显著**（50集时 p=0.21 不显著，加量才跨线） |
| C2-pin | 对齐 0.82 vs 错位 0.10(50集) | 悬殊 | ✅ 必要条件 |
| C2-t=0 | off 0.88 vs on 0.84(50集) | p=0.75 | ❌ 非必要 |
| C2-isolation | off 0.78 vs on 0.84(50集) | p=0.58 | ⚠️ 方向对、n=50 测不出 |

**当前最大软肋：主对比只在 poker 单任务上显著**，reviewer 会说"单任务侥幸"。正在补 2 个任务
（place_empty_cup、stack_blocks_two）重训对照+实验臂（每臂每任务 ~8h on 1×4 B200，训完测 100 集）。

## 五、代码与资源现状

- 投稿仓库：`fastwam-actionimage`(git remote: github.com/Lukebird17/fastwam-actionimage)。
  训练/评测入口 `scripts/`、`slurm/`；模型在 `src/fastwam/models/wan22/`；任务配置在 `configs/task/`。
- 设计文档与实验日志：`docs/iclr2027/`(PROJECT_DESIGN.md、ICLR2027_PAPER.md、EXPERIMENT_LOG.md、
  papers/ 17 篇参考 PDF + INDEX.md)。**先读这些再接管。**
- 集群：Slurm `b200-batch` 分区（B200 GPU)，单 ckpt≈92G（已加 save_total_limit=5 防磁盘爆）。
- eval:RoboTwin 闭环成功率，`eval_robotwin_poker.sbatch`，支持 `TASK_NAME`/`NUM_EPISODES` 参数化，
  同种子（4300000+）配对，统计用 McNemar（配对）/Fisher（不配对）。
- 环境：conda `curobo`(训练)/`test`（分析）;`actionimages` env 已坏勿用。

## 六、你需要输出的两块

**(a) 创新点评判与强化**：现有 C1 够不够 ICLR？C2 弱化后叙事怎么调？有没有我们能做、且没被占的
加强角度（提示：视角泛化、action-image 可解释性是 action-image 表示独有、纯 latent 方法做不了的
方向，调研确认偏蓝海；音频/触觉/不确定性已是红海，别碰）？

**(b) 实验计划**：排好优先级（P0 生死线 / P1 加分 / P2 可选），每个给出：假设、对照设计（保证单变量）、
所需 ckpt/数据/资源、工程量、预期如何写进论文。必须包含"主表扩到多任务显著"和"GeoProp baseline"
这两个 P0，其余由你判断补充。

约束：6 周窗口；优先复用现有 Wan2.2 + action-expert + RoboTwin/RLBench 代码栈；不碰红海；
任何新方向先论证"为什么空 + 为什么 6 周可行"再列入。

═══ PROMPT 结束 ═══
