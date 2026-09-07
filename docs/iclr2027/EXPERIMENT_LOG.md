# 实验日志 — Action Images @ WAM Scale (ICLR 2027)

> 每次实验追加一行到对应表。**任何人看这张表就能复现数字。**
> 固定字段：日期 / ckpt step / 数据集+seed / 成功率(n/N) / 统计 / 结论。
> eval 统一：RoboTwin `move_playingcard_away`、`demo_randomized`、同种子 4300005+、
> `ROBOTWIN_CAMERA_SHADER=default`、`SAPIEN_HEADLESS=1`。

## 原始主表：② vs ①（尚未匹配 token/计算量）

| 日期 | 臂 | ckpt | 任务 / n_ep | 成功率 | 统计 | 结论 |
|---|---|---|---|---|---|---|
| 2026-08-08 | ① scene_only | step_003660 | poker / 50 | 0.72 (36/50) | — | 控制组基线 |
| 2026-08-07 | ② action_image | step_003660 | poker / 50 | 0.84 (42/50) | vs ① McNemar p=0.21 | n=50 不显著 |
| 2026-08-09 | ① scene_only | step_003660 | poker / **100** | **0.74** (74/100) | — | job 328661 |
| 2026-08-09 | ② action_image | step_003660 | poker / **100** | **0.85** (85/100) | **vs ① McNemar p=0.0347** | job 328662，**显著 ✅** |

> **原始主表 100 集配对结果（2026-08-09，同 100 种子配对，McNemar exact）**：
> ②=0.85 vs ①=0.74，差 +0.11（11 个 episode）。仅②成功 c=17，仅①成功 b=6，
> discordant=23，**p=0.0347 < 0.05 显著**。这个检验说明两个原始系统确实不同，但②同时增加了
> action-image segment/token 和未来 action-image 辅助目标，因此不能把差异单独归因于像素条件接口。
> 下面的 token-matched 2×2 是正确的因果对照。

## Token-matched 2×2：Action Image 的输入与目标角色

四臂均使用 18 帧双 segment、相同数据窗口、effective batch、优化器和训练步数；只改变当前
Action Image 是否为空白，以及未来 Action Image 损失是否启用。

| 日期 | 当前 Action Image | 未来 Action Image 损失 | ckpt / n_ep | 成功率 | 作业 |
|---|---|---:|---|---:|---:|
| 2026-08-12 | 空白 | 0 | step_003660 / 100 | **0.86** | 349065 → 349399 |
| 2026-08-12 | 真实 | 0 | step_003660 / 100 | **0.85** | 349070 → 349400 |
| 2026-08-12 | 空白 | 0.05 | step_003660 / 100 | **0.89** | 349071 → 349398 |
| 2026-08-09 | 真实 | 0.05 | step_003660 / 100 | **0.85** | 已有主模型 |

> **结论（单训练 seed）**：真实 Action Image 输入没有收益（0.85 vs token-control 0.86）；
> input+target 也没有胜过 token-control。target-only 为 0.89，但相对 token-control 仅 +0.03，
> 100 episodes 下不足以构成稳定机制证据。当前结果不支持“像素条件接口利用视频预训练先验”的
> 原始 C1 表述。下一步必须在表面增益最大的 place_empty_cup 上重复 token-matched factorial；
> 若仍无因素稳定胜出，应停止 C1 叙事并转向受控诊断或相机诱导动作几何。

## 机制消融（C2：干净条件帧三条件）

| 日期 | 条件 | 设置 | 成功率(50集) | vs ②on 配对 McNemar | 结论 |
|---|---|---|---|---|---|
| 2026-08-06 | pin 位置 | 错位 (pre-fix) | 0.10 (5/50) | vs 对齐 0.82 | ✅ **pin 是必要条件**，错位致命 |
| 2026-08-06 | pin 位置 | 对齐 (post-fix) | 0.82 (41/50) | — | 参照 |
| 2026-08-10 | t=0 标记 off | `mark_clean_conditioning_frame:false` | 0.88 (44/50) | ②0.84 vs off 0.88，p=0.75 | ❌ **t=0 非必要**（off 略高但不显著） |
| 2026-08-10 | isolation off | `segment_first_frame_bidirectional` | 0.78 (39/50) | ②0.84 vs off 0.78，p=0.58 | ⚠️ **方向对但不显著**（off 降 6 点，n=50 测不出） |
| 待定 | 监督密度 λ | 0.05 / 1.0 | — | — | P2 |

> **C2 三条件配对小结（2026-08-10，均同 50 种子，②on=0.84 为参照）**：
> - **pin 位置（a）**：唯一被明确证实的必要条件（错位→0.10）。
> - **t=0 标记（b）**：off=0.88 ≈ on=0.84，p=0.75，**在本设定非必要**。C2 叙事需弱化此条。
> - **注意力隔离（c）**：off=0.78 低于 on=0.84（方向符合预期），但 p=0.58 不显著，n=50 测不出。
> **含义**：C2 不能写成"三个必要条件"，只有 pin 站得住。t=0/isolation 在 poker 单任务上
> 效果不显著。C2 不能作为独立创新点；C1 也必须以 token-matched 结果重新判断。

## ③ 全任务（scaling 支撑）

| 日期 | ckpt | 任务 | eval(50集) | 备注 |
|---|---|---|---|---|
| 2026-08-07 | step_017500 | poker | 0.44 (22/50) | 欠训(2.2%)，不可比 |
| 2026-08-11 | step_147500 | **poker** | **0.98** | 🔥 超单任务②(0.85)！多任务正迁移信号 |
| 2026-08-11 | step_147500 | stack_blocks_two | 0.08 | 崩，该任务对全任务模型极难 |
| 2026-08-11 | step_147500 | place_empty_cup | FAILED→待重测 | 文本缓存缺 prompt，已补(365917) |

> **观察（2026-08-11）**：③ 全任务在 poker 上为 0.98，高于单任务②的 0.85；但两者训练数据、
> 步数和任务组成均不同，而且没有 token-matched 多任务对照，因此不能归因于 Action Image 或
> “正迁移”。这里只保留为后续假设，不作为 C3 证据。新的 2×8 续训作业 367529 已于 2026-08-18
> 暂停，避免在论文机制未明确时继续消耗 16 GPU。

## 主表扩任务（C1 非单任务侥幸的证明）

| 任务 | ① scene_only | ② action_image | 状态 |
|---|---|---|---|
| poker (基准) | 0.74 | 0.85 (p=0.035) | 原始系统差异显著；factorial 未发现内容收益 |
| place_empty_cup | **0.35**（重测；首次 0.32） | **0.63**（两次均为 0.63） | ⚠️ 差异大，但尚未匹配 token |
| place_can_basket | 从 step_007500 恢复 (417084→417086) | 从 step_005000 恢复 (417085→417087) | ⏳ 维护结束后并行 |
| stack_blocks_two | step_007500(欠训) | **0.71** (step_012030) | ⚠️ 对照未完成，暂不作因果结论 |

## 待办（按 PROJECT_DESIGN §3 优先级）

- [x] P0: ②vs① 上 100 episode（原始系统 0.85 vs 0.74, p=0.035；存在 token/目标混杂）
- [x] P0: Poker token-matched 2×2 —— 0.86 / 0.85 / 0.89 / 0.85，无明确输入或耦合收益
- [~] P0: Place token-matched 2×2 —— 三个新臂训练 417097/417098/417099，评测 417100/417101/417102
- [~] P0: 主表扩任务 —— place_empty_cup 已完成原始对比；place_can_basket 恢复训练已排队
- [ ] P1: GeoProp baseline 训练臂（①已含同源 proprio，主对比公平，降为 P1）
- [x] P0: t=0 消融（off 0.88 ≈ on 0.84, 非必要）
- [x] P0: isolation 消融（off 0.78 vs on 0.84, p=0.58；100集 0.83 ≈ on, 非关键）
- [ ] P1: ③ place_empty_cup 重测（缓存已补）
- [ ] P1: attention map 可解释性图
- [ ] P2: 视角泛化
