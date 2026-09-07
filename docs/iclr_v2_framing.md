# Action Images @ WAM Scale — ICLR 投稿创新点 framing

> 内部讨论文档，2026-08-09 整理。目标会议：ICLR 2027（投稿截止约 2026-09 下旬）。
> 定位：**新内容**（v1 的延伸，非合并重投）。

## 0. 背景对齐

### v1（arXiv 2604.06168, Action Images, Haoyu Zhen 等）

- 核心声称：7-DoF 动作 → 3 个语义 3D 点 → 多视角 RGB Gaussian heatmap（gripper 编进蓝色通道），
  把 policy learning 变成多视角视频生成；**video backbone 本身就是 zero-shot policy，不需要 action module**；
  同一模型支持 video-action 联合生成 / action-conditioned video / action labeling。
- 评测：RLBench + 真机，单臂。
- **自认 limitation（原文）："not yet been fully developed into a closed-loop policy"**。

### FastWAM（arXiv 2603.16666, Yuan et al., Hang Zhao 组）

- 核心问题："Do WAMs need test-time future imagination?" 答案：不需要——**training-time
  video co-training 才是关键**；推理时只跑一遍 video DiT 当 world encoder，action expert 直接出动作。
- 架构：MoT = Wan2.2-TI2V-5B video DiT + ActionDiT（共享自注意力）。
- 与我们的关系：它是**架构/效率**工作，我们是**表示**工作；它留下了"proprio 以低维 token 进
  action expert，视觉先验用不上"的模态鸿沟——这是我们的切入点。

### 当前工作（本仓库）

- 把 Action Images 迁移进 FastWAM：双臂 16D（Aloha-Agilex），RoboTwin 50 任务；
  当前 proprio 渲染的 action image 作为条件帧 pin 进视频 latent 序列
  （`segment_first_frame_causal`，per-token t=0，causal isolation）。
- 对照实验（同一数据管线 `include_action_video` 单开关，action image 是唯一变量）：
  ① 纯 FastWAM vs ② +ActionImages vs ③ 全任务。

## 1. 核心问题（推荐的第一卖点）

> **"How should a World Action Model perceive the robot's own action state?"**
> （WAM 应该如何感知机器人自己的动作状态？）

FastWAM regime 下视频生成退到训练期，action image 的角色从 v1 的**生成目标**转为**条件接口**：

- **Proprioception-as-image**：把当前 proprio 渲染成像素空间的 action image，
  pin 进视频 latent 序列做条件。低维 proprio token 走不进视频 backbone 的视觉先验，
  渲染成图像就可以——模态鸿沟被表示层抹平。
- **同一表示，双向使用**：v1 证明 action 可以"被生成"（输出端）；
  v2 证明 action 可以"被渲染回来做条件"（输入端）。这是闭环 replan 的天然机制
  （每 32 步重渲染当前 proprio → 新条件帧 → 出下一 chunk）。
- **直接回应 v1 自认的 limitation**（not closed-loop）。

措辞红线：**不要再说 "zero-shot policy without action module"**（那是 v1 的 claim，
本版有 action expert，硬说会自相矛盾）。说"action image 作为 action expert 与视频 backbone
之间的像素级接口"。

## 2. 第二创新点：条件注入机制（把踩坑变成科学）

向视频 DiT 注入干净条件帧的三个必要条件——每条对应一个便宜 ablation：

| 机制 | 发现过程 | ablation |
|---|---|---|
| **Pin 位置** | `latent_t//2` 只在两段等长时碰巧对；通式 = "每个 segment 自己的第 0 帧"。双臂布局 `[scene×3 \| action×1]` 不等长，`//2` 全是 bug | pin 错位 vs 对齐（RoPE pin fix 的 pre/post 数据已有） |
| **干净帧时间标记** | `fuse_vae_embedding_in_latents` + per-token t=0（显式告诉模型"这帧是干净的"）vs 干净值顶着噪声级 t（σ→1 时调制信号自相矛盾） | t=0 标记 on/off |
| **注意力隔离** | conditioning 帧 causal isolation（不被未来噪声污染）vs 全双向（pin 帧 hidden state 混入 48 帧噪声） | isolation on/off |
| （附）监督密度 | 3 帧 λ=0.05 vs 12 帧 λ=1.0 | λ 扫描 |

包装：**"向 video DiT 注入 clean conditioning 的三个必要条件"**——结论超出机器人领域，
对任何 video diffusion 条件注入成立，拉高通用性分。

## 3. 第三创新点：双臂多任务扩展 + 严格对照

- v1：7-DoF 单臂 RLBench；v2：**16D 双臂、RoboTwin 50 任务**。
- ① vs ② 共享同一数据管线（同 545 episodes / 46752 windows / 16D 动作空间 / 归一化统计 / 文本缓存），
  **action image 是唯一变量**——比 v1 更强的因果证据。

## 4. Related work 划清界限

| 工作 | 区别 |
|---|---|
| **MV-VDP**（2604.03181） | 它把多视角 heatmap 当**生成目标**再解码成动作；我们是**条件接口**（v1 已有生成侧）。必须主动划清，别等 reviewer 抓 |
| **DAWN**（2509.22652） | pixel motion 当中间表示的层级式框架，不冲突，正常引用 |
| **UniPi / 早期 video-planning** | imagine-then-execute 需 test-time 生成；我们不需要 |
| **FastWAM** | 架构基座 + 动机来源（它的结论正是我们的动机） |

## 5. 实验-故事映射（~7 周最低可行包）

1. **主表**：① vs ②，poker + 2 个任务 × 100 episode（50 集下 p=0.21 不显著，必须加量）
2. **机制消融表**：pin 位置 / t=0 / isolation 三行（1x4 卡规模，每行几天）
3. **可解释性图**：action expert → action-image token 的 attention map；action image 可视化（渲染器现成）
4. **③ 全任务**：作为"多任务 regime 仍有效"的支撑，不追求 SOTA
5. （加分项）**视角泛化**：action image 是 view-grounded 的，换一个没见过的相机位姿测泛化

## 6. 当前实验证据状态（2026-08-09）

| 实验 | 状态 | 结果 |
|---|---|---|
| ① 纯 FastWAM 单任务 | 训完（step_003660） | eval 0.72（36/50） |
| ② +ActionImages 单任务 | 训完（step_003660） | eval 0.84（42/50），vs ① McNemar p=0.21（n=50 不显著） |
| ③ 全任务 | 续训排队中（从 step_017500） | 2.2% 训练量时 eval 0.44（欠训，不可直接比） |

## 7. 待定事项

- [ ] ③ `max_steps` 是否设 100000（≈1.24 epoch，3 个 24h 作业收口；当前 num_epochs=10 ≈ 23.7 天）
- [ ] fork 中 3 个未提交改动（`save_total_limit=5`、③ `save_every=5000`、trainer 裁剪实现）是否提交
- [ ] ② vs ① 是否上 100 episode 重测（建议：上）
- [ ] 主卖点最终定调：表示/接口故事（推荐）为主、机制故事为辅，还是反过来
