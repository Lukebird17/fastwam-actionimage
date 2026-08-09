# Action Images @ WAM Scale — ICLR 2027 投稿项目设计（长期维护版）

> **本文档是本投稿项目的唯一权威蓝图。** 后续所有实验、代码、写作都以此为准；
> 任何方向变更先改这里再动手。维护人：Haoyu Zhen。建立：2026-08-09。
>
> **会议目标**：ICLR 2027（California, 2027-04-26~30）。摘要截止按 ICLR 2026 类推
> **约 2026-09 中下旬**（官方页尚未放出确切日期，提交前以 https://iclr.cc/Conferences/2027 为准）。
> **定位**：Action Images v1（arXiv 2604.06168）的延伸新工作，非合并重投。

---

## 0. 一句话论文

> **World Action Model 该如何感知机器人自己的身体？我们把本体感觉（proprioception）渲染回
> 像素空间，作为条件帧注入预训练视频骨干，让动作状态也能吃到视频模型的视觉先验。**

这是对社区刚命名的开放问题（arXiv 2608.03052 枚举了 5 种 proprio 注入接口）给出的
**第 6 种、且是像素空间的答案**。

---

## 1. 研究问题（第一卖点，真空白）

**"How should a World Action Model perceive the robot's own action state?"**

FastWAM 范式下，视频生成退到训练期，action image 的角色从 v1 的**生成目标**转为 v2 的
**条件接口**：

- **Proprioception-as-image（条件侧）**：把当前 proprio 渲染成像素空间的 action image，
  pin 进视频 latent 序列做条件。低维 proprio token 走不进视频骨干的视觉先验，渲染成图像就可以
  —— 模态鸿沟在表示层被抹平。**调研确认：这一精确主张（像素帧进 video DiT 做条件）无人占。**
- **同一表示，双向使用**：v1 证明 action 可"被生成"（输出端）；v2 证明 action 可"被渲染回来
  做条件"（输入端）。这是闭环 replan 的天然机制（每 32 步重渲染当前 proprio → 新条件帧 → 下一 chunk）。
- **直接回应 v1 自认 limitation**（"not yet been fully developed into a closed-loop policy"）。

**措辞红线**：不要说 "zero-shot policy without action module"（那是 v1 的 claim，本版有
action expert，硬说自相矛盾）。说"action image 作为 action expert 与视频骨干之间的像素级接口"。

---

## 2. 贡献清单（焊死版，含拥挤度判断）

| # | 贡献 | 定位 | 拥挤度（2026-08 调研） | 必须引用/划界 |
|---|---|---|---|---|
| **C1** | 像素空间 proprio **条件接口**注入 video backbone | **主打 / headline** | **低（空白）**。最近邻 GeoProp(2607.07101) 是特征空间，非像素帧。 | GeoProp(2607.07101)、接口研究(2608.03052) |
| **C2** | 向 video DiT 注入**干净条件帧**的必要条件配方 + 逐条消融 | 副贡献 / 通用性加分 | 组件高（t=0 被 Diffusion Forcing 预见）/ 综合空白。只claim"综合+消融" | **Diffusion Forcing(2407.01392)**、Ca2-VDM、Causal Forcing、Faster-WAM(2608.02365) |
| **C3** | 双臂 16D / RoboTwin-50 / Aloha-Agilex **scaling** | 支撑结果，非头条 | 中（双臂视频生成被 CRAFT 占，具体组合空白） | CRAFT(2604.03552)、RoboTwin2(2506.18088) |

**明确放弃的贡献（红海，不写入贡献表）**：
- ❌ "跳过 test-time imagination"—— UVA(2503.00200) 抢占，FastWAM→Faster-WAM→Adaptive-WAM
  已是子领域。只在 Introduction 当**动机**引用。
- ❌ latent-action 世界模型路线（task2 设计文档）—— 挤进 DLAM/WALA/LaWAM 红海，且核心
  多视角一致性代码未实现，本届不做。

### C2 的三条件（每条配一个便宜消融）

| 机制 | 发现过程 | ablation |
|---|---|---|
| **Pin 位置** | `latent_t//2` 只在两段等长时碰巧对；通式="每个 segment 自己的第 0 帧"。双臂 `[scene×3‖action×1]` 不等长，`//2` 全是 bug | pin 错位 vs 对齐（RoPE pin fix 的 pre/post 数据已有：0.10→0.82） |
| **干净帧时间标记** | `fuse_vae_embedding_in_latents` + per-token t=0 vs 干净值顶着噪声级 t | t=0 标记 on/off |
| **注意力隔离** | conditioning 帧 causal isolation vs 全双向（pin 帧 hidden state 混入 48 帧噪声） | isolation on/off |
| （附）监督密度 | 3 帧 λ=0.05 vs 12 帧 λ=1.0 | λ 扫描 |

包装：**"向预训练 video DiT 注入 clean conditioning 的三个必要条件（面向'条件注入'而非
'生成历史'设定）"** —— 结论超出机器人领域，拉高通用性分。

---

## 3. 实验-故事映射（~6 周最低可行包）

> 优先级从上到下。P0 = 不做就拒稿；P1 = 显著加分；P2 = 加分项。

| 优先级 | 实验 | 规模 | 状态 (2026-08-09) |
|---|---|---|---|
| **P0** | **主表 ②vs①**：poker + 2 个任务 × **100 episode** | 3 任务 | 🔄 poker 100 集 eval 已挂（job 328661/328662）。50 集基线：①=0.72, ②=0.84 |
| **P1** | **GeoProp-style baseline**（特征空间 state 注入） | 1×4 卡 | ⚠️ 降为 P1：见下方"关键发现"，②vs① 已是公平对比，GeoProp 是锦上添花 |
| **P0** | **机制消融表**：pin 位置 / t=0 / isolation 三行 | 1×4 卡，每行几天 | pin 行已有(0.10→0.82)；t=0 训练 job 328712；isolation 用 `segment_first_frame_bidirectional` 干净单变量 |

> **关键发现（2026-08-09，已核实 config）**：① 和 ② **都喂了完全相同的 16D proprio**
> （`proprio_dim: 16`，两者一致），都走 `nn.Linear(proprio→text_dim)` 成全局 token concat 进
> 文本 context。**唯一区别是 ② 额外把同一个 proprio 渲染成 action image 当条件帧。**
> 因此 **②vs① 本身就是"像素空间条件 vs 向量空间注入"的公平对比**（信息同源、唯一变量）。
> GeoProp（特征空间 FiLM 注入）只是向量注入的一个更 fancy 变体 —— 有它能进一步证明像素接口
> 连"更 fancy 的向量注入"都打过，但**没有它主对比依然成立**。故 GeoProp 从 P0 降为 P1。
> 注意：GeoProp 论文需要相机内外参做投影接地；我们的 action image 渲染管线有相机参数
> （`datasets/robotwin/action_image.py`），可实现，但工程量 > 1 天。
| **P1** | ③ 全任务：多任务 regime 仍有效的支撑 | 2×8 续训中 | 续训 step ~112940，eval 17500 时 0.44（欠训） |
| **P1** | 可解释性图：action expert→action-image token attention map | 推理脚本 | 渲染器现成 |
| **P2** | 视角泛化：换一个没见过的相机位姿测泛化 | eval | action image 是 view-grounded 的，天然卖点 |
| **P2** | λ（监督密度）扫描 | 1×4 卡 | 附属于 C2 |

### 实验铁律（沿用既有约束）

- ① vs ② 必须**共享同一数据管线**（`include_action_video` 单开关，action image 是唯一变量），
  同 545 episodes / 46752 windows / 16D 动作空间 / 归一化统计 / 文本缓存 → 因果证据。
- eval 统一 `move_playingcard_away` 等任务、`demo_randomized`、同种子（4300005+）、
  `ROBOTWIN_CAMERA_SHADER=default`、`SAPIEN_HEADLESS=1`。
- 配对统计：同种子用 **McNemar exact**；不同种子用 Fisher exact。
- 所有训练 `save_total_limit=5`（已在 `configs/train.yaml:26` + `trainer.py` 裁剪实现），
  单 ckpt≈92G，防 /scratch 爆盘。
- 环境统一 conda；分析用 `envs/test/bin/python`，训练用 `envs/curobo/bin/python`；
  `actionimages` env 已坏勿用。

---

## 4. Related Work 划界（防 reviewer 抓）

| 工作 | 一句话区别 |
|---|---|
| **GeoProp (2607.07101)** | 它 state→图像平面但**特征空间 FiLM**；我们**像素空间条件帧**进 video DiT。最危险，主动划清。 |
| **接口研究 (2608.03052)** | 它枚举 5 种 proprio 接口、无像素帧；我们补第 6 种。正面引用，把它当"问题提出者"。 |
| **MV-VDP (2604.03181)** | 它多视角 heatmap 当**生成目标**解码动作；我们**条件接口**（v1 已有生成侧）。 |
| **Diffusion Forcing (2407.01392)** | per-token 噪声级/干净帧 t=0 的先验；我们综合成"条件注入"配方的必要性分析。**显眼引用，绝不抢首次。** |
| **UVA (2503.00200) / FastWAM (2603.16666)** | 跳 test-time imagination 的动机来源；不是我们的贡献。 |
| **CRAFT (2604.03552)** | 双臂视频扩散**数据生成**，非 policy。 |
| **DAWN (2509.22652)** | pixel motion 中间表示层级框架，正常引用。 |
| **UniPi / 早期 video-planning** | imagine-then-execute 需 test-time 生成；我们不需要。 |

---

## 5. 写作大纲（ICLR 8 页 + 无限附录）

1. **Intro**：WAM 崛起 → FastWAM 证 video co-training 够、inference 不需想象 → 但留下
   "proprio 怎么走不进视觉先验"的模态鸿沟（引 2608.03052 的开放问题）→ 我们：proprio-as-image
   条件接口。3 贡献对应 C1/C2/C3。
2. **Related Work**：按 §4 表格组织，GeoProp/Diffusion Forcing 放显眼位置。
3. **Method**：3.1 proprio-as-image 渲染与 pin 注入（C1）；3.2 干净条件帧三条件（C2）；
   3.3 闭环 replan；3.4 双臂扩展（C3）。
4. **Experiments**：主表（②vs①+GeoProp baseline）→ 机制消融（C2 三行）→ ③ 多任务 →
   可解释性/视角泛化。
5. **Limitation**：sim-only（真机未验）、单 embodiment 家族、Wan 骨干偏重。

---

## 6. 代码与工程维护规范（"以后一直用于维护"）

### 6.1 仓库布局（fastwam-actionimage = 唯一投稿仓库）

```
fastwam-actionimage/
├── docs/iclr2027/
│   ├── PROJECT_DESIGN.md      ← 本文档（唯一蓝图）
│   ├── papers/                ← 所有参考 PDF + INDEX.md
│   └── EXPERIMENT_LOG.md      ← 每次实验一行（见 §6.4）
├── configs/task/              ← 每个实验臂一个 yaml，命名 = 表里的行名
├── src/fastwam/               ← 模型/训练（trainer.py 含 _prune_checkpoints）
├── scripts/ , slurm/          ← 训练/评测入口
└── evaluate_results/robotwin/ ← eval 产物（_result_random.txt + mp4）
```

### 6.2 三个实验臂（对照变量 = action image 唯一）

| 臂 | task config | 说明 |
|---|---|---|
| ① 纯 FastWAM | `robotwin_poker_scene_only_1x4` | 无 action image（控制组） |
| ② +ActionImages | `robotwin_poker_action_image_1x4` | 单任务，唯一变量开 |
| ③ 全任务 | `robotwin_action_image_3cam_384_1e-4` | 50 任务 scaling |

新增 GeoProp baseline 时：复制 ① 的 config，把 state 注入从"无"换成"特征空间 FiLM 注入"，
其余严格不动，命名 `robotwin_poker_geoprop_1x4`。

### 6.3 提交前自检清单（每次大改跑一遍）

- [ ] `_prune_checkpoints` 仍 wired（`trainer.py` 里 save_checkpoint 调用它）
- [ ] ①② config 除 action image 开关外逐字段 diff 为空
- [ ] eval 端 RoPE pin = 训练端 pin（post-fix：latent_t=6, pins=(0,3)）
- [ ] 文本嵌入缓存覆盖 eval 可达指令集（防 Missing text embedding crash）
- [ ] wandb.enabled=true 且 `WANDB_API_KEY` 在 env_setup.sh（勿删）

### 6.4 实验日志（每次实验追加到 EXPERIMENT_LOG.md）

固定字段：日期 / 臂 / task config / ckpt step / 数据集+seed / 成功率(n/N) / 统计检验 / 一句话结论。
**任何人看到日志就能复现数字。** 当前基线见 §7。

---

## 7. 当前实验证据状态（2026-08-09 快照）

| 实验 | ckpt | eval | 统计 |
|---|---|---|---|
| ① 纯 FastWAM 单任务 | step_003660（训完） | 0.72 (36/50) | — |
| ② +ActionImages 单任务 | step_003660（训完） | 0.84 (42/50) | vs ① McNemar p=0.21（n=50 不显著） |
| ③ 全任务 | step_017500（2.2%，欠训） | 0.44 (22/50) | 不可直接比 |
| ② pin fix pre/post | — | 0.10 → 0.82 | C2 pin 行现成证据 |

---

## 8. 待定事项（逐条关闭，改这里）

- [x] **③ `max_steps`**：已设 100000（commit 175a313）。③ 实际已训到 step_125000（1.5 epoch），
      已超 100000，**建议不再续训**——其角色是"多任务仍有效"的支撑，1.5 epoch 足够。
- [x] **主表加量**：②vs① 100 episode 已挂（job 328661/328662）。
- [~] **GeoProp baseline**：**降为 P1**（2026-08-09 发现 ①已含同源 16D proprio，②vs① 即公平
      像素-vs-向量对比）。如需做：复制 ① config，把 Linear→token 换成 FiLM 调制，命名
      `robotwin_poker_geoprop_1x4`。GeoProp 论文需相机内外参（我们的 action_image 管线有）。
- [x] **C2 消融**：t=0（job 328712，`mark_clean_conditioning_frame:false`）+ isolation
      （`segment_first_frame_bidirectional`，干净单变量，commit 4ac003d）。
- [x] **fork 未提交改动**：`save_total_limit=5`、`save_every`、trainer 裁剪、`max_steps`、
      两个消融开关均已 commit。
- [ ] **两个交互式作业**（fastwam-grpo-local 等）是否清理。
- [ ] **主卖点最终定调**：已定 = C1 表示/接口为主，C2 机制为辅。（已定，勿反复）
- [ ] **投稿前 2-3 周重跑 arXiv 检索**（见 papers/INDEX.md 头部维护说明）。
- [ ] **isolation/t=0 消融训完后**，用 eval_robotwin_poker.sbatch 各测 50 集填入 EXPERIMENT_LOG。
