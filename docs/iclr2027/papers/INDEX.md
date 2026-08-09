# 参考文献库 — Action Images @ WAM Scale (ICLR 2027)

> 维护说明：本目录存放投稿相关的所有参考论文 PDF。**投稿前 2–3 周必须重新跑一次 arXiv 检索**
> （关键词：`world action model`、`proprioception + video diffusion`、`bimanual video generation`、
> `latent action`、`clean frame conditioning video diffusion`），因为该领域以"周"为单位在动。
> 下表"角色"列指该论文在我们论文里的定位（动机 / 划界 / 组件先验 / baseline / benchmark）。

| 文件 | arXiv | 角色 | 与我们 A 线的关系 |
|---|---|---|---|
| action-images-v1_2604.06168.pdf | 2604.06168 | **前身 (v1)** | 我们的表示来源；action image 作**生成目标**。v2 转为**条件接口**。 |
| fastwam_2603.16666.pdf | 2603.16666 | **架构基座 + 动机** | MoT 基座；其"proprio 走低维 token、视觉先验用不上"正是我们的切入点。 |
| geoprop_2607.07101.pdf | 2607.07101 | **最近邻 + 必做 baseline** | 把 state 投到图像平面但只在**特征空间** FiLM 调制。我们是**像素空间条件帧**。reviewer 必问，必须比。 |
| vla-proprio-interfaces_2608.03052.pdf | 2608.03052 | **开放问题来源** | 枚举 5 种 proprio 注入接口，**无一是像素帧进 video backbone**。我们 = 缺失的第 6 种。正面引用。 |
| uva-unified-video-action_2503.00200.pdf | 2503.00200 | **动机（非贡献）** | 最早"推理时只解码动作、跳视频生成"。**Claim 4 的抢占者**，只作动机引用。 |
| diffusion-forcing_2407.01392.pdf | 2407.01392 | **组件先验（红线）** | per-token 独立噪声级 / 干净历史帧 t=0 的**直接预见者**。Claim 2 的 (b) 必须引用它，禁止声称"首次"。 |
| faster-wam-future-conditioning_2608.04404.pdf | 2608.04404 | 划界 | 主张分布偏移下仍需 inference-time future conditioning（部分反驳 FastWAM）。 |
| faster-wam-deep-action_2608.02365.pdf | 2608.02365 | 划界 | RoPE realignment 用于 docking 接口 → Claim 2 (a) pin 位置的组件先验。 |
| adaptive-wam_2608.06008.pdf | 2608.06008 | 划界 | early-exit 避免全视频去噪 → WAM 效率红海的代表。 |
| lawam_2606.15768.pdf | 2606.15768 | 划界 | latent subgoal 替代未来视频。 |
| flowwam_2607.13017.pdf | 2607.13017 | 划界 | 光流视频作统一动作表示（生成侧）。 |
| craft-bimanual_2604.03552.pdf | 2604.03552 | 划界 | 双臂视频扩散做**数据生成**（非 policy）→ Claim 3 的占位者。 |
| robotwin2_2506.18088.pdf | 2506.18088 | **benchmark** | 50 双臂任务 + Aloha-Agilex，标准评测台。 |
| mixture-of-transformers_2411.04996.pdf | 2411.04996 | 组件先验 | MoT 架构来源（FastWAM 基座的理论基础）。 |
| dawn_2509.22652.pdf | 2509.22652 | 划界 | pixel motion 中间表示的层级框架。正常引用，不冲突。 |
| maskwam_2506.13515.pdf | 2506.13515 | 划界 | WAM 子领域代表之一。 |

## 一句话总结调研结论（2026-08-09）

- **空白（我们的主打）**：像素空间 proprio 条件注入 video backbone —— 没人占，且 2608.03052 刚把这个问题命名。
- **红海（避开当贡献）**:① "跳过 test-time imagination"(UVA 抢占 + FastWAM 子领域爆炸）;② latent-action(DLAM/WALA/LaWAM)。
- **红线**:per-token t=0 干净帧标记 = Diffusion Forcing 已预见，只能做"综合+消融"不能说"首次"。
