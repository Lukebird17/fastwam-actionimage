# 实验日志 — Action Images @ WAM Scale (ICLR 2027)

> 每次实验追加一行到对应表。**任何人看这张表就能复现数字。**
> 固定字段：日期 / ckpt step / 数据集+seed / 成功率(n/N) / 统计 / 结论。
> eval 统一：RoboTwin `move_playingcard_away`、`demo_randomized`、同种子 4300005+、
> `ROBOTWIN_CAMERA_SHADER=default`、`SAPIEN_HEADLESS=1`。

## 主表：② vs ①（action image 是唯一变量）

| 日期 | 臂 | ckpt | 任务 / n_ep | 成功率 | 统计 | 结论 |
|---|---|---|---|---|---|---|
| 2026-08-08 | ① scene_only | step_003660 | poker / 50 | 0.72 (36/50) | — | 控制组基线 |
| 2026-08-07 | ② action_image | step_003660 | poker / 50 | 0.84 (42/50) | vs ① McNemar p=0.21 | n=50 不显著，需上 100 |
| 待定 | GeoProp baseline | — | poker / 100 | — | — | P0，未实现 |

## 机制消融（C2：干净条件帧三条件）

| 日期 | 条件 | 设置 | 成功率 | 结论 |
|---|---|---|---|---|
| 2026-08-06 | pin 位置 | 错位 (pre-fix) | 0.10 (5/50) | pin 错位致命 |
| 2026-08-06 | pin 位置 | 对齐 (post-fix) | 0.82 (41/50) | C2 pin 行现成证据 |
| 2026-08-09 | t=0 标记 off | `mark_clean_conditioning_frame:false` | 训练中 (job 328712) | config `robotwin_poker_ablate_no_t0_1x4`，干净单变量 |
| 2026-08-09 | isolation off | `segment_first_frame_bidirectional` | 训练中 | config `robotwin_poker_ablate_isolation_1x4`，干净单变量（pin+t=0 保持，仅去隔离） |
| 待定 | 监督密度 λ | 0.05 / 1.0 | — | P2 |

## ③ 全任务（scaling 支撑）

| 日期 | ckpt | 训练进度 | eval | 备注 |
|---|---|---|---|---|
| 2026-08-07 | step_017500 | 2.2%（欠训） | 0.44 (22/50) | 不可直接比 |
| 2026-08-09 | step_125000 | ~1.5 epoch | 待测 | 作业 313725 TIMEOUT(24h)；已超 max_steps=100000，建议不再续训 |

## 待办（按 PROJECT_DESIGN §3 优先级）

- [x] P0: ②vs① 上 100 episode —— 已挂（job 328661 ① / 328662 ②），RUNNING
- [ ] P1: GeoProp baseline 训练臂 —— 降为 P1（①已含同源 proprio，主对比公平）
- [x] P0: t=0 消融 —— 已挂（job 328712）
- [x] P0: isolation 消融 —— 已实现 `segment_first_frame_bidirectional` 干净单变量，挂训练中
- [ ] P1: ③ step_125000 重测
- [ ] P1: attention map 可解释性图
- [ ] P2: 视角泛化
