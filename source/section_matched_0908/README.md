# 原配置 section 对照实验（2026-09-08）

本批次按用户纠正后的要求执行：以原 all-subject exhaustive 固定 encoder
实验为基准，唯一实验变量是 section 划分。之前的 section_pilot_0908 使用
mean / hidden-64，不属于本批次的基准或正式对照。

| 条件 | 训练 section | 留出 section |
|---|---|---|
| 历史基准 | 1,2,3,4,5,6,7 | 8,9 |
| holdout_45 | 1,2,3,6,7,8,9 | 4,5 |
| holdout_12 | 3,4,5,6,7,8,9 | 1,2 |

完全保留原来的 20 MEG × 35 fMRI 全组合流程、缓存与预处理、训练目标、
batch 64、60 轮 encoder、60 轮融合、lr 0.0003、weight decay 0.01、noise 0.1。
每个划分训练一个 seed-42 encoder；MLP hidden 512、attention hidden 767
分别运行融合种子 42/43/44，复用并冻结该划分的 encoder。

Encoder 按原 unified_benchmark 的平均特征 MedR 规则选模；融合头按原
train_subject_pair_exhaustive.py 的全组合平均 MedR 规则选模。
仍由两个留出 section 同时选模和报告，不新增验证分区。
历史 8/9 结果保留复用，不重复训练。

## 已完成任务

| Slurm 数组子任务 | 工作 |
|---|---|
| 1659650_0 | 留出 4/5，encoder seed 42 |
| 1659650_1 | 留出 1/2，encoder seed 42 |
| 1659652_0 / _1 / _2 | 留出 4/5，MLP seeds 42/43/44 |
| 1659652_3 / _4 / _5 | 留出 4/5，attention seeds 42/43/44 |
| 1659652_6 / _7 / _8 | 留出 1/2，MLP seeds 42/43/44 |
| 1659652_9 / _10 / _11 | 留出 1/2，attention seeds 42/43/44 |

1659650 和依赖它的 1659652 已全部成功完成；每批数组最多同时占用一张
GPU。14 个子任务均写出完成标记和预期结果，stderr 除 locale 警告外为空。
每个子任务上限 24 小时，与原 Slurm 设置一致。

服务器 SSH 别名：biomag
项目：/gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
日志：logs/section_matched_encoder_<job>_<index>.out/.err 和
logs/section_matched_heads_<job>_<index>.out/.err。
结果：results/section_matched_0908/<split>/<mode>/s<seed>/result.json。

## 核验

preflight.json 记录配置差异、数据划分检查及原代码 SHA-256。
配置仅允许 section 和 checkpoint/output 路径改变；训练/评估超参数保持一致。
实际缓存窗口无交叉且覆盖全部 2931 窗口：4/5 为 2342 训练 + 589 评估，
1/2 为 2336 训练 + 595 评估。
Encoder 入口复制自当前原程序，放开 7/2 划分限制、更新日志、在 encoder
阶段完成后保存来源信息并结束；训练函数和训练前执行顺序已用 AST 核对。
融合直接运行原训练脚本，每个任务启动检查原代码指纹，训练前检查 encoder
确实来自相同 section 划分；结果从原 checkpoint 保存的 val_metrics 导出。

原始基准配置：configs/unified_fixed_encoder_legacy_mlp.yaml、
configs/unified_fixed_encoder_legacy_attention.yaml。
原始基准结果：results/unified_fixed_encoder_legacy_mlp_comparison.md、
results/unified_fixed_encoder_legacy_attention_comparison.md。

## 结果摘要

以下是三个融合种子的均值；Top-10 越高越好，MedR 越低越好。8/9 为已
提交的历史同协议基准，4/5 和 1/2 为本批次结果。

| 留出 section | 方法 | Global Top-10 | Global MedR |
|---|---|---:|---:|
| 8/9 | fMRI | 3.37% | 243.24 |
| 8/9 | Late | 3.46% | 235.18 |
| 8/9 | MLP | 3.61% | 228.26 |
| 8/9 | Attention | 3.50% | 235.02 |
| 4/5 | fMRI | 5.04% | 185.49 |
| 4/5 | Late | 5.379% | 183.16 |
| 4/5 | MLP | 5.375% | 176.04 |
| 4/5 | Attention | 5.354% | 182.86 |
| 1/2 | fMRI | 4.65% | 205.09 |
| 1/2 | Late | 4.95% | 194.10 |
| 1/2 | MLP | 5.14% | 188.68 |
| 1/2 | Attention | 4.97% | 194.19 |

MLP 在三个划分的 Global MedR 都最低；在 8/9 和 1/2 的 Global Top-10
也最高。4/5 的 Late 与 MLP Top-10 相差约 0.004 个百分点，而 MLP 的
MedR 比 Late 低约 7.12。不同留出段的候选窗口数不同，绝对指标不应直接
解释为某个 section 更容易；主要比较是各划分内融合相对单模态的变化。
