---
publish: true
status: done
created: 2026-07-12
updated: 2026-07-12
type: paper-note
date: 2024-03-04
tags:
  - paper-note
  - recommender-system
  - ranking
  - exploration
  - contextual-bandit
  - long-term-value
  - online-experiment
  - wsdm-2024
---

# Long-Term Value of Exploration

## 1. 基本信息

- 标题：Long-Term Value of Exploration: Measurements, Findings and Algorithms
- 作者：Yi Su、Xiangyu Wang、Elaine Ya Le、Liang Liu、Yuening Li、Haokai Lu、Benjamin Lipshitz、Sriraj Badam、Lukasz Heldt、Shuchao Bi、Ed H. Chi、Cristos Goodrow、Su-Lin Wu、Lexi Baugher、Minmin Chen
- 机构：Google / Google DeepMind
- 时间：2024
- 会议：WSDM 2024
- arXiv：https://arxiv.org/abs/2305.07764
- DOI：https://doi.org/10.1145/3616855.3635833
- 本地 PDF：`C:\Users\chaol\Desktop\推荐论文阅读\re-ranking\longtermvalue\2305.07764 - rl探索优化长期体验.pdf`
- 分类：工业推荐 / pointwise ranking / 探索 / 长期价值评估

## 2. 相关论文

- [[A Long-term Value Prediction Framework In Video Ranking]]：2026 年的排名阶段 LTV 工作在相关工作中直接引用本文，并将本文的探索型 LTV 指标/内容池研究定位为与其效率型排序目标互补的路线；其新增重点是位置纠偏、后续观看归因和跨日作者价值。

已检查 vault 中的 CMR、MultiTRON、UnifiedRL、EMER、Pantheon 与重排综述。它们涉及多目标、线上收益或精排，但本文没有把它们作为直接基线、前置工作或比较对象；除上述被后续 ranking-LTV 工作直接引用的关系外，不新增宽泛主题相似的论文级 wikilink。

## 3. 一句话总结

这篇论文把“探索值得做”拆成可检验的因果链：**探索让更多新鲜/长尾内容在停止扶持后仍被发现（Discoverable Corpus 扩大），而减少可发现内容会长期降低满意 DAU**；在工程上，再用 Neural Linear Bandit（NLB）把不确定性采样接到既有深度精排模型上。

## 4. 问题：常规 A/B 测不到探索的长期收益

纯 exploitation 推荐会形成闭环：系统反复展示已知高分内容，用户只会对已曝光内容反馈，训练数据进一步把画像和内容池收窄。探索会把较不确定的内容展示给用户，短期点击、完播等指标可能中性甚至变差，却同时带来两类长期收益：

- **item exploration**：新鲜和长尾内容获得初始曝光，平台能发现原本不会浮现的高质量内容；
- **model learning**：原本未知的用户-内容对进入日志，覆盖更广、选择偏差更小，后续模型的不确定性降低。

难点是普通 user-diverted A/B 只随机分用户，两个实验臂仍共享同一内容池。Treatment 让某条内容火起来后，control 也会受益，内容池层的处理效应泄漏；因此只看短期 user metric 不能回答“探索是否扩大了平台长期可用的优质内容池”。

## 5. 系统背景：探索放在 ranking，不是 listwise re-ranking

![[Long-Term Value of Exploration.assets/fig1_multistage_system.png|800]]

**图 1** 给出作者研究的工业多阶段推荐：多个 nominator 从十亿级语料产生候选，pointwise ranking 对数百到数千候选逐个打分，最后 packing 才做列表级业务目标与多样化。本文把探索加在中间的 **pointwise ranking**，并没有重写最后的 setwise packing。

这一区分很重要：本文对“重排”的启发是给最终列表前的精排提供探索分数和更广内容池，而不是提出一个直接优化整条列表的 listwise reranker。文中多数对照也都保持后续 ranking/packing 不变，以隔离探索本身的影响。

## 6. 探索的长期价值：先测内容池，再连接满意度

### 6.1 Discoverable Corpus：只计停止探索后仍成功的内容

作者定义 $\operatorname{DC}_{\pi}@X,Y$：在系统 $\pi$ 下，内容结束探索扶持后，仍在 $Y$ 天内获得超过 $X$ 次正反馈的内容数量：

$$
\operatorname{DC}_{\pi}@X,Y
= \#\{c\mid c\text{ 在 post-exploration 的 }Y\text{ 天内获得}>X\text{ 次正反馈}\}.
$$

这里的 *post-exploration* 是关键隐藏条件。内容达到 graduation threshold $X'$ 后不再接受额外探索曝光，之后必须靠自身质量和相关性继续增长；否则 treatment 仅仅因为给了更多展示就会显得更好。小 $X$ 观察长尾内容是否被启动，大 $X$ 观察未来头部内容是否增长；论文用 7 天看短期、3 个月看长期。

### 6.2 User-Corpus-CoDiverted A/B：同时分用户和内容

![[Long-Term Value of Exploration.assets/fig2_user_corpus_codiverted.png|680]]

**图 2** 是为内容池效应设计的实验。作者把 $x\%$ 的内容池随机分给 control 与 treatment，同时也按相同比例将 $x\%$ 用户分给两臂；control 用户只接触 control 内容池，treatment 用户只接触 treatment 内容池。内容提供者也会整体分桶，避免同一创作者的内容跨臂传播效应。

这样既阻断内容池泄漏，又让“5% 用户探索 5% 内容池”的效果可外推到全量用户探索全量内容池。若只用 5% 流量探索 100% 内容，单个内容得到的探索强度过低，内容分布几乎不会改变，实验便无法测得该机制。

### 6.3 内容池确实扩大，而且不仅是低质内容

![[Long-Term Value of Exploration.assets/fig3_short_term_corpus_growth.png|760]]

**图 3** 是简单探索 treatment 的短期证据：预留若干 slot 给根据历史相似性召回的新鲜/长尾候选，其余 slot 保持 control 的系统。两张曲线分别是 7 天内 $\operatorname{DC}@100$ 与 $\operatorname{DC}@1000$；蓝色 treatment 随时间显著高于红色 control，说明更多内容在停止扶持后仍达到早期正反馈门槛。

![[Long-Term Value of Exploration.assets/table1_long_term_discoverable_corpus.png|720]]

**表 1** 把窗口拉到 3 个月。不同质量门槛的 Discoverable Corpus 均增加：$X_l=100$ 为 **+119.4%**，$1000$ 为 **+58.5%**，$10K$ 为 **+48.2%**，$1M$ 为 **+51.0%**，$10M$ 为 **+53.8%**。后面三个高阈值仍约 +50%，是“探索并非只推起低质冷启动内容”的核心数字。

![[Long-Term Value of Exploration.assets/fig4_corpus_quality_distribution.png|760]]

**图 4** 进一步检查这个解释。横轴是停止探索后正反馈数 $X$（log scale），纵轴是 3 个月内至少达到该 $X$ 的内容数。treatment 的柱子整体更高，但形状与 control 相近，尤其在 $X\ge10K$ 的高质量区域。这支持作者的判断：探索扩大了可发现内容的数量，同时发现的内容质量分布没有明显变差；不过它仍是分布证据，不代表逐内容证明因果质量提升。

### 6.4 内容池减少会长期伤害满意度

![[Long-Term Value of Exploration.assets/fig5_corpus_ablation_satisfaction.png|760]]

**图 5** 用反向实验闭合论证。两臂使用同一多阶段系统，但 treatment 随机从每个 nominator 的输出中过滤 $x\%$ 内容；对同一用户固定随机种子以保证其始终面对同一个缩小后的内容池，同时增加召回数，保证送到 ranking 的候选量相同。

左图显示，ablation 越大，满意日活（由满意度问卷预测的满意交互计数）下降越多，且负效应随 4 周时间累积。右图把可发现内容池缩减比例与满意 DAU 变化做线性插值，呈单调、近似线性的关系。作者因此提出“更大的 discoverable corpus 会带来长期满意度收益”；需保留它的边界：线性关系只在观测区间成立，内容池充分大时可能饱和，图也不能单独识别全部中介路径。

## 7. Neural Linear Bandit：将不确定性接进既有深度精排

### 7.1 从上下文 bandit 到候选级采样分数

每个时刻用户特征为 $u_t$，候选内容特征为 $a_t$，系统从候选集合 $\mathcal A$ 选择一个内容并得到点击、完播、点赞等奖励 $r_t$。目标是最小化相对事后最优内容 $a_t^*$ 的累计 regret：

$$
R_T(\pi)=\mathbb E\left[\sum_{t=1}^{T}
\bigl(r(u_t,a_t^*)-r(u_t,a_t)\bigr)\right].
$$

NLB 的折中是假设深度网络最后一层表征已经把复杂非线性问题变成近似线性回归：

$$
\mathbb E[r(u,a)] = \phi(u,a)^\top\beta,
\qquad
\Sigma_t=\epsilon I+\sum_{\tau=1}^{t}\phi_\tau\phi_\tau^\top,
\qquad
\hat\beta_t=\Sigma_t^\dagger\sum_{\tau=1}^{t}\phi_\tau r_\tau .
$$

其中 $\phi(u,a)\in\mathbb R^d$ 是一个**候选级向量**，不是请求级标量；$\Sigma_t^\dagger$ 在满秩时等于逆矩阵，欠秩时是给出最小 $L_2$ 范数解的伪逆。这个近似若不成立（例如最后层表征不足以线性解释平均奖励），后验方差就不再是可靠的不确定性。

给定高斯噪声 $\sigma^2$，每个候选的预测奖励后验为：

$$
r(u,a)\mid\mathcal D_t \sim
\mathcal N\left(
\phi(u,a)^\top\hat\beta_t,
\sigma^2\phi(u,a)^\top\Sigma_t^\dagger\phi(u,a)
\right).
$$

服务时并不是输出一个全局探索系数：对同一请求里的每个候选分别从该后验采样 $m_{TS}(u,a)$，再以 $p_{TS}(u,a)=\mu(m_{TS}(u,a))$ 作为排序分数，按分数排序。因此均值相近时，不确定性大的长尾/新鲜候选更有机会被抽高。

### 7.2 架构和梯度边界

![[Long-Term Value of Exploration.assets/fig6_neural_linear_bandit_architecture.png|780]]

**图 6** 左侧 control 是标准分类精排：拼接 $u,a$，经 DNN 得到最后层 embedding $\phi(u,a)$，经 logit $m(u,a)$ 和 sigmoid 输出原始概率 $p(u,a)$。右侧 treatment 复用同一 embedding，但在虚线处 **stop gradient**：NLB 分支利用冻结的 $\phi$、$\hat\beta$ 和 $\Sigma^\dagger$ 计算均值、方差、Thompson Sampling，再经 sigmoid 产生 $p_{TS}$。

这保证探索分支不反向改变原始监督目标的梯度；它只改变线上候选的展示概率和由此收集的数据。实验中使用 128 维最后层 embedding，并补接部分用户/内容特征以提升相关性和不确定性估计。分类任务的一个重要细节是：均值采用原模型二元预测 $\hat r$ 的 logit。由于 sigmoid 单调，均值排序与原模型排序一致；探索来自采样方差，而不是悄悄替换 exploitation 排序语义。

### 7.3 让 NLB 适应批训练和数值稳定性

工业模型不会逐样本在线重训。作者将更新拆成两段：

1. **训练阶段**：每个 batch 用当时的 $\phi$ 累积协方差矩阵 $\Sigma$，同时照常用 SGD 更新深度模型和 $\beta$；一个训练 run 结束后才计算一次 $\Sigma^\dagger$。
2. **服务阶段**：固定 $\Sigma^\dagger$，对每个请求、每个候选算后验均值和方差，采样后取 top-$K$，再把曝光反馈写回日志。

这种延后求伪逆的条件是服务可以接受“最近一次训练 run”的固定精度矩阵；若内容分布在两次导出之间快速漂移，不确定性会陈旧。对于近奇异的协方差，论文比较了伪逆和 Cholesky 分解：Cholesky 误差更低、更稳定，但伪逆明显更快；在 $d=128$ 的设置里作者选伪逆，接受少量估计精度换取训练速度。

![[Long-Term Value of Exploration.assets/fig7_uncertainty_estimator_tradeoff.png|760]]

**图 7** 左图以 NLB 预测与原神经网络预测的平均绝对差为误差代理，Cholesky 更低；右图以 global steps/sec 衡量训练速度，伪逆更高。它支撑的是上述工程权衡，不应误读为伪逆的方差估计更准确。

## 8. 线上实验：NLB 能扩大内容池并带来用户收益

作者在一个服务数十亿用户的短视频平台上做在线实验。主 A/B 将 control 与 NLB treatment 各分配 **0.3% 流量**、持续 6 周；协方差正则为 $\epsilon=10^{-6}$，噪声参数 $\sigma^2=10$，并用 5 个随机种子训练出的 ensemble 校准不确定性量级。

![[Long-Term Value of Exploration.assets/table2_freshness_metrics.png|650]]

**表 2** 显示新鲜内容的正反馈在所有 freshness bucket 均显著增长：1h **+1.49%**（95% CI [1.20, 1.77]）、3h **+1.51%**、12h **+1.45%**、1d **+1.43%**、3d **+2.55%**、12d **+1.16%**。作者还报告每位用户正反馈主题数增加 **+1.25%**，说明探索不只是在同一兴趣内做微扰。

![[Long-Term Value of Exploration.assets/fig8_user_satisfaction_gain.png|680]]

**图 8** 报告 6 周内满意 DAU 相对 control 的提升及 95% 置信区间，曲线整体为正。论文将其解释为：对新鲜/长尾内容的额外曝光既改变了内容池，也提供了有用的学习信号，因而可以转成用户满意度收益；图中未给出可脱离坐标精读的单一总 uplift 数，因此不把曲线峰值当成精确结论。

![[Long-Term Value of Exploration.assets/table3_uncertainty_correlations.png|620]]

**表 3** 是不确定性合理性的 sanity check。NLB 方差与内容年龄、内容累计正反馈的 Spearman 相关分别为 **-0.35 ± 0.003**、**-0.26 ± 0.003**，与用户活跃度为 **0.02 ± 0.008**。也就是模型更不确定于新鲜、较不热门内容，而不是简单把低活跃用户全部判为不确定；结果也与 ensemble 的约 -0.3 / 约 0 结果一致。

在 user-corpus-codiverted 评估里，NLB 相比 exploitation 系统还把 7 天 $\operatorname{DC}@100$ 提升 **+5.33%**、$\operatorname{DC}@1000$ 提升 **+5.66%**。这比仅看点击更直接地连接到论文提出的内容池指标。

## 9. 附录补强：探索也改善训练数据与模型质量

![[Long-Term Value of Exploration.assets/fig9_data_diverted_experiment.png|760]]

**图 9** 的 data-diverted A/B 不再比较两套正在服务的算法，而是将 control 与 treatment 的日志分开：Model A 只训练 control 日志，Model B 只训练探索日志；随后两者都在同一个原始 exploitation 线上系统里接受常规 A/B 评估。这样测的是“探索收集的数据是否训练出更好的模型”，排除了上线时直接加探索分数这一混杂因素。

![[Long-Term Value of Exploration.assets/fig10_model_learning_long_term_value.png|760]]

**图 10** 左图显示约 30M training steps 后，使用探索日志训练的模型不确定性低于 control；右图显示其长期满意度增益在 3 个月内继续上升。这里的 treatment Thompson Sampling 不确定性由 5 个不同随机种子的同构模型估计；为控制成本，底部表征层共享，只复制 head。该附录把“更多探索”补成更完整的链条：探索不仅扩内容，也得到更少选择偏差的训练样本。

## 10. 结论、边界与可复用判断

- **测量贡献大于算法新颖性**：论文最有价值的是内容池指标、user-corpus-codiverted 与 data-diverted 两种实验设计；它们回答了常规 A/B 会漏掉什么。
- **不要把相关性当作完整因果证明**：内容池 ablation 与满意度支持强关联，但可发现内容池的最优大小、饱和点与所有中介机制仍未识别。
- **NLB 的适用边界**：它依赖“最后层表征近似线性奖励”和可接受的离线精度矩阵更新频率；更复杂的深层非线性不确定性或剧烈漂移可能使后验失真。
- **任务边界**：实现面向单一分类奖励（文中为完播），没有解决现代推荐常见的多任务反馈、长期 reward 延迟和多目标冲突；作者把高效的 multi-task exploration 留为未来工作。
- **对重排的启发**：最终 packing 未改，但可以把 NLB 的候选级不确定性、内容池隔离实验和长期满意度指标带入 re-ranking 的探索 slot、长尾保护与 offline-online 对齐研究。

## 11. 记忆锚点

- **核心链条**：探索 → Discoverable Corpus → 长期满意度；另一路是探索日志 → 更低模型不确定性 → 更好模型质量。
- **核心指标**：$\operatorname{DC}@X,Y$ 必须在停止探索后统计，避免“给了更多曝光”本身造成假提升。
- **核心实验**：同时分用户和内容池防 treatment leakage；将日志分开、统一 serving 评估防 algorithm confounding。
- **核心算法**：DNN 最后一层 embedding 做 $\phi(u,a)$，NLB 对每个候选采样后验分数，方差推动探索。
- **工程取舍**：伪逆快、Cholesky 更准；离线更新精度矩阵、线上只做候选级方差与采样。

## 12. 图表覆盖检查

- 设计/流程图：图 1（多阶段系统）、图 2（user-corpus-codiverted）、图 6（NLB/control 架构）、图 9（data-diverted）均已嵌入并逐图解释。
- 内容池与长期价值图：图 3（短期增长）、表 1（3 个月各阈值增长）、图 4（质量分布）、图 5（内容池 ablation）均已覆盖。
- 算法与线上结果图：图 7（伪逆/Cholesky 取舍）、表 2（freshness）、图 8（满意度）、表 3（不确定性相关）、图 10（模型不确定性与长期质量）均已覆盖。
