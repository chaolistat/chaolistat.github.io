---
publish: true
status: done
created: 2026-07-12
updated: 2026-07-12
type: paper-note
date: 2026-07-12
tags:
  - paper-note
  - recommender-system
  - re-ranking
  - long-term-value
  - constrained-reinforcement-learning
  - actor-critic
  - short-video
  - multi-objective
---

# Two-Stage Constrained Actor-Critic

## 基本信息

- 标题：Two-Stage Constrained Actor-Critic for Short Video Recommendation
- 作者：Qingpeng Cai、Zhenghai Xue、Chi Zhang、Wanqi Xue、Shuchang Liu、Ruohan Zhan、Xueliang Wang、Tianyou Zuo、Wentao Xie、Dong Zheng、Peng Jiang、Kun Gai
- 机构：Kuaishou Technology；Hong Kong University of Science and Technology（Ruohan Zhan）；Kun Gai 标为 Unaffiliated
- 时间：2023
- 会议：WWW 2023
- arXiv：https://arxiv.org/abs/2302.01680
- DOI：https://doi.org/10.1145/3543507.3583259
- 代码：https://github.com/AIDefender/TSCAC
- 本地 PDF：C:\Users\chaol\Desktop\推荐论文阅读\re-ranking\longtermvalue\2302.01680 - Two-Stage Constrained Actor-Critic for Short Video Recommendation.pdf
- 分类：重排 / 长期价值 / 约束强化学习

## 相关论文

本次检查了 vault 中的多目标重排、Pareto 与 RL-MTF 笔记。CMR、MultiTRON、BatchRL-MTF 虽同样涉及多目标推荐，却不是本文直接引用、对比、继承或被其明确后续工作的关系；为避免把宽泛主题相似误写成论文关系，本笔记不新增 wikilink，也不改动它们的双向关联。

## 一句话总结

TSCAC 将短视频排序建模为一个以 WatchTime 为主目标、以点赞/评论/分享等稀疏互动为约束的 CMDP：先为每个辅助反馈训练独立 actor-critic，再训练 WatchTime actor，并用与这些辅助策略的 KL 接近性作软约束。它避免把密集时长与稀疏互动硬加成一个 reward；在 KuaiRand 上得到 WatchTime 13.14（相对 BC +2.23%），线上相对 LTR 取得 WatchTime +0.379%、Share +3.376%、Download +1.733%。

## 问题与背景

短视频信息流里，用户不断上滑并消费多个视频。平台最在意的是一个 session 内累计 WatchTime，因为它反映注意力和长期活跃；但 Like、Follow、Comment、Collect、Share 等互动又是体验和内容生态的 guardrail，不能为了时长被牺牲。

![[Two-Stage Constrained Actor-Critic.assets/fig1_short_video_platform.png|500]]

**Figure 1：短视频界面把“连续决策 + 多反馈”具体化。** 子图 (a) 是用户通过上滑切换视频的连续消费过程；子图 (b) 标出单次观看后可能出现的关注、点赞、评论、收藏、分享。它不是网络结构图，却说明本文的两个设计前提：WatchTime 几乎每次曝光都有，而互动常常隔很多次观看才出现；因此把所有信号简单求和，会让密集信号在 critic 中淹没稀疏信号。

传统做法在这里都不完全合适：

- 单目标 RL 只最大化时长，会忽略互动约束。
- 预设权重的 reward scalarization 需要手工找权重；多个约束时 Lagrange multiplier 的搜索空间会迅速变大。
- Pareto 最优并不保证业务最重视的 WatchTime 是最高的。
- 标准 RCPO 用一个 critic 评估加权 reward，但 TD target 只能自然使用一个折扣因子；不同响应频率和不同长期性混在一起时，值估计会互相干扰。

### CMDP：一个 action 在服务中如何变成推荐视频

![[Two-Stage Constrained Actor-Critic.assets/fig2_cmdp.png|620]]

**Figure 2：论文的 CMDP 时间线。** 用户打开 App 后开始一个 session；第 $t$ 个 request 中，系统从状态 $s_t$ 作出动作 $a_t$，展示视频后收到向量奖励 $r(s_t,a_t)=(r_1,\ldots,r_m)$，再转到 $s_{t+1}$。横向箭头表示同一 session 内多个 request，纵向多条反馈表示主目标与辅助目标同时存在；用户离开 App 时轨迹终止。

形式上，CMDP 写为 $(S,A,P,R,C,\rho_0,\Gamma)$。$R:S\times A\rightarrow\mathbb R^m$ 产生 $m$ 个行为信号，$\Gamma=(\gamma_1,\ldots,\gamma_m)$ 允许每个信号有自己的折扣因子。对策略 $\pi$，作者要解的是：

$$
\max_\pi\ \mathbb E_{s\sim\rho_\pi}[V_1^\pi(s)]
\quad\text{s.t.}\quad
\mathbb E_{s\sim\rho_\pi}[V_i^\pi(s)]\ge C_i,\ i=2,\ldots,m .
$$

这里第 1 个响应是主目标 WatchTime，$C_i$ 是第 $i$ 个辅助响应的最低要求。论文先用“动作 = 被推荐的视频”给出抽象 CMDP；但其生产实现将动作参数化为**连续的用户偏好向量**：actor 输出该向量，ranking function 对每个候选视频 embedding 做内积，分数最高的候选才是实际曝光视频。这个重参数化很关键——RL 学的是“==当前用户该偏向哪些主题/质量==”，而不是在百万视频 ID 上直接采样。

## 方法总览

TSCAC 有三层依赖关系：

1. 每种响应各自一个 critic，先把价值估准；
2. 第一阶段分别学好每个辅助目标的策略；
3. 第二阶段让 WatchTime 策略追逐主目标优势，同时不要离这些辅助策略太远。

这也是它的约束代理：不是在第二阶段直接检查每个 $V_i^\pi\ge C_i$，而是把“辅助最优策略附近的动作区域”当成较安全的策略域。

### 多 critic：先拆开密集和稀疏信号

对第 $i$ 个响应，独立 critic 为 $V_{\phi_i}$，TD advantage 为：

$$
A_i^{(k)}=r_i(s,a)+\gamma_i V_{\phi_i}^{(k)}(s')-V_{\phi_i}^{(k)}(s).
$$

这使 WatchTime、Like、Comment 等都有自己的 reward、折扣 $\gamma_i$ 和误差面。它解决的不是“多建几个 head 就更强”，而是不同时间尺度和观察频率不应共享一个 Bellman target。论文在一天真实短视频日志上比较了两种估值：

- joint：用 $V_{\mathrm{joint}}$ 估计 WatchTime 与 interaction 的和；
- separate：分别学 $V_w,V_i$，再以 $V_{\mathrm{separate}}=V_w+V_i$ 汇总。

与 session Monte Carlo 回报的相关性上，separate 分别比 joint 在 WatchTime 和 interaction 高 0.19%、0.14%；论文说明在该业务中 0.1% 已有显著性。这个小数值是后续两阶段设计的前提：若稀疏互动的 critic 本身不可靠，“约束”就只是噪声。

### 第一阶段：每个辅助反馈先学一条策略

对每个辅助响应 $i=2,\ldots,m$，用自己的 actor $\pi_{\theta_i}$ 最大化对应长期回报。critic 最小化 TD 残差，actor 做 advantage policy-gradient：

$$
\phi_i^{(k+1)}
\leftarrow
\arg\min_{\phi}\,
\mathbb E_{\pi_{\theta_i}^{(k)}}\!
\left[
\bigl(r_i(s,a)+\gamma_i V_{\phi_i}^{(k)}(s')-V_\phi(s)\bigr)^2
\right],
$$

$$
\theta_i^{(k+1)}
\leftarrow
\arg\max_\theta\,
\mathbb E_{\pi_{\theta_i}^{(k)}}\!
\left[A_i^{(k)}\log\pi_\theta(a\mid s)\right].
$$

输出不是一套把全部 feedback 加权求和的策略，而是 $\{\pi_{\theta_2},\ldots,\pi_{\theta_m}\}$：每一条都代表“若我要保护这项互动，动作应落在哪里”。若辅助目标有 $m-1$ 个，critic/actor 的训练与存储也随之线性增加；这正是用更高计算换更可信约束信号的代价。

### 第二阶段：主策略最大化 WatchTime，但软贴近辅助策略

第二阶段另建主目标 actor $\pi_{\theta_1}$ 和 critic $V_{\phi_1}$。critic 仍只用 WatchTime 更新；actor 的理想优化是：

$$
\max_\pi\ \mathbb E_\pi[A_1^{(k)}]
\quad\text{s.t.}\quad
D_{\mathrm{KL}}(\pi\Vert\pi_{\theta_i})\le\epsilon_i,\ i=2,\ldots,m .
$$

其拉格朗日解可写成：

$$
\pi^*(a\mid s)
=\frac{1}{Z(s)}
\left[
\prod_{i=2}^{m}
\pi_{\theta_i}(a\mid s)^{
\frac{\lambda_i}{\sum_{j=2}^{m}\lambda_j}}
\right]
\exp\left(
\frac{A_1^{(k)}}{\sum_{j=2}^{m}\lambda_j}
\right).
$$

这个式子的阅读顺序是：

1. 前面的加权几何积是辅助策略的共同偏好；某个动作若被某条辅助策略赋予很低概率，会被明显压低。
2. 指数项再按主目标 advantage 抬高 WatchTime 价值大的动作。
3. $Z(s)$ 只负责归一化，保证得到合法分布。

$\lambda_i$ 越大，主策略越被限制在辅助策略附近；越小，越接近只优化 WatchTime。生产实验为了可维护性把所有 $\lambda_i$ 设为同一个值，而没有逐目标细调。公式隐含两个条件：$\sum_i\lambda_i>0$，且辅助策略对需要保留的动作应有足够概率质量；否则加权几何积会把该动作几乎归零。论文用随机策略来写这一推导，因此可直接使用 KL 与概率密度；在连续确定性策略的扩展中，则改用动作相似度函数。

### 离线与连续动作扩展

从固定日志离线训练时，数据来自行为策略 $\pi_\beta$，而不是正在更新的 $\pi_{\theta_i}$。作者用 trajectory 上的 action-selection ratio 做一阶重要性采样，以校正分布错配；这要求行为策略对目标策略采到的动作不为零，否则比率不可估或方差爆炸。

生产场景里 action 是连续偏好 embedding，而不是离散 video ID。文中给出相似度例子：

$$
h(a_1,a_2)=\sum_{d=1}^{n}
\exp\left(-\frac{(a_{1d}-a_{2d})^2}{2}\right).
$$

它在两个偏好向量各维接近时更大。连续版本于是把“与辅助策略的 KL 接近”替换成“主 actor 输出与辅助 actor 输出相近”，同时用 $Q_{\phi_1}(s,\pi_{\theta_1}(s))$ 推高主目标价值。这里保持向量维度一致是必要条件：主、辅助 actor 必须在同一偏好空间输出，$h$ 才能逐维比较，且该向量必须能和 candidate embedding 做内积用于排序。

### Figure 4：生产中的推理—反馈—训练闭环

![[Two-Stage Constrained Actor-Critic.assets/fig4_production_workflow.png|900]]

**Figure 4 左侧（Inference of RL）。** 用户历史、属性和上下文特征组成 state；actor 采样 action，retrieval 提供 candidates，ranking 根据 action 选择曝光视频。曝光后的反馈进入 replay buffer，同时用户 history 更新，形成下一个 request 的 state。图明确了 TSCAC 插入的是候选排序决策，而不是替代召回。

**Figure 4 右侧（Training of RL）。** 训练从 replay buffer 取出 state、sampled action 和 feedback；critic 由 TD/critic loss 更新，actor 根据 critic 传回的 policy gradient 更新。两个虚线箭头表示 actor 不直接从真实标签监督，而是经 critic 的价值估计获得梯度。把它与左图连起来看，在线 collect 的 transition $(s,a,r,s')$ 是同一套闭环的桥梁。

## 实验与结果

### KuaiRand 离线实验

公共数据 KuaiRand 有 26,858 用户、10,221,515 items、68,148,288 samples；Like、Comment、Hate 的稀疏率分别为 1.61%、0.24%、0.048%，而 Click 为 37.70%。作者把同一用户日志拼成轨迹，每个用户取最常观看的 150 个视频；state 是 1044 维向量（用户特征 + 最近 20 个视频 + 150 个候选特征），主目标为 WatchTime，折扣设为 0.99，用 NCIS 作离线策略评估。

比较对象包括监督的 BC、Wide&Deep、DeepFM，传统约束 RL RCPO，拆 critic 的 RCPO-Multi-Critic，以及 Pareto。需要区分两项验证：RCPO-Multi-Critic 检验“拆 critic”是否有效；TSCAC 再检验“用辅助策略作软约束”是否优于直接 Lagrangian reward。

![[Two-Stage Constrained Actor-Critic.assets/table2_offline_kuairand.png|780]]

**Table 2：TSCAC 的主目标和多数辅助目标同时最好。** 相对 BC，TSCAC 的 Click +4.35%、Like +18.80%、Comment +15.6%、WatchTime +2.23%；Hate 从 2.304 降到 1.870（-18.83%，越低越好）。在所有方法中，它是 WatchTime 13.14 的最高值，并在 Click、Like、Comment 三项辅助指标最佳。Pareto 的 Hate 0.9915 最好，却把 WatchTime 拉到 11.90（-7.4%），正好印证“Pareto 不保证主业务目标”的批评。RCPO 的 WatchTime 到 13.07，但 Hate 2.951 是最差，显示稀疏信号会被密集时长主导；RCPO-Multi-Critic 改善了这个问题，却未达到 TSCAC 的整体平衡。

### $\lambda$ 消融：约束强度不是越大越好

![[Two-Stage Constrained Actor-Critic.assets/fig3_lambda_ablation.png|900]]

**Figure 3：五条曲线共享同一个 $\lambda$。** 作者测试 $10^{-1},10^{-2},10^{-3},10^{-4},10^{-5}$。约束从强到弱时，WatchTime 整体上升；但互动在 $\lambda=10^{-4}$ 达到更好平衡，继续减到 $10^{-5}$ 后互动反而下降。原因与公式一致：过大的 $\lambda$ 会过度锁住主策略；过小则几乎不再从辅助策略继承 interaction 偏好。图中不是证明一个普适常数，而是说明该软约束需要按业务误差容忍度校准。

附录的 TripAdvisor 多目标酒店模拟也给出同方向证据：TSCAC 的 Overall Rating 为 3.99（与 RCPO-Multi-Critic 并列最高），并在 7 个辅助评分中拿到 Service 3.43、Cleanliness 3.64、Value 3.37、Rooms 3.00、Location 2.98 五项最高。Pareto 在 Business、Check-in 上更高，但 Overall 只有 3.95，仍与“主目标优先”的设定不一致。

### 线上短视频实验

线上 actor 使用多元 Gaussian policy，均值和方差由 actor 输出；用户 state 包括历史互动、属性以及当前候选视频特征，动作是预测的 topic/quality 偏好 embedding。系统随机分用户 bucket：一个运行默认 LTR，其他分别运行 RCPO、只最大化互动的 Interaction-AC、TSCAC；模型先训练数天，再固定策略评估一天。

![[Two-Stage Constrained Actor-Critic.assets/table3_live_experiment.png|620]]

**Table 3：软约束比两种极端策略更平衡。** RCPO 仅优化 WatchTime 与加权互动，得到 WatchTime +0.309%，但 Share -0.707%、Comment -1.313%。Interaction-AC 把互动优先，Share +5.080%、Download +1.952%，WatchTime 只涨 +0.117%。TSCAC 把三者折中为 WatchTime +0.379%、Share +3.376%、Download +1.733%，均优于 RCPO；Comment 为 -0.619%，仍未能消除时长与评论之间的 trade-off。论文注明该平台中 WatchTime 0.1%、互动 1% 的变化已经有统计显著性，因此这些量级在生产中有业务意义。

![[Two-Stage Constrained Actor-Critic.assets/fig5_online_daily_gap.png|620]]

**Figure 5：两阶段学习在时间上的表现。** 前几天 Share、Download 很快转正并拉升，说明主策略先继承了 Interaction-AC 的限制；随后 WatchTime 稳步上行。Comment 则在第 5 天后转负，展示了这种“软”约束不是逐项指标的硬下界保证。相较之下，作者强调最终 TSCAC 在四项线上指标上均显著优于 RCPO；曲线最值得记住的是先建立互动安全域、再逐步取得 WatchTime，而非所有目标同步单调变好。

## 理解、局限与启发

- **用策略相近代理结果约束。** TSCAC 的创新不是为每个指标手工设一个 reward 权重，而是先学“守住某项互动”的策略，再让主策略靠近它们。这个代理在策略局部平滑、辅助策略质量足够好时有意义；若状态分布变化很大，策略接近并不严格等价于 $V_i^\pi\ge C_i$，所以不能把它当成硬安全证明。
- **multi-critic 是必要但有成本。** 拆开 critic 解决了 dense/sparse 混淆，也允许不同 $\gamma_i$；但每加入一种 guardrail 都需要额外 actor/critic、样本覆盖与超参数维护，目标很多时成本会增长。
- **约束强度仍依赖业务选择。** 线上为可维护性统一取 $\lambda$，但 Figure 3 说明最优平衡敏感于它。不同互动的价值、频率、可牺牲程度不同时，共用 $\lambda$ 可能不是最优。
- **离线证据受策略覆盖限制。** NCIS 和重要性采样只能评估行为日志已覆盖的 action 区域；对远离 $\pi_\beta$ 的新偏好向量仍可能高方差。公开 KuaiRand 与酒店模拟能验证方向，却不能复现快手线上复杂环境。
- **论文没有完全展开工程细节。** 生产 ranking function、候选规模、reward 标定和多目标 $\lambda_i$ 的选取规则没有给出足够细节；确定性连续 action 版本也被列为未来工作，因此端到端复现仍需要额外系统假设。

## 结论与记忆点

> 不要把“时长”与“互动”先加成一个 reward 再希望一个 critic 自己学会平衡；先分别学会保护稀疏互动的策略，再让追求 WatchTime 的主策略以可调强度靠近它们。

以后阅读长期价值 / 多目标重排论文时，可以用 TSCAC 检查四件事：

1. 密集主信号和稀疏 guardrail 是否由独立 value model 估计，并允许不同折扣？
2. 约束是直接作用在 outcome，还是以策略距离、动作区域或 reward penalty 代理？代理何时会失效？
3. 主目标策略如何从辅助策略中继承可行域：KL、动作相似度、硬边界还是共享 representation？
4. 离线日志是否覆盖新策略所需动作，线上 A/B 是否同时报告主指标和 guardrail 的真实 trade-off？
