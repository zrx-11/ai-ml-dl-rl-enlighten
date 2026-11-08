#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从零推导 02：贝尔曼方程 → 策略梯度。"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from deriv_common import build

OUT = "/home/user/Doubao/chats/38452296622422786/AI-点化学习系统/06-从零推导/02-贝尔曼到策略梯度.html"

TREE = r"""
<h1>贝尔曼方程到策略梯度：一条连续的推导链</h1>
<p>这一页回答一个问题：<b>为什么“评价一个状态有多好”最终会变成“直接调策略参数”</b>。中间没有跳跃，每一步都只有一次变形，你能逐行复核。</p>
<div class="chain">
  <div class="node jump" data-goto="s1">期望：把“平均”写成符号</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s2">回报与折扣：为什么要打折</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s3">价值函数：一个状态值多少</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s4">贝尔曼期望方程（完整推导）</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s5">贝尔曼最优方程：max 放哪</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s6">为什么还要直接对策略求导</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s7">log 导数技巧</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s8">策略梯度定理</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s9">REINFORCE 算一遍</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s10">基线：为什么不引入偏差</div>
</div>
<div class="box">
<h4>本节不涉及</h4>
<p>Q-learning、资格迹、PPO 的截断目标。这些是在这条链上加的分支，先把主干打通。</p>
</div>
"""

S1 = r"""
<h1>第 1 节　期望：把“平均”写成符号</h1>
<p>强化学习里所有量都是随机的。讲“这个状态值多少”，说的是“平均而言值多少”。所以要先能熟练地写期望。</p>
<h2>1.1 离散期望</h2>
<p>随机变量 $X$ 取值 $x_i$ 的概率是 $p_i$，则</p>
$$\mathbb{E}[X]=\sum_i p_i x_i$$
<p>数值例：掷一个偏心的骰子，1 点概率 0.5，6 点概率 0.5，其余 0。期望 $=0.5\cdot1+0.5\cdot6=3.5$。注意 3.5 从来不会出现——期望不是“最可能的值”，是加权平均。</p>
<h2>1.2 条件期望</h2>
<p>$\mathbb{E}[X\mid Y=y]$ 表示“已知 $Y=y$ 之后，$X$ 的平均”。强化学习里几乎所有期望都是条件期望：给定当前状态，未来平均能拿多少。</p>
<div class="box">
<h4>一个必须记牢的恒等式（全期望公式）</h4>
<p>$$\mathbb{E}[X]=\sum_y P(y)\,\mathbb{E}[X\mid Y=y]$$</p>
<p>意思是：总平均 = 先按 Y 分组求平均，再按 Y 的概率加权。贝尔曼方程的推导只用这一条。</p>
</div>
<h2>1.3 期望的线性性</h2>
<p>$\mathbb{E}[X+Y]=\mathbb{E}[X]+\mathbb{E}[Y]$，不需要独立。这条在后面拆回报时反复用。</p>
"""

S2 = r"""
<h1>第 2 节　回报与折扣：为什么未来的奖励要打折</h1>
<h2>2.1 回报的定义</h2>
<p>从时刻 $t$ 出发，未来所有奖励的折扣和：</p>
$$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}$$
<p>$\gamma\in[0,1)$ 是折扣因子。</p>
<h2>2.2 一个关键恒等式</h2>
<p>把第一项摘出来，剩下的正好是从 $t+1$ 开始的回报：</p>
$$G_t=R_{t+1}+\gamma G_{t+1}$$
<p>这不是技巧，是整个强化学习的结构基础：<b>回报自己套自己</b>。贝尔曼方程能成立，全靠这一行。</p>
<h2>2.3 为什么必须有 $\gamma$</h2>
<p>两个理由，一个是数学的，一个是语义的。</p>
<details class="fold"><summary>展开：数学理由（收敛）</summary><div class="fold-body">
<p>若每步奖励都不超过 $R_{\max}$，则</p>
$$G_t\le R_{\max}(1+\gamma+\gamma^2+\cdots)=\frac{R_{\max}}{1-\gamma}$$
<p>$\gamma&lt;1$ 时这个几何级数收敛。若 $\gamma=1$ 且任务永不结束，和会发散到无穷，价值函数没有定义。所以“没有终止状态的任务必须用 $\gamma&lt;1$”。</p>
</div></details>
<details class="fold"><summary>展开：语义理由（视野）</summary><div class="fold-body">
<p>$\gamma$ 决定“看多远”。$\gamma=0.9$ 时，10 步之后的奖励权重是 $0.9^{10}=0.349$，20 步后 $0.9^{20}=0.122$。$\gamma=0.99$ 时同样的 20 步还有 $0.818$。</p>
<p>换句话说，$\gamma$ 是“有效视野长度”的旋钮，大约为 $1/(1-\gamma)$ 步：$\gamma=0.9$ 约 10 步，$\gamma=0.99$ 约 100 步。</p>
</div></details>
<div class="box warn">
<h4>常见错误</h4>
<p>把 $\gamma$ 当成“学习率”。它俩毫无关系：$\gamma$ 在回报里，$\alpha$ 在更新里。</p>
</div>
"""

S3 = r"""
<h1>第 3 节　价值函数：一个状态到底值多少</h1>
<h2>3.1 状态价值</h2>
$$v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$$
<p>读法：从状态 $s$ 出发，之后都按策略 $\pi$ 行动，能拿到的回报的平均。</p>
<p>注意两件事：这个期望是<b>对策略的随机性和环境的随机性同时取</b>的；它依赖于 $\pi$——换策略，价值就变。</p>
<h2>3.2 动作价值</h2>
$$q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]$$
<p>多了一个“先选定动作 $a$”的条件。$q$ 比 $v$ 多一个维度，但信息更全：知道 $q$ 就能直接选动作（选最大的），只知道 $v$ 还得知道环境怎么转移。</p>
<h2>3.3 两者的关系</h2>
$$v_\pi(s)=\sum_a \pi(a\mid s)\,q_\pi(s,a)$$
<p>理由：从 $s$ 出发，先按 $\pi$ 抽一个动作，再取该动作的平均回报。这就是全期望公式。</p>
<h2>3.4 一个能算的数</h2>
<p>两个状态 $A$、$B$ 互相转移：$A\to B$ 得奖励 0，$B\to A$ 得奖励 1，确定性，$\gamma=0.9$。</p>
<p>设 $v(A)=x$，$v(B)=y$。由 $G_t=R_{t+1}+\gamma G_{t+1}$ 取期望：</p>
$$x=0+0.9y,\qquad y=1+0.9x$$
<details class="fold"><summary>展开：解这个方程组</summary><div class="fold-body">
<p>把 $y$ 代入第一式：$x=0.9(1+0.9x)=0.9+0.81x$。</p>
<p>移项：$x-0.81x=0.9$，即 $0.19x=0.9$，$x=4.7368$。</p>
<p>回代：$y=1+0.9\cdot4.7368=1+4.2632=5.2632$。</p>
<p>检查：$v(B)$ 比 $v(A)$ 大，因为 $B$ 下一步就能拿到奖励 1，合理。</p>
</div></details>
"""

S4 = r"""
<h1>第 4 节　贝尔曼期望方程：完整推导</h1>
<p>这是全页的核心。目标是把“无穷长的和”变成“一步 + 下一状态的价值”。</p>
<h2>4.1 从定义出发</h2>
$$v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$$
<p>用 $G_t=R_{t+1}+\gamma G_{t+1}$ 替换：</p>
$$v_\pi(s)=\mathbb{E}_\pi\big[R_{t+1}+\gamma G_{t+1}\mid S_t=s\big]$$
<p>由线性性拆开：</p>
$$v_\pi(s)=\underbrace{\mathbb{E}_\pi[R_{t+1}\mid S_t=s]}_{\text{即时奖励的平均}}+\gamma\underbrace{\mathbb{E}_\pi[G_{t+1}\mid S_t=s]}_{\text{未来回报的平均}}$$
<h2>4.2 把两层随机性写出来</h2>
<p>从 $s$ 出发，先按策略选动作 $a$，再按环境转移到 $s'$ 并拿到 $r$。用全期望公式逐层展开：</p>
$$v_\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\Big[r+\gamma\,\mathbb{E}_\pi[G_{t+1}\mid S_{t+1}=s']\Big]$$
<p>最后那个期望按定义就是 $v_\pi(s')$，于是：</p>
<div class="box key">
<h4>贝尔曼期望方程</h4>
$$v_\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\big[r+\gamma v_\pi(s')\big]$$
</div>
<p>这就是全部。它说的是：一个状态的价值 = 按策略选动作、按环境转移，拿到的即时奖励加下一状态价值的折扣，全部加权平均。</p>
<h2>4.3 用刚才的两状态例子验一遍</h2>
<p>$A$ 只有一个动作，必定到 $B$，$r=0$；$B$ 必定到 $A$，$r=1$。方程给：</p>
<p>$v(A)=\sum_a\pi(a|A)\cdot[0+0.9v(B)]=0.9v(B)$；$v(B)=1+0.9v(A)$。与第 3 节手解一致。</p>
<h2>4.4 动作用 $q$ 写的版本</h2>
$$q_\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)\Big[r+\gamma\sum_{a'}\pi(a'\mid s')q_\pi(s',a')\Big]$$
<p>两个版本的差别只是“先选动作”还是“先转移”。做题时用哪个方便就用哪个。</p>
<div class="box warn">
<h4>为什么它不能直接当求解公式用</h4>
<p>方程右边含 $v_\pi(s')$，是同一个未知函数在不同状态上的值。它是<b>方程组</b>而不是显式表达式。有模型时可以迭代求解（动态规划），没模型时只能采样估计（蒙特卡洛、TD）。</p>
</div>
"""

S5 = r"""
<h1>第 5 节　贝尔曼最优方程：max 该放在哪一维</h1>
<h2>5.1 从“按策略平均”到“取最好”</h2>
<p>最优价值定义为：</p>
$$v_*(s)=\max_\pi v_\pi(s)$$
<p>可以证明它满足：</p>
<div class="box key">
<h4>贝尔曼最优方程</h4>
$$v_*(s)=\max_a\sum_{s',r}p(s',r\mid s,a)\big[r+\gamma v_*(s')\big]$$
</div>
<p>对比期望方程：$\sum_a\pi(a\mid s)$ 换成了 $\max_a$。一个“按概率平均”，一个“挑最好的”。</p>
<h2>5.2 max 为什么在动作上</h2>
<p>因为策略的本质就是在每个状态选动作。要取最好，就是把这个选择换成 max。若把 max 放在状态上（$\max_{s'}$），意思就变成“运气最好时能到哪”，那不是智能体能控制的。</p>
<h2>5.3 同样的两状态例子，加上一个动作</h2>
<p>设在 $A$ 有两个动作：动作 1 到 $B$ 得 0；动作 2 到 $B$ 得 0.5。$B$ 不变（回 $A$ 得 1）。$\gamma=0.9$。</p>
<p>最优方程：$v(A)=\max(0+0.9v(B),\,0.5+0.9v(B))=0.5+0.9v(B)$，$v(B)=1+0.9v(A)$。</p>
<details class="fold"><summary>展开：解出最优价值</summary><div class="fold-body">
<p>$v(A)=0.5+0.9(1+0.9v(A))=0.5+0.9+0.81v(A)$。</p>
<p>移项：$0.19v(A)=1.4$，$v(A)=7.3684$，$v(B)=1+0.9\cdot7.3684=7.6316$。</p>
<p>对比第 3 节的 $v(A)=4.7368$：换一个更好的动作，价值从 4.74 涨到 7.37。这就是策略改进的全部含义。</p>
</div></details>
<h2>5.4 为什么它非线性、为什么没有闭式解</h2>
<p>max 是分段线性的：在每个状态取不同动作会得到不同的线性表达式。合起来是一个分段线性函数，不是线性方程组。解它只能迭代（值迭代、策略迭代），或者在大规模问题上用函数逼近近似。</p>
"""

S6 = r"""
<h1>第 6 节　既然有 $v$ 和 $q$，为什么还要直接对策略求导</h1>
<p>这是很多人卡住的地方：价值方法已经能选动作了，策略梯度图什么？</p>
<h2>6.1 三个具体麻烦</h2>
<ol>
<li><b>连续动作</b>：$q(s,a)$ 要对 $a$ 取 max 选动作。动作是连续值时，每步都要解一个优化问题，代价高。</li>
<li><b>随机最优策略</b>：在石头剪刀布这类博弈里，最优策略是“各 1/3 随机出”。确定性价值方法表达不了这种策略。</li>
<li><b>函数逼近下的 argmax 不稳定</b>：价值估计有噪声时，argmax 会跳来跳去，策略随之抖动。</li>
</ol>
<h2>6.2 换一个优化对象</h2>
<p>把策略写成参数的函数 $\pi(a\mid s,\theta)$，直接最大化期望回报：</p>
$$J(\theta)=\mathbb{E}_{\pi_\theta}[G_0]$$
<p>于是问题从“估计价值再隐式导出策略”变成“对 $\theta$ 做梯度上升”。动作空间长什么样都无所谓，只要能算出 $\pi$ 和它的梯度。</p>
<h2>6.3 难点在哪</h2>
<p>$\theta$ 既影响动作，也影响“之后会遇到什么状态”。所以</p>
$$\nabla_\theta J=\nabla_\theta\sum_s d_\theta(s)\sum_a\pi_\theta(a\mid s)q_\theta(s,a)$$
<p>里面 $\theta$ 出现了三次：状态分布 $d_\theta$、策略 $\pi_\theta$、价值 $q_\theta$。看起来要求三个导数，很麻烦。下一节的两步变形会把它化简到只剩一个。</p>
"""

S7 = r"""
<h1>第 7 节　log 导数技巧：一个恒等式救全场</h1>
<h2>7.1 恒等式</h2>
<p>对任意正函数 $p(\theta)$：</p>
$$\nabla_\theta\log p(\theta)=\frac{\nabla_\theta p(\theta)}{p(\theta)}$$
<p>移项就得到最常用的形式：</p>
$$\nabla_\theta p(\theta)=p(\theta)\,\nabla_\theta\log p(\theta)$$
<div class="box">
<h4>验一遍</h4>
<p>取 $p(\theta)=e^{2\theta}$。左边 $\nabla p=2e^{2\theta}$。右边：$\log p=2\theta$，$\nabla\log p=2$，乘 $p$ 得 $2e^{2\theta}$。一致。</p>
</div>
<h2>7.2 为什么这个恒等式有用</h2>
<p>求 $\nabla_\theta\pi_\theta(a\mid s)$ 时，$\pi$ 是 softmax 或高斯的密度，直接求导表达式很乱。用它换成 $\pi\nabla\log\pi$，而 $\log\pi$ 常常是简单形式：</p>
<ul>
<li>softmax：$\log\pi(a)=\theta_a-\log\sum_b e^{\theta_b}$，梯度是 $1-\pi(a)$（对 $\theta_a$）和 $-\pi(a)$（对其他）。</li>
<li>高斯：$\log\pi(a)=-\frac{(a-\mu)^2}{2\sigma^2}-\log\sigma-\frac12\log2\pi$，梯度非常干净。</li>
</ul>
<h2>7.3 二动作 softmax 的梯度（后面要用）</h2>
<p>设两个动作，参数 $\theta_1$、$\theta_2$，$\pi(a_1)=\dfrac{e^{\theta_1}}{e^{\theta_1}+e^{\theta_2}}$。</p>
<details class="fold"><summary>展开：算 $\nabla_{\theta_1}\log\pi(a_1)$</summary><div class="fold-body">
<p>$\log\pi(a_1)=\theta_1-\log(e^{\theta_1}+e^{\theta_2})$。</p>
<p>对 $\theta_1$ 求导：第一项给 1，第二项给 $\dfrac{e^{\theta_1}}{e^{\theta_1}+e^{\theta_2}}=\pi(a_1)$。</p>
$$\nabla_{\theta_1}\log\pi(a_1)=1-\pi(a_1)$$
<p>对称地，$\nabla_{\theta_1}\log\pi(a_2)=-\pi(a_1)$。</p>
<p>取 $\theta_1=\theta_2=0$：$\pi=0.5$，梯度为 $0.5$ 与 $-0.5$。</p>
</div></details>
"""

S8 = r"""
<h1>第 8 节　策略梯度定理</h1>
<h2>8.1 定理长什么样</h2>
$$\nabla_\theta J(\theta)=\sum_s d_\theta(s)\sum_a q_\theta(s,a)\,\nabla_\theta\pi_\theta(a\mid s)$$
<p>换成“采样期望”的写法：</p>
$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\Big[\sum_t q_\theta(S_t,A_t)\,\nabla_\theta\log\pi_\theta(A_t\mid S_t)\Big]$$
<h2>8.2 关键在于：状态分布那一项被消掉了</h2>
<p>完整推导要用到“状态分布对参数的导数”，那一项一般算不出来。定理之所以成立，是因为它在求和里被消掉了。这一节给出能验证的路线，不展开测度细节。</p>
<details class="fold"><summary>展开：为什么可以消掉（思路版）</summary><div class="fold-body">
<p>把回报按访问顺序展开。$\theta$ 影响回报只有两条途径：一是改变每一步选动作的概率；二是改变“走到哪些状态”。</p>
<p>第二条途径的贡献，可以写成“某个状态的未来回报”乘以“这个状态的访问概率对 $\theta$ 的导数”。而对一个概率分布求导再求和，得到的结构正好与第一条途径的写法合并，最终只有 $\nabla\log\pi$ 留下。</p>
<p>严格证明需要马尔可夫链的稳态分布与梯度交换次序，这里只需记住结论与它成立的代价：<b>要求策略处处可微、且采样来自当前策略</b>（同轨）。</p>
</div></details>
<h2>8.3 一个两动作的数值检查</h2>
<p>设两个动作的期望回报 $q(a_1)=10$，$q(a_2)=6$。$\theta_1=\theta_2=0$，故 $\pi=(0.5,0.5)$。</p>
<p>用定理的采样形式算梯度（对 $\theta_1$）：</p>
$$\nabla_{\theta_1}J\approx \pi(a_1)q(a_1)(1-\pi(a_1))+\pi(a_2)q(a_2)(-\pi(a_1))$$
<p>$=0.5\cdot10\cdot0.5+0.5\cdot6\cdot(-0.5)=2.5-1.5=1$。</p>
<p>直接对 $J(\theta)=\pi_1 q_1+\pi_2 q_2$ 求导验：$\dfrac{d\pi_1}{d\theta_1}=\pi_1(1-\pi_1)=0.25$，$\dfrac{d\pi_2}{d\theta_1}=-\pi_1\pi_2=-0.25$。所以 $\nabla J=10\cdot0.25+6\cdot(-0.25)=2.5-1.5=1$。一致。</p>
<div class="box key">
<h4>定理的一句话版本</h4>
<p>把每个动作的价值，乘上“让这个动作更容易被选中”的方向，加起来。价值高的动作，就往它那边推。</p>
</div>
"""

S9 = r"""
<h1>第 9 节　REINFORCE：用实际回报代替 $q$</h1>
<p>定理里有 $q_\theta$，实际中我们不知道它。最直接的办法：跑完一幕，用真实回报 $G_t$ 顶上去。</p>
<h2>9.1 更新式</h2>
$$\theta\leftarrow\theta+\alpha\sum_t G_t\,\nabla_\theta\log\pi_\theta(A_t\mid S_t)$$
<p>这就是 REINFORCE。它是无偏的（$G_t$ 的期望就是 $q$），代价是方差大。</p>
<h2>9.2 走一步，数字全给</h2>
<p>两个动作，$\theta=(\theta_1,\theta_2)=(0,0)$，所以 $\pi=(0.5,0.5)$。这一步实际选了动作 1，得到回报 $G=10$。学习率 $\alpha=0.01$。</p>
<details class="fold"><summary>展开：算这一次更新</summary><div class="fold-body">
<p>$\nabla_{\theta_1}\log\pi(a_1)=1-\pi_1=0.5$；$\nabla_{\theta_2}\log\pi(a_1)=-\pi_1=-0.5$。</p>
<p>乘 $G=10$：梯度估计为 $(5,\,-5)$。</p>
<p>更新：$\theta_1=0+0.01\cdot5=0.05$，$\theta_2=0-0.01\cdot5=-0.05$。</p>
<p>新策略：$\pi_1=\dfrac{e^{0.05}}{e^{0.05}+e^{-0.05}}=\dfrac{1.05127}{1.05127+0.95123}=0.52498$。</p>
<p>动作 1 的概率从 0.5 升到 0.525。方向对了：它刚带来高回报，就该更容易被选。</p>
</div></details>
<h2>9.3 它到底在干什么</h2>
<p>$\nabla\log\pi(a)$ 是“把动作 $a$ 的概率推高”的方向向量。$G$ 是权重。所以这一条规则可以读成：<b>把这次实际拿到的回报，当作这次动作的评分，评分高就把概率推上去，评分低就压下去。</b></p>
<div class="box warn">
<h4>为什么方差大</h4>
<p>$G$ 是一整幕的随机和，波动极大。同样的策略、同样的动作，两幕的 $G$ 可能差十倍。所以单次梯度估计噪声很大，需要很多回合才收敛。</p>
</div>
"""

S10 = r"""
<h1>第 10 节　基线：为什么不引入偏差</h1>
<h2>10.1 做法</h2>
<p>把 $G_t$ 换成 $G_t-b(S_t)$，其中 $b$ 只依赖状态，与动作无关。</p>
$$\theta\leftarrow\theta+\alpha\sum_t \big(G_t-b(S_t)\big)\nabla_\theta\log\pi_\theta(A_t\mid S_t)$$
<h2>10.2 期望不变的证明</h2>
<p>要证 $\mathbb{E}[b(S)\nabla\log\pi(A\mid S)]=0$。因为 $b$ 只依赖 $S$，可以先把 $A$ 的期望算掉：</p>
$$\sum_a\pi(a\mid s)\nabla\log\pi(a\mid s)=\sum_a \nabla\pi(a\mid s)=\nabla\sum_a\pi(a\mid s)=\nabla 1=0$$
<p>最后一步用了“概率之和恒为 1”，它对任何 $\theta$ 都成立，所以导数为 0。</p>
<h2>10.3 用数字验一遍</h2>
<p>$\pi=(0.5,0.5)$，$s$ 固定，取 $b=8$。</p>
<p>$\sum_a\pi(a)\nabla\log\pi(a)=0.5\cdot(1-0.5)+0.5\cdot(-0.5)=0.25-0.25=0$。确实是 0。</p>
<p>回到第 9 节的例子：$G=10$ 换成 $G-b=2$，梯度从 $(5,-5)$ 变成 $(1,-1)$，更新步长小了五倍，但期望方向没变。</p>
<h2>10.4 为什么它能降方差</h2>
<p>方差取决于 $G-b$ 的波动幅度。选 $b$ 接近 $\mathbb{E}[G]$ 时，$(G-b)$ 围绕 0 波动，幅度最小。最常用的 $b$ 就是状态价值 $v(s)$——这正是 actor-critic 里 critic 的作用。</p>
<div class="box">
<h4>一句话</h4>
<p>基线不改方向，只改步长的抖动幅度。减去一个“平均水平”，剩下的就是“比平均好多少”，这个量才是真正该被奖励的。</p>
</div>
"""

S11 = r"""
<h1>第 11 节　为什么这样设计</h1>
<h2>11.1 为什么用 $\log\pi$ 而不是直接对 $\pi$ 求导</h2>
<p>因为 $\log$ 把乘积变加法，梯度形式干净；而且 $\nabla\log\pi$ 有一个很好的解释：它是“提高该动作概率”的方向，与具体策略形式无关。这使同一套公式能用在 softmax、高斯、甚至混合策略上。</h2>
<h2>11.2 为什么策略梯度天然是同轨的</h2>
<p>公式里的期望是对当前策略 $\pi_\theta$ 取的。用别的策略采的数据，就要加重要性采样比率——这是另一条技术路线（离轨策略梯度）。</p>
<h2>11.3 为什么概率饱和后学不动</h2>
<p>二动作情形下 $\nabla_{\theta_1}J=\pi_1(1-\pi_1)(q_1-q_2)$。当 $\pi_1\to1$ 时，$\pi_1(1-\pi_1)\to0$，梯度消失。意思是：一旦策略几乎总选同一个动作，它就很难再改回来，即使那个动作其实不好。</p>
<p>这就是为什么要加熵正则：$J'=J+\beta H(\pi)$，鼓励策略保持一定随机性，避免过早饱和。</p>
<h2>11.4 为什么 actor-critic 要把两者结合</h2>
<p>纯策略梯度方差大、收敛慢；纯价值方法在连续动作上不好选动作、策略会抖。actor-critic 用 critic 提供低方差的 $q$ 或优势估计，actor 负责输出动作分布——两边各取所长。</h2>
<div class="box key">
<h4>四条设计动机</h4>
<p>用 $\log\pi$ 是为了形式统一；同轨是为了梯度无偏；熵正则为了防饱和；actor-critic 为了降方差。</p>
</div>
"""

S12 = r"""
<h1>第 12 节　动手：拖 $\theta$ 看策略与梯度</h1>
<p>这个演示把第 8、9 节的公式变成可以拖的东西。$\theta_1$ 变，动作 1 的概率变，期望回报变，梯度也变——你能看到梯度什么时候最大。</p>
<div class="demo">
  <div class="demo-row"><label>$\theta_1$ = <b id="thval">0.00</b></label>
  <input id="thslider" type="range" min="-4" max="4" step="0.05" value="0"></div>
  <div class="bars">
    <div class="barrow"><span class="barlab">动作 1（q=10）</span>
      <div class="bartrack"><div class="barfill" id="bar1"></div></div>
      <span class="barval" id="p1">50.0%</span></div>
    <div class="barrow"><span class="barlab">动作 2（q=6）</span>
      <div class="bartrack"><div class="barfill b2" id="bar2"></div></div>
      <span class="barval" id="p2">50.0%</span></div>
  </div>
  <div class="demo-out">
    <div>期望回报 J = <b id="jval">8.00</b></div>
    <div>梯度 dJ/dθ₁ = <b id="gval">1.000</b></div>
    <div>$\pi_1(1-\pi_1)$ = <b id="satval">0.250</b></div>
  </div>
  <div class="demo-btns">
    <button id="stepBtn">按 α=0.1 走一步</button>
    <button id="resetBtn">复位</button>
  </div>
  <p class="demo-note">$\theta_2=0$ 固定，$\pi_1=\dfrac{e^{\theta_1}}{e^{\theta_1}+1}$，$J=10\pi_1+6\pi_2$，$\dfrac{dJ}{d\theta_1}=\pi_1(1-\pi_1)(10-6)$。</p>
</div>
<h2>看什么</h2>
<ul>
<li>$\theta_1$ 在 0 附近时梯度最大（$0.25\cdot4=1$）；把 $\theta_1$ 拖到 4，$\pi_1=0.982$，$\pi_1(1-\pi_1)=0.0177$，梯度只剩 $0.071$。</li>
<li>这就是第 11.3 节说的“饱和后学不动”。拖到两端再点几次“走一步”，你会发现几乎不动。</li>
<li>把 $\theta_1$ 拖到 -4，$\pi_1$ 只有 1.8%，梯度也是接近 0——两边都会饱和，所以初始值很重要。</li>
</ul>
"""

S13 = r"""
<h1>第 13 节　自检与思考题</h1>
<h2>13.1 能不能一句话复述</h2>
<ul>
<li>回报自己套自己：$G_t=R_{t+1}+\gamma G_{t+1}$，这是一切的基础。</li>
<li>贝尔曼期望方程把无穷和变成“一步 + 下一状态价值”。</li>
<li>最优方程把“按策略平均”换成“在动作上取 max”，因此非线性、需迭代。</li>
<li>策略梯度：$\nabla J=\mathbb{E}[q\,\nabla\log\pi]$，状态分布那项被消掉。</li>
<li>基线不改期望只降方差，因为 $\sum_a\nabla\pi(a)=0$。</li>
</ul>
<h2>13.2 思考题</h2>
<details class="fold"><summary>题 1：$\gamma=0.5$ 时，第 4 步之后的奖励权重是多少</summary><div class="fold-body">
<p>从下一时刻算：$R_{t+1}$ 权重 1，$R_{t+k}$ 权重 $\gamma^{k-1}$。第 4 步是 $k=4$，权重 $0.5^3=0.125$。</p>
<p>所以 $\gamma=0.5$ 的有效视野约 $1/(1-0.5)=2$ 步，很近视。</p>
</div></details>
<details class="fold"><summary>题 2：两状态例子中若 $\gamma=0.99$，$v(A)$ 是多少</summary><div class="fold-body">
<p>$x=0.99(1+0.99x)=0.99+0.9801x$，移项 $0.0199x=0.99$，$x=49.75$，$y=1+0.99\cdot49.75=50.25$。</p>
<p>对比 $\gamma=0.9$ 时的 4.74：折扣越接近 1，价值越大——因为“反复拿奖励”累积得更久。</p>
</div></details>
<details class="fold"><summary>题 3：为什么 $\sum_a\pi(a)\nabla\log\pi(a)=0$ 这个结论必须依赖“和恒为 1”</summary><div class="fold-body">
<p>因为 $\sum_a\nabla\pi(a)=\nabla\sum_a\pi(a)=\nabla 1=0$。若 $\pi$ 不是合法概率分布（比如没归一化），这个和就不是常数，导数不为 0，基线就会引入偏差。所以基线的成立有前提：$\pi$ 必须是归一化的。</p>
</div></details>
<details class="fold"><summary>题 4：把基线取成 $b(s)=q(s,a)$（依赖动作）会怎样</summary><div class="fold-body">
<p>那就不是基线了。因为 $b$ 依赖 $a$ 时，$\sum_a\pi(a)\nabla\log\pi(a)$ 的推导不再成立，期望会变，梯度有偏。基线必须与动作无关——这是硬约束，不是建议。</p>
</div></details>
<details class="fold"><summary>题 5：REINFORCE 与 Q-learning 的根本差别</summary><div class="fold-body">
<p>REINFORCE 直接改策略参数，目标是期望回报，属策略方法、同轨、可用连续动作；Q-learning 学动作价值再用 max 导出策略，属值方法、离轨、需要离散动作（或对连续动作做优化）。前者方差大、后者在函数逼近下可能不稳定。</p>
</div></details>
<h2>13.3 下一步</h2>
<p>去读《强化学习》第 3 章（MDP 与贝尔曼）、第 13 章（策略梯度），再回《人工智能：现代方法》第 22 章看它在 AI 版图里的位置。第 1 页的反向传播与本页的关系是：策略梯度要落地，就得用反向传播算 $\nabla\log\pi$。</p>
"""

extra_css = r"""
<style>
.chain{display:flex;flex-direction:column;align-items:stretch;margin:16px 0 6px}
.node{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
  border-radius:9px;padding:11px 13px;font-size:14.5px;cursor:pointer}
.node:hover{background:var(--accent-soft)}
.arrow{text-align:center;color:var(--muted);font-size:15px;line-height:1.3;padding:2px 0}
.demo{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:12px;margin:16px 0}
.demo-row{display:flex;align-items:center;gap:10px;margin:6px 0;flex-wrap:wrap}
.demo-row input[type=range]{flex:1;min-width:140px;height:32px}
.bars{margin:10px 0}
.barrow{display:grid;grid-template-columns:108px 1fr 52px;align-items:center;gap:8px;margin:7px 0;font-size:13px}
.bartrack{height:16px;background:var(--code);border-radius:8px;overflow:hidden}
.barfill{height:100%;width:50%;background:var(--accent);transition:width .12s}
.barfill.b2{background:#b8860b}
.barval{font-variant-numeric:tabular-nums;text-align:right}
.demo-out{display:grid;grid-template-columns:1fr;gap:4px;font-size:14px;margin:8px 0;font-variant-numeric:tabular-nums}
.demo-btns{display:flex;gap:8px;margin-top:8px}
.demo-btns button{flex:1;min-height:44px;border:1px solid var(--line);background:var(--accent-soft);
  color:var(--fg);border-radius:9px;font-size:14px;cursor:pointer}
.demo-note{font-size:13px;color:var(--muted);margin:10px 0 2px}
@media (min-width:768px){.demo-out{grid-template-columns:1fr 1fr 1fr}.barrow{grid-template-columns:150px 1fr 60px}}
</style>
"""

extra_js = r"""
(function(){
  var th = document.getElementById('thslider');
  if (!th) return;
  var q1 = 10, q2 = 6;

  function sig(x){ return 1/(1+Math.exp(-x)); }

  function draw(){
    var t = parseFloat(th.value);
    document.getElementById('thval').textContent = t.toFixed(2);
    var p1 = sig(t), p2 = 1 - p1;
    var J = q1*p1 + q2*p2;
    var g = p1*(1-p1)*(q1-q2);
    document.getElementById('bar1').style.width = (p1*100).toFixed(1) + '%';
    document.getElementById('bar2').style.width = (p2*100).toFixed(1) + '%';
    document.getElementById('p1').textContent = (p1*100).toFixed(1) + '%';
    document.getElementById('p2').textContent = (p2*100).toFixed(1) + '%';
    document.getElementById('jval').textContent = J.toFixed(2);
    document.getElementById('gval').textContent = g.toFixed(3);
    document.getElementById('satval').textContent = (p1*(1-p1)).toFixed(3);
  }

  th.oninput = draw;
  document.getElementById('stepBtn').onclick = function(){
    var t = parseFloat(th.value);
    var p1 = sig(t);
    var nv = t + 0.1 * p1*(1-p1)*(q1-q2);
    nv = Math.max(-4, Math.min(4, nv));
    th.value = nv.toFixed(2);
    draw();
  };
  document.getElementById('resetBtn').onclick = function(){ th.value = 0; draw(); };
  document.querySelectorAll('.jump').forEach(function(el){
    el.onclick = function(){ show(CHAPTERS.indexOf(el.dataset.goto)); };
  });
  draw();
})();
"""

sections = [
    dict(id="s0", title="知识树与读法", html=TREE),
    dict(id="s1", title="期望与条件期望", html=S1),
    dict(id="s2", title="回报与折扣", html=S2),
    dict(id="s3", title="价值函数", html=S3),
    dict(id="s4", title="贝尔曼期望方程", html=S4),
    dict(id="s5", title="贝尔曼最优方程", html=S5),
    dict(id="s6", title="为什么直接对策略求导", html=S6),
    dict(id="s7", title="log 导数技巧", html=S7),
    dict(id="s8", title="策略梯度定理", html=S8),
    dict(id="s9", title="REINFORCE 算一遍", html=S9),
    dict(id="s10", title="基线", html=S10),
    dict(id="s11", title="为什么这样设计", html=S11),
    dict(id="s12", title="动手：拖 θ 看梯度", html=S12),
    dict(id="s13", title="自检与思考题", html=S13),
]

build(OUT, "贝尔曼方程到策略梯度：一条连续的推导链", "从零推导 · 第 2 页", sections,
      extra_css=extra_css, extra_js=extra_js, root_link="../index.html")
