#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从零推导 03：贝叶斯推断（条件概率 → 贝叶斯公式 → 共轭更新 → 朴素贝叶斯）。"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from deriv_common import build

OUT = "/home/user/Doubao/chats/38452296622422786/AI-点化学习系统/06-从零推导/03-贝叶斯推断.html"

TREE = r"""
<h1>贝叶斯推断：一条被误用最多的公式</h1>
<p>贝叶斯公式本身只有一行代数。真正难的是三件事：<b>为什么分母要那样算</b>、<b>先验到底影响多大</b>、<b>连续情形怎么更新</b>。这一页把这三件事拆开算给你看，每个数都能自己复核。</p>
<div class="chain">
  <div class="node jump" data-goto="s1">条件概率：把“已知”写进分母</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s2">乘法法则与全概率</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s3">贝叶斯公式（三种读法）</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s4">基础比率：一万人里数一遍</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s5">连续情形：Beta 与二项</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s6">序列更新与共轭</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s7">朴素贝叶斯分类器手算</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s8">先验到底多重要</div>
</div>
<div class="box">
<h4>本节不涉及</h4>
<p>MCMC 采样、变分推断、贝叶斯网的结构学习。先把“一个参数怎么被证据更新”这件事打通。</p>
</div>
"""

S1 = r"""
<h1>第 1 节　条件概率：把“已知”写进分母</h1>
<h2>1.1 从频率出发</h2>
<p>扔 100 次骰子，20 次是 6 点。$P(6)=20/100=0.2$。</p>
<p>现在加一个条件：只看“结果是偶数”的那些次。假设 100 次里有 48 次是偶数，其中 20 次是 6。那么在“已知是偶数”的前提下，6 点的比例是</p>
$$P(6\mid \text{偶数})=\frac{20}{48}=0.4167$$
<p>这就是条件概率：<b>把样本空间从“全部”缩到“满足条件的部分”，分母随之改变</b>。</p>
<h2>1.2 定义</h2>
$$P(A\mid B)=\frac{P(A\cap B)}{P(B)},\qquad P(B)&gt;0$$
<p>分母为什么是 $P(B)$？因为条件“已知 B”等价于把所有概率质量按 B 重新归一化。若分母不换成 $P(B)$，剩下的概率之和不等于 1，就不是概率了。</p>
<h2>1.3 一个必须警惕的点</h2>
<p>$P(A\mid B)$ 与 $P(B\mid A)$ 一般不相等，而且可能差得极远。这是贝叶斯公式存在的全部理由。第 4 节的医学检测例子会把这个差异放大到 6 倍。</p>
"""

S2 = r"""
<h1>第 2 节　乘法法则与全概率：贝叶斯公式的两块积木</h1>
<h2>2.1 乘法法则</h2>
<p>由定义直接移项：</p>
$$P(A\cap B)=P(A\mid B)P(B)=P(B\mid A)P(A)$$
<p>两边都等于同一个交集的概率，所以右边两个表达式相等。这一步是纯代数，没有任何假设。</p>
<h2>2.2 全概率公式</h2>
<p>若 $B_1,\dots,B_n$ 把样本空间切成互不重叠的几块（完备事件组），则</p>
$$P(A)=\sum_i P(A\mid B_i)P(B_i)$$
<p>读法：A 的总概率 = 每种情况下 A 的概率，按那种情况的概率加权。</p>
<p>数值例：两个盒子。盒 1 有 3 红 7 蓝，盒 2 有 8 红 2 蓝。随机选一个盒子（各 0.5），再抽一个球，抽到红的概率：</p>
$$P(\text{红})=0.5\cdot0.3+0.5\cdot0.8=0.15+0.4=0.55$$
<h2>2.3 把两块拼起来：贝叶斯公式</h2>
<p>从 $P(A\mid B)P(B)=P(B\mid A)P(A)$ 出发，两边除以 $P(B)$：</p>
<div class="box key">
<h4>贝叶斯公式</h4>
$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$$
</div>
<p>分母 $P(B)$ 通常用全概率展开算：$P(B)=\sum_i P(B\mid A_i)P(A_i)$。这一步就是“把分母补齐”。</p>
<div class="box warn">
<h4>为什么必须先算分母</h4>
<p>分母是“证据出现的总概率”。它保证后验是个合法概率（所有假设的后验加起来为 1）。跳过分母直接拿分子当概率，是最常见的错误，第 4 节会看到后果。</p>
</div>
"""

S3 = r"""
<h1>第 3 节　贝叶斯公式的三种读法</h1>
<h2>3.1 读法一：更新信念</h2>
$$\text{后验}=\frac{\text{似然}\times\text{先验}}{\text{证据}}$$
<p>先验是你看到证据之前的判断，似然是“如果假设成立，这个证据有多常见”，后验是看到证据之后的判断。三者的关系是乘法，不是加法。</p>
<h2>3.2 读法二：比例形式（最常用）</h2>
$$P(A\mid B)\propto P(B\mid A)P(A)$$
<p>比较两个假设时，分母相同，可以扔掉。于是“哪个假设更可能”只取决于分子大小：</p>
$$\frac{P(A_1\mid B)}{P(A_2\mid B)}=\frac{P(B\mid A_1)P(A_1)}{P(B\mid A_2)P(A_2)}$$
<p>这个比值叫后验几率。它说明：<b>证据强不强，要看它是否更能被某个假设解释；先验高低，直接乘进比值里。</b></p>
<h2>3.3 读法三：逆概率</h2>
<p>已知 $P(\text{检测}\mid\text{有病})$，想求 $P(\text{有病}\mid\text{检测})$。这两个方向在医学、法律、风控里经常被混淆，混淆的后果很具体。第 4 节算。</p>
"""

S4 = r"""
<h1>第 4 节　基础比率：在一万人里数一遍</h1>
<p>设某种病在人群中的患病率 1%。检测的灵敏度（有病时检出阳性）90%，假阳性率（没病却阳性）5%。现在某人检测阳性，他真的有病的概率是多少？</p>
<p>很多人会脱口而出“90%”。我们不用公式，直接数人。</p>
<h2>4.1 一万人表</h2>
<table>
<tr><th>人群</th><th>人数</th><th>阳性人数</th><th>阴性人数</th></tr>
<tr><td>有病</td><td>100</td><td>90</td><td>10</td></tr>
<tr><td>没病</td><td>9900</td><td>495</td><td>9405</td></tr>
<tr><td>合计</td><td>10000</td><td>585</td><td>9415</td></tr>
</table>
<p>阳性共 585 人，其中真病的只有 90 人：</p>
$$P(\text{有病}\mid +)=\frac{90}{585}=0.1538$$
<p>不是 90%，是 15.4%。差 5.85 倍，全部来自“没病的人太多，5% 的假阳性凑出了 495 人”。</p>
<h2>4.2 用公式算，核对同一个数</h2>
$$P(D\mid +)=\frac{P(+\mid D)P(D)}{P(+\mid D)P(D)+P(+\mid \lnot D)P(\lnot D)}=\frac{0.9\cdot0.01}{0.9\cdot0.01+0.05\cdot0.99}$$
<details class="fold"><summary>展开：分母逐项算</summary><div class="fold-body">
<p>分子：$0.9\cdot0.01=0.009$。</p>
<p>分母第一项：同样 $0.009$。第二项：$0.05\cdot0.99=0.0495$。</p>
<p>分母合计 $0.0585$。后验 $=0.009/0.0585=0.1538$。</p>
<p>与数人法完全一致（10000 人里 585 阳性，90 真病，比值相同）。</p>
</div></details>
<h2>4.3 这个数字说明什么</h2>
<p>低患病率下，一次阳性检测只能把“1%”提到“15%”，还远不足以确诊。这不是检测不行，是<b>基础比率太低，假阳性把阳性池子填满了</b>。临床上所以要复检、要结合症状，就是在压低假阳性。</p>
<div class="box key">
<h4>一条可迁移的规则</h4>
<p>当“阳性”的绝大多数来自“没病”的人群时，阳性结果的可信度必然很低。判断任何检测、任何告警、任何风控模型时，先问一句：阴性群体有多大。</p>
</div>
"""

S5 = r"""
<h1>第 5 节　连续情形：参数不是“有病/没病”，而是一个比例</h1>
<p>上一节假设是二选一。现在假设要估的是“一枚硬币正面朝上的概率 $\theta$”。$\theta$ 可以是 0 到 1 之间任何数，所以要用密度。</p>
<h2>5.1 连续版本的贝叶斯</h2>
$$p(\theta\mid \text{数据})=\frac{p(\text{数据}\mid\theta)\,p(\theta)}{\int p(\text{数据}\mid\theta)p(\theta)\,d\theta}$$
<p>结构与离散版完全一样，只是把求和换成积分。</p>
<h2>5.2 抛硬币：Beta 先验 + 二项似然</h2>
<p>抛 $n$ 次出现 $h$ 次正面的似然是 $p(h\mid\theta)\propto\theta^h(1-\theta)^{n-h}$。</p>
<p>取先验 $\theta\sim\mathrm{Beta}(a,b)$，密度正比于 $\theta^{a-1}(1-\theta)^{b-1}$。两者相乘：</p>
$$p(\theta\mid h)\propto\theta^{h+a-1}(1-\theta)^{n-h+b-1}$$
<p>这仍是 Beta 分布，只是参数变成 $a+h$ 与 $b+n-h$。所以</p>
<div class="box key">
<h4>共轭更新</h4>
$$\mathrm{Beta}(a,b)\ \xrightarrow{\ \text{观察到 } h \text{ 正 },\ n-h \text{ 反}\ }\ \mathrm{Beta}(a+h,\ b+n-h)$$
</div>
<h2>5.3 算一个具体的</h2>
<p>先验 $\mathrm{Beta}(2,2)$（均值 $2/(2+2)=0.5$，表示“认为大致公平但不确定”）。抛 10 次，7 次正面。</p>
<p>后验 $\mathrm{Beta}(2+7,\ 2+3)=\mathrm{Beta}(9,5)$，均值 $9/(9+5)=9/14=0.6429$。</p>
<details class="fold"><summary>展开：为什么先验取 (2,2) 而不是 (1,1)</summary><div class="fold-body">
<p>$\mathrm{Beta}(1,1)$ 是 [0,1] 上的均匀分布，均值 0.5，但“几乎什么都不知道”，等价于只加了 2 个虚拟观测（1 正 1 反）。</p>
<p>$\mathrm{Beta}(2,2)$ 等价于先看了 4 个虚拟观测（2 正 2 反），比均匀分布更有主张：更相信 $\theta$ 接近 0.5。</p>
<p>“虚拟观测个数”是先验强度，$a+b$ 越大，数据越难推翻它。这个解释很实用：它能让你把先验强度与数据量直接对比。</p>
</div></details>
<h2>5.4 与最大似然的对比</h2>
<p>最大似然只看数据：$\hat\theta=h/n=7/10=0.7$。贝叶斯后验均值 $0.643$，被先验往 0.5 拉了一点。数据越多，拉得越少——第 8 节给数字。</p>
"""

S6 = r"""
<h1>第 6 节　序列更新：今天的后验是明天的先验</h1>
<h2>6.1 一步一步来</h2>
<p>先验 $\mathrm{Beta}(2,2)$。观察到第 1 次是正面 → 后验 $\mathrm{Beta}(3,2)$，均值 $0.6$。</p>
<p>把这个后验当作下一次的先验，再观察到反面 → $\mathrm{Beta}(3,3)$，均值 $0.5$。</p>
<p>再观察 3 次正面 → $\mathrm{Beta}(6,3)$，均值 $0.667$。</p>
<h2>6.2 一次性算，结果一样</h2>
<p>总共观察到 4 正 1 反。直接更新：$\mathrm{Beta}(2+4,\ 2+1)=\mathrm{Beta}(6,3)$。同一步到位，结果完全相同。</p>
<div class="box key">
<h4>这条性质为什么重要</h4>
<p>它说明“顺序无关”。你可以来一条数据更新一次，也可以攒一批再更新；数学上等价。这为在线学习、流式更新提供了合法性。</p>
</div>
<h2>6.3 共轭的实际意义</h2>
<p>如果每次更新都要算一个积分，序列更新会贵得离谱。共轭先验让“更新”退化成“改两个整数”，一次加法就够了。这就是为什么 Beta-二项、Dirichlet-多项、高斯-高斯这几对组合在工程里被反复使用。</p>
<h2>6.4 什么时候没有共轭</p>
<p>大部分模型没有。这时只能近似：MCMC 采样、变分推断，或者用数值积分。第 5 节能算得这么干净，是因为选了共轭的一对——这是建模时的主动选择，不是自然规律。</p>
"""

S7 = r"""
<h1>第 7 节　朴素贝叶斯分类器：把公式用到一篇邮件上</h1>
<h2>7.1 模型</h2>
$$P(\text{类别}\mid \text{词}_1,\dots,\text{词}_k)\propto P(\text{类别})\prod_{i}P(\text{词}_i\mid \text{类别})$$
<p>“朴素”就是指这个连乘假设：给定类别后，各个词出现与否相互独立。现实中不成立（“机器”和“学习”经常一起出现），但分类效果常常还行。</p>
<h2>7.2 一份小训练集</h2>
<p>垃圾邮件 5 封，正常邮件 4 封。统计两个词的出现封数：</p>
<table>
<tr><th>词</th><th>垃圾邮件中出现</th><th>正常邮件中出现</th></tr>
<tr><td>免费</td><td>4 / 5</td><td>1 / 4</td></tr>
<tr><td>会议</td><td>2 / 5</td><td>3 / 4</td></tr>
</table>
<p>先验：$P(\text{垃圾})=5/9=0.5556$，$P(\text{正常})=4/9=0.4444$。</p>
<h2>7.3 加一平滑</h2>
<p>直接除会出 0：若某词在垃圾邮件里从未出现，连乘就变 0，整封邮件的后验被一票否决——这显然不合理。</p>
<p>每个词有“出现/不出现”两种取值，取 $\alpha=1$：</p>
$$P(\text{免费}\mid \text{垃圾})=\frac{4+1}{5+2}=\frac{5}{7}=0.7143$$
$$P(\text{会议}\mid \text{垃圾})=\frac{2+1}{5+2}=\frac{3}{7}=0.4286$$
$$P(\text{免费}\mid \text{正常})=\frac{1+1}{4+2}=\frac{2}{6}=0.3333$$
$$P(\text{会议}\mid \text{正常})=\frac{3+1}{4+2}=\frac{4}{6}=0.6667$$
<h2>7.4 判一封同时含“免费”和“会议”的邮件</h2>
<details class="fold"><summary>展开：两项分数与归一化</summary><div class="fold-body">
<p>垃圾得分：$0.5556\cdot0.7143\cdot0.4286$。</p>
<p>先算后两个：$0.7143\cdot0.4286=0.3061$。乘先验：$0.5556\cdot0.3061=0.1701$。</p>
<p>正常得分：$0.3333\cdot0.6667=0.2222$，乘 $0.4444$：$0.0988$。</p>
<p>归一化：$P(\text{垃圾}\mid \text{两词})=0.1701/(0.1701+0.0988)=0.1701/0.2689=0.6326$。</p>
<p>判为垃圾邮件，置信度约 63.3%。注意它不是 90%——因为“会议”这个词在正常邮件里更常见，把分数拉回去了。</p>
</div></details>
<h2>7.5 为什么要取对数</h2>
<p>一封邮件几百个词，每个概率都小于 1，连乘很快下溢到 0。改成对数相加：$\log P=\log P(\text{类别})+\sum_i\log P(\text{词}_i\mid\text{类别})$，数值稳定且把乘法变成加法。</p>
"""

S8 = r"""
<h1>第 8 节　先验到底多重要：用数字比</h1>
<p>“先验会不会带偏结论”这个问题，答案取决于先验强度与数据量的比例。给三组数字。</p>
<h2>8.1 数据很少时</h2>
<p>先验 $\mathrm{Beta}(2,2)$（强度 4，等价 2 正 2 反），数据 2 次正面。</p>
<p>后验 $\mathrm{Beta}(4,2)$，均值 $4/6=0.667$。最大似然给 $2/2=1.0$。</p>
<p>差 0.333。先验把“100% 正面”这个明显过头的判断压了下来——这正是我们想要的。</p>
<h2>8.2 数据很多时</h2>
<p>同样先验，数据 200 次正面。</p>
<p>后验 $\mathrm{Beta}(202,2)$，均值 $202/204=0.9902$。最大似然给 1.0。差 0.0098。</p>
<p>先验几乎被完全冲掉。数据量是先验强度的 50 倍，先验的影响可以忽略。</p>
<h2>8.3 一句话规则</h2>
<p>先验强度 $a+b$ 与数据量 $n$ 的比值决定先验还剩多少影响。$n\ll a+b$ 时先验主导，$n\gg a+b$ 时数据主导。所以“用了一个错的先验”在数据充足时不是致命问题，在数据稀少时是。</p>
<div class="box warn">
<h4>反过来也成立</h4>
<p>数据稀少时的“无信息先验”也不是中立的。$\mathrm{Beta}(1,1)$ 均匀先验在均值上是中立的，但在“相信极端值”这件事上并不中立。没有真正无信息的先验，只有影响大小的差别。</p>
</div>
"""

S9 = r"""
<h1>第 9 节　贝叶斯区间与频率区间：同一句话，两种含义</h1>
<p>两个学派都给出“95% 区间”，但含义不同，混用会出错。</p>
<h2>9.1 频率派：置信区间</h2>
<p>“如果重复采样很多次，95% 的区间会覆盖真值。”这句话描述的是<b>程序的性质</b>，不是“这个参数有 95% 概率落在区间内”。参数在频率派眼里是固定的，不是随机变量。</p>
<h2>9.2 贝叶斯派：可信区间</h2>
<p>“给定数据和先验，参数落在该区间的后验概率是 95%。”这句话直接对参数说概率，因为贝叶斯派把参数当作随机变量。</p>
<h2>9.3 数值上会接近，但解释不同</h2>
<p>数据多、先验弱时，两者数值常常几乎一样。差别主要体现在解释与小样本行为上——这也正是两派争论的焦点。</p>
<h2>9.4 实践上怎么选</h2>
<ul>
<li>只关心预测效果、数据量大 → 差别不大，看交叉验证更实在。</li>
<li>需要表达“我对这个参数有多确定”、要融入领域知识 → 贝叶斯更自然。</li>
<li>要严格的频率保证（如工业质检的合格率下限）→ 频率方法更合适。</li>
</ul>
"""

S10 = r"""
<h1>第 10 节　动手：拖三个滑块看后验怎么变</h1>
<p>这个演示是第 4 节那张表的活动版。你改患病率、灵敏度、假阳性率，右侧直接给出后验和“1000 人里真病几个”。</p>
<div class="demo">
  <div class="demo-row"><label>患病率 <b id="pv">1.0%</b></label>
  <input id="pslider" type="range" min="0.1" max="10" step="0.1" value="1"></div>
  <div class="demo-row"><label>灵敏度 <b id="sv">90%</b></label>
  <input id="sslider" type="range" min="50" max="100" step="1" value="90"></div>
  <div class="demo-row"><label>假阳性率 <b id="fv">5%</b></label>
  <input id="fslider" type="range" min="1" max="20" step="1" value="5"></div>

  <div class="bars">
    <div class="barrow"><span class="barlab">阳性者中真病</span>
      <div class="bartrack"><div class="barfill" id="barT"></div></div>
      <span class="barval" id="pt">15.4%</span></div>
    <div class="barrow"><span class="barlab">阳性者中误报</span>
      <div class="bartrack"><div class="barfill b2" id="barF"></div></div>
      <span class="barval" id="pf">84.6%</span></div>
  </div>

  <div class="demo-out">
    <div>每 1000 人阳性：<b id="numPos">59</b> 人</div>
    <div>其中真病：<b id="numTrue">9.0</b> 人</div>
    <div>其中误报：<b id="numFalse">49.5</b> 人</div>
  </div>
  <p class="demo-note">假设人群 1000 人，按 $P(D\mid+)=\dfrac{se\cdot p}{se\cdot p+(1-sp)(1-p)}$ 计算，$se$=灵敏度，$sp$=特异度=1−假阳性率。</p>
</div>
<h2>看什么</h2>
<ul>
<li>把患病率拖到 0.1%：后验会掉到个位数百分比。这就是低基础比率的威力。</li>
<li>把假阳性率拖到 1%：后验大幅回升。改善检测的“误报”比改善“漏报”更值钱——因为误报来自人数庞大的健康群体。</li>
<li>把患病率拖到 10% 再看：假阳性率 5% 时后验能到 60% 以上。同一个检测，在不同人群里的可信度完全不同。</li>
</ul>
"""

S11 = r"""
<h1>第 11 节　自检与思考题</h1>
<h2>11.1 能不能一句话复述</h2>
<ul>
<li>条件概率把样本空间按“已知”重新归一化。</li>
<li>贝叶斯公式 = 似然 × 先验 / 证据，分母靠全概率补齐。</li>
<li>基础比率低时，阳性结果的可信度会被大量假阳性拉垮。</li>
<li>Beta 先验 + 二项数据 → Beta 后验，更新只是改两个整数。</li>
<li>先验影响 ∝ 先验强度与数据量之比。</li>
</ul>
<h2>11.2 思考题</h2>
<details class="fold"><summary>题 1：患病率 1%、灵敏度 90%、假阳性 10%，阳性时患病概率是多少</summary><div class="fold-body">
<p>分子 $0.9\cdot0.01=0.009$。分母 $=0.009+0.1\cdot0.99=0.009+0.099=0.108$。</p>
<p>后验 $=0.009/0.108=0.0833$，约 8.3%。假阳性率从 5% 翻倍到 10%，后验从 15.4% 掉到 8.3%。</p>
</div></details>
<details class="fold"><summary>题 2：先验 $\mathrm{Beta}(1,1)$，观察 3 正 1 反，后验均值是多少</summary><div class="fold-body">
<p>后验 $\mathrm{Beta}(4,2)$，均值 $4/6=0.667$。最大似然是 $3/4=0.75$。先验（等价 1 正 1 反）把估计往 0.5 拉了 0.083。</p>
</div></details>
<details class="fold"><summary>题 3：为什么朴素贝叶斯要加平滑</summary><div class="fold-body">
<p>不加平滑时，某词在某个类别中从未出现会让该类的联合概率变成 0。乘法里一个 0 就毁掉全部证据——即使其他 20 个词都强烈支持该类。加一平滑给每个取值加一个虚拟计数，避免“一次没见到就判死刑”。</p>
</div></details>
<details class="fold"><summary>题 4：$P(A\mid B)$ 与 $P(B\mid A)$ 差得很大，这在现实中对应什么误判</summary><div class="fold-body">
<p>典型是“检察官谬误”：把“无辜者匹配到证据的概率很小”当成“匹配到证据者无辜的概率很小”，两者差别正比于先验比率。在医学上就是“灵敏度”与“阳性预测值”的混淆。任何“命中率 99% 所以几乎可以确定”的说法，都要先问阴性群体有多大。</p>
</div></details>
<details class="fold"><summary>题 5：为什么“今天的后验是明天的先验”在工程上很值钱</summary><div class="fold-body">
<p>它让更新可以做成流式、增量，不需要重算全部数据。存储上只需保存后验参数（如 Beta 的两个数），不必保存原始数据。这在推荐、风控、在线广告里是刚需。代价是：一旦模型结构变了，共轭关系失效，这套优势就没了。</p>
</div></details>
<h2>11.3 下一步</h2>
<p>去读《机器学习》第 7 章（贝叶斯分类器，含 EM 与朴素贝叶斯的完整推导），再看《人工智能：现代方法》第 12—13 章（概率推理与贝叶斯网）。这一页给的是“单参数如何被证据更新”，贝叶斯网给的是“多变量如何联合更新”，两者是同一公式的两次放大。</p>
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
.demo-row label{font-size:13.5px;min-width:96px}
.demo-row input[type=range]{flex:1;min-width:150px;height:32px}
.bars{margin:12px 0 6px}
.barrow{display:grid;grid-template-columns:104px 1fr 52px;align-items:center;gap:8px;margin:7px 0;font-size:13px}
.bartrack{height:16px;background:var(--code);border-radius:8px;overflow:hidden}
.barfill{height:100%;width:15%;background:var(--accent);transition:width .12s}
.barfill.b2{background:#b8860b}
.barval{font-variant-numeric:tabular-nums;text-align:right}
.demo-out{display:grid;grid-template-columns:1fr;gap:4px;font-size:14px;margin:8px 0;font-variant-numeric:tabular-nums}
.demo-note{font-size:13px;color:var(--muted);margin:10px 0 2px}
@media (min-width:768px){.demo-out{grid-template-columns:1fr 1fr 1fr}.barrow{grid-template-columns:150px 1fr 60px}}
</style>
"""

extra_js = r"""
(function(){
  var ps = document.getElementById('pslider');
  if (!ps) return;
  var ss = document.getElementById('sslider'), fs = document.getElementById('fslider');

  function draw(){
    var p = parseFloat(ps.value) / 100;
    var se = parseFloat(ss.value) / 100;
    var fpr = parseFloat(fs.value) / 100;
    var sp = 1 - fpr;

    document.getElementById('pv').textContent = (p*100).toFixed(1) + '%';
    document.getElementById('sv').textContent = (se*100).toFixed(0) + '%';
    document.getElementById('fv').textContent = (fpr*100).toFixed(0) + '%';

    var num = se * p;
    var den = num + fpr * (1 - p);
    var post = den > 0 ? num / den : 0;

    var posPer = den * 1000;          // 每 1000 人中的阳性数
    var truePos = num * 1000;
    var falsePos = fpr * (1 - p) * 1000;

    document.getElementById('barT').style.width = (post*100).toFixed(1) + '%';
    document.getElementById('barF').style.width = ((1-post)*100).toFixed(1) + '%';
    document.getElementById('pt').textContent = (post*100).toFixed(1) + '%';
    document.getElementById('pf').textContent = ((1-post)*100).toFixed(1) + '%';
    document.getElementById('numPos').textContent = posPer.toFixed(0);
    document.getElementById('numTrue').textContent = truePos.toFixed(1);
    document.getElementById('numFalse').textContent = falsePos.toFixed(1);
  }

  ps.oninput = draw; ss.oninput = draw; fs.oninput = draw;
  document.querySelectorAll('.jump').forEach(function(el){
    el.onclick = function(){ show(CHAPTERS.indexOf(el.dataset.goto)); };
  });
  draw();
})();
"""

sections = [
    dict(id="s0", title="知识树与读法", html=TREE),
    dict(id="s1", title="条件概率", html=S1),
    dict(id="s2", title="乘法法则与全概率", html=S2),
    dict(id="s3", title="贝叶斯公式的三种读法", html=S3),
    dict(id="s4", title="基础比率：一万人里数一遍", html=S4),
    dict(id="s5", title="连续情形：Beta 更新", html=S5),
    dict(id="s6", title="序列更新与共轭", html=S6),
    dict(id="s7", title="朴素贝叶斯手算", html=S7),
    dict(id="s8", title="先验到底多重要", html=S8),
    dict(id="s9", title="两种区间", html=S9),
    dict(id="s10", title="动手：拖滑块看后验", html=S10),
    dict(id="s11", title="自检与思考题", html=S11),
]

build(OUT, "贝叶斯推断：一条被误用最多的公式", "从零推导 · 第 3 页", sections,
      extra_css=extra_css, extra_js=extra_js, root_link="../index.html")
