#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从零推导 01：反向传播（链式法则 → 计算图 → 两层网络手算）。"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from deriv_common import build

OUT = "/home/user/Doubao/chats/38452296622422786/AI-点化学习系统/06-从零推导/01-反向传播.html"

TREE = r"""
<h1>反向传播：从链式法则到每一层该改多少</h1>
<p>这一节只做一件事：把“损失对每个权重的导数”算出来，并且每一步都能自己复核。上面那张图是依赖关系，点任意一个方块可以直接跳到那一节——如果某一节读不下去，说明它上面那层还没吃透。</p>
<div class="chain">
  <div class="node jump" data-goto="s1">导数：变化率</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s2">向量与矩阵（现场补）</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s3">复合函数：链式法则</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s4">多元：偏导与路径求和</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s5">计算图：把公式画成图</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s6">反向传播：倒着走一遍</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s7">两层网络手算（数字全给）</div>
  <div class="arrow">↓</div>
  <div class="node jump" data-goto="s8">为什么这么设计</div>
</div>
<div class="box">
<h4>本节不涉及</h4>
<p>卷积、残差连接、优化器（Adam）、批量归一化。这些是在这条主线之上加的东西，先把“梯度怎么来的”打通再看它们，否则只会记公式。</p>
</div>
<div class="box warn">
<h4>读法</h4>
<p>不要跳过推导框。每个折叠框展开后是完整代数过程，一行一行对着纸写。约 2.5 小时能读完并做完手算。</p>
</div>
"""

S1 = r"""
<h1>第 1 节　导数：一个数怎么随另一个数变</h1>
<p>你学过一个定义：</p>
$$f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}$$
<p>这个式子说的是：让输入动一点点 $h$，看输出动了多少，两者相除。它是“变化率”。</p>
<h2>先把“导数”变成能算的数</h2>
<p>取 $f(x)=x^2$，在 $x=3$ 处。不要用公式，用定义硬算一次：</p>
<details class="fold"><summary>展开：用定义算 $f'(3)$</summary><div class="fold-body">
<p>$f(3)=9$。取 $h=0.01$：$f(3.01)=9.0601$，差是 $0.0601$，商是 $6.01$。</p>
<p>取 $h=0.001$：$f(3.001)=9.006001$，差 $0.006001$，商 $6.001$。</p>
<p>取 $h=0.0001$：商 $6.0001$。h 越小商越靠近 6，所以 $f'(3)=6$。</p>
<p>用公式验：$f'(x)=2x$，$f'(3)=6$。一致。</p>
</div></details>
<p>这个“取小 h 算差商”的动作，后面会再出现一次——用来检查我们手推的梯度对不对，叫数值梯度检验。它不是装饰，是唯一能证伪自己推导的手段。</p>
<div class="box key">
<h4>记住这一条</h4>
<p>导数是“输入变一点点，输出变多少”的比例。链式法则要回答的就是：中间隔了好几层时，这个比例怎么乘起来。</p>
</div>
"""

S2 = r"""
<h1>第 2 节　向量与矩阵（现场补，不假设你学过）</h1>
<p>大一上来就缺这块，这里只补后面真正用得上的四件：向量、内积、矩阵乘法、转置。补完就够推反向传播了。</p>
<h2>2.1 向量就是一组数</h2>
<p>$x=(x_1,x_2,x_3)$ 是一个三维向量，可以理解为“三个特征”。</p>
<p><b>内积</b>（点积）：对应位置相乘再相加。</p>
$$x\cdot w = x_1w_1+x_2w_2+x_3w_3$$
<p>数值例：$x=(1,2,3)$，$w=(4,5,6)$，内积 $=1\cdot4+2\cdot5+3\cdot6=4+10+18=32$。</p>
<div class="box">
<h4>内积在神经网络里是什么</h4>
<p>神经元做的事就是内积加偏置：$z=w\cdot x+b$。所以“一个神经元的输出”= 一次内积 + 一次加法。</p>
</div>

<h2>2.2 矩阵是“一批向量排成矩形”</h2>
<p>一个 $2\times3$ 矩阵有两行三列：</p>
$$A=\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}$$
<p><b>转置</b> $A^\top$ 是行列互换，$2\times3$ 变成 $3\times2$：</p>
$$A^\top=\begin{pmatrix}1&4\\2&5\\3&6\end{pmatrix}$$
<p>记忆法：转置就是“行变列”。第一行 $(1,2,3)$ 变成第一列 $(1,2,3)^\top$。</p>

<h2>2.3 矩阵乘法：左边列数必须等于右边行数</h2>
<p>$(m\times n)\cdot(n\times p)=(m\times p)$。中间那个 $n$ 必须对上，对不上就是错的，没有例外。</p>
<p>手算一个：$A$ 是上面的 $2\times3$，$B=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}$ 是 $3\times2$。结果应是 $2\times2$。</p>
<details class="fold"><summary>展开：逐项算 $AB$</summary><div class="fold-body">
<p>结果第 1 行第 1 列 = $A$ 第 1 行与 $B$ 第 1 列的内积：$1\cdot1+2\cdot0+3\cdot1=4$。</p>
<p>第 1 行第 2 列 = $A$ 第 1 行与 $B$ 第 2 列：$1\cdot0+2\cdot1+3\cdot1=5$。</p>
<p>第 2 行第 1 列：$4\cdot1+5\cdot0+6\cdot1=10$。</p>
<p>第 2 行第 2 列：$4\cdot0+5\cdot1+6\cdot1=11$。</p>
$$AB=\begin{pmatrix}4&5\\10&11\end{pmatrix}$$
<p>检查形状：$2\times3$ 乘 $3\times2$ 得 $2\times2$，对上了。</p>
</div></details>

<h2>2.4 为什么反向传播里全是矩阵乘法</h2>
<p>一层有 100 个输入、50 个神经元，权重就是一个 $50\times100$ 矩阵。一次前向是矩阵乘向量；一次反向是转置矩阵乘向量。所以“懂矩阵乘法形状规则”是读懂反向传播公式的最低门槛。</p>
<div class="box warn">
<h4>只需记住这一条形状规则</h4>
<p>前向里 $W$ 的形状是（输出维度，输入维度）。反向算“损失对输入的梯度”时用的是 $W^\top$，形状反过来。写错形状是这类代码最常见的错。</p>
</div>
"""

S3 = r"""
<h1>第 3 节　复合函数：链式法则</h1>
<p>神经网络是“函数套函数”。$y=(2x+1)^2$ 里，先算 $u=2x+1$，再算 $y=u^2$。</p>
<h2>3.1 一层一层剥</h2>
<p>$y$ 对 $x$ 的变化率，应该等于“$y$ 对 $u$ 的变化率”乘“$u$ 对 $x$ 的变化率”：</p>
$$\frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dx}$$
<p>分别算：$\frac{dy}{du}=2u$，$\frac{du}{dx}=2$。合起来 $\frac{dy}{dx}=4u=4(2x+1)=8x+4$。</p>
<h2>3.2 代数字验一遍</h2>
<p>取 $x=1$：$u=3$，$y=9$。公式给 $\frac{dy}{dx}=8\cdot1+4=12$。</p>
<details class="fold"><summary>展开：用差商核对 12 这个数</summary><div class="fold-body">
<p>取 $h=0.001$：$x=1.001$，$u=3.002$，$y=3.002^2=9.012004$。</p>
<p>差 $=9.012004-9=0.012004$，除以 $0.001$ 得 $12.004$。</p>
<p>与解析值 12 只差 $0.004$，误差来自 $h$ 不够小（差商近似有 $O(h)$ 误差）。这说明 12 是对的。</p>
</div></details>
<h2>3.3 三层套起来会怎样</h2>
<p>$y=((2x+1)^2)^3$。设 $u=2x+1$，$v=u^2$，$y=v^3$：</p>
$$\frac{dy}{dx}=\frac{dy}{dv}\cdot\frac{dv}{du}\cdot\frac{du}{dx}=3v^2\cdot2u\cdot2=12\,u\,v^2$$
<p>代入 $x=1$：$u=3$，$v=9$，$\frac{dy}{dx}=12\cdot3\cdot81=2916$。</p>
<p>规律已经很清楚了：<b>每一层的局部导数乘起来</b>。有几层就乘几次。反向传播只是把这件事在多变量、有分支的情形下系统地做一遍。</p>
<div class="box key">
<h4>链式法则的一句话版本</h4>
<p>总变化率 = 沿路径把每一小段的局部变化率乘起来。</p>
</div>
"""

S4 = r"""
<h1>第 4 节　多元情形：偏导与“路径求和”</h1>
<p>神经网络里的变量不止一个，而且一个变量常常影响多个中间量。这时链式法则多出一条规则：<b>把每条路径的乘积加起来</b>。</p>
<h2>4.1 偏导是什么</h2>
<p>$f(u,v)=u^2+v$。对 $u$ 求偏导，就是把 $v$ 当常数：$\frac{\partial f}{\partial u}=2u$；对 $v$：$\frac{\partial f}{\partial v}=1$。</p>
<p>“偏”的意思是“只动这一个，别的按住不动”。</p>
<h2>4.2 一个变量走两条路</h2>
<p>设 $u=x+y$，$v=xy$，$f=u^2+v$。$x$ 同时影响 $u$ 和 $v$，所以 $f$ 通过两条路受 $x$ 影响：</p>
$$\frac{\partial f}{\partial x}=\frac{\partial f}{\partial u}\frac{\partial u}{\partial x}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial x}$$
<ul>
<li>$\frac{\partial f}{\partial u}=2u$，$\frac{\partial u}{\partial x}=1$，第一条路贡献 $2u$。</li>
<li>$\frac{\partial f}{\partial v}=1$，$\frac{\partial v}{\partial x}=y$，第二条路贡献 $y$。</li>
</ul>
<p>合起来 $\frac{\partial f}{\partial x}=2u+y=2(x+y)+y=2x+3y$。</p>
<details class="fold"><summary>展开：取 $x=1,y=2$ 核对</summary><div class="fold-body">
<p>$u=3$，$v=2$，$f=9+2=11$。公式给 $\frac{\partial f}{\partial x}=2\cdot3+2=8$。</p>
<p>数值核对：$x=1.001$ 时 $u=3.001$，$v=2.002$，$f=9.006001+2.002=11.008001$。差 $0.008001$，除以 $0.001$ 得 $8.001$。与 8 一致。</p>
</div></details>
<h2>4.3 这条规则就是反向传播的全部内容</h2>
<p>把上面两件事合起来：局部导数是“这一段的变化率”，路径求和是“同一个变量被多个下游影响时加总”。计算图上的反向传播，做的就是<b>对每个节点，把它所有下游传来的梯度加起来，再乘自己的局部导数</b>。</p>
<div class="box warn">
<h4>最容易错的地方</h4>
<p>忘记“加总”。一个变量被两处用到时，只算了其中一条路径的梯度，数值会明显偏小。后面第 5 节的例子里 $b$ 就走两条路，这是专门放进去的陷阱。</p>
</div>
"""

S5 = r"""
<h1>第 5 节　计算图：把公式画成图</h1>
<p>计算图把“先算什么、后算什么”画出来。节点是数，边是依赖关系。有了它，反向传播就变成“沿着边倒着走”。</p>
<h2>5.1 一个具体的表达式</h2>
$$e=(a+b)\cdot(b+1)$$
<p>取 $a=2$，$b=1$。先拆中间量：</p>
<ul>
<li>$c=a+b=3$</li>
<li>$d=b+1=2$</li>
<li>$e=c\cdot d=6$</li>
</ul>
<p>依赖关系：$a\to c$，$b\to c$，$b\to d$，$c\to e$，$d\to e$。注意 $b$ 有两条出边。</p>
<h2>5.2 每条边的局部导数</h2>
<p>边上的导数就是“上游动一点，下游动多少”：</p>
<table>
<tr><th>边</th><th>局部导数</th><th>值（在 a=2,b=1）</th></tr>
<tr><td>$a\to c$</td><td>$\partial c/\partial a=1$</td><td>1</td></tr>
<tr><td>$b\to c$</td><td>$\partial c/\partial b=1$</td><td>1</td></tr>
<tr><td>$b\to d$</td><td>$\partial d/\partial b=1$</td><td>1</td></tr>
<tr><td>$c\to e$</td><td>$\partial e/\partial c=d$</td><td>2</td></tr>
<tr><td>$d\to e$</td><td>$\partial e/\partial d=c$</td><td>3</td></tr>
</table>
<h2>5.3 倒着走：每个节点收集下游梯度再乘局部导数</h2>
<p>从 $e$ 开始，$\frac{\partial e}{\partial e}=1$。</p>
<details class="fold"><summary>展开：逐节点倒推</summary><div class="fold-body">
<p><b>$c$ 节点</b>：只有一条下游边来自 $e$，梯度 $=1\cdot\frac{\partial e}{\partial c}=1\cdot2=2$。</p>
<p><b>$d$ 节点</b>：梯度 $=1\cdot\frac{\partial e}{\partial d}=1\cdot3=3$。</p>
<p><b>$a$ 节点</b>：只有 $c$ 一条路径，梯度 $=2\cdot1=2$。</p>
<p><b>$b$ 节点</b>：两条路径，先各自算再相加：经 $c$ 是 $2\cdot1=2$，经 $d$ 是 $3\cdot1=3$，合计 $5$。</p>
</div></details>
<p>结论：$\frac{\partial e}{\partial a}=2$，$\frac{\partial e}{\partial b}=5$。</p>
<h2>5.4 数值核对</h2>
<p>对 $a$ 加 $0.001$：$c=3.001$，$e=3.001\cdot2=6.002$，差 $0.002$，商 $2.0$。对 $b$ 加 $0.001$：$c=3.001$，$d=2.001$，$e=3.001\cdot2.001=6.005001$，差 $0.005001$，商 $5.001$。两个都对得上。</p>
<div class="box key">
<h4>反向传播的三个动作</h4>
<p>1）前向：算出每个节点的值。2）倒着走：每个节点把下游传来的梯度<b>求和</b>。3）乘上该节点的局部导数，传给上游。</p>
</div>
"""

S6 = r"""
<h1>第 6 节　反向传播：倒着走一遍就是全部</h1>
<p>把第 5 节的做法换个说法：网络是一张计算图，损失在最后，权重在最前。反向传播 = 从损失出发，沿边反向传梯度。</p>
<h2>6.1 通用公式</h2>
<p>设某个权重 $w$ 通过一条链 $w\to z_1\to z_2\to\cdots\to L$ 影响损失，则</p>
$$\frac{\partial L}{\partial w}=\frac{\partial L}{\partial z_n}\cdot\frac{\partial z_n}{\partial z_{n-1}}\cdots\frac{\partial z_1}{\partial w}$$
<p>如果 $w$ 出现在多个位置（比如被两层共享），把每条链的结果加起来。</p>
<h2>6.2 为什么必须保存前向的中间值</h2>
<p>看上面公式：算 $\frac{\partial L}{\partial w}$ 时要用到 $\frac{\partial z_2}{\partial z_1}$，而这个局部导数常常依赖前向时的中间值。例如 $z_2=\sigma(z_1)$ 的导数是 $\sigma(z_1)(1-\sigma(z_1))$，必须知道 $z_1$ 或 $\sigma(z_1)$ 才能算。</p>
<p>所以框架里 `y = x * w` 这样的操作会把输入存下来。这是内存开销的主要来源。</p>
<h2>6.3 为什么不是“正向一遍”</h2>
<p>正向模式求导（前向 AD）一次能算出“一个输入对所有输出的影响”。网络里输入成百万、输出只有一个标量损失，正向模式要跑上百万遍；反向模式一遍就能得到全部输入的梯度。这就是“反向”被选中的唯一原因——不是因为它更聪明，是因为形状对得上。</p>
<div class="box">
<h4>一句话</h4>
<p>输出少、输入多，就从输出往回推。这是反向传播的全部设计动机。</p>
</div>
"""

S7 = r"""
<h1>第 7 节　两层网络手算：每个数都给出来</h1>
<p>现在把前面所有东西合成一个最小网络。所有数字都会算到最后一步，你可以逐行核对。</p>
<h2>7.1 网络结构</h2>
<p>输入一个数 $x$，一个隐层神经元（ReLU），一个输出神经元（线性），损失用平方误差。</p>
$$z=w_1x+b_1,\qquad a=\mathrm{ReLU}(z)=\max(0,z),\qquad o=w_2a+b_2,\qquad L=\tfrac12(o-y)^2$$
<p>参数：$w_1=2$，$b_1=-1$，$w_2=3$，$b_2=0.5$。数据：$x=1$，目标 $y=0$。</p>
<h2>7.2 前向</h2>
<p>$z=2\cdot1+(-1)=1$；$a=\max(0,1)=1$；$o=3\cdot1+0.5=3.5$；$L=\frac12(3.5-0)^2=\frac12\cdot12.25=6.125$。</p>
<h2>7.3 反向：从 L 出发</h2>
<details class="fold"><summary>展开：第 1 步，损失对输出</summary><div class="fold-body">
$$\frac{\partial L}{\partial o}=o-y=3.5-0=3.5$$
<p>理由：$L=\frac12(o-y)^2$，对 $o$ 求导得 $(o-y)$。这是平方误差最常见的一个导数，值得直接记住。</p>
</div></details>
<details class="fold"><summary>展开：第 2 步，输出层的两个参数</summary><div class="fold-body">
<p>$o=w_2a+b_2$，所以 $\frac{\partial o}{\partial w_2}=a=1$，$\frac{\partial o}{\partial b_2}=1$。</p>
$$\frac{\partial L}{\partial w_2}=\frac{\partial L}{\partial o}\cdot a=3.5\cdot1=3.5$$
$$\frac{\partial L}{\partial b_2}=3.5\cdot1=3.5$$
</div></details>
<details class="fold"><summary>展开：第 3 步，梯度传回隐层输出</summary><div class="fold-body">
$$\frac{\partial L}{\partial a}=\frac{\partial L}{\partial o}\cdot w_2=3.5\cdot3=10.5$$
<p>注意这里乘的是 $w_2$，因为 $o$ 对 $a$ 的局部导数是 $w_2$。</p>
</div></details>
<details class="fold"><summary>展开：第 4 步，穿过 ReLU</summary><div class="fold-body">
<p>$\mathrm{ReLU}$ 在 $z=1&gt;0$ 时导数为 1（在 $z&lt;0$ 时为 0）。</p>
$$\frac{\partial L}{\partial z}=\frac{\partial L}{\partial a}\cdot 1=10.5$$
<p>若 $z$ 是负数，这一步会得到 0，梯度就此断掉——这就是后面要讲的“死亡 ReLU”。</p>
</div></details>
<details class="fold"><summary>展开：第 5 步，隐层的两个参数</summary><div class="fold-body">
<p>$z=w_1x+b_1$，$\frac{\partial z}{\partial w_1}=x=1$，$\frac{\partial z}{\partial b_1}=1$。</p>
$$\frac{\partial L}{\partial w_1}=10.5\cdot1=10.5,\qquad \frac{\partial L}{\partial b_1}=10.5\cdot1=10.5$$
</div></details>
<h2>7.4 数值梯度检验</h2>
<p>拿 $w_1$ 检查。取 $\epsilon=0.0001$，把 $w_1$ 变成 $2.0001$：</p>
<p>$z=2.0001\cdot1-1=1.0001$，$a=1.0001$，$o=3\cdot1.0001+0.5=3.5003$，$L=\frac12\cdot3.5003^2=6.12605$。</p>
<p>差 $=6.12605-6.125=0.00105$，除以 $0.0001$ 得 $10.5$。与解析梯度 $10.5$ 一致。</p>
<h2>7.5 走一步梯度下降</h2>
<p>取学习率 $\alpha=0.01$：</p>
<ul>
<li>$w_1\leftarrow 2-0.01\cdot10.5=1.895$</li>
<li>$b_1\leftarrow -1-0.01\cdot10.5=-1.105$</li>
<li>$w_2\leftarrow 3-0.01\cdot3.5=2.965$</li>
<li>$b_2\leftarrow 0.5-0.01\cdot3.5=0.465$</li>
</ul>
<p>新前向：$z=1.895-1.105=0.79$，$a=0.79$，$o=2.965\cdot0.79+0.465=2.808$，$L=\frac12\cdot2.808^2\approx3.94$。损失从 6.125 降到 3.94。</p>
<div class="box warn">
<h4>如果学习率取 0.1 会怎样</h4>
<p>$w_1=2-1.05=0.95$，$b_1=-1-1.05=-2.05$，新 $z=0.95-2.05=-1.1$，ReLU 输出 0。这个神经元的梯度从此恒为 0，再也学不动——一步就把自己打死了。这不是理论吓唬人，加大学习率跑一次就能看到。</p>
</div>
"""

S8 = r"""
<h1>第 8 节　为什么这样设计</h1>
<p>推导完了，还有几个“不这么做会怎样”的问题需要回答。这些是你自己会想到但书上常不写的。</p>
<h2>8.1 为什么用 ReLU 而不是 sigmoid</h2>
<p>sigmoid 的导数是 $\sigma(z)(1-\sigma(z))$，最大值只有 $0.25$（在 $z=0$ 处）。十层连乘，梯度上界是 $0.25^{10}\approx9.5\times10^{-7}$——前面的层几乎收不到信号。</p>
<details class="fold"><summary>展开：$0.25^{10}$ 到底多小</summary><div class="fold-body">
<p>$0.25^2=0.0625$，$0.25^4=0.0039$，$0.25^8=1.53\times10^{-5}$，$0.25^{10}=9.5\times10^{-7}$。</p>
<p>意思是：输出层梯度若是 1，传到第 1 层只剩约百万分之一。学习率再大也救不回来。这就是梯度消失的算术版本。</p>
</div></details>
<p>ReLU 在正区间的导数是 1，连乘不衰减。代价是负区间导数为 0，可能“死掉”——所以有 LeakyReLU、ELU 这类改法。</p>
<h2>8.2 为什么损失要用平方而不是绝对值</h2>
<p>$L=\frac12(o-y)^2$ 的导数是 $(o-y)$，误差越大梯度越大，收敛快；绝对值损失 $|o-y|$ 的导数在 $o\neq y$ 时是常数 $\pm1$，误差很大时推力不变，收敛慢。系数 $\frac12$ 只是为了求导时把 2 消掉，不改变最优解。</p>
<h2>8.3 为什么必须“存”前向值</h2>
<p>因为局部导数依赖它。不存就得重算一遍前向，时间换空间；存了就是空间换时间。框架默认存，所以显存吃紧时可以用梯度检查点（重算换内存）。</p>
<h2>8.4 为什么深网络需要残差连接</h2>
<p>残差 $y=x+F(x)$ 的导数是 $1+F'(x)$。这个“1”让梯度可以原封不动地穿过去，不管 $F'$ 多小。它把连乘里加了一条恒等路径，从结构上绕开梯度消失。</p>
<div class="box key">
<h4>三条设计动机</h4>
<p>ReLU：让局部导数不衰减。平方损失：让梯度随误差增长。残差：让梯度有一条不衰减的直通路径。三者都在对付同一个敌人——连乘导致的衰减。</p>
</div>
"""

S9 = r"""
<h1>第 9 节　动手：调 $w$ 看梯度和损失</h1>
<p>这个演示只有一个参数。你拖滑块改 $w$，画面给出三样东西：损失曲线、当前位置、以及该点的切线斜率（就是梯度）。</p>
<div class="demo">
  <div class="demo-row"><label>w = <b id="wval">2.00</b></label>
  <input id="wslider" type="range" min="-1" max="5" step="0.01" value="2"></div>
  <div class="demo-row"><label>目标 y = <b id="yval">0.00</b></label>
  <input id="yslider" type="range" min="-2" max="4" step="0.5" value="0"></div>
  <svg id="plotsvg" viewBox="0 0 340 220" width="100%" height="220" role="img" aria-label="损失曲线与切线"></svg>
  <div class="demo-out">
    <div>当前损失 L = <b id="lval">0.000</b></div>
    <div>梯度 dL/dw = <b id="gval">0.000</b></div>
    <div>走一步后 w = <b id="nval">0.000</b></div>
  </div>
  <div class="demo-btns">
    <button id="stepBtn">按 α=0.1 走一步</button>
    <button id="resetBtn">复位</button>
  </div>
  <p class="demo-note">模型：$z=w\cdot x$，$x=1$，$L=\frac12(z-y)^2$。所以 $L=\frac12(w-y)^2$，$\frac{dL}{dw}=w-y$。最优 $w=y$，你能看到梯度为 0、曲线到谷底。</p>
</div>
<h2>看什么</h2>
<ul>
<li>把 $w$ 拖到 $y$ 的位置，梯度应显示 0.000——这是手算，不是玄学。</li>
<li>离 $y$ 越远，梯度绝对值越大，一步走得更远。这就是平方损失的特性。</li>
<li>把 $y$ 改成 4，再点“走一步”，看 $w$ 是不是朝 4 的方向挪了一大步（因为梯度大）。</li>
</ul>
"""

S10 = r"""
<h1>第 10 节　自检与思考题</h1>
<h2>10.1 能不能一句话复述</h2>
<ul>
<li>导数 = 输入动一点、输出动多少。</li>
<li>链式法则 = 沿路径把局部导数乘起来；一个变量走多条路径就加总。</li>
<li>反向传播 = 前向存值，反向按“下游求和 × 局部导数”逐层传。</li>
<li>选反向不选正向，是因为输出少输入多。</li>
</ul>
<h2>10.2 思考题（先自己做，再看答案）</h2>
<details class="fold"><summary>题 1：$y=(3x-1)^2$，求 $x=1$ 处的导数</summary><div class="fold-body">
<p>设 $u=3x-1$，$y=u^2$。$\frac{dy}{du}=2u$，$\frac{du}{dx}=3$，合起来 $6u=6(3x-1)$。$x=1$ 时 $u=2$，导数 $=12$。</p>
<p>数值核对：$x=1.001$ 时 $u=2.003$，$y=4.012009$，原值 4，差 0.012009，商 12.009。与 12 一致。</p>
</div></details>
<details class="fold"><summary>题 2：$f=(a\cdot b)+a$，求 $\frac{\partial f}{\partial a}$ 与 $\frac{\partial f}{\partial b}$</summary><div class="fold-body">
<p>$\frac{\partial f}{\partial a}=b+1$（两条路径：经 $a\cdot b$ 得 $b$，经 $+a$ 得 1）。$\frac{\partial f}{\partial b}=a$。</p>
<p>取 $a=2,b=3$：$\frac{\partial f}{\partial a}=4$。数值：$a=2.001$ 时 $f=2.001\cdot3+2.001=8.004$，原值 $6+2=8$，差 0.004，商 4.0。对得上。</p>
</div></details>
<details class="fold"><summary>题 3：为什么 $b$ 在例子里梯度是 5 而不是 2</summary><div class="fold-body">
<p>因为 $b$ 通过两条路径影响 $e$：一条经 $c=a+b$，贡献 $2$；一条经 $d=b+1$，贡献 $3$。只算一条会漏。任何“一个变量被两处使用”的结构都要加总。</p>
</div></details>
<details class="fold"><summary>题 4：把学习率调大为什么可能让 ReLU 神经元死掉</summary><div class="fold-body">
<p>步子太大，$w$ 与 $b$ 同时朝使 $z$ 变负的方向移动，$z&lt;0$ 后 ReLU 输出 0，局部导数为 0，梯度恒为 0，此后无论怎么更新这个神经元都不动。这就是“死亡 ReLU”。缓解办法：小学习率、LeakyReLU、或换掉 ReLU。</p>
</div></details>
<details class="fold"><summary>题 5：如果网络有两个输出、一个输入，反向传播还划算吗</summary><div class="fold-body">
<p>反向模式一次能算出“所有输入对每个输出的梯度”，但对多个输出要各跑一遍。输入少、输出多时，正向模式可能更省。这就是为什么反向传播适合“一个损失、百万权重”的训练场景，而不适合某些控制或物理仿真场景。</p>
</div></details>
<h2>10.3 下一步读什么</h2>
<p>把这一页吃透之后，去读《深度学习入门》第 5 章（误差反向传播法）做 NumPy 实现，再读《动手学深度学习》第 4 章（自动求导）。你会发现框架替你做的正是这一页里手算的那些事。</p>
"""

extra_css = r"""
<style>
.chain{display:flex;flex-direction:column;align-items:stretch;gap:0;margin:16px 0 6px}
.node{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
  border-radius:9px;padding:11px 13px;font-size:14.5px;cursor:pointer}
.node:hover{background:var(--accent-soft)}
.arrow{text-align:center;color:var(--muted);font-size:15px;line-height:1.3;padding:2px 0}
.demo{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:12px;margin:16px 0}
.demo-row{display:flex;align-items:center;gap:10px;margin:6px 0;flex-wrap:wrap}
.demo-row input[type=range]{flex:1;min-width:140px;height:32px}
.demo-out{display:grid;grid-template-columns:1fr;gap:4px;font-size:14px;margin:8px 0;font-variant-numeric:tabular-nums}
.demo-btns{display:flex;gap:8px;margin-top:8px}
.demo-btns button{flex:1;min-height:44px;border:1px solid var(--line);background:var(--accent-soft);
  color:var(--fg);border-radius:9px;font-size:14px;cursor:pointer}
.demo-note{font-size:13px;color:var(--muted);margin:10px 0 2px}
@media (min-width:768px){.demo-out{grid-template-columns:1fr 1fr 1fr}}
</style>
"""

extra_js = r"""
(function(){
  var wS = document.getElementById('wslider'), yS = document.getElementById('yslider');
  if (!wS) return;
  var svg = document.getElementById('plotsvg');
  var W = 340, H = 220, PADL = 42, PADB = 30, PADT = 12, PADR = 10;
  var wmin = -1, wmax = 5, ymin = -2, ymax = 4;

  function f(w, y){ return 0.5 * (w - y) * (w - y); }
  function df(w, y){ return w - y; }

  function X(w){ return PADL + (w - wmin) / (wmax - wmin) * (W - PADL - PADR); }
  function Y(v){ 
    var vmax = 9; // 0.5*(5-(-2))^2 = 24.5 太大，取 9 截顶
    var t = Math.min(v, vmax) / vmax;
    return H - PADB - t * (H - PADB - PADT);
  }

  function draw(){
    var w = parseFloat(wS.value), y = parseFloat(yS.value);
    document.getElementById('wval').textContent = w.toFixed(2);
    document.getElementById('yval').textContent = y.toFixed(2);
    var L = f(w, y), g = df(w, y);
    document.getElementById('lval').textContent = L.toFixed(3);
    document.getElementById('gval').textContent = g.toFixed(3);
    document.getElementById('nval').textContent = (w - 0.1 * g).toFixed(3);

    var parts = [];
    parts.push('<rect x="0" y="0" width="340" height="220" fill="none"/>');
    // 坐标轴
    parts.push('<line x1="'+PADL+'" y1="'+(H-PADB)+'" x2="'+(W-PADR)+'" y2="'+(H-PADB)+'" stroke="currentColor" stroke-opacity=".35"/>');
    parts.push('<line x1="'+PADL+'" y1="'+PADT+'" x2="'+PADL+'" y2="'+(H-PADB)+'" stroke="currentColor" stroke-opacity=".35"/>');
    for (var t = -1; t <= 5; t += 1){
      var px = X(t);
      parts.push('<line x1="'+px+'" y1="'+(H-PADB)+'" x2="'+px+'" y2="'+(H-PADB+4)+'" stroke="currentColor" stroke-opacity=".35"/>');
      parts.push('<text x="'+px+'" y="'+(H-PADB+16)+'" font-size="10" text-anchor="middle" fill="currentColor" fill-opacity=".6">'+t+'</text>');
    }
    parts.push('<text x="6" y="'+(H-PADB+16)+'" font-size="10" fill="currentColor" fill-opacity=".6">w</text>');
    parts.push('<text x="8" y="'+(PADT+10)+'" font-size="10" fill="currentColor" fill-opacity=".6">L</text>');

    // 曲线
    var d = '';
    for (var i = 0; i <= 200; i++){
      var wv = wmin + (wmax - wmin) * i / 200;
      var vv = f(wv, y);
      var xx = X(wv), yy = Y(vv);
      d += (i === 0 ? 'M' : 'L') + xx.toFixed(1) + ' ' + yy.toFixed(1) + ' ';
    }
    parts.push('<path d="'+d+'" fill="none" stroke="#0f6b52" stroke-width="2"/>');

    // 当前点
    var cx = X(w), cy = Y(L);
    parts.push('<circle cx="'+cx.toFixed(1)+'" cy="'+cy.toFixed(1)+'" r="5" fill="#b8860b"/>');
    // 切线：斜率 g，在像素坐标里斜率是 -g * (纵向像素/单位) / (横向像素/单位)
    var sx = (W - PADL - PADR) / (wmax - wmin);
    var sy = (H - PADB - PADT) / 9;
    var k = -g * sy / sx;
    var len = 46;
    var dx = len / Math.sqrt(1 + k*k), dy = k * dx;
    parts.push('<line x1="'+(cx-dx)+'" y1="'+(cy-dy)+'" x2="'+(cx+dx)+'" y2="'+(cy+dy)+'" stroke="#b8860b" stroke-width="2" stroke-dasharray="4 3"/>');
    parts.push('<text x="'+(cx+dx+3)+'" y="'+(cy+dy-2)+'" font-size="10" fill="#b8860b">斜率 '+g.toFixed(2)+'</text>');

    svg.innerHTML = parts.join('');
  }

  wS.oninput = draw; yS.oninput = draw;
  document.getElementById('stepBtn').onclick = function(){
    var w = parseFloat(wS.value), y = parseFloat(yS.value);
    var nw = w - 0.1 * df(w, y);
    nw = Math.max(wmin, Math.min(wmax, nw));
    wS.value = nw.toFixed(2);
    draw();
  };
  document.getElementById('resetBtn').onclick = function(){
    wS.value = 2; yS.value = 0; draw();
  };
  document.querySelectorAll('.jump').forEach(function(el){
    el.onclick = function(){ show(CHAPTERS.indexOf(el.dataset.goto)); };
  });
  draw();
  window.__afterShow = null;
})();
"""

sections = [
    dict(id="s0", title="知识树与读法", html=TREE),
    dict(id="s1", title="导数：变化率", html=S1),
    dict(id="s2", title="向量与矩阵（现场补）", html=S2),
    dict(id="s3", title="链式法则", html=S3),
    dict(id="s4", title="多元与路径求和", html=S4),
    dict(id="s5", title="计算图", html=S5),
    dict(id="s6", title="反向传播", html=S6),
    dict(id="s7", title="两层网络手算", html=S7),
    dict(id="s8", title="为什么这样设计", html=S8),
    dict(id="s9", title="动手：调 w 看梯度", html=S9),
    dict(id="s10", title="自检与思考题", html=S10),
]

build(OUT, "反向传播：从链式法则到每一层该改多少", "从零推导 · 第 1 页", sections,
      extra_css=extra_css, extra_js=extra_js,
      root_link="../index.html")
