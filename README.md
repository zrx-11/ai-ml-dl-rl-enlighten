# AI · 点化学习系统

五本 AI / ML / DL 教材的点化学习知识库，以一个总入口串起来。单文件、离线可用。

## 一分钟上手

双击 `index.html`。顶部切换书籍；左侧按「学习路径 / 章节 / 概念 / 测验 / 抽认卡 / 术语 / 模式 / 速查」浏览；进度自动存在浏览器本地，关掉再开还在。

只想学一本，直接打开对应目录里的「学习伴侣」HTML：内容同源，可单独发给别人。

## 目录结构

| 路径 | 说明 |
|---|---|
| `index.html` | 总学习手册，单文件统一入口（约 1 MB） |
| `index.md` | Markdown 导航页，链接到各书与跨书总览 |
| `00-跨书总览/` | 推荐顺序、跨书概念映射表、综合检验建议、推导页索引 |
| `01-人工智能现代方法/` | AIMA 第 4 版：28 章知识库 + `aima4学习伴侣.html` |
| `02-机器学习/` | 周志华：16 章 + `zhou-ml学习伴侣.html` |
| `03-深度学习入门/` | 斋藤康毅：8 章 + `dl-scratch学习伴侣.html` |
| `04-动手学深度学习/` | d2l：11 章 + `d2l学习伴侣.html` |
| `05-强化学习/` | Sutton 第 2 版：17 章 + `rl-sutton学习伴侣.html` |
| `06-从零推导/` | 三页独立交互推导：反向传播 / 贝尔曼到策略梯度 / 贝叶斯推断 |
| `_build/` | 生成脚本（分章、内容、KaTeX 内联、推导页骨架） |
| `各书/_work/` | 抽取原文与章节语料 |

每本书的目录内结构一致：`chapters/`、`concepts/`、`quiz.md`、`flashcards.md`、`glossary.md`、`patterns.md`、`cheatsheet.md`、`paths/learn.md`、`SKILL.md`、`manifest.json`。

## 三页从零推导

| 页面 | 讲什么 | 什么时候用 |
|---|---|---|
| `06-从零推导/01-反向传播.html` | 导数 → 链式法则 → 计算图 → 两层网络手算（含数值梯度核对） | 学完《深度学习入门》第 4 章仍不懂误差怎么传回去 |
| `06-从零推导/02-贝尔曼到策略梯度.html` | 期望 → 折扣回报 → 贝尔曼两方程 → log 导数技巧 → 策略梯度定理 → REINFORCE → 基线 | 读完《强化学习》第 3 章、准备进第 13 章 |
| `06-从零推导/03-贝叶斯推断.html` | 条件概率 → 贝叶斯公式 → 基础比率 → Beta 共轭更新 → 朴素贝叶斯手算 | 学完《机器学习》第 7 章前半、对先验没感觉 |

三页都是单文件，KaTeX 连字体一起内联，断网也能正常渲染公式。

## 部署到 GitHub Pages（可选）

```bash
# 1. 把本目录内容放到仓库根目录
git add . && git commit -m "add study handbook" && git push

# 2. 仓库 Settings → Pages → Source 选 main 分支、/(root)

# 3. 等 1 分钟，访问 https://<用户名>.github.io/<仓库名>/
```

`.nojekyll` 已经放好（空文件），它保证中文目录名和 `_build/`、`_work/` 这类下划线目录被原样托管。缺了它，Pages 会跳过这些目录，页面里的图片与章节文件会 404。

## 已知限制

- Mermaid 图需要联网从 CDN 加载。离线打开时图会退化成代码文本，章节正文与测验不受影响。
- 进度存在 `localStorage`，换浏览器或清缓存会丢。
- 五本书的测验共 199 题，都是单选 / 多选 / 填空，没有主观题自动评分。
- 三页推导页各约 670 KB，体积主要在 KaTeX 字体（约 270 KB），这是离线可用的代价。

## 重新构建

```bash
# 重新切分五本书（需要原始 PDF 在附件目录）
python3 _build/split_chapters.py

# 重新生成某本书的知识库与伴侣，以深度学习入门为例
python3 _build/gen_03.py
python3 /home/user/.doubao/agent_mode/workspace/.user_skills/book-enlighten/scripts/build_companion.py \
    --manifest 03-深度学习入门/manifest.json \
    --out 03-深度学习入门/dl-scratch学习伴侣.html

# 重新生成三页推导
python3 _build/deriv_01.py && python3 _build/deriv_02.py && python3 _build/deriv_03.py

# 重建总入口
python3 /home/user/.doubao/agent_mode/workspace/.user_skills/book-enlighten/scripts/build_handbook.py --root .
```

## 版权

五本教材的版权归原作者与出版方。本目录只包含**改写后的讲解、测验与图示**，不包含原文的连续复制。原始 PDF 不在交付包内，`_work/raw/` 是按章切分的抽取文本，仅供个人学习使用，不得再分发。
