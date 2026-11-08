#!/usr/bin/env python3
"""按章切分抽取出的整本 txt（v2）：处理"第\\n1 章"跨行标题与数字式目录。"""
import os, re, glob

ROOT = "/home/user/Doubao/chats/38452296622422786/AI-点化学习系统"

D2L_TITLES = ["深度学习简介", "预备知识", "深度学习基础", "深度学习计算", "卷积神经网络",
              "循环神经网络", "优化算法", "计算性能", "计算机视觉", "自然语言处理", "附录"]

def load(raw):
    t = open(raw, encoding="utf-8", errors="replace").read()
    t = re.sub(r"第\s*\n\s*(\d+)\s*章", r"第\1 章", t)          # 跨行章标题
    t = re.sub(r"第\s*\n\s*([IVX]+)\s*部分", r"第\1部分", t)     # 部标题
    return t.splitlines()

def cn_split(lines):
    starts, cur = [], 0
    for i, line in enumerate(lines):
        m = re.match(r"^\s*第\s*(\d+)\s*章\s*(.{0,26})$", line.strip())
        if not m:
            continue
        num = int(m.group(1))
        if num != cur + 1:
            continue
        raw_title = m.group(2)
        if re.search(r"[.·]{4,}", raw_title):
            continue
        if any(re.match(r"^[·.．…\s\d]{2,}$", lines[j].strip()) for j in range(i+1, min(i+3, len(lines)))):
            continue
        title = raw_title.strip(" 　.·")
        if not title and i + 1 < len(lines):
            nxt = lines[i + 1].strip(" 　.·")
            if nxt and len(nxt) <= 26 and not re.search(r"[.·]{4,}", nxt):
                title = nxt
        starts.append((num, i, title))
        cur = num
    return starts

def d2l_split(lines):
    def norm(x):
        for a, b in [("⾃","自"),("⾔","言"),("⼒","力"),("⼤","大"),("⽹","网"),("⻅","见"),("⾄","至")]:
            x = x.replace(a, b)
        return x.strip()
    starts, cur = [], 0
    for i, line in enumerate(lines):
        if i < 1300:
            continue
        s = norm(line)
        for k, t in enumerate(D2L_TITLES, 1):
            if s == t and k == cur + 1 and i > 0 and lines[i - 1].strip() == str(k):
                starts.append((k, i - 1, t))
                cur = k
    return starts

def run(book_dir, mode, titles=None):
    raw = [f for f in glob.glob(os.path.join(ROOT, book_dir, "_work/raw/*.txt")) if not os.path.basename(f).startswith("ch")][0]
    lines = load(raw)
    starts = cn_split(lines) if mode == "cn" else d2l_split(lines)
    out_dir = os.path.dirname(raw)
    for f in glob.glob(os.path.join(out_dir, "ch*.txt")):
        os.remove(f)
    rows = []
    for idx, (num, pos, title) in enumerate(starts):
        end = starts[idx + 1][1] if idx + 1 < len(starts) else len(lines)
        body = "\n".join(lines[pos:end])
        p = os.path.join(out_dir, f"ch{num:02d}.txt")
        open(p, "w", encoding="utf-8").write(body)
        rows.append((num, title, len(body)))
    md = [f"# {book_dir} 结构地图", "", f"源文件：`{os.path.basename(raw)}`", "",
          "| 章 | 标题 | 字符数 | 语料 |", "|---|---|---|---|"]
    for num, title, chars in rows:
        md.append(f"| {num} | {title} | {chars:,} | `ch{num:02d}.txt` |")
    os.makedirs(os.path.join(ROOT, book_dir, "_work"), exist_ok=True)
    open(os.path.join(ROOT, book_dir, "_work", "structure.md"), "w", encoding="utf-8").write("\n".join(md))
    print(f"{book_dir}: {len(rows)} 章 -> " + ", ".join(f"ch{n:02d}({c//1000}k)" for n, t, c in rows))

run("01-人工智能现代方法", "cn")
run("02-机器学习", "cn")
run("03-深度学习入门", "cn")
run("04-动手学深度学习", "d2l")
run("05-强化学习", "cn")
