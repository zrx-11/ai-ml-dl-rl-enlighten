#!/usr/bin/env python3
"""点化学习系统通用装配器。
输入一本书的结构化内容，输出：章节 md、概念 md、测验、卡片、术语、模式、速查、路径、总纲、manifest.json。
所有 id 由本脚本统一分配，避免多处手写不一致。
"""
import json, os, re, hashlib

def _slug(s):
    """生成稳定的章节文件名（只用序号，避免中文转写差异）。"""
    return s

def chapter_md(ch):
    meta = f"- **难度**：{'★' * ch.get('diff', 1)}{'☆' * (3 - ch.get('diff', 1))} | **预估学时**：{ch.get('hours', 2)} 小时 | **前置**：{ch.get('prereq', '无')}"
    return f"# 第 {ch['n']} 章：{ch['title']}\n\n{meta}\n\n" + ch["md"].strip() + "\n"

def build(book_dir, meta, chapters, concepts=None, patterns_md="", cheatsheet_md="",
          path_steps=None, extra_notes=""):
    concepts = concepts or []
    os.makedirs(os.path.join(book_dir, "chapters"), exist_ok=True)
    os.makedirs(os.path.join(book_dir, "concepts"), exist_ok=True)

    # ---- 章节文件 ----
    ch_entries = []
    for ch in chapters:
        cid = ch.get("id") or f"ch{ch['n']:02d}"
        fn = f"chapters/{cid}.md"
        open(os.path.join(book_dir, fn), "w", encoding="utf-8").write(chapter_md(ch))
        ch_entries.append({"id": cid, "title": f"第 {ch['n']} 章：{ch['title']}",
                           "file": fn, "difficulty": ch.get("diff", 1)})

    # ---- 概念文件 ----
    cp_entries = []
    for i, cp in enumerate(concepts, 1):
        cid = cp.get("id") or f"c{i:02d}"
        fn = f"concepts/{cid}.md"
        open(os.path.join(book_dir, fn), "w", encoding="utf-8").write(cp["md"].strip() + "\n")
        cp_entries.append({"id": cid, "title": cp["title"], "file": fn,
                           "chapterIds": cp.get("chapterIds", [])})

    # ---- 测验 / 卡片 / 术语 ----
    quizzes, cards, glossary = [], [], []
    qn = fn_ = 0
    for ch in chapters:
        cid = ch.get("id") or f"ch{ch['n']:02d}"
        for q in ch.get("quiz", []):
            qn += 1
            item = {"id": f"q{qn:02d}", "chapterId": cid, "type": q.get("type", "single"),
                    "question": q["q"], "explain": q["explain"]}
            if q.get("options"):
                item["options"] = q["options"]
                item["answer"] = q["answer"]
            else:
                item["options"] = []
                item["answer"] = q["answer"]
            quizzes.append(item)
        for c in ch.get("cards", []):
            fn_ += 1
            cards.append({"id": f"f{fn_:02d}", "chapterId": cid, "front": c[0], "back": c[1],
                          "tags": c[2] if len(c) > 2 else [cid]})
        for t in ch.get("terms", []):
            glossary.append({"term": t[0], "def": t[1], "chapterId": cid})

    # ---- 汇总 md ----
    q_md = [f"# {meta['title']} 测验\n"]
    for ch in chapters:
        cid = ch.get("id") or f"ch{ch['n']:02d}"
        if not ch.get("quiz"):
            continue
        q_md.append(f"## 第 {ch['n']} 章测验（检查点）\n")
        for i, q in enumerate(ch["quiz"], 1):
            typ = {"single": "单选", "multi": "多选", "blank": "填空"}.get(q.get("type", "single"), "简答")
            q_md.append(f"### Q{i}.（{typ}）{q['q']}")
            if q.get("options"):
                opts = " ".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(q["options"]))
                q_md.append(f"- {opts}")
            if isinstance(q.get("answer"), list):
                ans = "、".join(chr(65 + a) for a in q["answer"])
            elif isinstance(q.get("answer"), int):
                ans = chr(65 + q["answer"])
            else:
                ans = q["answer"]
            q_md.append(f"**答案**：{ans}")
            q_md.append(f"**解析**：{q['explain']}\n")
    q_md.append("## 综合测验（多章联考）\n")
    q_md.append(meta.get("final_quiz", "（见各章自测题；综合题见跨书总览的《综合检验建议》。）"))
    open(os.path.join(book_dir, "quiz.md"), "w", encoding="utf-8").write("\n".join(q_md) + "\n")

    c_md = [f"# {meta['title']} 抽认卡\n"]
    for ch in chapters:
        if not ch.get("cards"):
            continue
        cid = ch.get("id") or "ch%02d" % ch["n"]
        c_md.append(f"## 第 {ch['n']} 章\n")
        for front, back, *rest in ch["cards"]:
            tags = rest[0] if rest else [ch.get("id", "")]
            c_md.append(f"### 卡片：{front}")
            c_md.append(f"- **背面**：{back}")
            c_md.append(f"- **标签**：{', '.join(tags)}")
            c_md.append(f"- **章节**：{cid}\n")
    open(os.path.join(book_dir, "flashcards.md"), "w", encoding="utf-8").write("\n".join(c_md) + "\n")

    g_md = ["# 术语表\n", "| 术语 | 白话定义 | 首次出现 | 章节 |", "|---|---|---|---|"]
    seen = {}
    for g in glossary:
        if g["term"] in seen:
            continue
        seen[g["term"]] = 1
        ch_title = next((c["title"] for c in ch_entries if c["id"] == g["chapterId"]), "")
        g_md.append(f"| {g['term']} | {g['def']} | {g['chapterId']} | {ch_title} |")
    open(os.path.join(book_dir, "glossary.md"), "w", encoding="utf-8").write("\n".join(g_md) + "\n")

    open(os.path.join(book_dir, "patterns.md"), "w", encoding="utf-8").write(patterns_md.strip() + "\n")
    open(os.path.join(book_dir, "cheatsheet.md"), "w", encoding="utf-8").write(cheatsheet_md.strip() + "\n")

    # ---- 学习路径 ----
    if path_steps is None:
        path_steps = [{"step": f"{i:02d}", "title": c["title"], "chapterIds": [c["id"]],
                       "checkpoint": "完成本章自测题，正确率 ≥ 80%"}
                      for i, c in enumerate(ch_entries, 1)]
    path_md = [f"# 学习路径：{meta['title']}\n", meta.get("path_intro", "") + "\n", "## 路径总览\n"]
    path_md.append("```mermaid\nflowchart TD")
    for i, s in enumerate(path_steps):
        path_md.append(f"  S{i}[\"{s['step']} {s['title']}\"]")
        if i:
            path_md.append(f"  S{i-1} --> S{i}")
    path_md.append("```\n")
    path_md.append("## 逐步展开\n")
    for s in path_steps:
        path_md.append(f"### 步骤 {s['step']}：{s['title']}")
        path_md.append(f"**阅读**：{', '.join(s.get('chapterIds', []))}")
        path_md.append(f"**检查点**：{s.get('checkpoint', '')}")
        if s.get("goal"):
            path_md.append(f"**目标**：{s['goal']}")
        path_md.append("")
    path_md.append("## 学完全程后\n")
    path_md.append(meta.get("path_outro", ""))
    os.makedirs(os.path.join(book_dir, "paths"), exist_ok=True)
    open(os.path.join(book_dir, "paths", "learn.md"), "w", encoding="utf-8").write("\n".join(path_md) + "\n")

    # ---- 总纲 ----
    skill_md = f"""---
name: {meta['title']}（点化学习总纲）
---

# {meta['title']} 点化学习系统

## 这本书在讲什么
{meta['summary']}

## 全书心智模型
{meta.get('mindmap', '')}

## 学习路径（推荐顺序）
| 步骤 | 章节 | 主题 | 前置 | 检查点 |
|---|---|---|---|---|
"""
    for s in path_steps:
        ch_title = "、".join(next((c["title"] for c in ch_entries if c["id"] == cid), cid) for cid in s.get("chapterIds", []))
        skill_md += f"| {s['step']} | {', '.join(s.get('chapterIds', []))} | {s['title']} | {s.get('prereq', '按序')} | {s.get('checkpoint', '')} |\n"
    skill_md += "\n## 章节索引\n"
    for ch in chapters:
        cid = ch.get("id") or f"ch{ch['n']:02d}"
        skill_md += f"- 第 {ch['n']} 章 {ch['title']}（难度 {'★' * ch.get('diff', 1)}{'☆' * (3 - ch.get('diff', 1))}）→ chapters/{cid}.md\n"
    skill_md += "\n## 前置知识（缺口补讲位置）\n| 缺口 | 补讲位置 | 学完标志 |\n|---|---|---|\n"
    skill_md += meta.get("gaps", "| 无 | — | — |") + "\n"
    skill_md += """
## 术语表 / 模式库 / 速查表入口
- 术语表：`glossary.md`
- 模式库：`patterns.md`
- 速查表：`cheatsheet.md`
- 测验：`quiz.md`
- 抽认卡：`flashcards.md`
"""
    if extra_notes:
        skill_md += "\n" + extra_notes.strip() + "\n"
    open(os.path.join(book_dir, "SKILL.md"), "w", encoding="utf-8").write(skill_md)

    # ---- manifest ----
    manifest = {
        "slug": meta["slug"],
        "title": meta["title"],
        "subtitle": meta.get("subtitle", ""),
        "book": {"title": meta["book_title"], "author": meta["author"],
                 "source": meta.get("source", [])},
        "audience": meta.get("audience", "大一：会微积分与概率基础，线性代数需补"),
        "goal": meta.get("goal", "系统掌握"),
        "learningPath": path_steps,
        "chapters": ch_entries,
        "concepts": cp_entries,
        "glossary": glossary,
        "patterns": {"file": "patterns.md"},
        "cheatsheet": {"file": "cheatsheet.md"},
        "quizzes": quizzes,
        "flashcards": cards,
        "companion": f"{meta['slug']}学习伴侣.html",
    }
    open(os.path.join(book_dir, "manifest.json"), "w", encoding="utf-8").write(
        json.dumps(manifest, ensure_ascii=False, indent=2))

    # ---- 自检 ----
    problems = []
    for q in quizzes:
        if q["type"] in ("single", "multi") and not q.get("options"):
            problems.append(f"{q['id']} 缺 options")
        if q["type"] == "single" and not isinstance(q["answer"], int):
            problems.append(f"{q['id']} answer 类型不符")
        if q["type"] == "multi" and not isinstance(q["answer"], list):
            problems.append(f"{q['id']} answer 类型不符")
        if not q.get("explain"):
            problems.append(f"{q['id']} 缺解析")
    ids = [c["id"] for c in ch_entries]
    if len(ids) != len(set(ids)):
        problems.append("章节 id 重复")
    print(f"[{meta['slug']}] 章节 {len(ch_entries)} · 概念 {len(cp_entries)} · 测验 {len(quizzes)} "
          f"· 卡片 {len(cards)} · 术语 {len(seen)}")
    if problems:
        print("  ⚠ " + "; ".join(problems[:8]))
    else:
        print("  ✓ 结构自检通过")
    return manifest
