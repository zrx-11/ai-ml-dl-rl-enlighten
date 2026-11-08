#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给所有生成的 HTML 补 <link rel="icon">，消除项目页对域名根 favicon 的 404。

幂等：已含 rel="icon" 的文件跳过。每次 build_handbook / deriv_0x / build_companion
之后跑一次即可。
"""
import os, re, sys

ROOT = "/home/user/Doubao/chats/38452296622422786/AI-点化学习系统"
TAG = '<link rel="icon" type="image/x-icon" href="{href}">'


def patch(fp):
    rel = os.path.relpath(os.path.join(ROOT, "favicon.ico"), os.path.dirname(fp))
    href = rel.replace(os.sep, "/")
    with open(fp, "r", encoding="utf-8") as f:
        s = f.read()
    if re.search(r'<link[^>]+rel="icon"', s, re.I):
        return "skip"
    tag = TAG.format(href=href)
    if "</head>" in s:
        s = s.replace("</head>", tag + "\n</head>", 1)
    elif "<head>" in s:
        s = s.replace("<head>", "<head>\n" + tag, 1)
    else:
        return "no-head"
    with open(fp, "w", encoding="utf-8") as f:
        f.write(s)
    return "ok"


def main():
    n_ok = n_skip = n_bad = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if not name.lower().endswith(".html"):
                continue
            fp = os.path.join(dirpath, name)
            r = patch(fp)
            if r == "ok":
                n_ok += 1
            elif r == "skip":
                n_skip += 1
            else:
                n_bad += 1
                print("无 <head>：", os.path.relpath(fp, ROOT), file=sys.stderr)
    print(f"补图标：{n_ok} 个，已存在跳过 {n_skip} 个，失败 {n_bad} 个")


if __name__ == "__main__":
    main()
