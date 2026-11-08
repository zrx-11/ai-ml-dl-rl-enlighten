#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 KaTeX 0.16.9 的 CSS（含字体 base64）、JS、auto-render 合成可内联的片段。"""
import base64, os, re, sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "katex", "package", "dist")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)


UID = re.compile(r"url\(\s*['\"]?fonts/([^'\")]+?)['\"]?\s*\)")


def b64(name):
    fp = os.path.join(BASE, "fonts", name)
    data = base64.b64encode(open(fp, "rb").read()).decode()
    fmt = "woff2" if name.endswith(".woff2") else ("woff" if name.endswith(".woff") else "truetype")
    return "url(data:font/%s;base64,%s)" % (fmt, data)


def inline_css():
    """只保留 woff2（其他格式就地丢弃），避免同一字体被内联三份。"""
    css = open(os.path.join(BASE, "katex.min.css"), encoding="utf-8").read()

    def fix_src(m):
        seg = m.group(0)
        w2 = [n for n in UID.findall(seg) if n.endswith(".woff2")]
        if not w2:
            return seg
        from urllib.parse import unquote
        name = unquote(w2[0])
        return 'src:%s format("woff2");' % b64(name)

    css = re.sub(r"src:[^;]+;", fix_src, css)
    left = UID.findall(css)
    if left:
        print("警告：仍有未内联的 fonts 引用", len(left), file=sys.stderr)
    return css


def main():
    css = inline_css()
    open(os.path.join(OUT, "katex_inline.css"), "w", encoding="utf-8").write(css)
    js = open(os.path.join(BASE, "katex.min.js"), encoding="utf-8").read()
    open(os.path.join(OUT, "katex_inline.js"), "w", encoding="utf-8").write(js)
    ar = open(os.path.join(BASE, "contrib", "auto-render.min.js"), encoding="utf-8").read()
    open(os.path.join(OUT, "autorender_inline.js"), "w", encoding="utf-8").write(ar)
    for f in ("katex_inline.css", "katex_inline.js", "autorender_inline.js"):
        print(f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")


if __name__ == "__main__":
    main()
