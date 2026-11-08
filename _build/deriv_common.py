#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从零推导页的公共骨架：单文件 HTML、移动端优先、KaTeX 全内联、章节导航三处一致。"""
import os, json

BUILD = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BUILD, "assets")

PAGE_CSS = r"""
:root{
  --bg:#fbfaf7; --fg:#1d1d1f; --muted:#6b6b70; --line:#e3e0d8;
  --accent:#0f6b52; --accent-soft:#e7f2ee; --warn:#8a5a00; --warn-soft:#fdf4e3;
  --card:#ffffff; --code:#f4f1ea; --shadow:0 1px 2px rgba(0,0,0,.06);
}
@media (prefers-color-scheme: dark){
  :root{--bg:#141416;--fg:#eeeae2;--muted:#9a978f;--line:#2e2e32;--accent:#5fd0a8;
        --accent-soft:#17322b;--warn:#e6bd7a;--warn-soft:#2c2416;--card:#1c1c20;--code:#232327;}
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--bg);color:var(--fg);
  font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif;
  -webkit-text-size-adjust:100%;
}
.topbar{
  position:fixed;top:0;left:0;right:0;height:56px;z-index:30;
  display:flex;align-items:center;gap:8px;padding:0 10px;
  background:color-mix(in srgb,var(--bg) 86%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--line);
}
.topbar button{
  min-width:44px;min-height:44px;border:1px solid var(--line);background:var(--card);
  color:var(--fg);border-radius:10px;font-size:18px;cursor:pointer;
}
.tb-title{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  font-size:14.5px;font-weight:600;color:var(--fg)}
main{max-width:900px;margin:0 auto;padding:72px 16px 96px}
section.chapter{display:none;animation:fade .18s ease}
section.chapter.active{display:block}
@keyframes fade{from{opacity:.4}to{opacity:1}}
h1{font-size:22px;line-height:1.4;margin:8px 0 14px}
h2{font-size:19px;margin:30px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:16.5px;margin:22px 0 8px;color:var(--accent)}
p{margin:10px 0}
ul,ol{margin:8px 0 12px;padding-left:22px}
li{margin:4px 0}
code{background:var(--code);padding:1px 5px;border-radius:5px;font-size:14px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--code);padding:12px;border-radius:10px;overflow-x:auto;font-size:13.5px;
  line-height:1.6;border:1px solid var(--line)}
pre code{background:none;padding:0}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:14.5px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;white-space:nowrap}
th{background:var(--accent-soft);font-weight:600}
.katex-display{overflow-x:auto;overflow-y:hidden;padding:2px 0}
img,svg{max-width:100%;height:auto}
.box{border-left:4px solid var(--accent);background:var(--accent-soft);padding:10px 12px;margin:14px 0;border-radius:0 8px 8px 0}
.box.warn{border-color:var(--warn);background:var(--warn-soft)}
.box.key{border-color:#b8860b;background:var(--warn-soft);font-weight:600}
.box h4{margin:0 0 6px;font-size:14.5px;letter-spacing:.02em;color:var(--muted)}
details.fold{border:1px solid var(--line);border-radius:10px;margin:10px 0;background:var(--card);box-shadow:var(--shadow)}
details.fold summary{cursor:pointer;padding:10px 12px;font-size:14.5px;font-weight:600;list-style:none}
details.fold summary::before{content:"▸ ";color:var(--accent)}
details.fold[open] summary::before{content:"▾ "}
details.fold .fold-body{padding:2px 12px 12px;border-top:1px solid var(--line)}
.breadcrumb{font-size:12.5px;color:var(--muted);margin:0 0 6px}
.chapter-nav{display:flex;gap:10px;justify-content:space-between;margin:34px 0 0;
  padding-top:14px;border-top:1px dashed var(--line)}
.chapter-nav button{flex:1;min-height:44px;border:1px solid var(--line);background:var(--card);
  color:var(--fg);border-radius:10px;font-size:13.5px;cursor:pointer}
.chapter-nav button:disabled{opacity:.35;cursor:not-allowed}
#sidebar{
  position:fixed;top:0;bottom:0;left:0;width:82%;max-width:320px;z-index:40;
  background:var(--card);border-right:1px solid var(--line);
  transform:translateX(-102%);transition:transform .22s ease;overflow-y:auto;padding:14px 12px 40px;
}
#sidebar.open{transform:none}
#sidebar h4{margin:6px 4px 10px;font-size:13px;color:var(--muted);letter-spacing:.06em}
.toc-item{display:block;width:100%;text-align:left;padding:9px 10px;margin:2px 0;border:0;
  background:none;color:var(--fg);border-radius:8px;font-size:14px;cursor:pointer}
.toc-item:hover{background:var(--accent-soft)}
.toc-item.cur{background:var(--accent-soft);font-weight:600;color:var(--accent)}
#scrim{position:fixed;inset:0;background:rgba(0,0,0,.42);z-index:35;display:none}
#scrim.on{display:block}
#toTop{position:fixed;right:14px;bottom:18px;width:44px;height:44px;border-radius:50%;
  border:1px solid var(--line);background:var(--card);color:var(--fg);font-size:17px;cursor:pointer;z-index:25}
@media (min-width:768px){
  .topbar{display:none}
  #sidebar{transform:none;width:280px;box-shadow:none;padding-top:20px}
  #scrim{display:none!important}
  main{margin-left:280px;max-width:none;padding:28px 40px 120px}
  h1{font-size:27px}
  h2{font-size:21px}
  .chapter-nav{max-width:760px}
  section.chapter{max-width:760px}
}
"""

CORE_JS = r"""
const CHAPTERS = __CHAPTERS__;
const TITLES = __TITLES__;
let cur = 0;
const mainEl = document.getElementById('main');
const sidebar = document.getElementById('sidebar');
const scrim = document.getElementById('scrim');

function renderMath(el, tries){
  tries = tries || 0;
  if (!window.renderMathInElement || !window.katex){ 
    if (tries < 40) setTimeout(function(){renderMath(el, tries+1);}, 120);
    return;
  }
  try{
    window.renderMathInElement(el, {
      delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],
      throwOnError:false, strict:false, trust:false
    });
  }catch(e){ console.warn('math render', e); }
}

function fillNav(sec, i){
  const bc = sec.querySelector('.breadcrumb');
  if (bc) bc.textContent = '第 ' + (i) + ' 节 / 共 ' + (CHAPTERS.length-1) + ' 节 · ' + sec.dataset.title;
  const nav = sec.querySelector('.chapter-nav');
  if (!nav) return;
  nav.innerHTML = '';
  const prev = document.createElement('button');
  prev.textContent = i > 0 ? '← ' + TITLES[i-1] : '已是第一节';
  prev.disabled = i === 0;
  prev.onclick = function(){ show(i-1); };
  const next = document.createElement('button');
  next.textContent = i < CHAPTERS.length-1 ? TITLES[i+1] + ' →' : '已是最后一节';
  next.disabled = i === CHAPTERS.length-1;
  next.onclick = function(){ show(i+1); };
  nav.appendChild(prev); nav.appendChild(next);
}

function show(i){
  if (i < 0 || i >= CHAPTERS.length) return;
  const secs = document.querySelectorAll('section.chapter');
  secs.forEach(function(s){ s.classList.remove('active'); });
  cur = i;
  const sec = document.getElementById(CHAPTERS[i]);
  sec.classList.add('active');
  document.getElementById('tbTitle').textContent = TITLES[i];
  document.querySelectorAll('.toc-item').forEach(function(b){
    b.classList.toggle('cur', b.dataset.target === CHAPTERS[i]);
  });
  fillNav(sec, i);
  renderMath(sec, 0);
  const st = document.getElementById('prevBtn'), nt = document.getElementById('nextBtn');
  st.disabled = i === 0; nt.disabled = i === CHAPTERS.length-1;
  window.scrollTo({top:0, behavior:'instant'});
  closeMenu();
  if (window.__afterShow) window.__afterShow(sec, i);
}

function openMenu(){ sidebar.classList.add('open'); scrim.classList.add('on'); }
function closeMenu(){ sidebar.classList.remove('open'); scrim.classList.remove('on'); }

document.getElementById('menuBtn').onclick = openMenu;
scrim.onclick = closeMenu;
document.getElementById('prevBtn').onclick = function(){ show(cur-1); };
document.getElementById('nextBtn').onclick = function(){ show(cur+1); };
document.getElementById('toTop').onclick = function(){ window.scrollTo({top:0,behavior:'smooth'}); };
document.addEventListener('keydown', function(e){
  if (e.key === 'ArrowLeft') show(cur-1);
  if (e.key === 'ArrowRight') show(cur+1);
});
document.querySelectorAll('.toc-item').forEach(function(b){
  b.onclick = function(){ show(CHAPTERS.indexOf(b.dataset.target)); };
});
// 首次渲染：所有章节都渲染（包括 display:none 的），切章时再渲染一次
document.querySelectorAll('section.chapter').forEach(function(s, i){
  renderMath(s, 0);
  fillNav(s, i);
});
show(0);
window.showChapter = function(id){ var i = CHAPTERS.indexOf(id); if (i >= 0) show(i); };
window.__chapterIds = CHAPTERS;
"""


def build(path, title, subtitle, sections, extra_css="", extra_js="",
          footer_note="", root_link=None):
    """sections: list of dict(id, title, html)。第 0 节通常是知识树。"""
    ids = [s["id"] for s in sections]
    titles = [s["title"] for s in sections]
    katex_css = open(os.path.join(ASSETS, "katex_inline.css"), encoding="utf-8").read()
    katex_js = open(os.path.join(ASSETS, "katex_inline.js"), encoding="utf-8").read()
    ar_js = open(os.path.join(ASSETS, "autorender_inline.js"), encoding="utf-8").read()

    toc = "\n".join(
        '<button class="toc-item" data-target="%s">%d. %s</button>' % (s["id"], i, s["title"])
        for i, s in enumerate(sections))

    body = []
    for s in sections:
        body.append('<section class="chapter" id="%s" data-title="%s">' % (s["id"], s["title"]))
        body.append('<div class="breadcrumb"></div>')
        body.append(s["html"])
        body.append('<div class="chapter-nav"></div>')
        body.append('</section>')
    body = "\n".join(body)

    core = CORE_JS.replace("__CHAPTERS__", json.dumps(ids, ensure_ascii=False)) \
                  .replace("__TITLES__", json.dumps(titles, ensure_ascii=False))

    back = ('<p style="margin:26px 0 0"><a href="%s" style="color:var(--muted);font-size:13.5px">'
            '← 返回总入口</a></p>' % root_link) if root_link else ""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{katex_css}</style>
<style>{PAGE_CSS}{extra_css}</style>
</head>
<body>
<header class="topbar">
  <button id="menuBtn" aria-label="打开目录">☰</button>
  <div class="tb-title" id="tbTitle">{title}</div>
  <button id="prevBtn" aria-label="上一节">‹</button>
  <button id="nextBtn" aria-label="下一节">›</button>
</header>
<aside id="sidebar">
  <h4>{subtitle}</h4>
  {toc}
</aside>
<div id="scrim"></div>
<main id="main">
{body}
{back}
</main>
<button id="toTop" aria-label="回到顶部">↑</button>
<script>{katex_js}</script>
<script>{ar_js}</script>
<script>{core}</script>
<script>{extra_js}</script>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print("写出", path, os.path.getsize(path) // 1024, "KB")
    return path


def tree_section(intro_html, nodes, edges_js=""):
    """知识树：节点可点击跳到对应章节。nodes: list of (id, label, section_id)"""
    pass
