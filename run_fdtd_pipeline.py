# 09_FDTD电磁波数值仿真教学 — 课题编排规划
# 阶段一：文献下载 + 课题分析
# 阶段二：研究执行（仿真代码跑通+出图）

import os, json, shutil, time, subprocess
from datetime import datetime

# ── 路径配置 ──
BASE = "/mnt/c/Users/admin/Desktop/1号 Hermes 课题/09_FDTD电磁波数值仿真教学"
PHASE1 = f"{BASE}/01_课题分析"
PHASE2 = f"{BASE}/02_研究执行"
PHASE3 = f"{BASE}/03_论文输出"

os.makedirs(f"{PHASE1}/references/pdfs", exist_ok=True)
os.makedirs(f"{PHASE1}/references/metadata", exist_ok=True)

HERMES = shutil.which("hermes") or os.path.expanduser("~/.local/bin/hermes")
FEISHU = "feishu:oc_1fc8f0cdeb10f9b064110ed371f3f450"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def feishu_notify(title, files=None):
    cmd = [HERMES, "send", "-t", FEISHU, f"[{datetime.now().strftime('%H:%M:%S')}] {title}"]
    if files:
        for f in (files if isinstance(files, list) else [files]):
            if os.path.exists(f):
                cmd += ["--attach", f]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        log(f"  📤 飞书: {title} {'✅' if r.returncode == 0 else '❌'}")
    except Exception as e:
        log(f"  ⚠️ 飞书通知失败: {e}")

import urllib.request, urllib.parse, xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"

def arxiv_search(query, max_results=8):
    """arXiv API搜索，返回论文列表。零外部依赖。"""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    log(f"  📡 arXiv API: {query[:60]}...")
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Hermes-FDTD/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        log(f"  ✅ arXiv返回 ({time.time()-t0:.0f}s, {len(raw)} bytes)")
    except Exception as e:
        log(f"  ❌ arXiv请求失败: {e}")
        return []
    
    # 解析Atom XML
    ns = {"a": "http://www.w3.org/2005/Atom",
          "arxiv": "http://arxiv.org/schemas/atom"}
    papers = []
    try:
        root = ET.fromstring(raw)
        for entry in root.findall("a:entry", ns):
            title = entry.find("a:title", ns)
            title = title.text.strip().replace("\n", " ") if title is not None else "?"
            
            summary = entry.find("a:summary", ns)
            summary = summary.text.strip().replace("\n", " ") if summary is not None else ""
            
            authors = []
            for au in entry.findall("a:author", ns):
                name = au.find("a:name", ns)
                if name is not None:
                    authors.append(name.text)
            
            published = entry.find("a:published", ns)
            year = published.text[:4] if published is not None else "?"
            
            arxiv_id_full = entry.find("a:id", ns)
            arxiv_id = ""
            if arxiv_id_full is not None:
                arxiv_id = arxiv_id_full.text.split("/abs/")[-1]
                # 去掉版本号
                if "v" in arxiv_id:
                    arxiv_id = arxiv_id.split("v")[0]
            
            # arXiv论文都是未正式发表的，journal为空
            papers.append({
                "title": title,
                "authors": ", ".join(authors[:5]),
                "year": year,
                "arxiv_id": arxiv_id,
                "journal": "",
                "reason": summary[:300]
            })
    except Exception as e:
        log(f"  ❌ 解析XML失败: {e}")
        return []
    
    return papers

def hermes_chat(prompt, timeout=600):
    log(f"  🤖 调 Hermes (timeout={timeout}s)...")
    t0 = time.time()
    cmd = [HERMES, "chat", "-q", prompt, "--provider", "deepseek", "--model", "deepseek-v4-flash"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    log(f"  {'✅' if r.returncode == 0 else '❌'} Hermes 完成 ({time.time()-t0:.0f}s)")
    return {"ok": r.returncode == 0, "out": r.stdout[-3000:] if r.stdout else ""}

# ============================================================
# Phase 1: 课题分析 — 搜索并下载FDTD教学相关论文
# ============================================================

def phase1_literature_search():
    """阶段一：文献搜索与下载（arXiv API真实搜索）"""
    log("="*50)
    log("Phase 1: 文献搜索与下载")
    log("="*50)
    feishu_notify("📚 Phase 1: 开始搜索FDTD教学相关文献...")

    # 搜索关键词 — 教学类FDTD
    keywords = [
        "FDTD physics education simulation",
        "finite difference time domain teaching electromagnetics",
        "FDTD pedagogical tool undergraduate",
        "FDTD electromagnetic visualization teaching"
    ]
    
    all_papers = []
    seen_ids = set()
    for kw in keywords:
        log(f"\n搜索: {kw}")
        papers = arxiv_search(kw, max_results=8)
        for p in papers:
            pid = p["arxiv_id"]
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(p)
    
    log(f"\n共找到 {len(all_papers)} 篇去重论文")
    
    # 保存结果
    result = {"papers": all_papers}
    json_path = f"{PHASE1}/references/literature_search_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 同时生成可读Markdown
    md_lines = ["# FDTD教学论文文献搜索结果", f"搜索时间: {datetime.now()}", f"共 {len(all_papers)} 篇论文\n"]
    for i, p in enumerate(all_papers, 1):
        md_lines.append(f"## {i}. {p['title']}")
        md_lines.append(f"- 作者: {p['authors']}")
        md_lines.append(f"- 年份: {p['year']}")
        md_lines.append(f"- arXiv: {p['arxiv_id']}")
        md_lines.append(f"- 摘要: {p['reason']}\n")
    md_path = f"{PHASE1}/references/literature_search_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    
    feishu_notify(f"📚 Phase 1: 文献搜索完成，共{len(all_papers)}篇论文", files=[json_path, md_path])
    log(f"文献搜索结果保存至: {json_path}")
    log(f"Markdown版本: {md_path}")
    return True

def phase1_download_papers():
    """下载找到的论文PDF"""
    log("\n--- 下载论文PDF ---")
    feishu_notify("📥 Phase 1: 开始下载论文PDF...")
    
    target_papers = [
        {"id": "0806.4400", "title": "Sipos & Thompson 2008 - Electrodynamics on a grid (AJP)"},
        {"id": "2108.00607", "title": "FDTD teaching paper"},
        {"id": "1702.06072", "title": "Computational physics education"},
    ]
    
    pdf_dir = f"{PHASE1}/references/pdfs"
    downloaded = []
    
    for paper in target_papers:
        pdf_path = f"{pdf_dir}/arxiv_{paper['id']}.pdf"
        if os.path.exists(pdf_path):
            log(f"  已有: {paper['title']}")
            downloaded.append(pdf_path)
            continue
        
        url = f"https://arxiv.org/pdf/{paper['id']}.pdf"
        log(f"  下载: {paper['title']}")
        try:
            r = subprocess.run(["curl", "-sL", "-o", pdf_path, url], 
                             capture_output=True, timeout=60)
            if r.returncode == 0 and os.path.getsize(pdf_path) > 1000:
                log(f"    ✅ {os.path.getsize(pdf_path)} bytes")
                downloaded.append(pdf_path)
            else:
                log(f"    ❌ 下载失败")
        except Exception as e:
            log(f"    ❌ {e}")
    
    # 保存下载清单
    manifest = {p["id"]: p["title"] for p in target_papers}
    with open(f"{PHASE1}/references/download_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    feishu_notify(f"📥 Phase 1: 下载完成，共{len(downloaded)}篇PDF")
    return True

# ============================================================
# Phase 2: 研究执行 — 运行仿真代码
# ============================================================

def phase2_run_simulations():
    """阶段二：运行所有仿真场景，生成论文用图"""
    log("\n" + "="*50)
    log("Phase 2: 运行仿真代码")
    log("="*50)
    feishu_notify("🔬 Phase 2: 开始运行FDTD仿真...")
    
    code_path = f"{PHASE2}/code/fdtd_2d_tm.py"
    if not os.path.exists(code_path):
        log("  ❌ FDTD代码不存在")
        return False
    
    # 运行仿真+出图
    log("  运行FDTD仿真+生成论文用图...")
    r = subprocess.run(
        ["/home/mrcao/.hermes/venv/bin/python3", code_path],
        capture_output=True, text=True, timeout=600
    )
    
    if r.returncode != 0:
        log(f"  ❌ 仿真失败: {r.stderr[-500:]}")
        feishu_notify("🔬 Phase 2: FDTD仿真失败 ❌")
        return False
    
    log(f"  ✅ 仿真完成")
    
    # 检查输出图
    fig_dir = f"{PHASE3}/figures"
    figs = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
    log(f"  生成了 {len(figs)} 张图: {', '.join(figs)}")
    
    # 发送飞书通知+带图
    feishu_notify(f"🔬 Phase 2: FDTD仿真完成 ✅ 共{len(figs)}张图", 
                  files=[f"{fig_dir}/{f}" for f in figs[:3]])
    
    return True

# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        step = sys.argv[1]
    else:
        step = "all"
    
    if step in ("all", "phase1"):
        log("\n📚 Phase 1: 文献搜索与下载")
        phase1_literature_search()
        phase1_download_papers()
    
    if step in ("all", "phase2"):
        log("\n🔬 Phase 2: 仿真执行")
        phase2_run_simulations()
    
    log("\n✅ 完成")
