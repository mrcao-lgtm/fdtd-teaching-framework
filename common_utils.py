#!/usr/bin/env python3
"""
common_utils.py — Hermes编排器通用工具
=======================================
所有编排器/修复管线共享的通用函数：
- feishu_notify: 飞书通知+文件附件
- log: 带时间戳日志
- hermes_chat: 调Hermes CLI
"""

import os, json, subprocess, shutil, time
from datetime import datetime

# ============================================================
# 通用配置（改这里，所有脚本自动生效）
# ============================================================
FEISHU_CHAT = "feishu:oc_1fc8f0cdeb10f9b064110ed371f3f450"  # 飞书通知目标
HERMES_BIN = shutil.which("hermes") or os.path.expanduser("~/.local/bin/hermes")
MODEL = "deepseek-v4-flash"
PROVIDER = "deepseek"


def log(msg):
    """带时间戳的日志输出"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feishu_notify(title, files=None):
    """发送飞书通知+文件附件（15秒超时，失败不阻断）
    
    用法:
        feishu_notify("✅ 任务完成")
        feishu_notify("✅ 任务完成", files=["path/to/file.pdf"])
        feishu_notify("✅ 任务完成", files=["file1.pdf", "file2.json"])
    """
    if not HERMES_BIN:
        log("  ⚠️ hermes not found，跳过飞书通知")
        return False
    
    ts = datetime.now().strftime("%H:%M:%S")
    cmd = [HERMES_BIN, "send", "-t", FEISHU_CHAT, f"[{ts}] {title}"]
    
    if files:
        file_list = files if isinstance(files, list) else [files]
        for f in file_list:
            if f and os.path.exists(f):
                cmd += ["--attach", f]
    
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        ok = r.returncode == 0
        log(f"  📤 飞书: {title[:60]} {'✅' if ok else '❌'}")
        return ok
    except subprocess.TimeoutExpired:
        log(f"  ⚠️ 飞书通知超时(15s)，已跳过")
        return False
    except Exception as e:
        log(f"  ⚠️ 飞书通知失败: {e}")
        return False


def hermes_chat(prompt, timeout=600):
    """调Hermes CLI执行任务
    
    Args:
        prompt: 提示词文本
        timeout: 超时秒数（默认600=10分钟）
    
    Returns:
        {"ok": bool, "out": str, "err": str}
    """
    log(f"  🤖 调 Hermes (timeout={timeout}s)...")
    t0 = time.time()
    
    # 长prompt用文件传参
    if len(prompt) > 50000:
        tmpf = f"/tmp/hermes_prompt_{int(t0)}.txt"
        with open(tmpf, "w") as f:
            f.write(prompt)
        cmd = [HERMES_BIN, "chat", "-q", f"@{tmpf}",
               "--provider", PROVIDER, "--model", MODEL]
    else:
        cmd = [HERMES_BIN, "chat", "-q", prompt,
               "--provider", PROVIDER, "--model", MODEL]
    
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        ok = r.returncode == 0
        log(f"  {'✅' if ok else '❌'} Hermes 完成 ({dt:.0f}s, exit={r.returncode})")
        # 清理临时文件
        if len(prompt) > 50000 and os.path.exists(f"/tmp/hermes_prompt_{int(t0)}.txt"):
            os.remove(f"/tmp/hermes_prompt_{int(t0)}.txt")
        return {"ok": ok, "out": r.stdout[-3000:] if r.stdout else "",
                "err": r.stderr[-500:] if r.stderr else ""}
    except subprocess.TimeoutExpired:
        log(f"  ❌ Hermes 超时({timeout}s)")
        return {"ok": False, "out": "", "err": "timeout"}
    except Exception as e:
        log(f"  ❌ Hermes 异常: {e}")
        return {"ok": False, "out": "", "err": str(e)}


def ensure_dirs(*paths):
    """确保目录存在"""
    for p in paths:
        os.makedirs(p, exist_ok=True)


if __name__ == "__main__":
    # 测试飞书通知
    feishu_notify("🔧 common_utils.py 测试通知 — 通用通知模块就绪")
