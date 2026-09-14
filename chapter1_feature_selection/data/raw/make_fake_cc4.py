# data/raw/make_fake_cc4.py
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

random.seed(42)

OUT = Path(__file__).resolve().parent / "cc4_export.jsonl"

# 生成 1 小时的事件流（每5秒一条）
start = datetime(2026, 1, 1, 0, 0, 0)
events = []

for i in range(720):  # 720*5s = 3600s
    t = start + timedelta(seconds=5 * i)
    ts = int(t.timestamp())

    # 模拟攻击段：第 20min~35min
    is_attack = 1 if (20 * 60 <= 5 * i <= 35 * 60) else 0

    e = {
        "timestamp": ts,
        "agent_id": "A1",
        "event_type": random.choice(["flow", "process", "auth", "dns"]),
        "bytes_in": random.randint(100, 5000) + (2000 if is_attack else 0),
        "bytes_out": random.randint(100, 5000) + (3000 if is_attack else 0),
        "duration": random.random() * (3.0 if not is_attack else 8.0),
        "packet_count": random.randint(1, 50) + (20 if is_attack else 0),
        "src_ip": random.choice(["10.0.0.1", "10.0.0.2", "10.0.0.3"]),
        "dst_ip": random.choice(["192.168.1.10", "192.168.1.20", "192.168.1.30"]),
        "user": random.choice(["alice", "bob", "charlie"]),
        "process_name": random.choice(["chrome.exe", "svchost.exe", "powershell.exe"]),
        "command_line": "powershell -enc xxx" if is_attack else "normal cmd",
        "url": "http://malicious.example.com" if is_attack else "http://safe.example.com",
        "is_attack": is_attack
    }
    events.append(e)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for e in events:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"[OK] fake cc4_export.jsonl generated -> {OUT}")
