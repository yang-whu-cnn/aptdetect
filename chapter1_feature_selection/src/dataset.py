from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class WindowRecord:
    start_ts: int
    end_ts: int
    rows: pd.DataFrame
    y_in_window: int


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
        return data


def load_events_as_df(
    path: str,
    timestamp_key: str = "timestamp",
    agent_id_key: Optional[str] = None,
    event_type_key: Optional[str] = None,
    label_key: Optional[str] = None,
    attack_stage_key: Optional[str] = None,
) -> pd.DataFrame:
    raw = _read_json_or_jsonl(path)
    df = pd.DataFrame(raw)

    if timestamp_key not in df.columns:
        raise ValueError(f"JSON中找不到 timestamp_key={timestamp_key} 列")

    ts = df[timestamp_key].astype(np.int64)
    if ts.max() > 10**12:
        ts = (ts // 1000).astype(np.int64)
    df["timestamp"] = ts

    if agent_id_key and agent_id_key in df.columns:
        df["agent_id"] = df[agent_id_key].astype(str)
    else:
        df["agent_id"] = "global"

    if event_type_key and event_type_key in df.columns:
        df["event_type"] = df[event_type_key].astype(str)
    else:
        df["event_type"] = "event"

    if label_key and label_key in df.columns:
        df["is_attack"] = df[label_key].astype(int)
    elif attack_stage_key and attack_stage_key in df.columns:
        df["is_attack"] = df[attack_stage_key].notna().astype(int)
    else:
        df["is_attack"] = 0

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_windows(
    events: pd.DataFrame,
    window_size_sec: int = 300,
    step_sec: int = 300,
) -> List[WindowRecord]:
    ts_min = int(events["timestamp"].min())
    ts_max = int(events["timestamp"].max())

    windows: List[WindowRecord] = []
    t = ts_min
    while t <= ts_max:
        start = t
        end = t + window_size_sec
        sub = events[(events["timestamp"] >= start) & (events["timestamp"] < end)]
        y_in_window = int(sub["is_attack"].max()) if len(sub) > 0 else 0
        windows.append(WindowRecord(start_ts=start, end_ts=end, rows=sub, y_in_window=y_in_window))
        t += step_sec
    return windows


def build_labels(
    windows: List[WindowRecord],
    label_mode: str = "early_warning",
    early_warning_horizon_sec: int = 900,
    window_size_sec: int = 300,
) -> np.ndarray:
    y_in = np.array([w.y_in_window for w in windows], dtype=np.int64)

    if label_mode == "in_window":
        return y_in

    if label_mode == "early_warning":
        horizon_steps = max(1, early_warning_horizon_sec // window_size_sec)
        y = np.zeros_like(y_in)
        for i in range(len(windows)):
            future = y_in[i: i + horizon_steps]
            y[i] = int(future.max() > 0)
        return y

    raise ValueError(f"未知label_mode={label_mode}")


def _windows_to_df(windows: List[WindowRecord], y: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, w in enumerate(windows):
        rows.append(
            {
                "window_id": i,
                "start_ts": w.start_ts,
                "end_ts": w.end_ts,
                "y": int(y[i]),
                "events": json.dumps(w.rows.to_dict(orient="records"), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def save_windows_table(windows: List[WindowRecord], y: np.ndarray, out_path: str) -> str:
    """
    优先保存 parquet；如果缺少 pyarrow/fastparquet，则自动回退为 pickle。
    返回实际写入路径，方便后续脚本直接读取。
    """
    import os

    dfw = _windows_to_df(windows, y)
    out_path = str(out_path)
    try:
        dfw.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        alt_path = os.path.splitext(out_path)[0] + ".pkl"
        dfw.to_pickle(alt_path)
        return alt_path


def load_windows_table(path: str) -> pd.DataFrame:
    import os

    path = str(path)
    if os.path.exists(path):
        if path.endswith(".pkl"):
            return pd.read_pickle(path)
        try:
            return pd.read_parquet(path)
        except Exception:
            alt_path = os.path.splitext(path)[0] + ".pkl"
            if os.path.exists(alt_path):
                return pd.read_pickle(alt_path)
            raise

    alt_path = os.path.splitext(path)[0] + ".pkl"
    if os.path.exists(alt_path):
        return pd.read_pickle(alt_path)

    raise FileNotFoundError(f"找不到窗口文件: {path} 或 {alt_path}")


def save_windows_parquet(windows: List[WindowRecord], y: np.ndarray, out_path: str) -> None:
    save_windows_table(windows, y, out_path)