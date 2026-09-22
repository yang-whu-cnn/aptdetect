# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import os
import re

import numpy as np

# 兼容两种文件名：有的人项目里叫 actions.py，有的人叫 action_space.py
try:
    from src.actions import NAME2ID, ACTION_SPACE  # type: ignore
except Exception:
    try:
        from src.action_space import NAME2ID, ACTION_SPACE  # zip里常见是这个
    except Exception:
        # 最后兜底：保证能跑
        ACTION_SPACE = ["monitor", "light_evidence", "heavy_evidence", "local_mitigate", "strong_mitigate"]
        NAME2ID = {n: i for i, n in enumerate(ACTION_SPACE)}


def _action_names() -> List[str]:
    names: List[str] = []
    for a in ACTION_SPACE:
        # ACTION_SPACE 可能是 str 列表，也可能是 ActionDef dataclass 列表
        if hasattr(a, "name"):
            names.append(str(getattr(a, "name")))
        else:
            names.append(str(a))
    return names


def softmax(x: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """env_region.py 会 from src.llm_prior import softmax，所以必须存在。"""
    x = np.asarray(x, dtype=np.float32)
    t = float(max(1e-6, temperature))
    z = x / t
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


@dataclass
class PriorOutput:
    """
    build_evidence() 里会用：
      prior_out.plans
      prior_out.prior_scores
      prior_out.checkpoints
    所以必须是“有属性”的对象，不能返回 dict。
    """
    backend: str
    plans: List[List[int]]              # 每个 plan 是长度 H 的 action_id 序列
    prior_scores: List[float]           # 每个 plan 一个先验分
    checkpoints: List[Dict[str, Any]]   # 全局检查点（用于一致性覆盖率 C）


class LLMPriorBase:
    def generate(self, summary) -> PriorOutput:
        raise NotImplementedError


class DummyLLMPrior(LLMPriorBase):
    """
    本地 online（base_url=local）时用的“假 LLM prior”或 HTTP 失败时回退：
    - 不依赖任何 API / key
    - 输出 K 个候选计划，每个计划长度 H
    - 保证 plan 长度==H，避免 IndexError
    """
    def __init__(self, k_candidates: int = 6, plan_horizon: int = 5, temperature: float = 0.7, seed: int = 0):
        self.K = int(k_candidates)
        self.H = int(plan_horizon)
        self.temperature = float(temperature)
        self.rng = np.random.RandomState(int(seed))

        self.n_actions = len(ACTION_SPACE)

        # 按你当前 action_space.py 的动作名做映射（缺失时回退到0）
        self.id_monitor = int(NAME2ID.get("monitor", 0))
        self.id_light = int(NAME2ID.get("light_evidence", self.id_monitor))
        self.id_heavy = int(NAME2ID.get("heavy_evidence", self.id_light))
        self.id_local = int(NAME2ID.get("local_mitigate", self.id_heavy))
        self.id_strong = int(NAME2ID.get("strong_mitigate", self.id_local))

    def _pad_to_horizon(self, base: List[int]) -> List[int]:
        if not base:
            base = [self.id_monitor]
        plan = list(base)
        if len(plan) >= self.H:
            return plan[: self.H]
        while len(plan) < self.H:
            plan.append(self.id_monitor)
        return plan

    def _default_checkpoints(self, risk: float) -> List[Dict[str, Any]]:
        # 注意：这里是“全局检查点”，会对每个候选计划统一评估覆盖率 C
        if risk >= 0.7:
            return [
                {"cnt_port_scan": {"op": ">=", "value": 1}},
                {"cnt_outbound_conn": {"op": ">=", "value": 1}},
            ]
        if risk >= 0.3:
            return [
                {"cnt_port_scan": {"op": ">=", "value": 1}},
                {"cnt_proc_spawn": {"op": ">=", "value": 1}},
            ]
        return [{"cnt_auth_fail": {"op": ">=", "value": 1}}]

    def generate(self, summary) -> PriorOutput:
        raw_stats = getattr(summary, "raw_stats", {}) or {}
        risk = float(raw_stats.get("risk_proxy", 0.0))

        # 启发式 base plan（动作 id）
        if risk >= 0.7:
            base = [self.id_heavy, self.id_local, self.id_strong, self.id_heavy, self.id_monitor]
        elif risk >= 0.3:
            base = [self.id_light, self.id_heavy, self.id_monitor, self.id_monitor]
        else:
            base = [self.id_monitor, self.id_light]

        base = self._pad_to_horizon(base)

        plans: List[List[int]] = []
        prior_scores: List[float] = []
        checkpoints = self._default_checkpoints(risk)

        for _ in range(self.K):
            plan = base.copy()
            # 轻微扰动 1~2 个位置，保证多样性
            n_mut = 1 + int(self.rng.rand() < 0.4)
            for __ in range(n_mut):
                p = int(self.rng.randint(0, self.H))
                plan[p] = int(self.rng.randint(0, self.n_actions))
            plans.append(plan)

            # 很弱的先验分（主要给 posterior 一个偏好）
            score = 0.0
            if risk >= 0.7:
                score += 0.25 * sum(1 for a in plan if a in (self.id_local, self.id_strong))
                score += 0.10 * sum(1 for a in plan if a == self.id_heavy)
            elif risk >= 0.3:
                score += 0.15 * sum(1 for a in plan if a in (self.id_light, self.id_heavy))
            else:
                score += 0.10 * sum(1 for a in plan if a == self.id_monitor)
            prior_scores.append(float(score))

        return PriorOutput(backend="dummy", plans=plans, prior_scores=prior_scores, checkpoints=checkpoints)


class HTTPChatLLMPrior(LLMPriorBase):
    """
    OpenAI-compatible HTTP LLM prior（可用于 GPT / 千问兼容接口 / 本地 vLLM）。

    支持配置（都放在 cfg['llm_prior'] 下）：
      backend: http | openai | openai_compat | qwen | dashscope
      model: gpt-4.1-mini / qwen-plus 等
      api_key: 明文key（可选，不推荐）
      api_key_env: 从环境变量读取key（推荐）
      base_url: 如 https://api.openai.com/v1 或 https://dashscope.aliyuncs.com/compatible-mode/v1
      http_url: 直接给完整地址（兼容旧字段），例如 .../v1/chat/completions
      http_timeout_sec: 30
      max_new_tokens: 512
      temperature: 0.2  # 采样温度（生成候选计划建议低一点）
    """

    def __init__(self, cfg: dict):
        c = cfg.get("llm_prior", {}) or {}
        self.cfg = cfg
        self.K = int(c.get("k_candidates", 6))
        self.H = int(c.get("plan_horizon", (cfg.get("plan_score", {}) or {}).get("plan_horizon", 5)))
        self.gen_temperature = float(c.get("temperature", 0.2))
        self.max_new_tokens = int(c.get("max_new_tokens", 512))
        self.timeout = float(c.get("http_timeout_sec", c.get("timeout_sec", 30)))
        self.model = str(c.get("model", c.get("hf_model_name", ""))).strip()
        self.backend_name = str(c.get("backend", "http"))

        self.action_names = _action_names()
        self.n_actions = len(self.action_names)

        self.http_url = self._build_http_url(c)
        self.api_key = self._load_api_key(c)
        self.headers = self._build_headers(c)

        # 默认禁用 requests 从环境变量/系统继承代理（可通过配置 llm_prior.trust_env=true 打开）
        self.trust_env = bool(c.get("trust_env", False))
        self._session = None  # 延迟初始化 requests.Session

        # 失败回退 dummy，保证本地流程不被 API 问题卡死
        self._fallback = DummyLLMPrior(
            k_candidates=self.K,
            plan_horizon=self.H,
            temperature=float(c.get("temperature", 0.7)),
            seed=int(cfg.get("seed", 0)),
        )

    def _build_http_url(self, c: dict) -> str:
        http_url = str(c.get("http_url", "")).strip()
        if http_url:
            return http_url
        base_url = str(c.get("base_url", "")).strip().rstrip("/")
        if not base_url:
            # 给出一个保守默认，便于用户只填 model+env key 先试 OpenAI
            base_url = "https://api.openai.com/v1"
        return base_url + "/chat/completions"

    def _load_api_key(self, c: dict) -> str:
        key = str(c.get("api_key", "")).strip()
        if key:
            return key
        env_name = str(c.get("api_key_env", "")).strip()
        if env_name:
            return str(os.getenv(env_name, "")).strip()
        # 兼容常见环境变量（优先 OpenAI，再 DashScope）
        for name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
            v = os.getenv(name)
            if v:
                return str(v).strip()
        return ""

    def _build_headers(self, c: dict) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # 兼容自定义 header（可选）
        extra = c.get("headers", None)
        if isinstance(extra, dict):
            for k, v in extra.items():
                headers[str(k)] = str(v)
        return headers

    def _pad(self, ids: List[int]) -> List[int]:
        if not ids:
            ids = [int(NAME2ID.get("monitor", 0))]
        out = list(ids[: self.H])
        while len(out) < self.H:
            out.append(int(NAME2ID.get("monitor", 0)))
        return out

    def _default_checkpoints(self, risk: float) -> List[Dict[str, Any]]:
        # 如果 LLM 没返回 checkpoints，就给一套默认检查点
        return self._fallback._default_checkpoints(risk)

    def _build_messages(self, summary) -> List[Dict[str, str]]:
        raw_stats = getattr(summary, "raw_stats", {}) or {}
        risk = float(raw_stats.get("risk_proxy", 0.0))
        text = str(getattr(summary, "text", "") or "")
        vec = getattr(summary, "vec", None)
        vec_str = ""
        if vec is not None:
            try:
                vec_str = ",".join([f"{float(x):.4f}" for x in list(vec)])
            except Exception:
                vec_str = str(vec)

        # 用严格 JSON 输出约束，便于稳健解析
        system_prompt = (
            "你是网络安全防御规划助手。根据当前单区域态势摘要，生成多个候选高层动作计划。"
            "只输出 JSON，不要解释，不要 Markdown。"
        )
        user_prompt = {
            "task": "generate_candidate_plans",
            "k_candidates": self.K,
            "plan_horizon": self.H,
            "available_actions": self.action_names,
            "action_desc": {
                "monitor": "继续监测",
                "light_evidence": "轻量补证",
                "heavy_evidence": "重度补证",
                "local_mitigate": "局部处置",
                "strong_mitigate": "强处置"
            },
            "state_summary_text": text,
            "state_vector": vec_str,
            "raw_stats": raw_stats,
            "risk_hint": risk,
            "output_json_schema": {
                "plans": [
                    {
                        "actions": ["monitor", "light_evidence"],
                        "score": 0.0,
                        "reason": "简短原因"
                    }
                ],
                "checkpoints": [
                    {"cnt_port_scan": {"op": ">=", "value": 1}},
                    {"cnt_outbound_conn": {"op": ">=", "value": 1}}
                ]
            },
            "constraints": [
                "actions 长度必须等于 plan_horizon",
                "score 越大表示越优先",
                "若不确定也必须给满 k_candidates 个计划",
                "checkpoints 可为空数组"
            ]
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ]

    def _get_session(self):
        try:
            import requests
        except ImportError as e:
            raise ImportError("HTTP LLM backend 需要 requests：pip install requests") from e

        if self._session is None:
            sess = requests.Session()
            # 关键：默认不读取 HTTP(S)_PROXY / ALL_PROXY 等环境变量，避免被无效代理拦截
            sess.trust_env = self.trust_env
            self._session = sess
        return self._session

    def _post_chat(self, payload: dict) -> dict:
        sess = self._get_session()

        # 显式传 proxies={}，进一步避免 requests 在部分环境下使用代理
        resp = sess.post(self.http_url, headers=self.headers, json=payload, timeout=self.timeout, proxies={})
        # 把错误信息暴露清楚，便于你调 key/base_url/model
        if resp.status_code >= 400:
            try:
                detail = resp.text
            except Exception:
                detail = ""
            raise RuntimeError(f"LLM HTTP {resp.status_code}: {detail}")
        return resp.json()

    def _extract_content(self, resp_json: dict) -> str:
        # 兼容 OpenAI-compatible chat completions
        try:
            return str(resp_json["choices"][0]["message"]["content"])
        except Exception:
            # 兜底：把整个响应转字符串，后续解析器再尽量提取JSON
            return json.dumps(resp_json, ensure_ascii=False)

    def _extract_json_text(self, text: str) -> str:
        s = str(text).strip()
        # 去掉 ```json ... ```
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
            s = re.sub(r"```$", "", s).strip()

        # 先尝试整段就是 JSON
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            return s

        # 再提取第一个 JSON 对象/数组
        m_obj = re.search(r"\{[\s\S]*\}", s)
        m_arr = re.search(r"\[[\s\S]*\]", s)
        cand = None
        if m_obj and m_arr:
            cand = m_obj.group(0) if len(m_obj.group(0)) <= len(m_arr.group(0)) else m_arr.group(0)
        elif m_obj:
            cand = m_obj.group(0)
        elif m_arr:
            cand = m_arr.group(0)
        return cand or s

    def _action_to_id(self, x: Any) -> Optional[int]:
        # 支持动作名 / 整数 / 字符串数字
        if isinstance(x, (int, np.integer)):
            i = int(x)
            return i if 0 <= i < self.n_actions else None

        s = str(x).strip()
        if s == "":
            return None
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            i = int(s)
            return i if 0 <= i < self.n_actions else None

        # 标准动作名
        if s in NAME2ID:
            return int(NAME2ID[s])

        # 别名映射（避免模型偶尔输出近义词）
        alias = {
            "noop": "monitor",
            "scan": "light_evidence",
            "monitoring": "monitor",
            "collect_evidence": "light_evidence",
            "evidence": "light_evidence",
            "deep_evidence": "heavy_evidence",
            "mitigate": "local_mitigate",
            "local_isolate": "local_mitigate",
            "isolate": "strong_mitigate",
            "patch": "strong_mitigate",
            "strong_isolate": "strong_mitigate",
        }
        k = alias.get(s.lower())
        if k and k in NAME2ID:
            return int(NAME2ID[k])

        # 允许“1.monitor”这种格式
        s2 = re.sub(r"^[0-9]+[\).:\-\s]+", "", s)
        if s2 in NAME2ID:
            return int(NAME2ID[s2])
        k2 = alias.get(s2.lower())
        if k2 and k2 in NAME2ID:
            return int(NAME2ID[k2])
        return None

    def _parse_plans_from_text(self, text: str, summary) -> PriorOutput:
        raw_stats = getattr(summary, "raw_stats", {}) or {}
        risk = float(raw_stats.get("risk_proxy", 0.0))

        json_text = self._extract_json_text(text)
        obj: Any = None
        try:
            obj = json.loads(json_text)
        except Exception:
            obj = None

        plans_ids: List[List[int]] = []
        scores: List[float] = []
        checkpoints: List[Dict[str, Any]] = []

        # 1) 标准 schema：{"plans": [...], "checkpoints": [...]} 
        if isinstance(obj, dict):
            plans_obj = obj.get("plans", [])
            cps = obj.get("checkpoints", [])
            if isinstance(cps, list):
                checkpoints = [cp for cp in cps if isinstance(cp, dict)]

            if isinstance(plans_obj, list):
                for p in plans_obj:
                    if isinstance(p, dict):
                        acts = p.get("actions", p.get("plan", []))
                        score = p.get("score", 0.0)
                    else:
                        acts = p
                        score = 0.0
                    if not isinstance(acts, list):
                        continue
                    ids = []
                    for a in acts:
                        ai = self._action_to_id(a)
                        if ai is not None:
                            ids.append(ai)
                    if ids:
                        plans_ids.append(self._pad(ids))
                        try:
                            scores.append(float(score))
                        except Exception:
                            scores.append(0.0)

        # 2) 若根是数组，也尝试当作 plans 数组
        elif isinstance(obj, list):
            for p in obj:
                if isinstance(p, dict):
                    acts = p.get("actions", p.get("plan", []))
                    score = p.get("score", 0.0)
                else:
                    acts = p
                    score = 0.0
                if not isinstance(acts, list):
                    continue
                ids = []
                for a in acts:
                    ai = self._action_to_id(a)
                    if ai is not None:
                        ids.append(ai)
                if ids:
                    plans_ids.append(self._pad(ids))
                    try:
                        scores.append(float(score))
                    except Exception:
                        scores.append(0.0)

        # 3) 兜底：从自由文本里按动作名抓取，按 H 分组
        if not plans_ids:
            text_l = str(text)
            token_candidates = re.split(r"[\n,;|]+", text_l)
            ids_flat: List[int] = []
            for tok in token_candidates:
                ai = self._action_to_id(tok.strip())
                if ai is not None:
                    ids_flat.append(ai)
            if ids_flat:
                for i in range(0, len(ids_flat), self.H):
                    plans_ids.append(self._pad(ids_flat[i: i + self.H]))
                    scores.append(0.0)

        # 4) 仍为空：回退 dummy
        if not plans_ids:
            return self._fallback.generate(summary)

        # 补足/截断到 K
        rng = np.random.RandomState(int(self.cfg.get("seed", 0)))
        while len(plans_ids) < self.K:
            p = list(plans_ids[len(plans_ids) % len(plans_ids)])
            # 做一点扰动，避免完全重复
            pos = int(rng.randint(0, self.H))
            p[pos] = int(rng.randint(0, self.n_actions))
            plans_ids.append(p)
            scores.append(float(scores[(len(scores) - 1) % len(scores)]) if scores else 0.0)
        if len(plans_ids) > self.K:
            plans_ids = plans_ids[: self.K]
            scores = scores[: self.K]

        # 若 score 全是0，用一个弱启发式做 tie-break
        if len(scores) < len(plans_ids):
            scores += [0.0] * (len(plans_ids) - len(scores))
        if np.allclose(np.asarray(scores, dtype=np.float32), 0.0):
            scores2 = []
            id_monitor = int(NAME2ID.get("monitor", 0))
            id_local = int(NAME2ID.get("local_mitigate", id_monitor))
            id_strong = int(NAME2ID.get("strong_mitigate", id_local))
            id_heavy = int(NAME2ID.get("heavy_evidence", id_monitor))
            for plan in plans_ids:
                sc = 0.0
                if risk >= 0.7:
                    sc += 0.3 * sum(1 for a in plan if a in (id_local, id_strong))
                    sc += 0.1 * sum(1 for a in plan if a == id_heavy)
                elif risk >= 0.3:
                    sc += 0.15 * sum(1 for a in plan if a in (id_heavy, id_monitor))
                else:
                    sc += 0.1 * sum(1 for a in plan if a == id_monitor)
                scores2.append(sc)
            scores = scores2

        if not checkpoints:
            checkpoints = self._default_checkpoints(risk)

        return PriorOutput(
            backend=self.backend_name,
            plans=plans_ids,
            prior_scores=[float(x) for x in scores],
            checkpoints=checkpoints,
        )

    def generate(self, summary) -> PriorOutput:
        if not self.model:
            # 模型名没配时直接回退，避免卡流程
            print("[WARN] llm_prior.model is empty, fallback to DummyLLMPrior")
            return self._fallback.generate(summary)

        if not self.api_key and not ("127.0.0.1" in self.http_url or "localhost" in self.http_url):
            # 本地代理/本地vLLM可不带key；云接口一般需要key
            print("[WARN] llm_prior api_key empty, fallback to DummyLLMPrior")
            return self._fallback.generate(summary)

        payload = {
            "model": self.model,
            "messages": self._build_messages(summary),
            "temperature": self.gen_temperature,
            "max_tokens": self.max_new_tokens,
        }

        try:
            resp_json = self._post_chat(payload)
            content = self._extract_content(resp_json)
            out = self._parse_plans_from_text(content, summary)
            # 标识后端来源，便于日志里区分
            out.backend = self.backend_name
            return out
        except Exception as e:
            print(f"[WARN] HTTPChatLLMPrior failed ({e}); fallback to DummyLLMPrior")
            return self._fallback.generate(summary)


# 可选：本地 HF 模型占位（如你后续想接 transformers）
class HFLocalLLMPrior(LLMPriorBase):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._fallback = DummyLLMPrior(
            k_candidates=int((cfg.get("llm_prior", {}) or {}).get("k_candidates", 6)),
            plan_horizon=int((cfg.get("llm_prior", {}) or {}).get("plan_horizon", 5)),
            seed=int(cfg.get("seed", 0)),
        )

    def generate(self, summary) -> PriorOutput:
        # 这里先保留占位；你后续若想本地部署Qwen可再补 transformers pipeline
        print("[WARN] hf_local backend not implemented in this patch, fallback to DummyLLMPrior")
        return self._fallback.generate(summary)


def build_llm_prior(cfg: dict) -> LLMPriorBase:
    c = cfg.get("llm_prior", {}) or {}
    backend = str(c.get("backend", "dummy")).lower()

    # H 优先用 llm_prior.plan_horizon；没有的话再用 plan_score.plan_horizon
    H = c.get("plan_horizon", None)
    if H is None:
        H = (cfg.get("plan_score", {}) or {}).get("plan_horizon", 5)

    if backend == "dummy":
        return DummyLLMPrior(
            k_candidates=c.get("k_candidates", 6),
            plan_horizon=H,
            temperature=c.get("temperature", 0.7),
            seed=cfg.get("seed", 0),
        )

    if backend in {"http", "openai", "openai_compat", "qwen", "dashscope"}:
        return HTTPChatLLMPrior(cfg)

    if backend == "hf_local":
        return HFLocalLLMPrior(cfg)

    raise ValueError(f"Unknown llm_prior.backend={backend}")
