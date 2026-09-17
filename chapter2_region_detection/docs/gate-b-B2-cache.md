# Gate B2.5 — Persistent Prior Cache

日期：2026-09-17  
状态：**PASS / CLOSED (UNIT/PROTOCOL GATE)**

---

## 1. Purpose

LLM prior cache 是 formal PPO 的必要实验基础设施，不是可选性能优化。

目标：

```text
same formal state/model/prompt/config
-> same cache key
-> hit reuses frozen PriorBatch
-> no repeated API call during PPO optimizer epochs
```

不同 split/model/prompt/config/registry/state 必须隔离。

## 2. Source

```text
formal_experiments/ours/prior_cache.py
```

Canonical root：

```text
outputs/lwm_rl_v2/prior_cache/
```

## 3. Canonical key

Main experiment 禁止 state quantization。

Key material：

```text
cache format version
split
model alias
provider
exact model ID
registry version
registry SHA256
prompt version
A=4
K=6
H=4
generation config
generation-config SHA256
state dtype=float32_le
state shape=[27]
exact D27 float32 state SHA256
```

Final key：

```text
SHA256(canonical JSON key material)
```

Namespace：

```text
<root>/<split>/<model_alias>/registry_v<version>/<prompt_version>/<cache_key>.json
```

## 4. Value / safety contract

每个 formal live cache entry 保存：

- identity / cache key / format；
- PriorBatch plans / raw scores / normalized preferences / sources；
- `api_success`；
- `fallback_count`；
- prompt/response SHA256；
- latency；
- retry count；
- optional token/cost metadata；
- UTC creation time；
- entry checksum。

硬规则：

```text
api_success must be True
```

Provider failure 不能通过 fallback 被缓存为“成功 live prior”。

Forbidden serialization：

- API key；
- authorization headers；
- secrets；
- hidden CC4 truth；
- future reward/test labels；
- raw provider response。

## 5. Runtime semantics

```text
lookup
  valid hit -> return PriorBatch; provider calls unchanged
  miss      -> same-key lock
             -> re-check
             -> one logical provider transaction
             -> validate
             -> atomic temp write + fsync + replace
             -> reload/validate
```

Corrupt JSON/checksum/schema entry：

```text
quarantine to *.corrupt.<time_ns>
no silent reuse
```

Incompatible format version：hard fail；no silent migration。

## 6. Acceptance tests

```text
tests/test_prior_cache.py
```

7 tests：

1. exact float32 state hashing/no quantization；
2. split/model/prompt/config/registry key isolation；
3. miss once / hit zero-provider-call semantics；
4. sensitive metadata rejection；
5. `api_success=false` rejection/no entry；
6. corrupt quarantine + regeneration；
7. atomic/checksum/PriorBatch roundtrip/no temp-lock leftovers。

用户本地与 B2 registry/state-bank tests 联合执行：

```text
15/15 PASS
```

其中 cache unit gate 7 tests 全部包含在这次通过结果中。

## 7. Close decision

因此：

```text
B2.5 cache implementation : PASS
Key isolation              : PASS
Failed-provider rejection  : PASS
Corruption handling        : PASS
Atomic write               : PASS
Sensitive serialization    : PASS

FINAL STATUS: PASS / CLOSED
```

这里的 CLOSED 只表示 cache unit/protocol 本身通过。真正的 provider transaction metadata 会在下一步三模型 live preflight 和 240-state prior generation 中继续集成验证。

Next：

```text
three-model live capability preflight
-> verified capability freeze
-> 240-state multi-LLM prior-quality
```
