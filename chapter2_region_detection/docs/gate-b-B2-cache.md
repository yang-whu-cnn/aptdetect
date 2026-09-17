# Gate B2.5 — Persistent Prior Cache

日期：2026-09-17  
状态：**SOURCE READY / LOCAL TEST PENDING**

---

## 1. Purpose

LLM prior cache 是 formal PPO 的必要实验基础设施，不是可选性能优化。

目标：

```text
same formal state/model/prompt/config
-> same cache key
-> hit reuses frozen PriorBatch
-> no API call during PPO gradient epochs
```

不同 split/model/prompt/config/registry/state 必须隔离。

## 2. Source

```text
formal_experiments/ours/prior_cache.py
```

Canonical runtime root：

```text
outputs/lwm_rl_v2/prior_cache/
```

## 3. Canonical key

No state quantization.

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

Namespace path：

```text
<root>/<split>/<model_alias>/registry_v<version>/<prompt_version>/<cache_key>.json
```

## 4. Value schema

Each formal LLM cache entry stores：

- format version / cache key / identity；
- PriorBatch plans/raw scores/normalized preferences/sources；
- `api_success`；
- `fallback_count`；
- prompt/response SHA256；
- latency；
- retry count；
- optional usage/token/cost metadata；
- UTC creation time；
- entry SHA256 checksum。

Required metadata is validated before write.

`fallback_count` must exactly match non-LLM sources in PriorBatch.

Formal cache **硬性要求 `api_success=True`**。`api_success=False` 的 provider transaction 会被拒绝，且不会写入 cache；provider failure 不能通过 fallback 被伪装成可复用的 live success。

Forbidden serialization：

- API key；
- authorization headers；
- secrets；
- hidden CC4 truth；
- future reward/test labels。

Raw provider response is not stored by this cache module.

## 5. Runtime semantics

```text
lookup
  valid hit -> return PriorBatch, zero provider calls
  miss      -> acquire same-key lock
             -> re-check
             -> exactly one logical generator transaction
             -> validate
             -> atomic temp write + fsync + replace
             -> reload/validate
```

This means PPO optimizer epochs only read the cache; they do not repeatedly call the LLM API for an already-seen key.

## 6. Corruption / version / concurrency

### Corrupt JSON or checksum/schema mismatch

```text
rename to *.corrupt.<time_ns>
raise PriorCacheCorruptError
```

`get_or_generate()` treats the quarantined corrupt entry as a hard miss and regenerates once under lock.

### Incompatible cache format version

Hard fail with `PriorCacheFormatError`; no silent migration.

### Concurrent same-key writes

Same-key lock file serializes provider generation/write. After lock acquisition the cache is rechecked, so a second process does not intentionally repeat a logical generation after a first valid write has appeared.

## 7. Acceptance

Must pass：

- exact float32 state hash, no quantization；
- split/model/prompt/generation-config/registry change => different key；
- hit => generator call count unchanged；
- miss => one generator call；
- `api_success=false` => reject and no cache entry；
- corrupt entry quarantined and not reused；
- sensitive metadata rejected；
- atomic write leaves no temp/lock file；
- entry checksum roundtrip；
- PriorBatch roundtrip exact；
- A4/K6/H4/D27/registry provenance in identity。

## 8. Tests

```text
tests/test_prior_cache.py
```

7 tests cover：

1. exact state hashing；
2. namespace/key isolation including registry/config；
3. miss-once / hit-zero-call semantics；
4. sensitive metadata rejection；
5. failed provider transaction rejection；
6. corrupt quarantine + regeneration；
7. atomic/checksum/roundtrip/no temp-lock leftovers。

## 9. Current close condition

B2.5 remains OPEN until local tests pass. No live API calls are required for this cache unit gate.

After B2.2/B2.3/B2.5 local PASS, next step is provider/model live preflight on a tiny validation subset, then formal prior-quality generation on the frozen 240-state bank.
