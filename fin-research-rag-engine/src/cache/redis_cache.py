"""缓存层：Redis 兼容接口，无 Redis 自动降级为内存 LRU。

RAG 的延迟大头在多路召回与重排上，而**真实业务里的提问是高度重复的**：
「开户要什么材料」「R2 能买什么」这类问题一天会被问几十次。
因此缓存不是优化，是必需。

为什么做成可替换后端
--------------------
生产用 Redis（多实例共享、可持久化），本地 / CI 用内存 LRU（零依赖、确定性）。
两者实现同一套 `get / set / stats` 接口，因此**引擎代码里没有"有没有 Redis"的分支**。

缓存键必须包含**全部影响结果的参数**（问题、过滤条件、TopK、检索模式）。
只按问题文本做键是最常见的错误：同一个问题加了 `year >= 2024` 过滤之后，
命中旧缓存会返回**没过滤的结果**——用户以为筛选生效了，实际上没有。
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import CACHE_MAX_ENTRIES, CACHE_TTL_S, REDIS_URL
from ..utils.jsonable import to_plain
from ..utils.text import stable_hash

__all__ = ["CacheStats", "MemoryLRUCache", "RedisCache", "build_cache", "cache_key"]


def cache_key(question: str, expr: str = "", top_k: int = 0, mode: str = "hybrid", route: str = "") -> str:
    """构造缓存键：**所有影响结果的参数都要进键**。"""
    payload = json.dumps(
        {"q": question.strip(), "expr": expr.strip(), "top_k": int(top_k), "mode": mode, "route": route},
        ensure_ascii=False,
        sort_keys=True,
    )
    return "finrag:" + stable_hash(payload, size=16)


@dataclass
class CacheStats:
    """缓存命中统计（命中率是回答"响应时间怎么压到 50ms"的关键证据）。"""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    expired: int = 0
    backend: str = "memory"

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "expired": self.expired,
            "hit_rate": round(self.hit_rate, 4),
        }


class MemoryLRUCache:
    """带 TTL 的内存 LRU 缓存。"""

    name = "memory"

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES, ttl_s: int = CACHE_TTL_S) -> None:
        self.max_entries = int(max_entries)
        self.ttl_s = int(ttl_s)
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self.stats = CacheStats(backend=self.name)

    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        item = self._data.get(key)
        if item is None:
            self.stats.misses += 1
            return None
        expires_at, value = item
        if expires_at and expires_at < time.time():
            self._data.pop(key, None)
            self.stats.expired += 1
            self.stats.misses += 1
            return None
        self._data.move_to_end(key)
        self.stats.hits += 1
        return value

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None) -> None:
        ttl = self.ttl_s if ttl_s is None else int(ttl_s)
        expires_at = (time.time() + ttl) if ttl > 0 else 0.0
        self._data[key] = (expires_at, value)
        self._data.move_to_end(key)
        self.stats.sets += 1
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)
            self.stats.evictions += 1

    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "entries": len(self._data), "max_entries": self.max_entries, "ttl_s": self.ttl_s}


class RedisCache:
    """Redis 后端（可选依赖）。连不上时**自动降级**为内存缓存，而不是抛错。"""

    name = "redis"

    def __init__(self, url: str = REDIS_URL, ttl_s: int = CACHE_TTL_S) -> None:
        self.url = url
        self.ttl_s = int(ttl_s)
        self.stats = CacheStats(backend=self.name)
        self._fallback = MemoryLRUCache(ttl_s=ttl_s)
        self._client = None
        self._connected = False
        self._error = ""
        self._connect()

    def _connect(self) -> None:
        if not self.url:
            self._error = "未配置 REDIS_URL"
            return
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(self.url, decode_responses=True, socket_timeout=2.0)
            client.ping()
            self._client = client
            self._connected = True
        except Exception as exc:  # noqa: BLE001 - 连不上就降级，不要让服务起不来
            self._error = f"{type(exc).__name__}: {exc}"
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        if not self._connected or self._client is None:
            value = self._fallback.get(key)
            self.stats.hits, self.stats.misses = self._fallback.stats.hits, self._fallback.stats.misses
            return value
        try:
            raw = self._client.get(key)
        except Exception:  # noqa: BLE001 - 运行期断连同样降级
            self._connected = False
            return self._fallback.get(key)
        if raw is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None) -> None:
        ttl = self.ttl_s if ttl_s is None else int(ttl_s)
        if not self._connected or self._client is None:
            self._fallback.set(key, value, ttl)
            self.stats.sets += 1
            return
        try:
            self._client.setex(key, ttl, json.dumps(to_plain(value), ensure_ascii=False))
            self.stats.sets += 1
        except Exception:  # noqa: BLE001
            self._connected = False
            self._fallback.set(key, value, ttl)
            self.stats.sets += 1

    def delete(self, key: str) -> bool:
        if not self._connected or self._client is None:
            return self._fallback.delete(key)
        try:
            return bool(self._client.delete(key))
        except Exception:  # noqa: BLE001
            return False

    def clear(self) -> None:
        self._fallback.clear()
        if self._connected and self._client is not None:
            try:
                for key in self._client.scan_iter("finrag:*"):
                    self._client.delete(key)
            except Exception:  # noqa: BLE001
                pass

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "connected": self._connected,
            "url": self.url or "(未配置)",
            "error": self._error,
            "ttl_s": self.ttl_s,
        }


def build_cache(prefer: str = "auto", url: str = REDIS_URL):
    """构造缓存：auto 时优先 Redis，不可用回落内存。"""
    key = (prefer or "auto").lower()
    if key == "memory":
        return MemoryLRUCache()
    if key in ("redis", "auto"):
        cache = RedisCache(url=url)
        if cache.connected:
            return cache
        if key == "redis":
            # 显式要 Redis 但连不上：仍然降级，但在 describe() 里如实说明
            return cache
        return cache
    return MemoryLRUCache()
