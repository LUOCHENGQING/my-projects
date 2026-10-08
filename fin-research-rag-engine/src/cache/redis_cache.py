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

在 RAG 全链路中的位置
--------------------
    切分 / 索引 → 【查缓存】 → FAQ → 主体闸门 → 三路召回 → 重排 → 生成 → 校验 → 【回填缓存】 → 接口

被谁调用：`src/engine.py`（`build_cache()` 建实例、`cache_key()` 造键、`ask()` 里 `get` / `set`；
FAQ 直出与主体闸门拒答两个分支**也会回填**，因为这两类结果同样是确定性的）；
`src/serve.py` 的 `/health` 读 `describe()`；`tests/test_cache_faq.py` 校验两种后端行为一致。

键设计（`cache_key`）
-------------------
    `"finrag:" + stable_hash(payload, size=16)`
    payload = `json.dumps({"q": 问题, "expr": 过滤表达式, "top_k": int, "mode": 检索模式, "route": 路由},
                          ensure_ascii=False, sort_keys=True)`
    其中 `stable_hash` 是 blake2b、`digest_size=16` 字节 → 32 位十六进制，**跨进程稳定**
    （刻意不用内置 `hash()`，它带 PYTHONHASHSEED 随机化，重启后缓存全失效）。
    键里含 `route`，因此「同一问题、不同路由」会各自缓存，不会互相串味。

降级路径（**任何一步失败都不抛异常，只降级**）
------------------------------------------
    1. `REDIS_URL` 为空 → 不 import redis，`_error="未配置 REDIS_URL"`，`_connected=False`；
    2. 未安装 redis 或 `ping()` 失败 → `_error` 记下异常，`_connected=False`，读写走 `_fallback`；
    3. 运行期断连（`get` / `set` / `delete` 抛异常）→ 把 `_connected` 置回 False 并改走内存 LRU，
       也就是说**Redis 挂掉后服务照跑**，只是退回单机缓存；
    4. `build_cache(prefer)` 负责选后端：`memory` 直接用内存；`redis` / `auto` 都先试 Redis，
       连不上时返回的仍是 `RedisCache` 对象（它自带内存降级），差别只在 `describe()` 里如实报错。

容量与 TTL：`CACHE_MAX_ENTRIES`（默认 512，超出按 LRU 淘汰）、`CACHE_TTL_S`（默认 600 秒）。
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
    """构造缓存键：**所有影响结果的参数都要进键**。

    参数：question 问题（strip 后入键）；expr 元数据过滤表达式（strip）；
          top_k 最终取几条（转 int）；mode 检索模式，默认 "hybrid"；route 问题类型路由。
    返回：str —— `"finrag:" + stable_hash(payload, size=16)`，即 `finrag:` 加 32 位十六进制，
          跨进程 / 重启后依然稳定；`sort_keys=True` 保证字段顺序变化不改变键。
    副作用/异常：无（`top_k` 传 None 会抛 TypeError，调用方 `engine.ask()` 已用 `top_k or self.final_top_k` 兜住）。
    """
    payload = json.dumps(
        {"q": question.strip(), "expr": expr.strip(), "top_k": int(top_k), "mode": mode, "route": route},
        ensure_ascii=False,
        sort_keys=True,
    )
    return "finrag:" + stable_hash(payload, size=16)


@dataclass
class CacheStats:
    """缓存命中统计（命中率是回答"响应时间怎么压到 50ms"的关键证据）。

    字段：
        hits        命中次数（内存后端在 `get()` 里累加；Redis 后端在 Redis 命中时累加）
        misses      未命中次数（**过期淘汰也算一次 miss**）
        sets        写入次数
        evictions   LRU 淘汰条数（只有内存后端会累加）
        expired     TTL 过期条数（只有内存后端会累加）
        backend     后端名：`MemoryLRUCache.name` = "memory"、`RedisCache.name` = "redis"

    关键派生属性：`hit_rate` = `hits / (hits + misses)`，一次请求都没有时返回 0.0（不是 1.0）。
    注：Redis 后端降级到内存时，只有 hits / misses 会从 `_fallback` 同步回来，
    sets / expired / evictions 不回流，因此那段时间 `describe()` 里的 sets 会偏低。
    """

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    expired: int = 0
    backend: str = "memory"

    @property
    def hit_rate(self) -> float:
        """缓存命中率。参数：无。

        返回：float —— `hits / (hits + misses)`；没有任何请求时返回 0.0（避免把"没流量"显示成 100% 命中）。
        副作用/异常：无。
        """
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """导出为 dict（backend / 五个计数 / hit_rate），供 `/health` 与评测报告。

        参数：无。
        返回：Dict[str, Any]；hit_rate 保留 4 位小数。
        副作用/异常：无。
        """
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
    """带 TTL 的内存 LRU 缓存（本地 / CI 默认后端，零依赖、确定性）。

    关键属性：
        name         固定为 "memory"，会写进 `CacheStats.backend` 与 `describe()`
        max_entries  条数上限，超出按 LRU 淘汰最久未使用的键
        ttl_s        默认 TTL（秒）；**传入 ≤ 0 表示永不过期**（过期时间戳记 0.0，判空即跳过）
        _data        `OrderedDict[key] = (expires_at, value)`；每次命中把它 move_to_end，
                     因此"最前面 = 最久没用"
        stats        `CacheStats`，本对象持有，供 `/health` 汇总

    线程安全说明：只用 `OrderedDict` 的原子单步操作，**没有加锁**——多线程并发下计数可能少记，
    但不会读到坏数据；生产多实例场景应换成 `RedisCache`。
    """

    name = "memory"

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES, ttl_s: int = CACHE_TTL_S) -> None:
        """初始化内存缓存。

        参数：max_entries 条数上限，默认 `CACHE_MAX_ENTRIES`（配置默认 512）；
              ttl_s 默认存活秒数，默认 `CACHE_TTL_S`（配置默认 600）。
        返回：无（构造函数）。
        副作用/异常：无 IO、无网络（这也是它作为 CI 默认后端的原因）。
        """
        self.max_entries = int(max_entries)
        self.ttl_s = int(ttl_s)
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self.stats = CacheStats(backend=self.name)

    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        """读缓存。

        参数：key 缓存键（由 `cache_key()` 生成）。
        返回：命中的值（**原对象引用**，不做深拷贝）；未命中 / 已过期返回 None。
        副作用：命中时把该键移到队尾（LRU 语义）并 `stats.hits += 1`；
                命中过期项时删除它并 `stats.expired += 1`、`stats.misses += 1`。
        异常：无。
        """
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
        """写缓存。

        参数：key 缓存键；value 任意对象（**按引用存放，不序列化、不拷贝**）；
              ttl_s 本条记录的存活秒数，None 表示用构造时的 `ttl_s`；≤ 0 表示永不过期。
        返回：None。
        副作用：覆盖写并移到队尾；`stats.sets += 1`；超出 `max_entries` 时循环淘汰最旧键并累加 `evictions`。
        异常：无。
        """
        ttl = self.ttl_s if ttl_s is None else int(ttl_s)
        expires_at = (time.time() + ttl) if ttl > 0 else 0.0
        self._data[key] = (expires_at, value)
        self._data.move_to_end(key)
        self.stats.sets += 1
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)
            self.stats.evictions += 1

    def delete(self, key: str) -> bool:
        """删除单个键。参数：key 缓存键；返回：bool —— 真的删掉了 True，键本来不存在 False。副作用：无统计变化。"""
        return self._data.pop(key, None) is not None

    def clear(self) -> None:
        """清空全部条目。参数：无；返回：None；副作用：清空 `_data`，**统计计数不清零**（命中率是历史累计值）。"""
        self._data.clear()

    def __len__(self) -> int:
        """当前条目数（含尚未被访问到的过期项）。参数：无；返回：int；副作用/异常：无。"""
        return len(self._data)

    def describe(self) -> Dict[str, Any]:
        """自述信息，进 `/health` 的 `cache` 段。

        参数：无。
        返回：Dict[str, Any] —— backend（"memory"）/ entries（当前条数）/ max_entries / ttl_s。
        副作用/异常：无。
        """
        return {"backend": self.name, "entries": len(self._data), "max_entries": self.max_entries, "ttl_s": self.ttl_s}


class RedisCache:
    """Redis 后端（可选依赖）。连不上时**自动降级**为内存缓存，而不是抛错。

    关键属性：
        name        固定为 "redis"（即使当前已降级，`CacheStats.backend` 仍如实写 "redis"）
        url         连接串（来自 `REDIS_URL`；空串表示未配置）
        ttl_s       写入用的默认 TTL（秒），最终由 `setex` 生效
        _fallback   内存 LRU 兜底实例（`MemoryLRUCache(ttl_s=ttl_s)`，容量取配置默认值）
        _client     redis 客户端；未连接时为 None
        _connected  当前是否真的连着 Redis（**构造时探测一次，之后断连会被置回 False**）
        _error      最近一次连接失败的原因，写进 `describe()`；空串表示没出错
        stats       `CacheStats(backend="redis")`

    对外契约与 `MemoryLRUCache` 完全一致（get / set / delete / clear / describe），
    因此上层代码里**没有任何"有没有 Redis"的分支**。
    """

    name = "redis"

    def __init__(self, url: str = REDIS_URL, ttl_s: int = CACHE_TTL_S) -> None:
        """初始化并**立即探测一次连接**。

        参数：url Redis 连接串，默认 `REDIS_URL`；ttl_s 默认 TTL 秒数，默认 `CACHE_TTL_S`。
        返回：无（构造函数）。
        副作用：构造时就延迟导入 redis 并发起 `ping()`——连不上只记录 `_error`，
                不抛异常，以免"缓存不可用"导致整个服务起不来。
        异常：无（所有失败都在 `_connect()` 内被吞成 `_error`）。
        """
        self.url = url
        self.ttl_s = int(ttl_s)
        self.stats = CacheStats(backend=self.name)
        self._fallback = MemoryLRUCache(ttl_s=ttl_s)
        self._client = None
        self._connected = False
        self._error = ""
        self._connect()

    def _connect(self) -> None:
        """建立 Redis 连接（仅构造时调用一次）。

        参数：无。
        返回：None。
        副作用：成功则设置 `_client` 与 `_connected=True`；失败或未配置 url 则写 `_error` 并保持 `_connected=False`。
                 连接参数固定 `decode_responses=True`（取回 str 而非 bytes）与 `socket_timeout=2.0`（避免卡住请求）。
        异常：无——`import redis` 失败（未装可选依赖）与 `ping()` 失败走同一个 except，只降级不抛错。
        """
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
        """当前是否连着 Redis。参数：无；返回：bool（**运行期断连后会变 False**）。副作用/异常：无。"""
        return self._connected

    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        """读缓存；Redis 不可用时改读内存兜底缓存。

        参数：key 缓存键。
        返回：命中的值（JSON 能解出来的按 JSON 解，解不出来原样返回）；未命中返回 None。
        副作用：Redis 命中 `stats.hits += 1`、未命中 `stats.misses += 1`；
                降级分支会把 `_fallback.stats` 的 hits / misses **同步回 self.stats**
                （这样命中率统计不会因为降级而失真；sets / expired / evictions 不同步）。
                运行期异常会把 `_connected` 置回 False（后续请求不再尝试 Redis）。
        异常：无（断连、JSON 解析失败都被处理成返回值或降级）。
        """
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
        """写缓存；Redis 不可用时写内存兜底缓存。

        参数：key 缓存键；value 任意对象（写 Redis 前会经 `to_plain()` 净化 numpy 标量再 JSON 序列化）；
              ttl_s 本条记录的存活秒数，None 表示用构造时的 `ttl_s`。
        返回：None。
        副作用：Redis 分支用 `setex` 原子写入并 `stats.sets += 1`；运行期异常时置
                `_connected=False`、转写兜底缓存并同样 `stats.sets += 1`（**不抛异常，用户无感**）。
        异常：无。
        """
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
        """删除单个键。

        参数：key 缓存键。
        返回：bool —— Redis 分支返回 `bool(DELETE 结果)`；降级分支返回兜底缓存的删除结果；
              **运行期异常时返回 False**（当作"没删掉"，不做二次降级尝试）。
        副作用：无统计变化。
        异常：无。
        """
        if not self._connected or self._client is None:
            return self._fallback.delete(key)
        try:
            return bool(self._client.delete(key))
        except Exception:  # noqa: BLE001
            return False

    def clear(self) -> None:
        """清空缓存：先清内存兜底，再按 `finrag:*` 前缀扫描删除 Redis 里的键。

        参数：无。
        返回：None。
        副作用：删键（Redis 用 `scan_iter` 而不是 `KEYS`，避免在大库上阻塞 Redis）；Redis 分支异常被静默忽略。
        异常：无。
        """
        self._fallback.clear()
        if self._connected and self._client is not None:
            try:
                for key in self._client.scan_iter("finrag:*"):
                    self._client.delete(key)
            except Exception:  # noqa: BLE001
                pass

    def describe(self) -> Dict[str, Any]:
        """自述信息，进 `/health` 的 `cache` 段——**必须能一眼看出当前是 Redis 还是降级**。

        参数：无。
        返回：Dict[str, Any] —— backend（"redis"）/ connected / url（未配置时写 "(未配置)"）/
              error（最近一次连接失败原因）/ ttl_s。**不回显任何凭据**（只回显完整 URL，含密码的 URL 需自行注意）。
        副作用/异常：无。
        """
        return {
            "backend": self.name,
            "connected": self._connected,
            "url": self.url or "(未配置)",
            "error": self._error,
            "ttl_s": self.ttl_s,
        }


def build_cache(prefer: str = "auto", url: str = REDIS_URL):
    """构造缓存：auto 时优先 Redis，不可用回落内存。

    参数：prefer 期望后端（大小写不敏感）：`"memory"` 直接用内存 LRU；
          `"redis"` / `"auto"` 都先尝试 Redis；其他任何取值一律回落内存；
          url Redis 连接串，默认 `REDIS_URL`。
    返回：`MemoryLRUCache` 或 `RedisCache` —— 两者对外契约一致（get / set / delete / clear / describe）。
    副作用：`"redis"` / `"auto"` 分支会尝试连接 Redis（构造时 `ping()` 一次，失败只记录错误）。
    异常：无。
    注：实际实现为 `if key == "redis": return cache` 与紧随其后的 `return cache` 返回**同一个对象**——
        「显式要 Redis 但连不上」并不会换成 `MemoryLRUCache`，而是返回自带内存降级的 `RedisCache`；
        两种取值真正的差别只体现在 `describe()` 里能否看出"没连上"（`connected=False` + `error`）。
    """
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
