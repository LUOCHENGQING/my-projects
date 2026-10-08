"""缓存子包。

    redis_cache     Redis 兼容缓存 + 内存 LRU 降级 + 命中率统计

在 RAG 全链路中的位置
--------------------
    切分 / 索引 → 【缓存查询】 → FAQ 直出 → 主体闸门 → 三路召回 → 重排 → 生成 → 校验 → 【缓存回填】

被谁调用：`src/engine.py`（`build_cache` 建实例、`cache_key` 造键、`ask()` 里 `get` / `set`）；
`src/serve.py` 的 `/health` 读 `describe()`；`tests/test_cache_faq.py` 断言两种后端行为一致。

对外关键对象：`CacheStats`（命中率证据）、`MemoryLRUCache`（本地降级后端）、`RedisCache`（生产后端）、
`build_cache(prefer="auto")`（工厂）、`cache_key(question, expr, top_k, mode, route)`（键设计见下）。

缓存键必须包含所有影响结果的参数（问题 / 过滤条件 / TopK / 检索模式 / 路由），
否则「加了过滤条件却命中旧缓存」会返回错误结果，且极难发现。
"""

from __future__ import annotations

from .redis_cache import CacheStats, MemoryLRUCache, RedisCache, build_cache, cache_key

# 公开契约：上层只应通过这里 import，不要绕过它直接引用 redis_cache 内部名字。
__all__ = ["CacheStats", "MemoryLRUCache", "RedisCache", "build_cache", "cache_key"]
