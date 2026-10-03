"""缓存子包。

    redis_cache     Redis 兼容缓存 + 内存 LRU 降级 + 命中率统计

缓存键必须包含所有影响结果的参数（问题 / 过滤条件 / TopK / 检索模式），
否则「加了过滤条件却命中旧缓存」会返回错误结果，且极难发现。
"""

from __future__ import annotations

from .redis_cache import CacheStats, MemoryLRUCache, RedisCache, build_cache, cache_key

__all__ = ["CacheStats", "MemoryLRUCache", "RedisCache", "build_cache", "cache_key"]
