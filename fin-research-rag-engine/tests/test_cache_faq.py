"""缓存与 FAQ 测试。"""

from __future__ import annotations

import time

import pytest

from src.cache import MemoryLRUCache, RedisCache, build_cache, cache_key
from src.faq import FAQIndex, identifiers


# ---------------------------------------------------------------------------
# 缓存键
# ---------------------------------------------------------------------------
def test_cache_key_changes_with_question():
    assert cache_key("问题A") != cache_key("问题B")


def test_cache_key_changes_with_filter():
    """过滤条件必须进键，否则「加了过滤却命中旧缓存」会返回错误结果。"""
    assert cache_key("同一问题", expr="") != cache_key("同一问题", expr="year = 2024")


def test_cache_key_changes_with_topk_and_mode_and_route():
    base = cache_key("同一问题")
    assert base != cache_key("同一问题", top_k=3)
    assert base != cache_key("同一问题", mode="dense")
    assert base != cache_key("同一问题", route="clause")


def test_cache_key_is_stable_across_calls():
    assert cache_key("问题", expr="x", top_k=2) == cache_key("问题", expr="x", top_k=2)


def test_cache_key_has_prefix():
    assert cache_key("问题").startswith("finrag:")


# ---------------------------------------------------------------------------
# 内存 LRU
# ---------------------------------------------------------------------------
def test_cache_set_and_get(cache):
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}


def test_cache_miss_returns_none(cache):
    assert cache.get("missing") is None
    assert cache.stats.misses == 1


def test_cache_hit_rate(cache):
    cache.set("k", 1)
    cache.get("k")
    cache.get("missing")
    assert cache.stats.hit_rate == pytest.approx(0.5)


def test_cache_evicts_oldest_when_full():
    small = MemoryLRUCache(max_entries=2, ttl_s=60)
    small.set("a", 1)
    small.set("b", 2)
    small.set("c", 3)
    assert small.get("a") is None
    assert small.get("c") == 3
    assert small.stats.evictions >= 1


def test_cache_expires_entries():
    short = MemoryLRUCache(max_entries=4, ttl_s=1)
    short.set("k", 1, ttl_s=1)
    time.sleep(1.05)
    assert short.get("k") is None
    assert short.stats.expired == 1


def test_cache_delete_and_clear(cache):
    cache.set("k", 1)
    assert cache.delete("k") is True
    assert cache.delete("k") is False
    cache.set("k", 1)
    cache.clear()
    assert len(cache) == 0


def test_cache_describe(cache):
    described = cache.describe()
    assert described["backend"] == "memory"
    assert described["max_entries"] == 8


def test_cache_to_dict_stats(cache):
    cache.set("k", 1)
    cache.get("k")
    payload = cache.stats.to_dict()
    assert payload["hits"] == 1
    assert payload["sets"] == 1


# ---------------------------------------------------------------------------
# Redis 降级
# ---------------------------------------------------------------------------
def test_redis_cache_degrades_without_server(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    cache = RedisCache(url="")
    assert not cache.connected
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}
    assert cache.describe()["error"]


def test_redis_cache_degrades_on_unreachable_server(monkeypatch):
    cache = RedisCache(url="redis://127.0.0.1:6399/0")
    assert not cache.connected or cache.get("k") is None
    cache.set("k", 1)
    assert cache.get("k") == 1


def test_build_cache_default_is_memory():
    cache = build_cache("memory")
    assert cache.name == "memory"


def test_build_cache_auto_never_raises(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    cache = build_cache("auto", url="")
    cache.set("k", 1)
    assert cache.get("k") == 1


# ---------------------------------------------------------------------------
# FAQ
# ---------------------------------------------------------------------------
def test_faq_identifiers_extraction():
    ids = identifiers("C2 客户购买 WY2024-01，期限 90 天")
    assert "C2" in ids and "WY2024-01" in ids and "90" in ids


def test_faq_hits_exact_question(faq):
    match = faq.best("开户需要准备哪些材料？")
    assert match is not None
    assert match.entry.faq_id == "FAQ-001"


def test_faq_hits_similar_question(faq):
    match = faq.best("双录资料要保存多久？")
    assert match is not None
    assert "二十" in match.answer


def test_faq_rejects_off_topic_question(faq):
    assert faq.best("明天大盘会涨还是会跌？") is None
    assert faq.best("请帮我写一段产品营销文案，要能吸引客户") is None


def test_faq_identifier_gate_blocks_near_miss(faq):
    """「C2 客户可以买什么」不能被「R2 可以卖给 C1 吗」这条 FAQ 抢答。"""
    match = faq.best("C2 客户可以购买哪些风险等级的产品？")
    assert match is None or "C1" not in match.answer or "C2" in match.answer


def test_faq_answer_payload_shape(faq):
    payload = faq.answer("开户需要准备哪些材料？")
    assert payload["source"] == "faq"
    assert payload["updated_at"]
    assert payload["score"] > 0


def test_faq_answer_returns_none_for_long_tail(faq):
    assert faq.answer("示例集团应收账款增速为什么值得关注？") is None


def test_faq_search_returns_ranked_candidates(faq):
    matches = faq.search("合格投资者标准是什么？", top_k=3)
    assert matches
    assert matches[0].score >= matches[-1].score


def test_faq_describe_and_len(faq):
    assert len(faq) == len(faq.entries)
    assert faq.describe()["threshold"] > 0


def test_faq_empty_index():
    index = FAQIndex([])
    assert index.best("任意问题") is None
    assert index.search("任意问题") == []
    assert index.answer("任意问题") is None


def test_faq_match_to_dict(faq):
    match = faq.best("开户需要准备哪些材料？")
    payload = match.to_dict()
    assert payload["faq_id"] == "FAQ-001"
    assert 0.0 <= payload["score"] <= 1.0
