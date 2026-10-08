"""缓存与 FAQ 测试。

覆盖的被测模块
--------------
    src.cache   cache_key 的键构成与可复现性、MemoryLRUCache（命中/淘汰/TTL/删除/统计）、
                RedisCache 无服务时的内存降级、build_cache 的后端选择
    src.faq     identifiers 标识符抽取、FAQIndex 的词面+语义双路打分、阈值命中与拒答、
                标识符闸门，以及 best / search / answer 的排序与返回结构

覆盖策略
--------
    正常路径：键随参数变化且可复现、写入后能命中、FAQ 精确提问与近义提问都能直出并带出处。
    边界：未命中、容量满触发淘汰、TTL 到期、重复删除、空 FAQ 索引、长尾问题。
    异常 / 降级：没有 REDIS_URL 与 Redis 地址不可达两种情况都必须降级为可用（不抛异常），
                build_cache("auto") 在无 Redis 环境下同样不能把异常抛给调用方。
    对抗 / 拒答：跑题问题（大盘涨跌、代写营销文案）与近邻标识符（C2 vs C1）不得被直出，
                宁可交回 RAG 也不能"答得又快又错"。
全部用例运行在 mock 环境（conftest._force_mock_env）下，缓存与 FAQ 都使用本地确定性后端。
"""

from __future__ import annotations

import time

import pytest

from src.cache import MemoryLRUCache, RedisCache, build_cache, cache_key
from src.faq import FAQIndex, identifiers


# ---------------------------------------------------------------------------
# 缓存键
# ---------------------------------------------------------------------------
def test_cache_key_changes_with_question():
    """键必须随问题文本变化：不同问题绝不能共用缓存条目（否则会串答案）。"""
    assert cache_key("问题A") != cache_key("问题B")


def test_cache_key_changes_with_filter():
    """过滤条件必须进键，否则「加了过滤却命中旧缓存」会返回错误结果。

    不变式：所有影响结果的检索参数都是键的一部分，漏一个就等于埋一次错答。
    """
    # 同一个问题，只差一个过滤条件——这正是最容易漏进键的那类参数
    assert cache_key("同一问题", expr="") != cache_key("同一问题", expr="year = 2024")


def test_cache_key_changes_with_topk_and_mode_and_route():
    """TopK、检索模式、路由三个参数都必须各自进键（少一个就会拿到口径不符的旧结果）。"""
    # 从默认键出发，每次只改一个参数，逐项定位被漏掉的字段
    base = cache_key("同一问题")
    assert base != cache_key("同一问题", top_k=3)
    assert base != cache_key("同一问题", mode="dense")
    assert base != cache_key("同一问题", route="clause")


def test_cache_key_is_stable_across_calls():
    """相同入参必须产出相同键：键用稳定哈希（blake2b），跨进程也一致，否则缓存等于失效。"""
    assert cache_key("问题", expr="x", top_k=2) == cache_key("问题", expr="x", top_k=2)


def test_cache_key_has_prefix():
    """键必须带命名空间前缀，便于在共享 Redis 实例里识别归属与批量清理。"""
    assert cache_key("问题").startswith("finrag:")


# ---------------------------------------------------------------------------
# 内存 LRU
# ---------------------------------------------------------------------------
def test_cache_set_and_get(cache):
    """写入后必须能原样取回值（含 dict 这类结构化载荷），缓存不能改变结果形状。"""
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}


def test_cache_miss_returns_none(cache):
    """未命中必须返回 None 并计入 misses：不能用空值冒充命中，否则命中率会失真。"""
    assert cache.get("missing") is None
    assert cache.stats.misses == 1


def test_cache_hit_rate(cache):
    """命中率口径必须是 hits/(hits+misses)：一命中一未命中即 0.5。"""
    cache.set("k", 1)
    cache.get("k")
    cache.get("missing")
    assert cache.stats.hit_rate == pytest.approx(0.5)


def test_cache_evicts_oldest_when_full():
    """容量超限必须淘汰最久未使用的条目并记入 evictions，不能无限增长。"""
    # 容量取 2，写入第 3 条即可触发一次淘汰
    small = MemoryLRUCache(max_entries=2, ttl_s=60)
    small.set("a", 1)
    small.set("b", 2)
    small.set("c", 3)
    assert small.get("a") is None
    assert small.get("c") == 3
    assert small.stats.evictions >= 1


def test_cache_expires_entries():
    """TTL 到期的条目必须失效并计入 expired，绝不能把过期结果当命中返回给用户。"""
    # TTL 取 1 秒再真实睡过 1.05 秒：用真实时间触发过期分支，不 mock 时钟以免掩盖实现
    short = MemoryLRUCache(max_entries=4, ttl_s=1)
    short.set("k", 1, ttl_s=1)
    time.sleep(1.05)
    assert short.get("k") is None
    assert short.stats.expired == 1


def test_cache_delete_and_clear(cache):
    """delete 必须返回是否真的删掉（重复删返回 False），clear 必须清空全部条目。"""
    cache.set("k", 1)
    assert cache.delete("k") is True
    assert cache.delete("k") is False
    cache.set("k", 1)
    cache.clear()
    assert len(cache) == 0


def test_cache_describe(cache):
    """describe 必须自报后端类型与容量上限，运维排查「缓存为什么没生效」靠它。"""
    described = cache.describe()
    assert described["backend"] == "memory"
    # 容量来自夹具构造参数（8），用于确认 describe 反映的是实例而非全局配置
    assert described["max_entries"] == 8


def test_cache_to_dict_stats(cache):
    """统计必须可序列化：hits / sets 等计数要能直接进报告与接口响应。"""
    cache.set("k", 1)
    cache.get("k")
    payload = cache.stats.to_dict()
    assert payload["hits"] == 1
    assert payload["sets"] == 1


# ---------------------------------------------------------------------------
# Redis 降级
# ---------------------------------------------------------------------------
def test_redis_cache_degrades_without_server(monkeypatch):
    """未配置 Redis 时必须降级为内存实现：功能照常可用，并把降级原因写进 describe。"""
    # 显式删掉环境变量，模拟 CI / 本地「没有 Redis」的默认环境
    monkeypatch.delenv("REDIS_URL", raising=False)
    cache = RedisCache(url="")
    assert not cache.connected
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}
    assert cache.describe()["error"]


def test_redis_cache_degrades_on_unreachable_server(monkeypatch):
    """Redis 地址不可达时不得抛异常：读写都要落到内存兜底，引擎侧无需感知。"""
    # 指向一个几乎不可能有服务的端口，触发连接失败分支
    cache = RedisCache(url="redis://127.0.0.1:6399/0")
    assert not cache.connected or cache.get("k") is None
    cache.set("k", 1)
    assert cache.get("k") == 1


def test_build_cache_default_is_memory():
    """显式指定 memory 必须拿到内存缓存：它是与 Redis 路线对照的基线后端。"""
    cache = build_cache("memory")
    assert cache.name == "memory"


def test_build_cache_auto_never_raises(monkeypatch):
    """auto 模式在无 Redis 环境下必须静默降级且可读写，绝不把连接异常抛给调用方。"""
    monkeypatch.delenv("REDIS_URL", raising=False)
    cache = build_cache("auto", url="")
    # 降级后仍要能正常写入与读回，否则「无 Redis 就不可用」会直接拖垮整条链路
    cache.set("k", 1)
    assert cache.get("k") == 1


# ---------------------------------------------------------------------------
# FAQ
# ---------------------------------------------------------------------------
def test_faq_identifiers_extraction():
    """标识符抽取必须覆盖风险等级、产品代码与数字——金融问答里它们不可替换。"""
    ids = identifiers("C2 客户购买 WY2024-01，期限 90 天")
    assert "C2" in ids and "WY2024-01" in ids and "90" in ids


def test_faq_hits_exact_question(faq):
    """与 FAQ 条目字面一致的提问必须命中该条目（直出的基本前提）。"""
    match = faq.best("开户需要准备哪些材料？")
    assert match is not None
    assert match.entry.faq_id == "FAQ-001"


def test_faq_hits_similar_question(faq):
    """换一种问法的近义提问也必须命中：词面覆盖率与二元组 Jaccard 取较大者来兜住改写。"""
    match = faq.best("双录资料要保存多久？")
    assert match is not None
    assert "二十" in match.answer


def test_faq_rejects_off_topic_question(faq):
    """分数低于阈值的问题必须拒答交回 RAG：跑题问题被直出等于「答得又快又错」。"""
    assert faq.best("明天大盘会涨还是会跌？") is None
    assert faq.best("请帮我写一段产品营销文案，要能吸引客户") is None


def test_faq_identifier_gate_blocks_near_miss(faq):
    """「C2 客户可以买什么」不能被「R2 可以卖给 C1 吗」这条 FAQ 抢答。

    不变式：问题里的等级 / 代码 / 数字必须与该 FAQ 条目有交集，否则分数被惩罚压到阈值以下，
    宁可走慢一点的 RAG，也不能直出一条看起来相关、实际客群错位的答案。
    """
    match = faq.best("C2 客户可以购买哪些风险等级的产品？")
    # C2 与 C1 是不同客群，答案不可互换；命中的答案里不得出现只属于 C1 的口径
    assert match is None or "C1" not in match.answer or "C2" in match.answer


def test_faq_answer_payload_shape(faq):
    """直出答案的载荷必须与 RAG 答案对齐：标明来源、带更新时间与非零匹配得分。"""
    payload = faq.answer("开户需要准备哪些材料？")
    assert payload["source"] == "faq"
    assert payload["updated_at"]
    assert payload["score"] > 0


def test_faq_answer_returns_none_for_long_tail(faq):
    """长尾问题没有 FAQ 依据时必须返回 None 交给 RAG，不能硬凑一条相近条目直出。"""
    assert faq.answer("示例集团应收账款增速为什么值得关注？") is None


def test_faq_search_returns_ranked_candidates(faq):
    """search 不套阈值但必须按分数降序返回候选，便于人工观察边界案例的排序。"""
    matches = faq.search("合格投资者标准是什么？", top_k=3)
    assert matches
    assert matches[0].score >= matches[-1].score


def test_faq_describe_and_len(faq):
    """len(faq) 与 describe() 的口径必须与实际条目数一致，阈值要能被订阅方读到。"""
    assert len(faq) == len(faq.entries)
    assert faq.describe()["threshold"] > 0


def test_faq_empty_index():
    """空 FAQ 索引必须安全降级：best/search/answer 一律返回空结果而不是抛异常。"""
    index = FAQIndex([])
    assert index.best("任意问题") is None
    assert index.search("任意问题") == []
    assert index.answer("任意问题") is None


def test_faq_match_to_dict(faq):
    """命中记录必须可序列化且得分归一在 [0,1]：它会被直接回传给前端解释「为什么判成 FAQ」。"""
    match = faq.best("开户需要准备哪些材料？")
    payload = match.to_dict()
    assert payload["faq_id"] == "FAQ-001"
    assert 0.0 <= payload["score"] <= 1.0
