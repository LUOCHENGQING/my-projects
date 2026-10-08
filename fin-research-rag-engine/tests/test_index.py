"""索引层测试：BM25、稠密/稀疏表示、Milvus 语义的内存向量库。

覆盖的被测模块
--------------
    src.index.bm25          Okapi BM25 的排序、分数数组形状、IDF 语义、命中词、空语料安全性
    src.index.embedding     LocalHashingBackend 的稠密编码（确定性、L2 归一化、余弦可比性）、
                            稀疏词权重（次线性加权）、get_backend 的降级链
    src.index.vector_store  Milvus 语义的内存实现：集合生命周期、插入与幂等、维度校验、
                            稠密/稀疏检索、标量表达式过滤（字符串与数值）、按表达式删除、
                            度量方式（COSINE / IP / L2）、命中对象序列化

覆盖策略
--------
    正常      用 4 条制度短句与 3 条记录组成的迷你语料，断言排名次序与检索命中。
    边界      空语料（BM25Index([])）、空查询（token 为空 / 稀疏权重为空字典）、
              空批量编码、零向量归一化、删除全部记录、top_k 大于候选数。
    异常      向量维度不符抛 ValueError、未知度量方式抛 ValueError、取不存在的集合抛 KeyError。
    对抗      幂等与稳定性：同 record_id 重复插入不增加条数、同文本两次编码需完全一致、
              过滤条件必须在打分前生效（被过滤掉的高分记录不得进入结果）。
    非功能    确定性（同输入两次编码一致）与量纲统一（L2 转成负距离，保证「越大越相似」）——
              这两点分别是「离线可复现」与「三路分数可融合」的前提。

为什么这些用例能离线跑：向量库是内存实现（对齐 Milvus 的接口语义），
嵌入后端是确定性哈希实现，两者都不需要 docker-compose、模型下载或网络。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.index import (
    BM25Index,
    Collection,
    LocalHashingBackend,
    MilvusLiteClient,
    VectorRecord,
    cosine_scores,
    get_backend,
    l2_normalize,
    sparse_dot,
    sparse_from_text,
)
from src.utils import tokenize


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
def test_bm25_finds_document_with_exact_term(bm25):
    """字面命中目标文档时，该文档必须排在首位（BM25 是条款号/专有名词那一路的保障）。"""
    # 注：实际实现为——本用例只断言「目标文档排第一」，`search()` 仍会返回其它有 token 重叠
    # 的文档（语料里「投资者」「资产」等单字-二元组会部分命中），因此它不是「唯一命中」的断言。
    results = bm25.search("合格投资者 金融资产")
    assert results
    assert results[0][0] == 0


def test_bm25_ranks_by_idf_not_just_presence():
    """罕见词必须靠 IDF 把「只含罕见词」的短文档顶上去，而不是按是否出现来排序。"""
    # 三篇文档刻意构造成「罕见词的文档频次递减 + 文档长度递减」，
    # 于是 IDF 与长度归一化的效果同向叠加，可以稳定断言 scores[2] > scores[1] > scores[0]
    docs = [["合格", "投资者"], ["合格", "投资者", "罕见词"], ["罕见词"]]
    index = BM25Index(docs)
    scores = index.score_array(["罕见词"])
    assert scores[2] > scores[1] > scores[0]


def test_bm25_score_array_shape(bm25):
    """分数数组长度必须恒等于语料篇数，且下标与文档一一对应（融合阶段按此对齐）。"""
    # 用 tokenize 而不是原始字符串：score_array 只接受已分词的 token 序列
    scores = bm25.score_array(tokenize("冷静期"))
    assert scores.shape == (bm25.doc_count,)
    assert scores.max() > 0


def test_bm25_empty_query_returns_zeros(bm25):
    """空查询必须得到全零分数（长度仍与语料一致），不能抛错也不能给出任意高分。"""
    assert not bm25.score_array([]).any()


def test_bm25_matched_terms(bm25):
    """命中词只能来自该文档倒排表里真实存在的 token，未登录词不得出现在命中列表。"""
    # 文档 3 是「冷静期不少于二十四小时」，注释里那句「单字 + 二元组」正是命中形态的由来
    hits = bm25.matched_terms(tokenize("冷静期 二十四小时"), 3)
    assert "冷静" in hits and "冷" in hits
    assert "不存在词组" not in hits


def test_bm25_term_idf_unknown_term_is_zero(bm25):
    """未登录词的 IDF 为 0（不是抛错、也不是负值），因此它不会给任何文档加分。"""
    assert bm25.term_idf("完全不存在的词") == 0.0
    assert bm25.term_idf("冷静") > 0.0


def test_bm25_vocabulary_size(bm25):
    """词表规模来自倒排表键数，必须为正（体检时用它判断分词是否过碎或索引是否空建）。"""
    assert bm25.vocabulary_size > 10


def test_bm25_empty_corpus_is_safe():
    """空语料必须安全：doc_count 为 0，打分返回形状 (0,) 的空数组而不是除零异常。"""
    # 空语料会让 avgdl 为 0，是最容易触发除零的边界，故单独建一个空索引
    index = BM25Index([])
    assert index.doc_count == 0
    assert index.score_array(tokenize("任意")).shape == (0,)


# ---------------------------------------------------------------------------
# 稠密 / 稀疏表示
# ---------------------------------------------------------------------------
def test_local_backend_is_deterministic():
    """同一文本必须编码出完全相同的向量（跨实例也一致），否则检索与评测都无法复现。"""
    # 两个实例分别编码同一句：验证确定性来自算法本身，而不是"同一个对象里的缓存"
    backend = LocalHashingBackend(dim=128)
    a = backend.encode(["合格投资者金融资产不低于三百万元"])[0]
    b = LocalHashingBackend(dim=128).encode(["合格投资者金融资产不低于三百万元"])[0]
    assert np.allclose(a, b)


def test_local_backend_vectors_are_normalised():
    """编码输出必须是单位向量，余弦相似度才退化成点积、分数区间才可比。"""
    vec = LocalHashingBackend(dim=64).encode_one("适当性管理")
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-9


def test_similar_text_scores_higher_than_unrelated():
    """同句查询的余弦分必须高于无关句，稠密那一路才有排序价值。"""
    # 两条候选刻意一条与查询同文、一条是别的话题，保证比较的是"相似 vs 不相似"而不是偶然重合
    backend = LocalHashingBackend(dim=512)
    matrix = backend.encode(["合格投资者的金融资产标准", "冷静期不少于二十四小时"])
    query = backend.encode_one("合格投资者的金融资产标准")
    scores = cosine_scores(query, matrix)
    assert scores[0] > scores[1]


def test_encode_empty_batch_returns_empty_matrix():
    """空批量必须返回形状 (0, dim) 的矩阵：保持二维，否则上层按轴取分的语义会错位。"""
    matrix = LocalHashingBackend(dim=32).encode([])
    assert matrix.shape == (0, 32)


def test_l2_normalize_handles_zero_vector():
    """零向量归一化不能产生 NaN/Inf，必须原样（全零）返回。"""
    matrix = np.zeros((1, 4))
    assert np.allclose(l2_normalize(matrix), matrix)


def test_sparse_from_text_uses_sublinear_weights():
    """稀疏权重必须按 1 + ln(tf) 次线性增长：重复堆词不能按线性倍数刷分。"""
    # 刻意让「客户」出现 3 次、「产品」出现 1 次，用 1+ln(3) 与 1.0 的对比把口径钉死
    weights = sparse_from_text("客户客户客户产品")
    assert weights["客户"] == pytest.approx(1.0 + np.log(3))
    assert weights["产品"] == pytest.approx(1.0)


def test_sparse_dot_and_empty_cases():
    """稀疏点积的基本口径：同词相乘求和，任一侧为空则得 0（而不是抛错或按 1 处理）。"""
    assert sparse_dot({"a": 1.0}, {"a": 2.0}) == pytest.approx(2.0)
    assert sparse_dot({}, {"a": 1.0}) == 0.0
    assert sparse_dot({"a": 1.0}, {}) == 0.0


def test_get_backend_falls_back_to_local_when_model_missing():
    """请求真实模型（bge-m3）而不可用时必须降级到本地确定性后端，而不是抛错或返回空后端。"""
    # 传一个生产用模型名：验证降级链而不是本地后端的自身行为
    backend = get_backend("bge-m3")
    assert backend.dim > 0
    assert backend.available
    assert "fallback" in backend.describe()["model"] or backend.describe()["backend"] == "local"


def test_backend_encode_sparse_batch_length():
    """稀疏批量编码的输出条数必须与输入文本条数一致，且每条都是非空权重表。"""
    backend = LocalHashingBackend(dim=64)
    sparse = backend.encode_sparse(["合格投资者", "冷静期"])
    assert len(sparse) == 2
    assert sparse[0]


# ---------------------------------------------------------------------------
# 向量库
# ---------------------------------------------------------------------------
def _records(dim=8):
    """构造向量库用例的迷你数据：3 条记录，文本互不相关，且每条带不同的标量元数据。

    为什么这样构造：
        * 文本各不相同且主题互斥（金融资产 / 双录期限 / 冷静期），排名断言才不会被
          token 偶然重叠影响；
        * 三条的 `doc_type` 与 `year` 刻意形成 2:1 的分组（含一条 2024 年），
          因此字符串表达式与数值表达式（含 >= 比较）都能用同一份数据覆盖；
        * 文本顺序与元数据顺序严格一一对应（texts[i] 配 metas[i]），
          断言里因此可以硬编码 r0 / r1 / r2，不必反查元数据。

    返回：(backend, records)——backend 留给用例自己编码查询向量，records 用于插入。
    """
    backend = LocalHashingBackend(dim=dim)
    texts = ["合格投资者的金融资产标准", "双录资料保存期限", "冷静期安排"]
    metas = [
        {"source_id": "A", "doc_type": "监管政策", "year": 2024},
        {"source_id": "B", "doc_type": "内部制度", "year": 2023},
        {"source_id": "C", "doc_type": "监管政策", "year": 2023},
    ]
    dense = backend.encode(texts)
    sparse = backend.encode_sparse(texts)
    return backend, [
        VectorRecord(record_id=f"r{i}", dense=dense[i], sparse=sparse[i], meta=metas[i])
        for i in range(len(texts))
    ]


def test_collection_insert_and_count(client):
    """插入返回值是本次处理的记录数，count 必须与之相等（插入与计数不得各说各话）。"""
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    assert collection.insert(records) == 3
    assert collection.count() == 3


def test_collection_rejects_wrong_dimension(client):
    """维度不符必须在写入前抛 ValueError——否则会在矩阵堆叠阶段才报错，堆栈指向无关代码。"""
    collection = client.create_collection("t", dim=8)
    with pytest.raises(ValueError):
        collection.insert([VectorRecord(record_id="x", dense=np.zeros(4))])


def test_collection_insert_is_idempotent_by_id(client):
    """同 record_id 重复插入按覆盖处理，条数不变（重复灌数据不能让同一块被命中两次）。"""
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    collection.insert(records)
    collection.insert(records)
    assert collection.count() == 3


def test_dense_search_returns_most_similar_first(client):
    """稠密检索必须按相似度降序返回，且与查询同文的 r0 排第一。"""
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    collection.insert(records)
    # 查询直接用 r0 的原文重新编码：保证"最相似"是构造出来的事实，而不是对哈希碰撞的期待
    query = backend.encode_one("合格投资者的金融资产标准")
    hits = collection.search_dense(query, top_k=3)
    assert hits[0].record_id == "r0"
    assert hits[0].dense_score >= hits[-1].dense_score


def test_dense_search_with_metadata_expression(client):
    """字符串标量过滤必须在打分前收窄候选：只剩 doc_type 为「内部制度」的 r1。"""
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    collection.insert(records)
    # 查询词取自 r1（保存期限），但真正决定结果的仍是过滤条件——这样可以验证"过滤优先于相似度"
    query = backend.encode_one("保存期限")
    hits = collection.search_dense(query, top_k=3, expr='doc_type = "内部制度"')
    assert [h.record_id for h in hits] == ["r1"]


def test_dense_search_with_year_expression(client):
    """数值比较表达式（year >= 2024）必须按数值语义生效，而不是按字符串比较。"""
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    collection.insert(records)
    # 查询向量用全 1：本用例比的是过滤而不是相似度，向量本身不该影响结果集
    hits = collection.search_dense(np.ones(8), top_k=5, expr="year >= 2024")
    assert {h.record_id for h in hits} == {"r0"}


def test_sparse_search_returns_hits(client):
    """稀疏检索能命中共享 token 的记录，且与查询同文的 r2 排第一。"""
    collection = client.create_collection("t", dim=8)
    _, records = _records()
    collection.insert(records)
    hits = collection.search_sparse(sparse_from_text("冷静期"), top_k=3)
    assert hits and hits[0].record_id == "r2"


def test_sparse_search_empty_query(client):
    """空稀疏查询（无任何 token 权重）必须返回空列表，不能把全库零分当成有效召回。"""
    collection = client.create_collection("t", dim=8)
    _, records = _records()
    collection.insert(records)
    assert collection.search_sparse({}, top_k=3) == []


def test_collection_query_by_expression(client):
    """标量查询只按元数据取记录（不走向量打分），结果应含 2 条「监管政策」。"""
    collection = client.create_collection("t", dim=8)
    _, records = _records()
    collection.insert(records)
    rows = collection.query('doc_type = "监管政策"')
    assert {r["record_id"] for r in rows} == {"r0", "r2"}


def test_collection_delete_by_expression(client):
    """按表达式删除：返回被删条数，且集合内其余记录保持可检索。"""
    collection = client.create_collection("t", dim=8)
    _, records = _records()
    collection.insert(records)
    removed = collection.delete("year = 2023")
    assert removed == 2
    assert collection.count() == 1


def test_collection_delete_all(client):
    """不带表达式调用 delete 表示清空集合：返回全部条数，count 归零。"""
    collection = client.create_collection("t", dim=8)
    _, records = _records()
    collection.insert(records)
    assert collection.delete() == 3
    assert collection.count() == 0


def test_collection_l2_metric_uses_negative_distance(client):
    """L2 度量统一成负距离返回，保证与 COSINE/IP 一样「越大越相似」，融合层才不需要特判。"""
    collection = client.create_collection("t", dim=4, metric="L2")
    # 零向量与 [1,0,0,0] 的欧氏距离恰为 1，取负后即 -1.0，正好把"负距离"口径钉死
    collection.insert([VectorRecord(record_id="a", dense=np.array([0.0, 0, 0, 0]))])
    hits = collection.search_dense(np.array([1.0, 0, 0, 0]), top_k=1)
    assert hits[0].dense_score == pytest.approx(-1.0)


def test_collection_rejects_unknown_metric(client):
    """不支持的度量方式必须在建集合时就抛 ValueError，避免静默按错误口径算相似度。"""
    with pytest.raises(ValueError):
        client.create_collection("bad", dim=4, metric="HAMMING")


def test_client_collection_lifecycle(client):
    """集合生命周期：存在性判断 → 创建 → 列出 → 删除 → 再取必须抛 KeyError。"""
    assert not client.has_collection("x")
    client.create_collection("x", dim=4)
    assert client.has_collection("x")
    assert client.list_collections() == ["x"]
    assert client.drop_collection("x")
    with pytest.raises(KeyError):
        client.get_collection("x")


def test_client_create_collection_is_idempotent_but_checks_dim(client):
    """同名同维创建幂等（复用实例），同名异维必须报错——静默复用会把错误推迟到插入阶段。"""
    client.create_collection("x", dim=4)
    assert client.create_collection("x", dim=4).dim == 4
    with pytest.raises(ValueError):
        client.create_collection("x", dim=8)


def test_client_describe(client):
    """客户端体检输出必须带上每个集合的名称与实时条数（供引擎 stats 与接口使用）。"""
    collection = client.create_collection("x", dim=4)
    collection.insert([VectorRecord(record_id="a", dense=np.zeros(4))])
    described = client.describe()
    assert described[0]["count"] == 1
    assert described[0]["name"] == "x"


def test_search_hit_to_dict(client):
    """命中对象序列化必须给出 record_id 与 score，供接口与轨迹复用同一份结构。"""
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    collection.insert(records)
    hit = collection.search_dense(backend.encode_one("冷静期"), top_k=1)[0]
    payload = hit.to_dict()
    assert payload["record_id"] and "score" in payload
