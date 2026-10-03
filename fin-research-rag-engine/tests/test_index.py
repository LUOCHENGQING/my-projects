"""索引层测试：BM25、稠密/稀疏表示、Milvus 语义的内存向量库。"""

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
    results = bm25.search("合格投资者 金融资产")
    assert results
    assert results[0][0] == 0


def test_bm25_ranks_by_idf_not_just_presence():
    docs = [["合格", "投资者"], ["合格", "投资者", "罕见词"], ["罕见词"]]
    index = BM25Index(docs)
    scores = index.score_array(["罕见词"])
    assert scores[2] > scores[1] > scores[0]


def test_bm25_score_array_shape(bm25):
    scores = bm25.score_array(tokenize("冷静期"))
    assert scores.shape == (bm25.doc_count,)
    assert scores.max() > 0


def test_bm25_empty_query_returns_zeros(bm25):
    assert not bm25.score_array([]).any()


def test_bm25_matched_terms(bm25):
    """分词是「单字 + 二元组」，因此命中的是 '冷静' 这种二元组而不是 '冷静期' 整词。"""
    hits = bm25.matched_terms(tokenize("冷静期 二十四小时"), 3)
    assert "冷静" in hits and "冷" in hits
    assert "不存在词组" not in hits


def test_bm25_term_idf_unknown_term_is_zero(bm25):
    assert bm25.term_idf("完全不存在的词") == 0.0
    assert bm25.term_idf("冷静") > 0.0


def test_bm25_vocabulary_size(bm25):
    assert bm25.vocabulary_size > 10


def test_bm25_empty_corpus_is_safe():
    index = BM25Index([])
    assert index.doc_count == 0
    assert index.score_array(tokenize("任意")).shape == (0,)


# ---------------------------------------------------------------------------
# 稠密 / 稀疏表示
# ---------------------------------------------------------------------------
def test_local_backend_is_deterministic():
    backend = LocalHashingBackend(dim=128)
    a = backend.encode(["合格投资者金融资产不低于三百万元"])[0]
    b = LocalHashingBackend(dim=128).encode(["合格投资者金融资产不低于三百万元"])[0]
    assert np.allclose(a, b)


def test_local_backend_vectors_are_normalised():
    vec = LocalHashingBackend(dim=64).encode_one("适当性管理")
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-9


def test_similar_text_scores_higher_than_unrelated():
    backend = LocalHashingBackend(dim=512)
    matrix = backend.encode(["合格投资者的金融资产标准", "冷静期不少于二十四小时"])
    query = backend.encode_one("合格投资者的金融资产标准")
    scores = cosine_scores(query, matrix)
    assert scores[0] > scores[1]


def test_encode_empty_batch_returns_empty_matrix():
    matrix = LocalHashingBackend(dim=32).encode([])
    assert matrix.shape == (0, 32)


def test_l2_normalize_handles_zero_vector():
    matrix = np.zeros((1, 4))
    assert np.allclose(l2_normalize(matrix), matrix)


def test_sparse_from_text_uses_sublinear_weights():
    weights = sparse_from_text("客户客户客户产品")
    assert weights["客户"] == pytest.approx(1.0 + np.log(3))
    assert weights["产品"] == pytest.approx(1.0)


def test_sparse_dot_and_empty_cases():
    assert sparse_dot({"a": 1.0}, {"a": 2.0}) == pytest.approx(2.0)
    assert sparse_dot({}, {"a": 1.0}) == 0.0
    assert sparse_dot({"a": 1.0}, {}) == 0.0


def test_get_backend_falls_back_to_local_when_model_missing():
    backend = get_backend("bge-m3")
    assert backend.dim > 0
    assert backend.available
    assert "fallback" in backend.describe()["model"] or backend.describe()["backend"] == "local"


def test_backend_encode_sparse_batch_length():
    backend = LocalHashingBackend(dim=64)
    sparse = backend.encode_sparse(["合格投资者", "冷静期"])
    assert len(sparse) == 2
    assert sparse[0]


# ---------------------------------------------------------------------------
# 向量库
# ---------------------------------------------------------------------------
def _records(dim=8):
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
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    assert collection.insert(records) == 3
    assert collection.count() == 3


def test_collection_rejects_wrong_dimension(client):
    collection = client.create_collection("t", dim=8)
    with pytest.raises(ValueError):
        collection.insert([VectorRecord(record_id="x", dense=np.zeros(4))])


def test_collection_insert_is_idempotent_by_id(client):
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    collection.insert(records)
    collection.insert(records)
    assert collection.count() == 3


def test_dense_search_returns_most_similar_first(client):
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    collection.insert(records)
    query = backend.encode_one("合格投资者的金融资产标准")
    hits = collection.search_dense(query, top_k=3)
    assert hits[0].record_id == "r0"
    assert hits[0].dense_score >= hits[-1].dense_score


def test_dense_search_with_metadata_expression(client):
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    collection.insert(records)
    query = backend.encode_one("保存期限")
    hits = collection.search_dense(query, top_k=3, expr='doc_type = "内部制度"')
    assert [h.record_id for h in hits] == ["r1"]


def test_dense_search_with_year_expression(client):
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    collection.insert(records)
    hits = collection.search_dense(np.ones(8), top_k=5, expr="year >= 2024")
    assert {h.record_id for h in hits} == {"r0"}


def test_sparse_search_returns_hits(client):
    collection = client.create_collection("t", dim=8)
    _, records = _records()
    collection.insert(records)
    hits = collection.search_sparse(sparse_from_text("冷静期"), top_k=3)
    assert hits and hits[0].record_id == "r2"


def test_sparse_search_empty_query(client):
    collection = client.create_collection("t", dim=8)
    _, records = _records()
    collection.insert(records)
    assert collection.search_sparse({}, top_k=3) == []


def test_collection_query_by_expression(client):
    collection = client.create_collection("t", dim=8)
    _, records = _records()
    collection.insert(records)
    rows = collection.query('doc_type = "监管政策"')
    assert {r["record_id"] for r in rows} == {"r0", "r2"}


def test_collection_delete_by_expression(client):
    collection = client.create_collection("t", dim=8)
    _, records = _records()
    collection.insert(records)
    removed = collection.delete("year = 2023")
    assert removed == 2
    assert collection.count() == 1


def test_collection_delete_all(client):
    collection = client.create_collection("t", dim=8)
    _, records = _records()
    collection.insert(records)
    assert collection.delete() == 3
    assert collection.count() == 0


def test_collection_l2_metric_uses_negative_distance(client):
    collection = client.create_collection("t", dim=4, metric="L2")
    collection.insert([VectorRecord(record_id="a", dense=np.array([0.0, 0, 0, 0]))])
    hits = collection.search_dense(np.array([1.0, 0, 0, 0]), top_k=1)
    assert hits[0].dense_score == pytest.approx(-1.0)


def test_collection_rejects_unknown_metric(client):
    with pytest.raises(ValueError):
        client.create_collection("bad", dim=4, metric="HAMMING")


def test_client_collection_lifecycle(client):
    assert not client.has_collection("x")
    client.create_collection("x", dim=4)
    assert client.has_collection("x")
    assert client.list_collections() == ["x"]
    assert client.drop_collection("x")
    with pytest.raises(KeyError):
        client.get_collection("x")


def test_client_create_collection_is_idempotent_but_checks_dim(client):
    client.create_collection("x", dim=4)
    assert client.create_collection("x", dim=4).dim == 4
    with pytest.raises(ValueError):
        client.create_collection("x", dim=8)


def test_client_describe(client):
    collection = client.create_collection("x", dim=4)
    collection.insert([VectorRecord(record_id="a", dense=np.zeros(4))])
    described = client.describe()
    assert described[0]["count"] == 1
    assert described[0]["name"] == "x"


def test_search_hit_to_dict(client):
    collection = client.create_collection("t", dim=8)
    backend, records = _records()
    collection.insert(records)
    hit = collection.search_dense(backend.encode_one("冷静期"), top_k=1)[0]
    payload = hit.to_dict()
    assert payload["record_id"] and "score" in payload
