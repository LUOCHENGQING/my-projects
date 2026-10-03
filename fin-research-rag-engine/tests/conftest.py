"""pytest 公共夹具。

会话级夹具缓存语料 / 切分 / 检索器 / 引擎，避免每个用例重复建索引
（一次建索引约 0.3 秒，上百个用例重复建会白等半分钟）；
运行级夹具用 tmp_path 隔离轨迹文件，保证测试不污染项目里的 runs/ 目录。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _force_mock_env():
    """测试全程强制 mock：不依赖网络、不依赖 API Key，结果完全可复现。"""
    os.environ["MOCK_LLM"] = "1"
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("REDIS_URL", None)
    yield


from src.answer import AnswerGenerator, CitationLedger, LLMClient  # noqa: E402
from src.cache import MemoryLRUCache, cache_key  # noqa: E402
from src.chunking import build_chunks, chunk_stats  # noqa: E402
from src.engine import RAGEngine  # noqa: E402
from src.faq import FAQIndex  # noqa: E402
from src.index import BM25Index, MilvusLiteClient, get_backend  # noqa: E402
from src.ingest import apply_masking, clean_corpus, load_corpus  # noqa: E402
from src.retrieve import HybridRetriever, RetrievalPipeline, get_reranker  # noqa: E402
from src.utils import tokenize  # noqa: E402


@pytest.fixture(scope="session")
def corpus():
    """原始语料（未清洗、未脱敏），用于解析层测试。"""
    return load_corpus()


@pytest.fixture(scope="session")
def cleaned_documents(corpus):
    """清洗 + 脱敏后的文档（深拷贝会话语料的副本，避免污染其他夹具）。"""
    from src.config import DATA_DIR
    from src.ingest import load_corpus as reload_corpus

    fresh = reload_corpus(DATA_DIR)
    apply_masking(fresh.documents)
    clean_corpus(fresh.documents)
    return fresh.documents


@pytest.fixture(scope="session")
def chunks(cleaned_documents):
    return build_chunks(cleaned_documents)


@pytest.fixture(scope="session")
def retriever(chunks):
    parents, children = chunks
    return HybridRetriever(parents, children, backend=get_backend("local"))


@pytest.fixture(scope="session")
def pipeline(retriever):
    return RetrievalPipeline(retriever, reranker=get_reranker("local"))


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    """一个完整引擎（轨迹写入临时目录，避免污染项目 runs/）。"""
    runs_dir = tmp_path_factory.mktemp("runs")
    return RAGEngine.build(runs_dir=runs_dir, quiet=True)


@pytest.fixture
def generator():
    return AnswerGenerator(LLMClient(force_mock=True))


@pytest.fixture
def ledger():
    return CitationLedger()


@pytest.fixture
def cache():
    return MemoryLRUCache(max_entries=8, ttl_s=60)


@pytest.fixture
def key():
    return cache_key("测试问题", expr="year >= 2024", top_k=5)


@pytest.fixture
def bm25():
    docs = [
        "合格投资者的金融资产不低于 300 万元",
        "C2 客户仅可购买 R1、R2 级产品",
        "双录资料保存期限不少于二十年",
        "冷静期不少于二十四小时",
    ]
    return BM25Index([tokenize(d) for d in docs])


@pytest.fixture
def client():
    return MilvusLiteClient()


@pytest.fixture
def faq(corpus):
    return FAQIndex(corpus.faq, backend=get_backend("local"))
