"""pytest 公共夹具（tests/ 下全部测试共享，本文件自身不含测试用例）。

提供什么
--------
被测链路需要的重资产对象，按「会话级缓存 / 函数级轻对象」两层组织：

    会话级（scope="session"）：语料 corpus、清洗后文档 cleaned_documents、切分结果 chunks、
        检索器 retriever、检索管线 pipeline、完整引擎 engine——建一次索引供全部用例复用
        （一次建索引约 0.3 秒，上百个用例重复建会白等半分钟）。
    函数级（默认 scope="function"）：generator / ledger / cache / key / bm25 / client / faq
        ——构造只要毫秒级，且自带可变状态（引用账本、命中统计），每例新建可避免用例之间串状态。

覆盖策略
--------
1. 全程 mock：`_force_mock_env` 自动生效，不联网、不需要 API Key、结果完全可复现，
   因此断言可以卡死在精确数值上。
2. 分层覆盖：语料层（原始 -> 清洗 -> 切分）逐级派生，同一份 data/ 示例库被解析、
   切分、召回、评测各层复用；过滤器则用内联小数据覆盖正常/边界/异常（非法表达式）三类输入。
3. 状态隔离：运行级产物（轨迹 JSONL、评测报告）统一写进 tmp_path / tmp_path_factory，
   保证测试不污染项目里的 runs/ 与 eval/ 目录。

夹具消费者索引（改动夹具前用它评估影响面）
----------------------------------------
    corpus            -> test_ingest.py、test_retrieve.py，以及本文件的 cleaned_documents / faq
    cleaned_documents -> test_chunking.py、test_engine_api.py，以及本文件的 chunks
    chunks            -> 本文件的 retriever
    retriever         -> test_retrieve.py，以及本文件的 pipeline
    pipeline          -> test_retrieve.py
    engine            -> test_engine_api.py
    generator/ledger  -> test_answer.py
    cache/key         -> test_cache_faq.py（注：实际实现为——`key` 目前没有用例直接请求）
    bm25/client       -> test_index.py
    faq               -> test_cache_faq.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 让 tests 直接 import src.*，不要求项目被 pip install 成包（CI 里只装依赖不装本项目）
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _force_mock_env():
    """测试全程强制 mock：不依赖网络、不依赖 API Key，结果完全可复现。

    提供：把环境变量 MOCK_LLM 置为 "1"，并清掉 OPENAI_API_KEY / REDIS_URL——
          于是 LLM 走抽取式作答、缓存走内存 LRU 降级（两条降级路径的判定仍在各自用例里）。
    作用域：session，且 autouse=True，因此所有测试文件**隐式**使用，无需显式声明参数。
    """
    os.environ["MOCK_LLM"] = "1"
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("REDIS_URL", None)
    yield


# 故意放在 sys.path 注入之后（而非文件头部）：先保证 src 可导入，再引用 src 里的名字，
# 因此这些 import 都带 noqa: E402
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
    """原始语料（未清洗、未脱敏），用于解析层测试。

    提供：`src.ingest.load_corpus()` 返回的 Corpus（documents / faq / issues / stats）。
    作用域：session，整个测试进程只读一次盘。
    被使用：test_ingest.py 的解析与语料级用例、test_retrieve.py 的过滤条件推断用例，
            以及本文件的 cleaned_documents 与 faq（两者都由它派生）。
    注：刻意保持未清洗状态——清洗/脱敏会改写原文，解析层断言必须对着原始文本。
    """
    return load_corpus()


@pytest.fixture(scope="session")
def cleaned_documents(corpus):
    """清洗 + 脱敏后的文档（深拷贝会话语料的副本，避免污染其他夹具）。

    提供：`List[SourceDocument]`——重新读盘后依次 apply_masking() 与 clean_corpus() 的结果。
    作用域：session。
    被使用：test_chunking.py 的文档级/统计/元数据用例、test_engine_api.py 的轨迹文件用例，
            以及本文件的 chunks（再往下供 test_retrieve.py 走全链路）。
    注：实际实现为——不是复制 corpus 对象，而是用 DATA_DIR 重新读盘一份，
        因此清洗/脱敏不会回头改动会话级 corpus（两个夹具互不影响）。
    """
    from src.config import DATA_DIR
    from src.ingest import load_corpus as reload_corpus

    fresh = reload_corpus(DATA_DIR)
    apply_masking(fresh.documents)
    clean_corpus(fresh.documents)
    return fresh.documents


@pytest.fixture(scope="session")
def chunks(cleaned_documents):
    """切分结果 (parents, children)：索引层真正消费的两级块。

    提供：`build_chunks()` 的返回值，会话内只切一次（切分是测试里最贵的一步）。
    作用域：session。
    被使用：本文件的 retriever（test_retrieve.py 因此间接受益）。
    """
    return build_chunks(cleaned_documents)


@pytest.fixture(scope="session")
def retriever(chunks):
    """本地后端的混合检索器：BM25 + 稠密 + 稀疏三路，索引已建好。

    提供：`HybridRetriever` 实例（backend=get_backend("local")，确定性、无需模型下载）。
    作用域：session。
    被使用：test_retrieve.py 的召回/过滤/路由用例，以及本文件的 pipeline。
    """
    parents, children = chunks
    return HybridRetriever(parents, children, backend=get_backend("local"))


@pytest.fixture(scope="session")
def pipeline(retriever):
    """检索管线：多路召回 + 去重 + 本地重排。

    提供：`RetrievalPipeline` 实例（reranker=get_reranker("local")，可复现的重排实现）。
    作用域：session。
    被使用：test_retrieve.py 的 pipeline 与 Evidence 序列化用例。
    """
    return RetrievalPipeline(retriever, reranker=get_reranker("local"))


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    """一个完整引擎（轨迹写入临时目录，避免污染项目 runs/）。

    提供：`RAGEngine.build()` 组装好的引擎（语料 + 索引 + FAQ + 缓存 + 追踪）。
    作用域：session，整套链路只建一次。
    被使用：test_engine_api.py 的端到端问答、统计与 HTTP handler 用例。
    """
    runs_dir = tmp_path_factory.mktemp("runs")
    return RAGEngine.build(runs_dir=runs_dir, quiet=True)


@pytest.fixture
def generator():
    """强制 mock 的答案生成器：只走确定性抽取式作答路径。

    提供：`AnswerGenerator(LLMClient(force_mock=True))`。
    作用域：function——每例独立实例，避免上一次的引用账本/计时状态串场。
    被使用：test_answer.py 的生成类用例（结构化答案、拒答、降级、序列化）。
    """
    return AnswerGenerator(LLMClient(force_mock=True))


@pytest.fixture
def ledger():
    """空的引用账本：编号从 1 开始，一次问答一个。

    提供：`CitationLedger()`。
    作用域：function，保证每个用例的 [n] 编号都从 1 起、互不干扰。
    被使用：test_answer.py 的账本分配、校验与渲染用例。
    """
    return CitationLedger()


@pytest.fixture
def cache():
    """小容量内存 LRU：容量 8 条 / TTL 60 秒。

    提供：`MemoryLRUCache(max_entries=8, ttl_s=60)`——容量刻意取小，几个 set 就能触发淘汰。
    作用域：function，命中统计（CacheStats）需要干净起点。
    被使用：test_cache_faq.py 的内存 LRU 用例；test_cache_describe 断言的 max_entries=8
            就来自这里的构造参数（而不是配置默认值）。
    """
    return MemoryLRUCache(max_entries=8, ttl_s=60)


@pytest.fixture
def key():
    """一个固定的缓存键：覆盖「问题 + year 过滤表达式 + TopK」三种入参组合。

    提供：`cache_key("测试问题", expr="year >= 2024", top_k=5)`。
    作用域：function。
    被使用：无（注：实际实现为——当前 tests/ 下没有用例直接请求本夹具，缓存键用例都直接调用
            `cache_key()`；它保留给本地调试与后续用例）。
    """
    return cache_key("测试问题", expr="year >= 2024", top_k=5)


@pytest.fixture
def bm25():
    """4 条制度短句构成的迷你 BM25 索引。

    提供：`BM25Index`，语料刻意短，四条各含不同业务术语（金融资产 / 产品风险等级 /
          双录期限 / 冷静期），便于精确断言 IDF 与排序，而不是验证语料真实性。
    作用域：function。
    被使用：test_index.py 的 BM25 用例。
    """
    docs = [
        "合格投资者的金融资产不低于 300 万元",
        "C2 客户仅可购买 R1、R2 级产品",
        "双录资料保存期限不少于二十年",
        "冷静期不少于二十四小时",
    ]
    return BM25Index([tokenize(d) for d in docs])


@pytest.fixture
def client():
    """进程内的 Milvus 兼容向量库客户端（无需真实 Milvus 服务）。

    提供：`MilvusLiteClient()`；每个用例自己建 collection，跑完即丢。
    作用域：function，避免跨用例的表/数据残留（也保证删除类用例互不干扰）。
    被使用：test_index.py 的向量库与稠密/稀疏检索用例。
    """
    return MilvusLiteClient()


@pytest.fixture
def faq(corpus):
    """基于示例语料 FAQ 条目构建的 FAQ 索引。

    提供：`FAQIndex(corpus.faq, backend=get_backend("local"))`——本地嵌入后端 + 默认阈值。
    作用域：function；依赖会话级 corpus，因此不会重复读盘，又能每例拿到干净索引。
    被使用：test_cache_faq.py 的 FAQ 命中、拒答、标识符闸门与排序用例。
    """
    return FAQIndex(corpus.faq, backend=get_backend("local"))
