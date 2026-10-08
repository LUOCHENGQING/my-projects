"""RAG（检索增强）子包：整个投研系统的「证据层」。

层次与职责
----------
本子包位于「原始语料 -> 可引用证据」这一层：向下读 data/*.md，向上通过
tools/builtin.py 的 search_filings / get_financial_metric 两个工具，把证据交给
RetrieverAgent、AnalystAgent 与 WriterAgent。本层不调用 LLM、不下结论，全部输出
都是确定性的（同一份语料必然得到同一份证据），因此 trace 可复现、评测可回归。

    embedding   确定性哈希向量（手写，无外部 API）
    bm25        Okapi BM25 稀疏检索
    chunking    父子块切分
    corpus      文档加载与解析（front-matter / Markdown 表格）
    facts       结构化财务事实抽取
    retriever   BM25 + 向量 混合检索与加权重排

两条互补的证据通路
------------------
1. **非结构化通路**（chunking -> bm25 / embedding -> retriever）：
   回答「哪一段话与问题相关」，产出 RetrievedSnippet（子块原文 text + 父块上下文
   context + 章节标题 + source_id），供 Writer 生成带编号的引用。
2. **结构化通路**（corpus -> facts）：
   把 Markdown 表格解析成 MetricFact，数值型结论一律走 get_financial_metric 查表 +
   tools/builtin.py 中 calc_ratio / RATIO_DEFS 的确定性比率公式，而不是让 LLM 从
   自然语言里"读"数字——这就从结构上消灭了数字幻觉，LLM 只负责解释数字。

对外接口
--------
输入：无（本模块是纯声明式再导出，不含任何实现逻辑）。
输出：下列公共名字，供上层一行 `from .rag import ...` 完成装配。
被谁调用：src/orchestrator.py（ResearchPipeline.__init__ 里构建检索器与事实库）、
src/tools/builtin.py（search_filings / get_financial_metric 的依赖）、tests/ 与
eval/run_eval.py。

注：BM25Index、子模块 bm25 本体，以及 embedding.cosine_scores / token_weights
并未在此重导出，它们由 HybridRetriever 在包内部直接 import 使用。
"""

from __future__ import annotations

from .chunking import ChildChunk, ParentChunk, build_chunks, split_document
from .corpus import Document, DocumentStore, load_documents
from .embedding import cosine_similarity, embed, embed_matrix
from .facts import FactStore, MetricFact, build_fact_store
from .retriever import HybridRetriever, RetrievedSnippet

__all__ = [
    "ChildChunk",
    "ParentChunk",
    "split_document",
    "build_chunks",
    "build_fact_store",
    "Document",
    "DocumentStore",
    "load_documents",
    "embed",
    "embed_matrix",
    "cosine_similarity",
    "FactStore",
    "MetricFact",
    "HybridRetriever",
    "RetrievedSnippet",
]
