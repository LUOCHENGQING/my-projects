"""答案生成与校验子包。

    llm           OpenAI 兼容客户端 + 提示词构造（无 key 自动降级）
    citations     引用编号分配与溯源校验（悬空引用当场剔除）
    generator     抽取式作答 / 真实模型作答两条路径 + 拒答策略
    faithfulness  数字忠实度、句子支撑率、答案相关性（RAGAS 风格、确定性可回归）

这一层只做两件事：**把证据组织成有出处的答案**，以及**验证答案没有编造**。
检索质量不在这里解决——那是 retrieve 子包的职责，混在一起就再也说不清问题出在哪。

在 RAG 全链路中的位置
--------------------
    切分 / 索引 → 三路召回 → 去重重排 → 【本子包：组织答案 + 校验答案】 → 缓存 / 接口 / 轨迹

上游：`src.engine.RAGEngine.ask()` 在检索拿到 `Evidence` 列表之后调用本子包的
`AnswerGenerator.generate(question, evidence, plan_note)`。
下游：`src.engine` 把 `GeneratedAnswer` 与 `FaithfulnessReport` 组装成 `AnswerResult`，
写缓存（`src.cache.redis_cache`）并落轨迹（`src.tracing`），最终由 `src.serve` / `src.demo` 输出；
离线评测 `eval/run_eval.py` 直接复用本子包的 `evaluate_faithfulness` 与 `claim_body`。

对外关键对象（`__all__` 里的名字即本层公开契约）
--------------------------------------------
    入口    AnswerGenerator.generate()
    输入    question（原问题）+ evidence（`retrieve.pipeline.Evidence` 序列：子块精确命中 + 父块上下文）
    输出    GeneratedAnswer（答案正文 / 引用列表 / CitationCheck / 是否拒答 / notes / 耗时）
    另出    FaithfulnessReport（数字忠实度、句子支撑率、相关性、要点覆盖）

反幻觉做成**结构**，而不是提示词里的叮嘱
--------------------------------------
    1) 引用编号只由 `citations.CitationLedger` 签发，其他任何地方不许自己拼 `[n]`；
    2) 输出前 `CitationLedger.validate()` 做确定性校验，**账本之外的编号当场剔除**并记入 `dropped`；
    3) `faithfulness.evaluate_faithfulness()` 复核数字与句子（阈值口径见该模块 docstring）。

另有一道**主体闸门**在上游、不在本子包内：`src.engine` 的 `unknown_entities()` 会先用
`ENTITY_RE` 抽出问题里的公司/机构，凡不在资料库 `known_entities` 里的**直接拒答**，
防止拿 B 公司的数字回答 A 公司的问题。本子包不重复实现这道闸门，只负责「有证据时的引用与忠实度」。

导入约定：本文件只做 re-export，不放任何逻辑，避免 `import src.answer` 时产生副作用。
"""

from __future__ import annotations

from .citations import Citation, CitationCheck, CitationLedger
from .faithfulness import (
    FaithfulnessReport,
    NumberCheck,
    answer_relevance,
    check_numbers,
    claim_body,
    evaluate_faithfulness,
    keyphrase_coverage,
    sentence_support,
    strip_meta,
)
from .generator import REFUSAL_TEXT, AnswerGenerator, GeneratedAnswer, best_sentence
from .llm import SYSTEM_PROMPT, LLMClient, LLMResult, build_user_prompt, render_evidence_block

# 公开契约清单：调用方（src.engine / src.serve / tests / eval）只应依赖这里列出的名字。
__all__ = [
    "LLMClient",
    "LLMResult",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "render_evidence_block",
    "Citation",
    "CitationCheck",
    "CitationLedger",
    "AnswerGenerator",
    "GeneratedAnswer",
    "best_sentence",
    "REFUSAL_TEXT",
    "check_numbers",
    "NumberCheck",
    "sentence_support",
    "answer_relevance",
    "keyphrase_coverage",
    "evaluate_faithfulness",
    "FaithfulnessReport",
    "claim_body",
    "strip_meta",
]
