"""答案生成与校验子包。

    llm           OpenAI 兼容客户端 + 提示词构造（无 key 自动降级）
    citations     引用编号分配与溯源校验（悬空引用当场剔除）
    generator     抽取式作答 / 真实模型作答两条路径 + 拒答策略
    faithfulness  数字忠实度、句子支撑率、答案相关性（RAGAS 风格、确定性可回归）

这一层只做两件事：**把证据组织成有出处的答案**，以及**验证答案没有编造**。
检索质量不在这里解决——那是 retrieve 子包的职责，混在一起就再也说不清问题出在哪。
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
