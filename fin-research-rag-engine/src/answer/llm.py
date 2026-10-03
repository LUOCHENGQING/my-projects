"""LLM 客户端（OpenAI 兼容）+ 提示词构造。

两条原则
--------
1. **没有 Key 也必须能跑通**。RAG 引擎的价值在于检索链路，不该因为没配 API Key
   就整个项目无法演示 / 无法回归测试。因此无 Key 时自动降级为
   `answer.generator.MockGenerator` 的确定性抽取式作答，离线全流程可跑。
2. **提示词里写死"数字只能来自证据"**。模型自己算比例、自己补金额是金融场景的
   致命错误，提示词只是第一道防线，第二道是 `citations.validate()` 的结构性校验。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..config import LLM_TEMPERATURE, LLM_TIMEOUT_S, MODEL_NAME, current_api_key, current_base_url, use_mock_llm
from ..retrieve.pipeline import Evidence

__all__ = ["LLMResult", "LLMClient", "SYSTEM_PROMPT", "build_user_prompt", "render_evidence_block"]

SYSTEM_PROMPT = """你是一名金融资料检索助手，服务于银行、证券机构的投研、信贷、风控与合规场景。

必须遵守的规则：
1. 只能依据「参考资料」作答，不得使用参考资料之外的知识，不得推测。
2. 每一个结论后面必须标注来源编号，格式为 [1]、[2]，编号必须来自参考资料。
3. 涉及金额、比例、期限、等级、条款号时，必须原样引用参考资料中的数值，
   禁止自行计算、换算或补全。
4. 如果参考资料不足以回答，直接说明「现有资料不足以回答该问题」，
   并指出还缺哪一类资料。不要编造。
5. 最后输出「出处与时效」清单，逐条列出引用编号、资料名称、资料编号与更新/生效日期。

输出结构：
【结论】一段话直接回答。
【依据】分条列出，每条都带 [n]。
【出处与时效】逐条列出引用来源及其更新日期。"""


@dataclass
class LLMResult:
    """一次 LLM 调用的结果。mock 与真实模式返回同一结构。"""

    text: str
    mode: str = "mock"                     # mock | openai
    model: str = MODEL_NAME
    latency_ms: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and not self.error

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 3),
            "chars": len(self.text),
            "usage": self.usage,
            "error": self.error,
        }


def render_evidence_block(evidence: Sequence[Evidence], limit: int = 1200) -> str:
    """把证据渲染成提示词里的「参考资料」段。每条都带编号、出处与日期。

    证据文本做了长度截断：把整章父块全塞进提示词会瞬间吃满上下文窗口，
    真正有用的句子反而被淹没。需要更多上下文时，应该靠"父块回溯"取，而不是全量灌。
    """
    blocks: List[str] = []
    for item in evidence:
        body = item.text.strip()
        if len(body) > limit:
            body = body[:limit] + "…"
        blocks.append(
            f"[{item.evidence_id}] 出处：{item.citation_label}\n"
            f"资料编号：{item.source_id}　资料类型：{item.doc_type or '未标注'}　"
            f"更新/生效日期：{item.updated_at}　版本：{item.version or '未标注'}\n"
            f"内容：{body}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, evidence: Sequence[Evidence], plan_note: str = "") -> str:
    """构造用户提示词。把检索计划也带上，便于模型理解证据为什么是这些。"""
    parts = [f"【问题】{question.strip()}"]
    if plan_note:
        parts.append(f"【检索说明】{plan_note}")
    if evidence:
        parts.append("【参考资料】\n" + render_evidence_block(evidence))
    else:
        parts.append("【参考资料】（空）")
    parts.append("请按【结论】【依据】【出处与时效】三段作答，所有结论都要带引用编号。")
    return "\n\n".join(parts)


class LLMClient:
    """OpenAI 兼容客户端。无 Key 或强制 mock 时 `available` 为 False。"""

    def __init__(self, force_mock: Optional[bool] = None, model: str = MODEL_NAME) -> None:
        self.model = model
        self.force_mock = use_mock_llm() if force_mock is None else force_mock
        self._client = None

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return (not self.force_mock) and bool(current_api_key())

    @property
    def mode(self) -> str:
        return "openai" if self.available else "mock"

    def describe(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "model": self.model,
            "base_url": current_base_url() if self.available else "(mock)",
            "has_key": bool(current_api_key()),
            "forced_mock": self.force_mock,
        }

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # 延迟导入：mock 模式下完全不碰第三方 SDK

            self._client = OpenAI(
                api_key=current_api_key(),
                base_url=current_base_url(),
                timeout=LLM_TIMEOUT_S,
            )
        return self._client

    # ------------------------------------------------------------------
    def complete(self, system: str, user: str, temperature: float = LLM_TEMPERATURE) -> LLMResult:
        """调用一次对话补全。mock 模式下会返回空文本，由上层切换到抽取式作答。"""
        started = time.perf_counter()
        if not self.available:
            return LLMResult(
                text="",
                mode="mock",
                model="(mock)",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="mock 模式：不发起真实请求",
            )

        try:
            client = self._ensure_client()
            resp = client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            usage: Dict[str, int] = {}
            if getattr(resp, "usage", None) is not None:
                usage = {
                    "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
                }
            return LLMResult(
                text=text,
                mode="openai",
                model=self.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                usage=usage,
            )
        except Exception as exc:  # noqa: BLE001 - 外部服务不可用不应让整条链路崩掉
            # 网络 / 配额 / 鉴权失败时**降级而不是抛出**：上层会切到抽取式作答，
            # 用户拿到的仍然是一条带出处的回答，而不是一个 500。
            return LLMResult(
                text="",
                mode="openai",
                model=self.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
