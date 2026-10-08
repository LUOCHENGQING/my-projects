"""LLM 客户端（OpenAI 兼容）+ 提示词构造。

两条原则
--------
1. **没有 Key 也必须能跑通**。RAG 引擎的价值在于检索链路，不该因为没配 API Key
   就整个项目无法演示 / 无法回归测试。因此无 Key 时自动降级为
   `answer.generator.MockGenerator` 的确定性抽取式作答，离线全流程可跑。
   注：实际实现为 `answer/generator.py` 的 `AnswerGenerator._extractive()`——
   项目里**没有** `MockGenerator` 这个类，降级由 `AnswerGenerator.generate()` 按 `LLMClient.available` 分流。
2. **提示词里写死"数字只能来自证据"**。模型自己算比例、自己补金额是金融场景的
   致命错误，提示词只是第一道防线，第二道是 `citations.validate()` 的结构性校验。

在 RAG 全链路中的位置
--------------------
    切分 / 索引 → 三路召回 → 去重重排 → 【本模块：提示词构造 + 一次模型调用】 → 引用校验 → 缓存 / 接口

被谁调用：`answer/generator.py` 的 `AnswerGenerator.generate()`（构造提示词并调用 `complete()`）；
          `src/engine.py` 只用 `LLMClient` 的 `available` / `mode` / `model` / `describe()` 做体检与自述；
          `tests/test_answer.py` 会注入假的 `LLMResult` 来测降级路径。

输入：问题 + `Evidence` 序列 + 检索说明 → 渲染成「【问题】【检索说明】【参考资料】」提示词。
输出：`LLMResult`（text / mode / model / latency_ms / usage / error）——**任何失败都塞进 error 字段，不抛异常**。
关键配置来自 `..config`：`MODEL_NAME`、`LLM_TEMPERATURE`、`LLM_TIMEOUT_S`、`current_api_key()`、
`current_base_url()`、`use_mock_llm()`。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..config import LLM_TEMPERATURE, LLM_TIMEOUT_S, MODEL_NAME, current_api_key, current_base_url, use_mock_llm
from ..retrieve.pipeline import Evidence

__all__ = ["LLMResult", "LLMClient", "SYSTEM_PROMPT", "build_user_prompt", "render_evidence_block"]

# 系统提示词：5 条规则与项目里的闸门一一对应，改这里要同步想清楚下游校验。
#   规则 1（不得用参考资料之外的知识） → 兜底靠 `faithfulness.check_numbers` 与句子支撑率；
#   规则 2（每条结论带 [n]，编号必须来自参考资料） → 兜底靠 `CitationLedger.validate()` 剔除悬空编号；
#   规则 3（数字原样引用，禁止自行计算） → 兜底靠 `check_numbers`（一个编造的数字都不容忍）；
#   规则 4（资料不足就明说） → 与 `REFUSAL_TEXT` 的拒答策略对齐；
#   规则 5（输出出处与时效清单） → 与 `CitationLedger.render()` 的格式对齐。
# 注意：这是**提示词**不是契约，模型可能不遵守，所以上面每条都另有确定性校验兜底。
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
    """一次 LLM 调用的结果。mock 与真实模式返回同一结构。

    字段：
        text        模型输出正文（mock 或失败时为空串）
        mode        `"mock"`（mock | openai 两种取值）；**失败时仍是 "openai"**，靠 `error` 区分成败
        model       模型名，默认取自 `MODEL_NAME`
        latency_ms  本次调用耗时（毫秒）
        usage       token 用量：`{"prompt_tokens": int, "completion_tokens": int}`（缺省为空 dict）
        error       错误描述（`"类型名: 消息"` 或 mock 提示），空串表示无错

    关键派生属性：`ok` = 正文非空且无 error。
    """

    text: str
    mode: str = "mock"                     # mock | openai
    model: str = MODEL_NAME
    latency_ms: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        """本次调用是否可用。参数：无；返回：bool（text strip 后非空 **且** error 为空）。副作用/异常：无。"""
        return bool(self.text.strip()) and not self.error

    def to_dict(self) -> Dict[str, object]:
        """导出为 dict（mode / model / latency_ms / chars / usage / error），供轨迹与体检报告。

        参数：无。
        返回：Dict[str, object]；`chars` 是正文字符数（**不落全文**，避免把长答案塞进轨迹），
              latency_ms 保留 3 位小数。
        副作用/异常：无。
        """
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

    每条证据的格式（模型靠 `[evidence_id]` 复述编号，靠「资料编号/日期」写出处清单）：
        `[E1] 出处：机构 · 标题 · 章节` + `资料编号：…　资料类型：…　更新/生效日期：…　版本：…` + `内容：…`

    参数：evidence 证据序列（空序列返回空串）；limit 单条证据正文的字符上限，默认 1200。
    返回：str —— 各条以空行分隔拼成的「参考资料」正文片段。
    副作用/异常：无；不修改入参证据对象（只在本地截断副本上操作）。
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
    """构造用户提示词。把检索计划也带上，便于模型理解证据为什么是这些。

    参数：question 原问题（会 strip）；evidence 证据序列；plan_note 检索说明（空串则整段省略）。
    返回：str —— 由「【问题】」「【检索说明】(可选)」「【参考资料】」与末尾的作答格式要求
          以空行连接而成；无证据时参考资料写「（空）」而不是整段消失，
          这样模型能明确看出"是没检索到"，而不是以为提示词被截断了。
    副作用/异常：无。
    """
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
    """OpenAI 兼容客户端。无 Key 或强制 mock 时 `available` 为 False。

    关键属性：
        model       模型名（构造时定，写进 `LLMResult.model` 与 `describe()`）
        force_mock  是否强制走 mock；None 时取 `config.use_mock_llm()`
        _client     惰性创建的 `openai.OpenAI` 实例（mock 路径下**永远为 None**，不碰第三方 SDK）

    三条铁律：没 Key 不报错（`available=False` 交给上层降级）；调用失败不抛异常（错误进 `error`）；
    真实请求只在 `complete()` 里发生，且只在 `available` 为真时才发生。
    """

    def __init__(self, force_mock: Optional[bool] = None, model: str = MODEL_NAME) -> None:
        """初始化客户端（**不建立任何连接**）。

        参数：force_mock 是否强制 mock（None 表示读配置 `use_mock_llm()`）；model 模型名，默认 `MODEL_NAME`。
        返回：无（构造函数）。
        副作用/异常：无；不 import openai、不读 Key、不发请求（真正的连接推迟到 `_ensure_client()`）。
        """
        self.model = model
        self.force_mock = use_mock_llm() if force_mock is None else force_mock
        self._client = None

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """是否具备真实调用条件。参数：无。

        返回：bool —— `not force_mock` 且 `current_api_key()` 非空（**只看 Key，不试探网络**，
              因此这里不会因为断网而阻塞；连不通的情况由 `complete()` 的异常分支兜住）。
        副作用/异常：无。
        """
        return (not self.force_mock) and bool(current_api_key())

    @property
    def mode(self) -> str:
        """当前工作模式。参数：无；返回：str —— `"openai"` 或 `"mock"`。

        注：`AnswerResult` / `GeneratedAnswer.mode` 里的 "mock-extractive" 等取值由生成层另加，
        本属性只表示「客户端能不能发请求」。
        副作用/异常：无。
        """
        return "openai" if self.available else "mock"

    def describe(self) -> Dict[str, object]:
        """导出自述信息，供 `/health` 与体检报告显示「到底用的哪个模型、有没有 Key」。

        参数：无。
        返回：Dict[str, object] —— mode / model / base_url（mock 时写 "(mock)"，**不回显 Key**）/
              has_key / forced_mock。
        副作用/异常：无。
        """
        return {
            "mode": self.mode,
            "model": self.model,
            "base_url": current_base_url() if self.available else "(mock)",
            "has_key": bool(current_api_key()),
            "forced_mock": self.force_mock,
        }

    def _ensure_client(self):
        """惰性创建并复用 OpenAI 客户端。

        参数：无。
        返回：`openai.OpenAI` 实例（同一个 `LLMClient` 内复用同一实例，避免每次调用重建连接池）。
        副作用：首次调用时 import openai 并构造客户端；超时取自 `LLM_TIMEOUT_S`。
        异常：未安装 openai / Key 或 base_url 非法时抛 `ImportError` / SDK 异常，
              由 `complete()` 捕获后转成 `LLMResult.error`（**不在本方法内吞异常**）。
        """
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
        """调用一次对话补全。mock 模式下会返回空文本，由上层切换到抽取式作答。

        参数：system 系统提示词（生产路径传 `SYSTEM_PROMPT`）；user 用户提示词（`build_user_prompt` 的产物）；
              temperature 采样温度，默认 `LLM_TEMPERATURE`（配置默认 0.0，金融问答要可复现）。
        返回：`LLMResult` —— 成功时 text 为模型输出、usage 带 token 数；
              不可用时 text 空 + `error="mock 模式：不发起真实请求"`；
              调用失败时 text 空 + `error="类型名: 消息"`（**永不抛异常**，上层据此降级）。
        副作用：真实模式下发起一次 HTTP 请求；`_client` 会被首次调用创建并缓存。
        异常：无（`Exception` 已在函数内捕获，注释见下方 except 分支）。
        """
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
