"""ClientProfilingAgent —— 客户画像与硬约束提取。

职责边界（与投研、AML 项目里的 Agent 都不同）
--------------------------------------------
本 Agent **不产出任何投资观点**，它只做一件事：把客户档案 + 风险测评问卷
翻译成**可执行的硬约束与软偏好**，并按「从严原则」决定最终生效的风险等级。

- 硬约束：风险等级上限、投资期限、流动性下限、三类集中度上限、禁止项、
  币种、税收优惠额度、合格投资者准入、已具备经验的品类
- 软偏好：收益目标、回撤容忍度、费率预算
- 从严原则：问卷折算等级低于档案登记等级时，取孰低者作为有效等级，并留痕说明

工具白名单：`profile.read_client` / `profile.parse_questionnaire` / `profile.derive_constraints`
权限集合：`profile:read`、`profile:derive`、`trace:write`
"""

from __future__ import annotations

from typing import Any, Mapping

from ..schemas import ClientProfile
from ..utils import pct
from .base import BaseAgent, Stopwatch
from .tools import (
    PERM_PROFILE_DERIVE,
    PERM_PROFILE_READ,
    PERM_TRACE_WRITE,
    AgentSpec,
)

SYSTEM_PROMPT = """你是财富管理机构的客户画像引擎，只负责把客户档案与风险测评问卷
转换成可执行的硬约束与软偏好，禁止给出任何投资建议、产品推荐或收益判断。

你必须遵守：
1. 只使用输入中出现的字段，不得推断或补全客户未提供的信息；
2. 硬约束（风险等级、期限、流动性、集中度、禁止项、币种、准入、经验）必须逐条列示，不允许遗漏；
3. 当问卷折算等级低于档案登记等级时，按「从严原则」取孰低者，并明确说明；
4. 输出为结构化 JSON，字段固定，不得附加解释性文字。

输出字段：summary（客户画像摘要）、consistency_note（等级一致性说明）。"""

SPEC = AgentSpec(
    name="ClientProfilingAgent",
    role="客户画像与硬约束提取",
    system_prompt=SYSTEM_PROMPT,
    tools=("profile.read_client", "profile.parse_questionnaire", "profile.derive_constraints", "profile.compose_profile_text"),
    permissions=frozenset({PERM_PROFILE_READ, PERM_PROFILE_DERIVE, PERM_TRACE_WRITE}),
    can_write_state=("client", "effective_client", "profile"),
)


class ClientProfilingAgent(BaseAgent):
    """客户画像 Agent。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(SPEC, **kwargs)

    # ------------------------------------------------------------------
    def run(self, *, client: ClientProfile, questionnaire: Mapping[str, Any]) -> dict[str, Any]:
        """提取硬约束与软偏好，并返回按从严原则收紧后的有效客户档案。"""
        stopwatch = Stopwatch()
        tool_calls: list[str] = []

        loaded = self.call_tool("profile.read_client", client=client)
        tool_calls.append("profile.read_client")

        derived = self.call_tool(
            "profile.derive_constraints", client=loaded, questionnaire=questionnaire
        )
        tool_calls.extend(["profile.parse_questionnaire", "profile.derive_constraints"])

        effective_level = int(derived["effective_level"])
        effective_client = loaded.model_copy(update={"risk_capacity": effective_level}, deep=True)

        text = self.call_tool(
            "profile.compose_profile_text",
            llm=self.llm,
            task="client_profile_summary",
            context={
                "display_name": client.display_name,
                "age": client.age,
                "risk_level": effective_level,
                "archived_level": derived["archived_level"],
                "questionnaire_level": derived["questionnaire"]["level"],
                "horizon_years": client.investment_horizon_years,
                "investable_amount": client.investable_amount,
                "liquidity_floor": client.liquidity_floor_ratio,
                "caps": {
                    "product": client.max_single_product_ratio,
                    "asset_class": client.max_single_class_ratio,
                    "issuer": client.max_single_issuer_ratio,
                },
                "prohibited": list(client.prohibited_categories) + list(client.prohibited_product_ids),
                "experience": list(client.experienced_categories),
            },
        )
        tool_calls.append("profile.compose_profile_text")

        profile = {
            "client_id": client.client_id,
            "display_name": client.display_name,
            "age": client.age,
            "is_elderly": client.is_elderly,
            "hard_constraints": derived["hard_constraints"],
            "soft_preferences": derived["soft_preferences"],
            "protection_flags": derived["protection_flags"],
            "questionnaire": derived["questionnaire"],
            "archived_level": derived["archived_level"],
            "effective_level": effective_level,
            "tightened_by_questionnaire": derived["tightened_by_questionnaire"],
            "summary": text["data"].get("summary", ""),
            "consistency_note": text["data"].get("consistency_note", ""),
            "constraint_digest_lines": self._digest_lines(client, effective_level),
        }

        output = {"client": client, "effective_client": effective_client, "profile": profile}
        self.trace(
            "profile",
            payload_in={"client": client.model_dump(), "questionnaire": dict(questionnaire.get("responses", {}))},
            payload_out={"profile": profile, "effective_level": effective_level},
            tool_calls=tool_calls,
            latency_ms=stopwatch.ms(),
            extra={"effective_risk_level": effective_level, "tightened": derived["tightened_by_questionnaire"]},
        )
        return self.guard_output(output)

    # ------------------------------------------------------------------
    @staticmethod
    def _digest_lines(client: ClientProfile, effective_level: int) -> list[str]:
        """生成便于 CLI 打印的约束摘要行（确定性）。"""
        return [
            f"风险承受等级上限：R{effective_level}（档案登记 R{client.risk_capacity}）",
            f"投资期限：{client.investment_horizon_years:g} 年",
            f"流动性资产占比下限：{pct(client.liquidity_floor_ratio)}",
            (
                "集中度上限：单一产品 "
                f"{pct(client.max_single_product_ratio)}／单一类别 {pct(client.max_single_class_ratio)}／"
                f"同一发行人 {pct(client.max_single_issuer_ratio)}"
            ),
            f"内控预警线：{pct(client.warning_ratio)}",
            f"禁止项：{'、'.join(client.prohibited_categories + client.prohibited_product_ids) or '无'}",
            f"已具备投资经验：{'、'.join(client.experienced_categories) or '无'}",
            (
                f"币种：{client.currency}；合格投资者：{'是' if client.qualified_investor else '否'}；"
                f"税收优惠额度：{client.tax_advantaged_quota:,.0f} 元"
            ),
            (
                f"软偏好：收益目标 {pct(client.return_target)}；"
                f"回撤容忍 {pct(client.max_drawdown_tolerance)}；"
                f"费率预算 {pct(client.annual_fee_budget_ratio, 3)}"
            ),
            (
                f"投资者保护标记：{'高龄客户' if client.is_elderly else '非高龄'}；"
                f"双录{'已完成' if client.dual_record_completed else '未完成'}"
            ),
        ]
