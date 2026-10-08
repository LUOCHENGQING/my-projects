"""ClientProfilingAgent —— 客户画像与硬约束提取。

所属层次
--------
Agent 层（`src/agents/`），流水线第 1 环（`profile` 节点）。
被 `src/pipeline.py` 的 `AdvisoryPipeline.node_profile` 调用，
经 `src/agents/__init__.py` 的 `build_agents` 构造。

解决什么问题
------------
投顾的第一道问题是「这位客户到底能买什么、不能买什么」，而不是「买哪只产品」。
本模块把客户档案 + 风险测评问卷翻译成**可执行的硬约束与软偏好**，
并按「从严原则」定出最终生效的风险等级，供下游筛选与闸门使用。

职责边界（与投研、AML 项目里的 Agent 都不同）
--------------------------------------------
本 Agent **不产出任何投资观点**，它只做一件事：把客户档案 + 风险测评问卷
翻译成**可执行的硬约束与软偏好**，并按「从严原则」决定最终生效的风险等级。

- 硬约束：风险等级上限、投资期限、流动性下限、三类集中度上限、禁止项、
  币种、税收优惠额度、合格投资者准入、已具备经验的品类
- 软偏好：收益目标、回撤容忍度、费率预算
- 从严原则：问卷折算等级低于档案登记等级时，取孰低者作为有效等级，并留痕说明

与其他 Agent 的互斥关系
-----------------------
- 只写 `client` / `effective_client` / `profile` 三个键（`can_write_state`），
  **不写** `screening` / `portfolio` / `gate` / `narrative`——"能买什么"的判定
  归 `ProductScreeningAgent`，"买多少"归 `PortfolioOptimizerAgent`，
  "合不合规"归 `SuitabilityOfficerAgent`，本 Agent 一律不越界；
- 模型只用于组织画像措辞（`profile.compose_profile_text`），
  硬约束数值全部来自确定性工具，因此本 Agent **无否决权**，
  也不能放宽任何约束（放宽约束的唯一合法路径是闸门下发 `TightenSpec`，而那是收紧）。

工具白名单：`profile.read_client` / `profile.parse_questionnaire` / `profile.derive_constraints`
权限集合：`profile:read`、`profile:derive`、`trace:write`

注：实际实现为——`SPEC.tools` 还包含 `profile.compose_profile_text`（复用统一的
文案工具生成画像摘要），即白名单共 **4** 个工具；上面列出的是"硬约束提取"这一步
用到的前三个。权限集合与 `SPEC.permissions` 一致。
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

#: 本 Agent 的能力契约（名称/角色/提示词/工具白名单/权限集合/可写状态键）
#: 权限只需要 `profile:*` 与 `trace:write`；`can_write_state` 三个键与其它 Agent 不相交。
SPEC = AgentSpec(
    name="ClientProfilingAgent",
    role="客户画像与硬约束提取",
    system_prompt=SYSTEM_PROMPT,
    tools=("profile.read_client", "profile.parse_questionnaire", "profile.derive_constraints", "profile.compose_profile_text"),
    permissions=frozenset({PERM_PROFILE_READ, PERM_PROFILE_DERIVE, PERM_TRACE_WRITE}),
    can_write_state=("client", "effective_client", "profile"),
)


class ClientProfilingAgent(BaseAgent):
    """客户画像 Agent。

    职责：读档案 → 解析问卷 → 抽取硬约束/软偏好 → 按从严原则产出有效风险等级
    → 生成画像摘要文案，最终写出 `client` / `effective_client` / `profile`。

    边界：不筛选产品、不建组合、不做合规判定、不提任何收益观点；
    无否决权，也不具备 `screening:*` / `optimize:*` / `suitability:*` / `narrative:*` 权限，
    因此即便代码写错也无法触碰下游状态键（会被 `BaseAgent.guard_output` 拦下）。
    """

    def __init__(self, **kwargs: Any) -> None:
        """以固定契约构造画像 Agent。

        参数：
            **kwargs: 透传给 `BaseAgent.__init__`，常用 `llm=` 与 `tracer=`；
                不接受 `spec`（本类已固化 `SPEC`，避免运行期被替换掉权限边界）。

        返回：None。
        副作用/异常：无。
        """
        super().__init__(SPEC, **kwargs)

    # ------------------------------------------------------------------
    def run(self, *, client: ClientProfile, questionnaire: Mapping[str, Any]) -> dict[str, Any]:
        """提取硬约束与软偏好，并返回按从严原则收紧后的有效客户档案。

        参数（keyword-only）：
            client: 客户档案（Raw 输入，含档案登记风险等级 `risk_capacity`、
                期限、集中度上限、禁止项、经验等）。
            questionnaire: 风险测评问卷原始数据，形如
                `{"items": [...], "responses": {client_id: {题号: 分值}}, ...}`；
                缺少该客户作答时折算等级为 None，此时**不收紧**。

        返回：
            经 `guard_output` 校验的字典，含三个键：
            - `client`：原样回写的输入档案（不收紧，保留档案登记等级）；
            - `effective_client`：把 `risk_capacity` 换成从严原则生效等级的**深拷贝**，
              下游筛选/构建/闸门都以它为准；
            - `profile`：结构化画像（硬约束、软偏好、保护标记、问卷折算结果、
              档案登记等级、有效等级、是否被问卷收紧、摘要与一致性说明、CLI 摘要行）。

        副作用：
            - 调用 `profile.compose_profile_text`，可能触发 LLM（无 Key 时自动 mock）；
            - 写一条 `node="profile"` 的 trace，`extra` 里带生效等级与是否收紧标记。

        异常：
            PermissionError: 越权调用工具或越权写共享状态（能力契约被改坏时）。
            KeyError: 工具返回值缺少约定字段（如 `derived["effective_level"]`）。
        """
        stopwatch = Stopwatch()
        tool_calls: list[str] = []

        # 读取档案：工具内部返回深拷贝，避免下游误改缓存对象
        loaded = self.call_tool("profile.read_client", client=client)
        tool_calls.append("profile.read_client")

        # 抽取硬约束与软偏好（从严原则在此工具内落地：effective = min(档案, 问卷)）
        derived = self.call_tool(
            "profile.derive_constraints", client=loaded, questionnaire=questionnaire
        )
        # 注：实际实现为 `profile.derive_constraints` 的 handler 内部调用
        # `profile.parse_questionnaire`，本方法并未直接调用它；这里一并记入
        # tool_calls，是为了让 trace 反映"实际发生过的工具行为"。
        tool_calls.extend(["profile.parse_questionnaire", "profile.derive_constraints"])

        effective_level = int(derived["effective_level"])
        # 有效客户档案 = 原档案 + 生效风险等级（深拷贝，不动入参）
        effective_client = loaded.model_copy(update={"risk_capacity": effective_level}, deep=True)

        # 文案生成：数字全部取自确定性结果，模型只负责组织语言
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

        # 结构化画像：下游（闸门、建议书、评估脚本）读的就是这份字典
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
        """生成便于 CLI 打印的约束摘要行（确定性）。

        参数：
            client: 客户档案（这里读的是**档案登记**等级 `risk_capacity`，
                用于与生效等级对照，显示"从严"是否触发）。
            effective_level: 从严原则生效后的风险等级（问卷低于档案时取问卷值）。

        返回：
            中文摘要行列表，固定顺序：风险等级上限、投资期限、流动性下限、
            集中度上限、内控预警线、禁止项、已有经验、币种/合格投资者/税优额度、
            软偏好、投资者保护标记。

        副作用/异常：
            无副作用；纯格式化，不调用模型、不读文件，同样的输入必然得到同样的输出。
        """
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
