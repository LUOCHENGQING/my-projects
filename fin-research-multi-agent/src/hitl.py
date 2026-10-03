"""人机协同（Human-in-the-loop）。

触发条件（由 RiskCheckerAgent 判定）：
    1. 风险等级为 high；或
    2. 反思循环重算次数达到上限仍未消除数据缺口（结论支撑不足）。

触发后图会在 human_review 节点暂停，把「待确认摘要」打印到 CLI 并等待人工输入：
    y / yes / approve  -> 批准，继续走到 WriterAgent 出简报
    n / no  / reject   -> 驳回，流程终止（不产出简报）
    其他输入           -> 视为驳回，并记录原因

`--auto` 参数（或非交互式 stdin）会跳过交互，自动放行并如实标注 source="auto"，
保证 CI / 评测 / 演示环境不会被卡住。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

__all__ = ["HumanDecision", "HumanReviewer"]

_APPROVE = {"y", "yes", "approve", "ok", "1", "是", "同意", "批准"}
_REJECT = {"n", "no", "reject", "0", "否", "驳回", "拒绝"}


@dataclass
class HumanDecision:
    """人工确认结果。"""

    decision: str          # approved / rejected
    reason: str
    source: str            # interactive / auto / non-interactive
    raw_input: str = ""

    @property
    def approved(self) -> bool:
        return self.decision == "approved"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "source": self.source,
            "raw_input": self.raw_input,
        }


class HumanReviewer:
    """把待确认信息呈现给人工，并收集决策。"""

    def __init__(
        self,
        auto: bool = False,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[..., None] = print,
    ) -> None:
        self.auto = auto
        self._input = input_fn
        self._print = print_fn

    # ------------------------------------------------------------------
    def _render(self, payload: Dict[str, Any]) -> None:
        line = "=" * 68
        self._print("")
        self._print(line)
        self._print("⚠  高风险结论，需要人工确认（Human-in-the-loop）")
        self._print(line)
        self._print(f"研究问题   : {payload.get('question', '')}")
        self._print(f"公司 / 年度: {payload.get('company', '')} / {payload.get('year', '')}")
        self._print(f"风险等级   : {payload.get('risk_level', '')}")
        self._print(f"升级原因   : {payload.get('reason', '')}")

        findings = payload.get("risk_findings") or []
        if findings:
            self._print("-" * 68)
            self._print("风险规则命中明细：")
            for idx, item in enumerate(findings, 1):
                self._print(
                    f"  {idx}. [{item.get('level', '').upper()}] {item.get('title', '')} "
                    f"（{item.get('metric_display', '')}，阈值 {item.get('threshold', '')}）"
                )
        gaps = payload.get("gaps") or []
        if gaps:
            self._print("-" * 68)
            self._print("未消除的数据缺口：")
            for idx, gap in enumerate(gaps, 1):
                self._print(f"  {idx}. [{gap.get('code', '')}] {gap.get('problem', '')}")
                self._print(f"     要求补正：{gap.get('required_fix', '')}")
        self._print(line)

    # ------------------------------------------------------------------
    def review(self, payload: Dict[str, Any]) -> HumanDecision:
        """执行一次人工确认。"""
        if self.auto:
            self._print("[HITL] --auto 已开启，自动放行（不阻塞演示与评测）。")
            return HumanDecision(
                decision="approved",
                reason="auto 模式自动放行：高风险结论已标注，交由人工事后复核。",
                source="auto",
            )

        if not sys.stdin or not sys.stdin.isatty():
            self._print("[HITL] 当前为非交互式环境（stdin 非 TTY），按流程自动放行并标记。")
            return HumanDecision(
                decision="approved",
                reason="非交互式环境自动放行：无法取得人工输入，已在简报中保留风险标注。",
                source="non-interactive",
            )

        self._render(payload)
        try:
            raw = self._input("是否批准输出该投研简报？[y/N] > ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""

        lowered = raw.lower()
        if lowered in _APPROVE:
            return HumanDecision("approved", "人工确认通过。", "interactive", raw)
        if lowered in _REJECT or not lowered:
            return HumanDecision("rejected", "人工未批准（默认驳回）。", "interactive", raw)
        return HumanDecision("rejected", f"人工输入无法识别（{raw!r}），按驳回处理。", "interactive", raw)


def render_payload(state: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """从共享状态里提取给人工看的待确认摘要。"""
    plan = state.get("plan") or {}
    companies = plan.get("companies") or []
    risk_report = state.get("risk_report") or {}
    return {
        "question": state.get("question", ""),
        "company": "、".join(companies) if companies else "",
        "year": plan.get("year", ""),
        "risk_level": risk_report.get("overall_level", state.get("risk_level", "")),
        "reason": reason,
        "risk_findings": [
            f for f in (risk_report.get("findings") or []) if f.get("level") in {"high", "medium"}
        ][:8],
        "gaps": state.get("revision_gaps") or [],
        "revision_round": state.get("revision_round", 0),
    }
