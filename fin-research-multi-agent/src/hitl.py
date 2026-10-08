"""人机协同（Human-in-the-loop）。

触发条件（由 RiskCheckerAgent 判定）：
    1. 风险等级为 high；或
    2. 反思循环重算次数达到上限仍未消除数据缺口（结论支撑不足）。

触发后图会在 human_review 节点暂停，把「待确认摘要」打印到 CLI 并等待人工输入：
    y / yes / approve / ok / 1 / 是 / 同意 / 批准  -> 批准，继续走到 WriterAgent 出简报
    n / no  / reject / 0 / 否 / 驳回 / 拒绝        -> 驳回，流程终止（不产出简报）
    其他输入 / 空输入                              -> 视为驳回，并记录原因

`--auto` 参数（或非交互式 stdin）会跳过交互，自动放行并在 source 字段里如实标注来源，
保证 CI / 评测 / 演示环境不会被卡住。
注：实际实现为 auto 模式标 source="auto"，非交互式环境标 source="non-interactive"
（两者都会放行，但事后可从 source 区分「流程自动放行」与「真人工批准」）。

架构层次与职责：
    本模块是「人机协同层」，只负责两件事——把升级原因渲染成人类可读的待确认摘要、
    以及收集人工决策。它自己不做任何风险判定：「是否该转人工」由 RiskCheckerAgent
    给出的 risk_verdict == "escalate" 决定（见 src/agents/risk_checker.py），编排层
    只在 human_review 节点调用本模块。

对外关键对象：
    HumanReviewer     渲染 + 收集决策，review() 是唯一入口
    HumanDecision     决策结果的数据载体（decision / reason / source / raw_input）
    render_payload()  从共享状态抽取「给人工看的摘要」

主要输入输出：
    输入：orchestrator 传入的 payload（question / company / year / risk_level / reason /
          risk_findings / gaps / revision_round）。
    输出：HumanDecision，经 to_dict() 写回共享状态的 human_decision 字段
          （orchestrator 的后置条件边 route_after_human 据此选 writer 还是 END）。

被谁调用：
    * orchestrator.ResearchPipeline._build_spec() 内部的 human_review_node：
      先 render_payload(state, reason) 组装 payload，再 self.reviewer.review(payload)；
    * ResearchPipeline.__init__ 构造 HumanReviewer(auto=..., print_fn=...)；
    * tests/conftest.py 通过 auto=True 的 pipeline 夹具间接使用（不阻塞测试）。

触发的两种原因（由 RiskCheckerAgent 侧产生，本模块只负责展示）：
    1. 风险等级为 high：risk_verdict == "escalate" 但当前没有数据缺口
       （reason 为空且 gaps 为空时取该文案）；
    2. 反思循环达上限仍有缺口：state["revision_gaps"] 非空
       （reason 取「反思循环已达上限（N 轮），仍有 M 项数据缺口未消除」）。

三级降级（保证任何环境都不阻塞）：
    1. auto=True      -> 直接放行，source="auto"；
    2. stdin 非 TTY   -> 直接放行，source="non-interactive"；
    3. 交互式 TTY     -> 打印摘要并读一行输入；空输入或无法识别一律按驳回处理。

注：实际实现为在 human_review 节点内**同步阻塞**读取 stdin（self._input(...)），
而不是 LangGraph 的 interrupt/Command 式「暂停-恢复」：它不会把图挂起后交给外部驱动器，
而是就地等这一次输入；EOFError / KeyboardInterrupt 会被当作空输入（即驳回）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

__all__ = ["HumanDecision", "HumanReviewer"]

_APPROVE = {"y", "yes", "approve", "ok", "1", "是", "同意", "批准"}
_REJECT = {"n", "no", "reject", "0", "否", "驳回", "拒绝"}
# 为什么维护两份词表而不是简单 startswith("y")：投行/合规场景下「确认放行」是高危动作，
# 只认白名单里的明确表达，其余一律落到 _REJECT 分支，避免误把 "yeah?" / 乱码当同意。


@dataclass
class HumanDecision:
    """人工确认结果。

    职责：把「谁、以什么方式、做了什么决定」压成可落 trace、可写进共享状态的载体
    （orchestrator 会调用 to_dict() 存进 state["human_decision"]，并整份写进 JSONL 轨迹）。

    关键属性：
        decision   决策结果，只取 "approved" / "rejected" 两种值
        reason     人类可读的决策说明（自动放行时会写明「无法取得人工输入」等实情）
        source     决策来源，interactive / auto / non-interactive 三选一，
                   用于事后区分「真人工批准」与「流程自动放行」
        raw_input  交互模式下人工输入的原始字符串（自动放行时为空串）

    状态流转：由 HumanReviewer.review() 构造 -> 写进 state["human_decision"] ->
    被条件边 route_after_human 读取（approved 走 writer，其余走 END）。
    """

    decision: str          # approved / rejected
    reason: str
    source: str            # interactive / auto / non-interactive
    raw_input: str = ""

    @property
    def approved(self) -> bool:
        """是否为「批准」（decision == "approved"）；只做判断，无副作用。"""
        return self.decision == "approved"

    def to_dict(self) -> Dict[str, Any]:
        """导出可 JSON 序列化的四个字段，供写状态 / 写 trace 使用。"""
        return {
            "decision": self.decision,
            "reason": self.reason,
            "source": self.source,
            "raw_input": self.raw_input,
        }


class HumanReviewer:
    """把待确认信息呈现给人工，并收集决策。

    职责：HITL 的唯一执行体——负责渲染摘要、按 auto / 非交互 / 交互三种情形取得决策，
    并把「输入不可靠」的情况降级成放行或驳回（见 review()）。

    关键属性：
        auto      是否跳过交互直接放行（由 ResearchPipeline(auto=...) 传入）
        _input    读入一行的函数，默认内置 input；测试可注入假输入实现全自动用例
        _print    输出函数，默认 print；orchestrator 在 quiet=True 时注入空函数静音

    状态流转/副作用：本类不持有跨调用状态；每次 review() 只读 payload、只写 stdout
    （以及交互模式下的阻塞等待），决策结果通过返回值交给调用方写进共享状态。
    """

    def __init__(
        self,
        auto: bool = False,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[..., None] = print,
    ) -> None:
        """初始化人工确认器。

        参数：
            auto      为 True 时 review() 直接放行且不读 stdin（CI/评测/演示用）。
            input_fn  读入一行的可注入函数，签名 (prompt) -> str，默认内置 input。
            print_fn  输出函数，签名与 print 兼容，默认 print（quiet 时由编排层换成空函数）。

        返回：无。
        副作用：仅保存三个属性，不做 IO。
        """
        self.auto = auto
        self._input = input_fn
        self._print = print_fn

    # ------------------------------------------------------------------
    def _render(self, payload: Dict[str, Any]) -> None:
        """把待确认摘要打印到 CLI（只输出，不返回、不修改状态）。

        参数：payload —— render_payload() 的产物，本方法只读其中的 question / company /
        year / risk_level / reason / risk_findings / gaps；缺失字段用空值兜底。

        返回：None。
        副作用：向 self._print 输出多行文本；对 payload 无修改。
        """
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
        """执行一次人工确认。

        参数：
            payload  待确认摘要（render_payload() 的产物），只在交互分支被 _render 打印。

        返回：
            HumanDecision —— 三分支之一：
                auto=True         -> approved / source="auto"；
                stdin 非 TTY      -> approved / source="non-interactive"；
                交互式            -> 命中 _APPROVE 则 approved，命中 _REJECT、空输入或
                                     无法识别的输入一律 rejected（source="interactive"）。

        副作用：
            可能打印摘要并向 stdin 阻塞读一行（交互模式），不修改 payload 与共享状态。
        异常：
            内置 read 抛出的 EOFError / KeyboardInterrupt 在内部被吞掉并当作空输入，
            因此本方法不会把异常抛给图执行器（HITL 不应炸掉整条流水线）。
        """
        if self.auto:
            self._print("[HITL] --auto 已开启，自动放行（不阻塞演示与评测）。")
            return HumanDecision(
                decision="approved",
                reason="auto 模式自动放行：高风险结论已标注，交由人工事后复核。",
                source="auto",
            )

        # 为什么用 isatty 而不是 try/except：CI 里 stdin 是管道或已关闭，读它会直接 EOF，
        # 与其反复处理异常，不如提前判定「拿不到人」并显式标注放行来源，保证流程可跑完。
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
            # 为什么静默成空串：Ctrl-C / EOF 只代表「这次没给出批准」，不是系统故障，
            # 按默认值（驳回）处理即可，不应中断整条流水线。
            raw = ""

        lowered = raw.lower()
        if lowered in _APPROVE:
            return HumanDecision("approved", "人工确认通过。", "interactive", raw)
        if lowered in _REJECT or not lowered:
            return HumanDecision("rejected", "人工未批准（默认驳回）。", "interactive", raw)
        return HumanDecision("rejected", f"人工输入无法识别（{raw!r}），按驳回处理。", "interactive", raw)


def render_payload(state: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """从共享状态里提取给人工看的待确认摘要。

    参数：
        state   共享状态（只读 plan / risk_report / risk_level / revision_gaps /
                revision_round / question，不修改）。
        reason  升级原因文案，由调用方（orchestrator 的 human_review_node）决定：
                「反思循环已达上限……」或「风险等级为 high……」。

    返回：
        扁平字典 {question, company, year, risk_level, reason, risk_findings, gaps,
        revision_round}。其中 company 是多公司用「、」连接；risk_findings 只取
        risk_report.findings 里 level 为 high / medium 的前 8 条（避免刷屏）；
        gaps 直接取 revision_gaps 原文供人工逐条核对补正要求。

    副作用 / 异常：
        纯函数，无副作用、不抛异常（全部用 get 兜底）。
        注：实际实现为 risk_level 优先取 risk_report["overall_level"]，
        取不到才回退到 state["risk_level"]。
    """
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
