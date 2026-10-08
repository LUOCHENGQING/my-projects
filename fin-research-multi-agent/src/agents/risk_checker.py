"""RiskCheckerAgent —— 风险与合规核查（反思循环的驱动者）。

层次与职责：
    位于 agents 层，是图节点 ``risk_checker``。它是 Analyst 与 Writer 之间**唯一的闸门**：
    pass 才允许写报告，revise 会把控制流打回 Analyst，escalate 会转入 human_review 节点。
    也就是说"一份结论能不能进最终报告"由它裁决，而不是由生成结论的 Analyst 自评。

这是整个系统里唯一有权「打回上游」的 Agent，也是唯一能触发人机协同的 Agent。

它做两件事，且刻意分开：
    1. **确定性核查门（gate）** —— 结论是否有证据、是否有量化支撑、异常指标是否做了
       交叉验证、引用能否回溯到原文。这部分是合规闸门，**不交给模型判断**。
    2. **风险规则扫描** —— 调用 check_risk_rules 工具，按阈值规则给出风险等级与命中明细。

反思循环（最多 MAX_REVISION_ROUNDS 轮）：
    gate 发现缺口 且 未超轮次  -> verdict = revise    -> 条件边回到 AnalystAgent 重算
    gate 仍有缺口 且 已超轮次  -> verdict = escalate  -> 转人工确认（需人工确认）
    无缺口 但整体风险等级 high -> verdict = escalate  -> 转人工确认
    其余                       -> verdict = pass      -> 走到 WriterAgent

为什么循环不会死循环：
    * 轮次计数 revision_round 由状态持有，每次 revise 递增；
    * 图本身还设了 recursion_limit 作为最后一道保险；
    * 超限后不是继续重算，而是降级为「需人工确认」，把不确定性交给人，而不是让模型
      无休止地自我说服。

互斥点（与其它四个 Agent）：
    * vs AnalystAgent（职责对立）：Analyst 立论、RiskChecker 质疑；前者产出 findings，
      后者专门找 findings 的漏洞（无证据 / 无量化 / 异常未交叉验证 / 来源不可回溯）。
      两者的判断立场天然相反，因此**裁决权只在这里**，Analyst 不能给自己发通过。
    * vs PlannerAgent：RiskChecker 不改 route（那是入口决策），只在既有拓扑上选择
      "放行 / 打回 / 转人工"。
    * vs RetrieverAgent：它不检索，只**核对**证据是否真实可回溯（是否在本次 evidence
      与 document_store 的 source_ids 之内）。
    * vs WriterAgent：它不写正文、不分配引用编号，只把裁决、缺口与风险明细交给 Writer。

对外关键类 / 函数：
    * ``RiskCheckerAgent``（``name = "risk_checker"``）—— 唯一对外类；
    * 模块级常量 ``OUTLIER_RULES`` —— 触发"必须交叉验证"的异常指标阈值表（合规口径）。

主要输入输出：
    输入：``state["findings"]`` / ``state["metrics"]`` / ``state["evidence"]`` /
    ``state["risk_report"]``（取历史 gate_history）/ ``state["revision_round"]`` /
    ``state["max_revision_rounds"]`` / ``state["plan"]``。
    输出（写回新 state）：``risk_report``（工具结果 + 叙述 + 缺口 + 历史 + 双裁决）、
    ``risk_verdict``（**编排层真正读的字段**）、``risk_level``、``needs_human``、
    ``revision_gaps``、``revision_requests`` / ``revision_round``（仅在 revise 时递增）、
    ``llm_stats["risk_checker"]``。

被谁调用：
    * ``src/orchestrator.py`` 注册为节点 ``"risk_checker"``，其路由函数
      ``route_after_risk`` 读 ``risk_verdict`` 决定去 analyst / human_review / writer；
    * ``tests/test_risk_loop.py`` **直接实例化本类**做反思循环的单元验证。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..state import ResearchState, evidence_ids
from ..tools.registry import PermissionLevel
from .base import BaseAgent

#: 触发「必须做交叉验证」的异常指标规则
#: key   —— 对应 metrics 里的指标键（与 analyst.py 的计划键一致，两边必须同名）
#: test  —— 判定函数：入参是该指标的数值（已归一为小数口径），返回 True 表示"异常"
#: reason —— 写进缺口说明的金融含义，会直接展示给用户与人工复核者，所以要说清"为什么要看一眼"
#: 阈值取的是常见的审慎口径（不是法定标准），含义是"跌破/突破这条线就得结合趋势解释"，
#: 而不是"低于此即违规"。判定为异常 ≠ 结论错误，只是要求补充交叉验证。
OUTLIER_RULES: List[Dict[str, Any]] = [
    {"key": "cash_conversion", "test": lambda v: v < 0.50,
     "reason": "净利润现金含量低于 0.50 倍，盈利的现金支撑可能不足"},
    {"key": "debt_to_asset", "test": lambda v: v > 0.65,
     "reason": "资产负债率高于 65%，杠杆水平需要结合趋势判断"},
    {"key": "revenue_growth", "test": lambda v: v < 0.0,
     "reason": "营业收入出现负增长，需要解释原因"},
    {"key": "gross_margin", "test": lambda v: v < 0.15,
     "reason": "毛利率低于 15%，需要判断是否可持续"},
    {"key": "current_ratio", "test": lambda v: v < 1.20,
     "reason": "流动比率低于 1.20，短期偿债能力需要交叉验证"},
]


class RiskCheckerAgent(BaseAgent):
    """风险与合规核查 Agent：确定性裁决 + 风险规则扫描 + 人机协同触发。

    职责：
        1. ``_gate`` 做确定性核查，产出结构化缺口 gaps；
        2. 调 ``check_risk_rules`` 工具做阈值规则扫描，得到 overall_level 与命中明细；
        3. ``_decide`` 用 gaps / 轮次 / 风险等级算出**唯一生效的** verdict；
        4. 再请 LLM 写一段核查叙述（其 verdict 只作为对照，见 verdict_override）；
        5. 处理轮次与打回要求：revise 时把 gaps 的 required_fix 转成下轮指令并递增轮次，
           否则清空指令、保持轮次不变。

    关键属性（类属性）：
        name = "risk_checker" —— 写进 trace 与 steps。
        role = "风险与合规核查、结论打回"。
        allowed_tools = ("check_risk_rules",) —— 只此一个，无法检索、无法计算、无法写文件。
        permissions = {PermissionLevel.RESTRICTED_READ} —— 读受控的合规规则库需要显式授权。

    状态流转（本项目唯一会"回退"的节点）：
        pass      -> 状态原样交给 Writer（revision_requests 被清空）。
        revise    -> revision_round += 1，并把 gaps 的 required_fix 写进 revision_requests，
                     条件边回 Analyst 重算；Analyst 靠 revision_round > 0 判断要补交叉验证口径。
        escalate  -> needs_human = True，交给 human_review 节点等人工决定是否继续写。
        每次裁决都会追加一条 gate_history，因此"为什么被打回"在收敛后依然可复盘。

    设计要点：
        合规闸门是**代码判断**（``_decide``），不是模型判断。LLM 的 verdict 只写入
        ``risk_report["llm_verdict"]`` 并置 ``verdict_override`` 标记，供人观察"模型是否
        与规则一致"，但**不影响路由**——因为路由读的是 ``risk_verdict``（确定性结果）。
    """

    name = "risk_checker"
    role = "风险与合规核查、结论打回"
    allowed_tools = ("check_risk_rules",)
    permissions = frozenset({PermissionLevel.RESTRICTED_READ})

    def _execute(self, state: ResearchState) -> ResearchState:
        """核查门 + 风险扫描 + 确定性裁决 + 叙述生成。

        参数：
            state: 上游共享状态。读取 ``plan``（company / year / period）、``metrics``、
                ``findings``、``evidence``、``risk_report``（只为取 ``gate_history`` 续写）、
                ``revision_round``、``max_revision_rounds``。

        返回：
            新 state，写入 ``risk_report`` / ``risk_verdict`` / ``risk_level`` /
            ``needs_human`` / ``revision_gaps`` / ``llm_stats["risk_checker"]``，
            并按裁决结果更新 ``revision_requests`` 与 ``revision_round``。

        副作用 / 异常：
            * 调用一次 ``check_risk_rules`` 工具（只读合规规则库 + 传入本地指标值）；
            * 调用一次 LLM（``task="risk_review"``）；
            * 工具失败不中断裁决：记入 errors 后按 ``overall_level = "info"`` 继续，
              即"规则库不可用"不会让流程卡死，但会在 errors 与 trace 里留痕。
            * 本方法不主动抛异常。
        """
        new_state = self._copy(state)
        plan = state.get("plan") or {}
        company = (plan.get("companies") or [""])[0]
        year = int(plan.get("year") or 0)
        period = str(plan.get("period") or "年度")
        metrics: Dict[str, Any] = dict(state.get("metrics") or {})
        findings: List[Dict[str, Any]] = list(state.get("findings") or [])
        revision_round = int(state.get("revision_round") or 0)
        max_rounds = int(state.get("max_revision_rounds") or self.ctx.config.max_revision_rounds)

        # ---- 1) 确定性核查门 ----
        # numeric 只保留数值型指标的原始值（小数口径），作为规则扫描的"额外事实"
        # 与异常判定（OUTLIER_RULES）的输入；display 之类的展示字段不参与。
        numeric = {k: float(v["value"]) for k, v in metrics.items() if isinstance(v.get("value"), (int, float))}
        gaps = self._gate(state, findings, metrics, numeric)

        # ---- 2) 风险规则扫描（工具） ----
        risk_result = self.call_tool(
            "check_risk_rules",
            {
                "company": company,
                "year": year,
                "period": period,
                # extra_facts 让规则库能吃到"本次现算的比率"（如 cash_conversion），
                # 而不只依赖库内预置事实；min_level="low" 表示从最低等级开始完整返回命中项。
                "extra_facts": numeric,
                "min_level": "low",
            },
            )
        risk_data: Dict[str, Any] = dict(risk_result.data or {}) if risk_result.ok else {}
        if not risk_result.ok:
            # 失败降级：risk_data 为空 -> overall 落回 "info"，流程继续但错误被显式记录。
            self._merge_errors(
                new_state, "risk_checker",
                [f"风险规则工具失败：{(risk_result.error or {}).get('message', '')}"],
            )
        overall = str(risk_data.get("overall_level") or "info")

        # ---- 3) 确定性裁决（合规闸门不委托给模型） ----
        verdict = self._decide(gaps, revision_round, max_rounds, overall)

        # ---- 4) LLM 生成核查叙述（与裁决交叉校验） ----
        # 注意：payload 里已经把 deterministic_verdict 告诉模型，模型只能"解释"这个裁决；
        # 即使它给出不同意见，也只会被记进 llm_verdict / verdict_override 供人观察。
        payload = {
            "company": company,
            "year": year,
            "period": period,
            "gate": {
                "gaps": gaps,
                "round": revision_round,
                "max_rounds": max_rounds,
                "deterministic_verdict": verdict,
            },
            "risk": {
                "overall_level": overall,
                "triggered_count": risk_data.get("triggered_count", 0),
                "findings": risk_data.get("findings", []),
                "facts_used": risk_data.get("facts_used", {}),
            },
            "findings": [
                {
                    "id": f.get("id"),
                    "title": f.get("title"),
                    "statement": f.get("statement"),
                    "ratio_refs": f.get("ratio_refs"),
                    "evidence_ids": f.get("evidence_ids"),
                    # 只传交叉验证的条数：模型无需复述全文，避免叙述与结论正文重复。
                    "cross_checks": len(f.get("cross_checks") or []),
                }
                for f in findings
            ],
            # metrics 只传 display（展示串），不传原始 value：叙述是给人看的，
            # 且避免模型拿裸数值自行做二次运算。
            "metrics": {k: v.get("display") for k, v in metrics.items()},
        }
        response = self.ctx.llm.chat("risk_review", payload)
        llm_data: Dict[str, Any] = dict(response.data or {})
        llm_verdict = str(llm_data.get("verdict") or verdict)

        # ---- 5) 写回状态 ----
        narrative = str(llm_data.get("narrative") or "")
        escalation_reason = str(llm_data.get("escalation_reason") or "")
        if not escalation_reason and verdict == "escalate":
            # escalate 必须带原因：人工确认界面要靠它解释"为什么把我叫来"，
            # 模型漏写时给一句兜底文案（而不是留空让人对着空白页判断）。
            escalation_reason = "核查判定需要人工确认，但模型未给出原因，请人工复核风险明细。"

        # 追加本轮核查记录：最终状态只保留「当前缺口」，但历史必须留痕，
        # 否则"为什么被打回"这个信息会在循环收敛后丢失，事后无法复盘。
        history = list((state.get("risk_report") or {}).get("gate_history") or [])
        history.append(
            {
                "round": revision_round,
                "verdict": verdict,
                "risk_level": overall,
                "gaps": gaps,
                "narrative": narrative,
            }
        )

        new_state["risk_report"] = {
            **risk_data,
            "narrative": narrative,
            "escalation_reason": escalation_reason,
            "gaps": gaps,
            "gate_history": history,
            "gate_round": revision_round,
            "gate_max_rounds": max_rounds,
            "deterministic_verdict": verdict,
            # llm_verdict 与 verdict_override 是"模型是否同意规则裁决"的观测项，
            # 不参与路由（路由只读 risk_verdict），留痕的目的是评估模型与规则的一致性。
            "llm_verdict": llm_verdict,
            "verdict_override": llm_verdict != verdict,
        }
        new_state["risk_verdict"] = verdict
        new_state["risk_level"] = overall
        # needs_human 只是"本步判定需要人工"的标记；真正的人工节点由编排层的条件边决定。
        new_state["needs_human"] = verdict == "escalate"
        new_state["revision_gaps"] = gaps
        new_state["llm_stats"] = {**(state.get("llm_stats") or {}), "risk_checker": self._llm_stats_fragment(response)}

        if verdict == "revise":
            # 把结构化缺口翻译成给 Analyst 的自然语言整改要求，供下一轮 payload 使用。
            new_state["revision_requests"] = [str(g.get("required_fix", "")) for g in gaps if g.get("required_fix")]
            new_state["revision_round"] = revision_round + 1
        else:
            # 非 revise 一律清空指令并**保持轮次不变**：避免轮次在 pass/escalate 时被动增长，
            # 从而让 max_revision_rounds 的语义恒为"允许被打回的次数"。
            new_state["revision_requests"] = []
            new_state["revision_round"] = revision_round

        return new_state

    # ------------------------------------------------------------------
    @staticmethod
    def _decide(gaps: List[Dict[str, Any]], revision_round: int, max_rounds: int, overall: str) -> str:
        """确定性裁决。

        参数：
            gaps: 核查门产出的缺口列表（空列表表示"这一轮没有发现缺口"）。
            revision_round: 当前已完成的打回轮次（从 0 开始）。
            max_rounds: 允许的最大打回轮次。
            overall: 风险规则扫描给出的整体等级（info / low / medium / high 之类）。

        返回：
            ``"revise"`` / ``"escalate"`` / ``"pass"`` 三者之一。

        判定优先级（顺序即优先级，不能调换）：
            1. 有缺口且还有轮次预算 -> revise（给上游一次自我修正的机会）；
            2. 有缺口但轮次用尽   -> escalate（**不再让模型自我说服**，交给人）；
            3. 无缺口但风险等级为 high -> escalate（合规上高风险事项不允许自动发布）；
            4. 其余 -> pass。

        副作用 / 异常：无（纯函数）。overall 的比较是精确字符串比较，
            因此大小写/近义词（如 "High"）不会被识别为高风险——这是既有实现。
        """
        if gaps and revision_round < max_rounds:
            return "revise"
        if gaps and revision_round >= max_rounds:
            return "escalate"
        if overall == "high":
            return "escalate"
        return "pass"

    # ------------------------------------------------------------------
    def _gate(
        self,
        state: ResearchState,
        findings: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        numeric: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """确定性核查门：数据是否足够、结论是否被支撑。

        参数：
            state: 当前状态（用于取 ``evidence`` 与 ``document_store.source_ids``）。
            findings: Analyst 产出的结论列表。
            metrics: 比率字典。
                注：实际实现为 —— 该参数**在本方法内未被读取**，异常值判定统一走 ``numeric``。
                保留它是为了与调用处签名一致（也便于将来放开"按 metrics 展示口径"的检查）。
            numeric: 指标键 -> 数值（小数口径），用于 OUTLIER_RULES 与收入质量判定。

        返回：
            缺口列表，每项形如
            ``{"code", "target", "problem", "required_fix", "severity"}``：
                * code —— GAP-NO-FINDING / GAP-EVIDENCE / GAP-TRACE / GAP-METRIC / GAP-XCHECK
                  （前四类来自逐条结论检查，GAP-XCHECK 来自异常值与收入质量检查）；
                * target —— 结论 id，或触发规则的指标键；
                * severity —— blocker / high / medium，表示合规上的严重程度。
            返回空列表表示"这一轮没有发现缺口"。

        副作用 / 异常：
            无副作用（只读 state / findings / numeric，不修改任何入参）。
            对 ``rule["test"](value)`` 的调用包了 try/except：判定函数本身出错时**跳过该规则**
            而不是让整步失败（规则表是可扩展的，不能假设每个 lambda 都安全）。
        """
        gaps: List[Dict[str, Any]] = []
        # 证据"真实存在"的判定基准有两个来源，缺一不可：
        #   valid_evidence —— 本次检索真正recall出来的 child_id（防凭空编号）；
        #   known_sources  —— 资料库里真实存在的 source_id（防跨库/伪造来源，即可回溯性）。
        valid_evidence = evidence_ids(state)
        doc_store = self.ctx.document_store
        known_sources = set(doc_store.source_ids) if doc_store is not None else set()
        evidence_index = {str(e.get("child_id")): e for e in state.get("evidence", [])}

        # 没有任何结论属于阻断级缺口：Report 将无内容可写，必须打回而不是放行。
        if not findings:
            gaps.append(
                {
                    "code": "GAP-NO-FINDING",
                    "target": "-",
                    "problem": "AnalystAgent 未产出任何分析结论。",
                    "required_fix": "基于已检索证据重新计算指标并给出至少一条有量化支撑的结论。",
                    "severity": "blocker",
                }
            )
            return gaps

        for finding in findings:
            fid = str(finding.get("id"))
            evs = [str(e) for e in (finding.get("evidence_ids") or [])]

            # (1) 证据支撑
            # 分两种情况报同一 code，是因为整改动作不同：一种是"没挂证据"，
            # 另一种是"挂了但这条证据不在本次检索结果里"（多半是引用幻觉）。
            if not evs:
                gaps.append(
                    {
                        "code": "GAP-EVIDENCE",
                        "target": fid,
                        "problem": f"结论「{finding.get('title')}」没有任何原文证据支撑。",
                        "required_fix": "为该结论补挂至少一条来源于资料库的证据（child_id）。",
                        "severity": "blocker",
                    }
                )
            elif not set(evs) & valid_evidence:
                gaps.append(
                    {
                        "code": "GAP-EVIDENCE",
                        "target": fid,
                        "problem": f"结论「{finding.get('title')}」引用的证据不在本次检索结果中。",
                        "required_fix": "重新检索并绑定真实存在的证据片段。",
                        "severity": "blocker",
                    }
                )

            # (2) 引用可回溯：证据必须来自资料库内的真实文档
            # 这一条是"引用可追溯"的兜底：即使 child_id 在 evidence 里，
            # 也要能顺着它查到一个资料库里真实存在的 source_id。
            for eid in evs:
                src = str((evidence_index.get(eid) or {}).get("source_id", ""))
                # 仅在"资料库已知来源集合非空且该证据确有 source_id"时才判定不可回溯，
                # 避免资料库未装载（known_sources 为空）时把一切都误判为不可回溯。
                if known_sources and src and src not in known_sources:
                    gaps.append(
                        {
                            "code": "GAP-TRACE",
                            "target": fid,
                            "problem": f"证据 {eid} 的资料编号 {src} 无法在资料库中回溯。",
                            "required_fix": "替换为可回溯的资料来源。",
                            "severity": "high",
                        }
                    )

            # (3) 量化支撑
            # 没有 ratio_refs 的结论在合规上属于"定性断言"，不予放行——这也是本项目
            # "数字由工具产生"原则在闸门侧的体现。
            if not (finding.get("ratio_refs") or []):
                gaps.append(
                    {
                        "code": "GAP-METRIC",
                        "target": fid,
                        "problem": f"结论「{finding.get('title')}」没有任何量化指标支撑。",
                        "required_fix": "用 calc_ratio / get_financial_metric 补齐量化指标后再下结论。",
                        "severity": "high",
                    }
                )

        # (4) 异常指标必须做交叉验证（反思循环的主要触发点）
        # cross_checked_keys 的构造含义：只要某个指标出现在**任意一条带 cross_checks 的结论**的
        # ratio_refs 里，就认为它已被交叉验证过（本系统的交叉验证挂在结论上，而非指标上）。
        cross_checked_keys = {
            str(ref)
            for finding in findings
            if (finding.get("cross_checks") or [])
            for ref in (finding.get("ratio_refs") or [])
        }
        for rule in OUTLIER_RULES:
            key = str(rule["key"])
            value: Optional[float] = numeric.get(key)
            if value is None:
                # 该指标本轮没算出来 -> 无法判定异常，跳过（它更可能因"缺量化支撑"在 (3) 被记）。
                continue
            try:
                triggered = bool(rule["test"](value))
            except Exception:  # noqa: BLE001
                # 规则函数本身异常时跳过该条，不影响其它规则与整体裁决。
                continue
            if not triggered:
                continue
            if key in cross_checked_keys:
                continue
            gaps.append(
                {
                    "code": "GAP-XCHECK",
                    "target": key,
                    "problem": f"指标 {key}={value:.4f} 属于异常值：{rule['reason']}，但结论未做交叉验证。",
                    "required_fix": (
                        f"针对 {key} 补充上期（上年同期）对比，并与相关科目（如应收账款增速、"
                        "经营活动现金流）做交叉印证后再下结论。"
                    ),
                    "severity": "medium",
                }
            )

        # (5) 收入质量：应收账款增速显著高于收入增速时，同样要求交叉验证
        # 金融含义：应收增速远超收入增速往往意味着"收入靠赊销堆出来"或放宽信用政策，
        # 需要与经营现金流对照来判断收入确认质量。10 个百分点是经验性审慎阈值。
        ar_growth = numeric.get("receivable_growth")
        rev_growth = numeric.get("revenue_growth")
        if ar_growth is not None and rev_growth is not None and (ar_growth - rev_growth) > 0.10:
            # 这里检查的是 cash_conversion 是否已被交叉验证过（而非应收增速本身），
            # 因为"收入质量"的正解是用现金流来印证，所以缺口也指向现金流指标。
            if "cash_conversion" not in cross_checked_keys:
                gaps.append(
                    {
                        "code": "GAP-XCHECK",
                        "target": "cash_conversion",
                        "problem": (
                            f"应收账款增速 {ar_growth * 100:.2f}% 显著高于营业收入增速 "
                            f"{rev_growth * 100:.2f}%，收入质量存疑，但结论未做交叉验证。"
                        ),
                        "required_fix": "用应收账款增速与经营现金流对照，验证收入确认质量。",
                        "severity": "high",
                    }
                )

        return gaps

    # ------------------------------------------------------------------
    def _trace_input(self, state: ResearchState) -> Dict[str, Any]:
        """trace 输入摘要：记录待核查的结论 ID、当前轮次与轮次上限。

        参数：state —— 进入本步时的状态。
        返回：``{"findings": [id...], "revision_round": int, "max_revision_rounds": int|None}``。
        副作用 / 异常：无。
        """
        return {
            "findings": [f.get("id") for f in (state.get("findings") or [])],
            "revision_round": state.get("revision_round", 0),
            "max_revision_rounds": state.get("max_revision_rounds"),
        }

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        """trace 输出摘要：完整记录裁决、缺口、逐轮历史与风险命中。

        参数：state —— 本步执行后的状态，读 ``risk_report`` / ``risk_verdict`` / ``risk_level``。
        返回：``{"verdict", "risk_level", "gaps", "gate_history"(每轮只留缺口 code),
            "risk_findings"(rule_id / level / title), "narrative", "escalation_reason",
            "verdict_override"}``。
            注：gate_history 里的 gaps 在这里被压成 code 列表以控体积，state 里存的是完整缺口。
        副作用 / 异常：无；``risk_report`` 缺失时按空字典处理。
        """
        report = state.get("risk_report") or {}
        return {
            "verdict": state.get("risk_verdict"),
            "risk_level": state.get("risk_level"),
            "gaps": report.get("gaps", []),
            "gate_history": [
                {"round": h.get("round"), "verdict": h.get("verdict"),
                 "gaps": [g.get("code") for g in (h.get("gaps") or [])]}
                for h in (report.get("gate_history") or [])
            ],
            "risk_findings": [
                {"rule_id": f.get("rule_id"), "level": f.get("level"), "title": f.get("title")}
                for f in (report.get("findings") or [])
            ],
            "narrative": report.get("narrative", ""),
            "escalation_reason": report.get("escalation_reason", ""),
            "verdict_override": report.get("verdict_override", False),
        }

    def _trace_extra(self, state: ResearchState) -> Dict[str, Any]:
        """trace 补充字段：把"循环往哪走"的关键数字单独摊平，便于按轮次分析。

        参数：state —— 本步执行后的状态。
        返回：``{"verdict", "gaps_count", "risk_level", "next_revision_round"}``；
            其中 ``next_revision_round`` 是**本步写回后**的 ``revision_round``，
            revise 时已递增 1，因此它表示"下一轮 Analyst 将看到的轮次"。
        副作用 / 异常：无。
        """
        report = state.get("risk_report") or {}
        return {
            "verdict": state.get("risk_verdict"),
            "gaps_count": len(report.get("gaps") or []),
            "risk_level": state.get("risk_level"),
            "next_revision_round": state.get("revision_round"),
        }
