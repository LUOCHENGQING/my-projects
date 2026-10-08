"""WriterAgent —— 结构化投研简报生成。

层次与职责：
    位于 agents 层的最末端（图节点 ``writer``），是**唯一的产物出口**。
    从 planner 出发有三条路径都汇到它：直达（寒暄类问题）、risk_checker --pass-->、
    human_review --approved-->；另有 writer -> END 的固定边收尾。
    进入本 Agent 就意味着"内容已定稿"，它只做组织与排版，不再产生新的判断。

职责边界：
    * 唯一持有 `cite_source`（write 权限）的 Agent：只有它能把资料编号解析成正式引用编号。
      这也意味着引用编号的分配是**单一职责**的，不会出现多个 Agent 各自编号导致错乱。
    * 报告骨架（章节、表格、引用清单）由 Agent 确定性地拼装，
      LLM 只负责核心结论与论证段落的语言组织。
      这样做的好处：无论模型怎么发挥，**引用编号格式与可追溯性都不会被破坏**。

引用可追溯的三段式：
    finding.evidence_ids -> evidence.source_id -> cite_source 分配的引用编号 [n]
    最终报告里的每个 [n] 都能在「五、引用来源」里找到对应的 source_id 与文件路径。

互斥点（与其它四个 Agent）：
    * **独占引用编号**：cite_source 只在 Writer 的白名单里，因此 [n] 的分配权与写权限
      都是它独有的；Analyst / Retriever 产生的都是 source_id 级别的原始标识，
      不存在"两处各自编号"的可能。
    * vs PlannerAgent：Writer 不改 route，也不重做任务分解。
    * vs RetrieverAgent：Writer 不检索，只从既有 evidence 反查 source_id。
    * vs AnalystAgent：Writer 不计算、不下新结论；它只把 findings / metrics / facts
      排版成报告，结论文本来自 Analyst（必要时用 finding.statement 兜底）。
    * vs RiskCheckerAgent：Writer 不做合规裁决；风险等级、缺口、叙述原样引用
      risk_report 的内容，逐字呈现而不做二次评判。

对外关键类 / 函数：
    * ``WriterAgent``（``name = "writer"``）—— 唯一对外类；
    * 模块级函数 ``_fmt_num``（数值格式化）、``_has_citation``（正文引用检测）；
    * 模块级类 ``_SectionCounter``（章节序号计数器，处理"可选章节"导致的编号错位）。

主要输入输出：
    输入：``state["plan"]``、``state["findings"]``、``state["evidence"]``、
    ``state["facts"]``、``state["metrics"]``、``state["risk_report"]``、
    ``state["analysis_summary"]``、``state["human_decision"]``、``state["config"]``、``state["run_id"]``。
    输出（写回新 state）：``citations``（引用清单）、``citation_index``（source_id -> 编号）、
    ``report``（Markdown 简报全文）、``llm_stats["writer"]``；
    引用分配失败会写进 ``errors``。

被谁调用：
    * ``src/orchestrator.py`` 注册为节点 ``"writer"``，其后接 END；
    * ``tests/test_citation_traceability.py`` 直接实例化做引用可追溯性验证。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..state import ResearchState
from ..tools.registry import PermissionLevel
from .base import BaseAgent


def _fmt_num(value: Any) -> str:
    """数值格式化：大数加千分位，小数保留合理位数。

    参数：
        value: 任意值；``None`` 表示"该指标没有数据"。

    返回：
        展示用字符串：
            * ``None``   -> ``"-"``（表格里占位，而不是空单元格或 "None"）；
            * ``bool``   -> ``str(value)``，即 "True"/"False"（**先于 int 判断**，
              否则 True 会被当成 1 而显示成 "1"）；
            * ``int``    -> 千分位整数（如 ``1,234``）；
            * ``float``  -> 绝对值 >= 1000 用两位小数 + 千分位；>= 1 用 4 位小数并去掉尾随 0；
              否则（< 1，如比率）用 6 位小数并去掉尾随 0 与孤立小数点；
            * 其它类型 -> ``str(value)``。

    副作用 / 异常：无（纯函数，不抛异常）。
        注：``bool`` 是 ``int`` 的子类，这里靠判断顺序先拦下 bool，是有意为之的顺序依赖。
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        # 量级分档：大数看千分位，中等数看 4 位，比率类小数看 6 位——
        # 目的是让"万元级金额"和"0.0123 的比率"在同一张表里都可读又不丢精度。
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        if abs(value) >= 1:
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


class WriterAgent(BaseAgent):
    """结构化投研简报撰写与引用绑定 Agent：产物出口。

    职责：
        1. 统计各 source_id 被结论引用的次数，按"被引越多编号越小"确定引用顺序；
        2. 逐个调用 ``cite_source``（唯一的 write 权限工具）换取正式引用编号，形成
           ``citations`` 与 ``citation_index``；
        3. 把 finding -> 引用编号映射成 ``citation_map``；
        4. 请 LLM 组织标题、核心结论与论证段落（只写"话"，不写"编号格式"）；
        5. 确定性拼装整份 Markdown 报告（章节、指标表、比率表、风险提示、引用清单、免责声明）。

    关键属性（类属性）：
        name = "writer" —— 写进 trace 与 steps。
        role = "结构化投研简报撰写与引用绑定"。
        allowed_tools = ("cite_source",) —— 只有引用编号分配这一个工具。
        permissions = {PermissionLevel.WRITE} —— **全项目唯一的 write 权限持有者**。
            注：该工具的作用是"分配引用编号并返回编号与出处元数据"；
            ``report`` 本身是写回 state（由编排层落 trace/返回结果），并非由 Agent 直接写文件。

    状态流转：
        入口态：findings / evidence / metrics / facts / risk_report 都已定稿
            （可能是 pass 放行，也可能是人工 approved）。
        出口态：citations / citation_index / report 三件套，随后图走到 END。
        本 Agent 是**终态节点**，没有任何条件边从它出发（只有 writer -> END），
        因此它不会被重跑；报告中出现的每个 [n] 都是这一刻确定下来的。
    """

    name = "writer"
    role = "结构化投研简报撰写与引用绑定"
    allowed_tools = ("cite_source",)
    permissions = frozenset({PermissionLevel.WRITE})

    def _execute(self, state: ResearchState) -> ResearchState:
        """分配引用编号 -> 建立结论到编号的映射 -> LLM 组织语言 -> 确定性拼装报告。

        参数：
            state: 上游共享状态。读取 plan（companies[0] / year / period）、findings、
                evidence、risk_report、analysis_summary、human_decision、config、run_id、question。

        返回：
            新 state，写入 ``citations`` / ``citation_index`` / ``report`` /
            ``llm_stats["writer"]``；引用分配失败的信息通过 ``_merge_errors`` 写入 ``errors``。

        副作用 / 异常：
            * 对每个被引用的 source_id 调用一次 ``cite_source``（write 级操作）；
              单个失败只记入 ``citation_failures`` 并跳过，不影响其它引用与报告生成——
              宁可少一条引用，也不让整份报告写不出来。
            * 调用一次 LLM（``task="write"``）；
            * 本方法不主动抛异常。
        """
        new_state = self._copy(state)
        plan = state.get("plan") or {}
        companies = plan.get("companies") or [""]
        company = companies[0] if companies else ""
        year = int(plan.get("year") or 0)
        period = str(plan.get("period") or "年度")
        findings: List[Dict[str, Any]] = list(state.get("findings") or [])
        evidence: List[Dict[str, Any]] = list(state.get("evidence") or [])
        risk_report: Dict[str, Any] = dict(state.get("risk_report") or {})

        # child_id -> 证据片段，用于把"结论引用的证据"翻译成"资料来源编号"。
        evidence_index = {str(e.get("child_id")): e for e in evidence}

        # ---- 1) 为所有被引用的资料分配引用编号（write 权限工具） ----
        # 排序策略：被结论引用次数最多的资料优先拿到小编号（主要来源 = [1]），
        # 次数相同时按首次出现顺序，保证结果确定、可复现。
        usage: Dict[str, int] = {}
        first_seen: Dict[str, int] = {}
        order = 0
        for finding in findings:
            for eid in finding.get("evidence_ids") or []:
                src = str((evidence_index.get(str(eid)) or {}).get("source_id", ""))
                # 证据 ID 找不到对应片段（或被剔除过）时跳过：不允许产生无出处的引用编号。
                if not src:
                    continue
                usage[src] = usage.get(src, 0) + 1
                if src not in first_seen:
                    first_seen[src] = order
                    order += 1
        # 第二轮：把"本轮召回到但没被任何结论引用"的资料也纳入编号范围。
        # 目的是让报告的引用清单能覆盖全部证据来源，避免出现"正文没引但资料确实用过"的空白。
        for item in evidence:
            src = str(item.get("source_id", ""))
            if src and src not in first_seen:
                first_seen[src] = order
                order += 1
                usage.setdefault(src, 0)

        # 主键是 (-被引次数, 首次出现顺序)：次数多的在前，同次数按出现顺序，结果确定可复现。
        used_source_ids: List[str] = sorted(first_seen, key=lambda s: (-usage.get(s, 0), first_seen[s]))

        citations: List[Dict[str, Any]] = []
        citation_index: Dict[str, int] = {}
        citation_failures: List[str] = []
        for source_id in used_source_ids:
            # 取该资料在证据里的第一个章节标题作为引用定位信息（用于把 [n] 指到具体章节）。
            section_keyword = next(
                (str(e.get("section_title")) for e in evidence if str(e.get("source_id")) == source_id),
                "",
            )
            result = self.call_tool(
                "cite_source",
                {"source_id": source_id, "section_keyword": section_keyword or None},
            )
            if not result.ok:
                citation_failures.append(f"{source_id}: {(result.error or {}).get('message', '')}")
                continue
            data = dict(result.data or {})
            citation_index[source_id] = int(data.get("citation_no"))
            citations.append(data)

        # ---- 2) 结论 -> 引用编号映射 ----
        # 这一步是"引用可追溯"的中间桥：finding.evidence_ids -> source_id -> citation_no。
        citation_map: Dict[str, List[int]] = {}
        for finding in findings:
            nos: List[int] = []
            for eid in finding.get("evidence_ids") or []:
                src = str((evidence_index.get(str(eid)) or {}).get("source_id", ""))
                no = citation_index.get(src)
                # 同一结论的多条证据可能指向同一资料，去重后编号保持升序（正文里 [1][3] 更好读）。
                if no is not None and no not in nos:
                    nos.append(no)
            if not nos and citations:
                # 兜底：结论没有任何可解析的引用来源时，挂上首个引用编号 [1]。
                # 这是**排版兜底**而非溯源声明——它保证正文不缺引用标记，
                # 而"该结论是否有证据"已由 RiskChecker 的 GAP-EVIDENCE 在闸门处把过关。
                nos = [citations[0]["citation_no"]]
            citation_map[str(finding.get("id"))] = sorted(nos)

        # ---- 3) LLM 组织语言 ----
        # 给模型的是已绑定的 citation_map 与 citations，所以它只能在既定编号体系里"用"编号，
        # 不能自己发明编号——编号的分配权始终在 Agent 手上（见 _assemble 里的补挂逻辑）。
        payload = {
            "question": state.get("question", ""),
            "company": company,
            "year": year,
            "period": period,
            "findings": findings,
            "citation_map": citation_map,
            "citations": citations,
            "risk": risk_report,
            "analysis_summary": state.get("analysis_summary", ""),
            # 人工决策一并交给模型，使它能在正文里体现"已人工确认"这一事实。
            "human_decision": state.get("human_decision"),
        }
        response = self.ctx.llm.chat("write", payload)
        data: Dict[str, Any] = dict(response.data or {})

        # ---- 4) 确定性拼装报告 ----
        # 无论模型返回什么，最终报告的骨架与引用格式都由 _assemble 决定。
        report = self._assemble(state, data, findings, citation_map, citations, company, year, period, risk_report)

        new_state["citations"] = citations
        new_state["citation_index"] = citation_index
        new_state["report"] = report
        new_state["llm_stats"] = {**(state.get("llm_stats") or {}), "writer": self._llm_stats_fragment(response)}
        # 引用分配失败必须显式暴露：报告会缺 [n]，人要能顺着 errors 找到是哪个 source_id 没编上号。
        self._merge_errors(new_state, "writer", citation_failures)
        return new_state

    # ------------------------------------------------------------------
    def _assemble(
        self,
        state: ResearchState,
        llm_data: Dict[str, Any],
        findings: List[Dict[str, Any]],
        citation_map: Dict[str, List[int]],
        citations: List[Dict[str, Any]],
        company: str,
        year: int,
        period: str,
        risk_report: Dict[str, Any],
    ) -> str:
        """确定性拼装 Markdown 简报全文。

        参数：
            state: 共享状态（取 question / run_id / config / facts / metrics / human_decision）；
                注意此处读的 facts / metrics 是**上游已写入 state 的值**，本方法不改 state。
            llm_data: LLM 返回的结构化正文（title / executive_summary / analysis_paragraphs）。
            findings: 已校验的结论列表。
            citation_map: finding id -> 引用编号列表。
            citations: cite_source 返回的引用条目（含 citation_no / source_id / section / path 等）。
            company / year / period: 报告抬头信息。
            risk_report: 风险明细（overall_level / narrative / findings / gaps）。

        返回：
            完整的 Markdown 字符串，章节顺序为：
            一、核心结论 → 二、关键财务指标 → 三、关键比率指标（仅当有 metrics）→
            分析与论证 → 风险提示 → 引用来源，末尾附免责声明。
            章号由 ``_SectionCounter`` 连续分配，因此"三、关键比率指标"缺失时后续章号不会错位。

        副作用 / 异常：
            无副作用（不发 LLM、不调工具、不改 state）。不抛异常；
            所有取值都带 ``or`` / ``.get`` 兜底，缺失字段退化为空串或占位符。

        关于引用编号的两处"补挂"（保持"正文必有引用标记"）：
            * 核心结论条目若自身不含 ``[n]``，补上**第一条结论**的引用编号（最多 2 个）；
            * 论证段落若不含 ``[n]``，补上该结论自己的引用编号（最多 2 个）。
            这两处只做标记补全，不改变编号分配结果，也不会伪造不存在的编号。
        """
        # 抬头：优先用 LLM 给的标题，缺失时按"公司 + 年份 + 报告期"确定性生成。
        title = str(llm_data.get("title") or f"{company} {year} 年{period}投研简报")
        # config 里的 mock_llm 决定抬头怎么写"生成方式"：是离线确定性大脑，还是具体模型名。
        mode = (state.get("config") or {}).get("mock_llm", True)
        mode_desc = "确定性 mock 大脑（离线模式）" if mode else str((state.get("config") or {}).get("model_name", "LLM"))
        risk_level = str(risk_report.get("overall_level") or state.get("risk_level") or "info")
        human_decision = state.get("human_decision") or {}

        lines: List[str] = []
        lines.append(f"# {title}")
        lines.append("")
        counter = _SectionCounter()
        lines.append(f"> **研究问题**：{state.get('question', '')}")
        lines.append(
            f"> **运行编号**：{state.get('run_id', '')} ｜ **生成方式**：{mode_desc} ｜ **整体风险等级**：{risk_level.upper()}"
        )
        # 合规声明：资料库为虚构演示数据，必须在报告显著位置声明，避免被当成真实投研结论使用。
        lines.append(
            "> **数据来源**：本地演示资料库（公司名称、财务数据与事件均为虚构，"
            "仅用于技术演示，不构成任何投资建议）"
        )
        # 人工确认信息只在真的发生过人机协同时才出现（escalate 后 approved）。
        if human_decision:
            lines.append(
                f"> **人工确认**：{human_decision.get('decision', '')}"
                f"（来源：{human_decision.get('source', '')}；说明：{human_decision.get('reason', '')}）"
            )
        lines.append("")

        # ---- 一、核心结论 ----
        lines.append(f"## {counter.next()}、核心结论")
        lines.append("")
        executive: List[str] = list(llm_data.get("executive_summary") or [])
        if not executive:
            # 兜底：模型没给核心结论段落时，直接用 findings 的 statement 拼，
            # 并给每条挂上它自己的引用编号——保证"有结论且可追溯"不依赖模型发挥。
            executive = [
                f"{f.get('statement', '')}{''.join(f'[{n}]' for n in citation_map.get(str(f.get('id')), []))}"
                for f in findings
            ]
        for idx, item in enumerate(executive, 1):
            text = str(item).strip()
            if not _has_citation(text):
                # 补挂策略用的是**第一条结论**的编号（而非该条自己的），
                # 因为核心结论段落与 finding 之间没有可靠的一一对应关系；
                # findings 为空时 fallback 为空列表，即不补挂（正文保持无标记）。
                fallback = citation_map.get(str(findings[0].get("id")) if findings else "", [])
                text = text + "".join(f"[{n}]" for n in fallback[:2])
            lines.append(f"{idx}. {text}")
        lines.append("")

        # ---- 二、关键财务指标 ----
        # 这一节来自 facts（工具取到的科目原值），是"原始数据"层；
        # 下一节才是有计算口径的比率层，二者分开是为了让读者能核对原值。
        lines.append(f"## {counter.next()}、关键财务指标")
        lines.append("")
        lines.append("| 指标 | 数值 | 单位 | 出处 |")
        lines.append("| --- | --- | --- | --- |")
        rows = self._fact_rows(state.get("facts") or {})
        if rows:
            for metric, value, unit, source_id in rows:
                lines.append(f"| {metric} | {_fmt_num(value)} | {unit} | {source_id} |")
        else:
            # 明确写"（无）"而不是留空表：让人一眼看出是"没取到数"，而非排版漏了。
            lines.append("| （无） | - | - | - |")
        lines.append("")

        # ---- 三、关键比率 ----
        # 注意整节都在 `if metrics:` 里，包括 counter.next() 的调用——
        # 这正是 _SectionCounter 存在的理由：可选章节不能占用章号。
        metrics: Dict[str, Any] = state.get("metrics") or {}
        if metrics:
            lines.append(f"## {counter.next()}、关键比率指标")
            lines.append("")
            lines.append("| 比率 | 数值 | 计算口径 | 出处 |")
            lines.append("| --- | --- | --- | --- |")
            for key, item in metrics.items():
                label = item.get("label") or key
                # display 缺省时回落到 _fmt_num(value)，保证既不空白也不输出 None。
                display = item.get("display") or _fmt_num(item.get("value"))
                lines.append(
                    f"| {label} | {display} | {item.get('formula', '')} | {item.get('source_id', '') or '-'} |"
                )
            lines.append("")

        # ---- 分析与论证 ----
        # 每条结论一个小节；cross_checks 作为要点列在段落下，展示"做过交叉验证"的证据。
        lines.append(f"## {counter.next()}、分析与论证")
        lines.append("")
        paragraphs: Dict[str, str] = dict(llm_data.get("analysis_paragraphs") or {})
        for finding in findings:
            fid = str(finding.get("id"))
            lines.append(f"### {fid} {finding.get('title', '')}")
            lines.append("")
            # 段落缺失时退回 conclusion 原文（statement），话术可以没有，结论不能丢。
            text = str(paragraphs.get(fid) or finding.get("statement") or "").strip()
            if not _has_citation(text):
                text += "".join(f"[{n}]" for n in citation_map.get(fid, [])[:2])
            lines.append(text)
            lines.append("")
            for extra in finding.get("cross_checks") or []:
                lines.append(f"- 交叉验证：{extra}")
            # 只在确实有交叉验证条目时补一个空行，避免出现连续空行。
            if finding.get("cross_checks"):
                lines.append("")

        # ---- 风险提示 ----
        # 原样引用 RiskChecker 的叙述与规则命中，Writer 不做二次判断，保证风险表述不被改写。
        lines.append(f"## {counter.next()}、风险提示")
        lines.append("")
        narrative = str(risk_report.get("narrative") or "").strip()
        if narrative:
            lines.append(narrative)
            lines.append("")
        risk_findings = risk_report.get("findings") or []
        if risk_findings:
            lines.append("| 等级 | 风险项 | 指标值 | 触发阈值 | 说明 | 出处 |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for item in risk_findings:
                lines.append(
                    f"| {str(item.get('level', '')).upper()} | {item.get('title', '')} | "
                    f"{item.get('metric_display', '')} | {item.get('threshold', '')} | "
                    f"{item.get('detail', '')} | {item.get('source_id', '') or '-'} |"
                )
            lines.append("")
        gaps = risk_report.get("gaps") or []
        if gaps:
            # 走到 Writer 却仍有缺口 = 只可能是"轮次用尽后人工放行"的情形，
            # 因此这一节明确标注"需人工确认"，把未消除的不确定性留在报告里而不是藏起来。
            lines.append("**未消除的数据缺口（需人工确认）**")
            lines.append("")
            for gap in gaps:
                lines.append(f"- `{gap.get('code', '')}` {gap.get('problem', '')} → 要求补正：{gap.get('required_fix', '')}")
            lines.append("")

        # ---- 引用来源 ----
        # 按 citation_no 升序输出，与正文 [n] 一一对应，且带 source_id 与文件路径，
        # 这是"引用可追溯"的最后一环。
        lines.append(f"## {counter.next()}、引用来源")
        lines.append("")
        if citations:
            for item in sorted(citations, key=lambda x: int(x.get("citation_no", 0))):
                lines.append(
                    f"[{item.get('citation_no')}] {item.get('company', '')} {item.get('period', '')} · "
                    f"{item.get('section', '')} —— `{item.get('source_id', '')}`"
                    f"（{item.get('doc_type', '')}，文件：{item.get('path', '')}）"
                )
        else:
            lines.append("（本次运行未产生引用）")
        lines.append("")

        lines.append("---")
        lines.append("")
        # 强制免责声明：产物面向人，必须明确"虚构数据 + 不构成投资建议"。
        lines.append(
            "> 免责声明：本简报由多智能体系统自动生成，资料库内容为虚构演示数据，"
            "不构成任何投资建议；请勿据此做出投资决策。"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def _fact_rows(facts: Dict[str, Any]) -> List[Tuple[str, Any, str, str]]:
        """把事实缓存整理成表格行（同一年度的同一指标只保留一条）。

        参数：
            facts: ``AnalystAgent`` 写回的科目原值缓存，键形如 ``"营业收入|2024|年度"``，
                值可能是科目字典，也可能是表示"取数失败"的 ``None``。

        返回：
            ``[(展示名, 数值, 单位, 出处), ...]`` 的列表，最多 24 行。
            展示名形如 ``"营业收入（年度）"``；出处缺失时用 ``"-"``。

        副作用 / 异常：无（纯函数，不修改 facts）。不抛异常。

        去重与顺序：
            * 去重键是 ``(metric, year, period)``——因为 Analyst 会为上一年度（同比对照）
              也取同科目，若不按年份去重，表格里会出现同一指标的多行；
            * 遍历用 ``sorted(facts.keys())`` 而非插入顺序，让同一份数据每次生成
              完全相同的表格（可复现，也便于逐字节比对报告）。
            * 截断到 24 行是排版约束：关键财务指标表只做"概览"，不追求列全。
        """
        seen = set()
        rows: List[Tuple[str, Any, str, str]] = []
        for key in sorted(facts.keys()):
            item = facts.get(key)
            # 跳过错缓存（None）与非字典值：负缓存项没有可展示的内容。
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric") or "")
            year = item.get("year")
            dedup = (metric, year, item.get("period"))
            if not metric or dedup in seen:
                continue
            seen.add(dedup)
            rows.append(
                (
                    # period 缺失时退回 year 展示，避免出现"营业收入（）"这种空括号。
                    f"{metric}（{item.get('period', year)}）",
                    item.get("value"),
                    str(item.get("unit") or ""),
                    str(item.get("source_id") or "-"),
                )
            )
        return rows[:24]

    # ------------------------------------------------------------------
    def _trace_input(self, state: ResearchState) -> Dict[str, Any]:
        """trace 输入摘要：记录待撰写的结论、证据与放行裁决。

        参数：state —— 进入本步时的状态。
        返回：``{"findings": [id...], "evidence": [child_id...], "risk_verdict": str}``。
        副作用 / 异常：无。
        """
        return {
            "findings": [f.get("id") for f in (state.get("findings") or [])],
            "evidence": [e.get("child_id") for e in (state.get("evidence") or [])],
            "risk_verdict": state.get("risk_verdict"),
        }

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        """trace 输出摘要：记录引用编号表与报告规模（不落全文）。

        参数：state —— 本步执行后的状态，读 ``citations`` 与 ``report``。
        返回：``{"citations": [{no, source_id, section}], "report_chars": int,
            "report_head": 前 400 字符}``。
            注：``report_head`` 截断是为了控制 trace 体积；完整报告在 state["report"] 与
            运行结果里，不依赖 trace 保存。
        副作用 / 异常：无。
        """
        return {
            "citations": [
                {"no": c.get("citation_no"), "source_id": c.get("source_id"), "section": c.get("section")}
                for c in (state.get("citations") or [])
            ],
            "report_chars": len(state.get("report") or ""),
            "report_head": (state.get("report") or "")[:400],
        }


def _has_citation(text: str) -> bool:
    """判断一段文本里是否已经包含 [n] 形式的引用编号。

    参数：
        text: 待检测文本；``None`` 会被 ``text or ""`` 归一为空串。

    返回：
        ``bool`` —— 命中 ``\\[\\d+\\]`` 即 True。用于判断"要不要给这段话补挂引用标记"。

    副作用 / 异常：无。
        注：``import re`` 写在函数体内（模块顶层没有导入 re），
        属于"用到才导入"的局部导入；功能上每次调用都会走一次已缓存的模块查找。
    """
    import re

    return bool(re.search(r"\[\d+\]", text or ""))


_CN_NUM = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]


class _SectionCounter:
    """章节序号计数器，保证章节编号连续且不会因为可选章节而错位。

    职责：
        ``_assemble`` 里"关键比率指标"整节是可选的（无 metrics 就整段不写）。
        如果章号写死，缺一节就会导致后面的"分析与论证/风险提示/引用来源"编号跳号；
        用计数器并在**真正输出的位置**才调用 ``next()``，编号自然连续。

    关键属性：
        ``_n``: 内部计数，从 0 开始，每次 ``next()`` 后自增。

    状态流转：
        一次报告拼装对应一个实例；实例不可重置，也没有"回退"能力。

    注：实际实现为 —— 只支持到十（``_CN_NUM`` 长度 10），
        第 11 章及以后会退化为阿拉伯数字字符串（如 "11"）。
    """

    def __init__(self) -> None:
        """初始化计数器。

        参数：无。
        返回：None。
        副作用：把内部计数 ``_n`` 置为 0。
        """
        self._n = 0

    def next(self) -> str:
        """取下一个章节序号并推进计数。

        参数：无。
        返回：中文序号字符串（"一" / "二" …）；超出 ``_CN_NUM`` 长度时返回
            ``str(self._n + 1)``，即阿拉伯数字。
        副作用：``self._n`` 自增 1（**每次调用都会消耗一个序号**，
            因此只能在确定要输出该章节时调用）。
        异常：无。
        """
        label = _CN_NUM[self._n] if self._n < len(_CN_NUM) else str(self._n + 1)
        self._n += 1
        return label
