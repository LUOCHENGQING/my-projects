"""AnalystAgent —— 财务指标计算与分析。

层次与职责：
    位于 agents 层的中段（图节点 ``analyst``）。上游是 Retriever 给的 ``evidence``，
    下游是 RiskChecker 的核查门与 Writer 的报告拼装；在反思循环里它是**唯一会被打回重算**
    的节点（条件边 risk_checker --revise--> analyst）。每次重算都会读到新的
    ``revision_round`` 与 ``revision_requests``，从而补充交叉验证所需的口径。

职责边界（本项目最重要的一条设计原则）：
    **数字由工具产生，语言由模型产生。**

    * Agent 自己决定「算哪些指标、用哪两个科目、用什么口径」——这是领域策略，
      必须确定、可审计、可复现，不能交给模型即兴发挥。
    * 所有数值一律经由 `get_financial_metric`（结构化事实库）与 `calc_ratio`（比率计算器）
      产生；模型只负责解释这些数字之间的关系，并给出结论措辞。
    * 结论里出现的每一个 `ratio_refs` / `evidence_ids` 都会被二次校验，
      不在清单内的（= 幻觉出来的）一律剔除，并记入 errors。

它只持有 `get_financial_metric` + `calc_ratio`，拿不到 write 权限，
从权限层面杜绝"分析师顺手改数据/写文件"。

互斥点（与其它四个 Agent）：
    * vs PlannerAgent：Planner 定"算哪些维度"（targets），Analyst 定"用什么科目算"，
      Analyst 从不改 route。
    * vs RetrieverAgent：Analyst **只消费不召回**——它绝不自己去检索原文，
      引用证据只能从 ``state["evidence"]`` 里挑（挑不到就等着被核查门打回）。
    * vs RiskCheckerAgent（职责对立）：Analyst 是"立论方"，目标是产出有支撑的结论并倾向
      给出解释；RiskChecker 是"质疑方"，目标是找出证据缺口、量化缺失与异常未交叉验证。
      二者的立场刻意相反，且 Analyst **无权**决定自己是否通过——裁决权在 RiskChecker。
      被打回时 Analyst 只是"照单补算"，不能反过来判定自己的结论已合格。
    * vs WriterAgent：Analyst 只产出 findings / metrics，不生成引用编号、不写报告正文。

对外关键类 / 函数：
    * ``AnalystAgent``（``name = "analyst"``）—— 唯一对外类；
    * 模块级计划常量 ``RATIO_PLAN`` / ``RISK_BASELINE_PLAN`` / ``REVISION_PLAN`` /
      ``PRIOR_COMPARATIVES`` —— 把"算什么"从代码逻辑里抽出来，便于审阅与扩展；
    * 模块级工具函数 ``_value_of`` / ``_direct_label``。

主要输入输出：
    输入：``state["plan"]``（companies / year / period / targets）、``state["evidence"]``、
    ``state["revision_round"]``、``state["revision_requests"]``。
    输出（写回新 state）：``facts``（原始指标缓存）、``metrics``（算出的比率）、
    ``findings``（经校验的结论）、``analysis_summary``、``llm_stats["analyst"]``，
    并把被剔除的幻觉引用与缺失指标写入 ``errors``。

被谁调用：
    * ``src/orchestrator.py`` 注册为节点 ``"analyst"``，并被 risk_checker 的 revise 分支回指；
    * ``tests/test_citation_traceability.py`` 直接实例化做单步测试。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..state import ResearchState
from ..tools.registry import PermissionLevel
from .base import BaseAgent

#: 分析维度 -> 指标计算计划。
#: kind=ratio  : 计算 分子科目 / 分母科目
#: kind=growth : 计算 (本期 - 上期) / |上期|
#: kind=direct : 直接取披露值（监管指标等）
#:
#: 字段约定（数值只由工具产生，所以这里只声明"口径"，不写死任何数字）：
#:     key          本维度内的指标唯一键，也是 findings.ratio_refs 的合法取值之一（同时用于去重）
#:     ratio_name   传给 calc_ratio 的比率类型名（growth 类统一用 "growth_rate"）
#:     numerator / denominator  ratio 类的分子 / 分母科目名，必须与事实库中的科目名一致
#:     metric       growth / direct 类要取的候选科目名
#:     label        展示名（growth 类会一并传给 calc_ratio 作为标签）
#:     pct          direct 类专用：披露值本身是百分数（如"不良贷款率"），需除以 100 归一到小数
#: 注意：维度名必须与 Planner 的默认 targets（"盈利能力" / "偿债能力" / "风险合规"）对齐，
#: 否则 RATIO_PLAN.get(target) 取不到计划条目，会直接落到基线兜底分支。
RATIO_PLAN: Dict[str, List[Dict[str, Any]]] = {
    "盈利能力": [
        {"key": "net_margin", "kind": "ratio", "ratio_name": "net_margin",
         "numerator": "净利润", "denominator": "营业收入"},
        {"key": "gross_margin", "kind": "ratio", "ratio_name": "gross_margin",
         "numerator": "毛利润", "denominator": "营业收入"},
        {"key": "roe", "kind": "ratio", "ratio_name": "roe",
         "numerator": "净利润", "denominator": "所有者权益"},
        {"key": "rnd_intensity", "kind": "ratio", "ratio_name": "rnd_intensity",
         "numerator": "研发费用", "denominator": "营业收入"},
    ],
    "偿债能力": [
        {"key": "debt_to_asset", "kind": "ratio", "ratio_name": "debt_to_asset",
         "numerator": "负债总额", "denominator": "资产总额"},
        {"key": "current_ratio", "kind": "ratio", "ratio_name": "current_ratio",
         "numerator": "流动资产", "denominator": "流动负债"},
    ],
    "成长性": [
        {"key": "revenue_growth", "kind": "growth", "ratio_name": "growth_rate",
         "metric": "营业收入", "label": "营业收入同比增速"},
        {"key": "profit_growth", "kind": "growth", "ratio_name": "growth_rate",
         "metric": "净利润", "label": "净利润同比增速"},
    ],
    "现金流质量": [
        {"key": "cash_conversion", "kind": "ratio", "ratio_name": "cash_conversion",
         "numerator": "经营活动现金流净额", "denominator": "净利润"},
    ],
    "营运效率": [
        {"key": "asset_turnover", "kind": "ratio", "ratio_name": "asset_turnover",
         "numerator": "营业收入", "denominator": "资产总额"},
    ],
    "银行监管指标": [
        # 银行口径下净利率仍按"净利润 / 营业收入"计算；其余三项为监管直接披露值（pct=True）。
        {"key": "net_margin", "kind": "ratio", "ratio_name": "net_margin",
         "numerator": "净利润", "denominator": "营业收入"},
        {"key": "npl_ratio", "kind": "direct", "ratio_name": "npl_ratio",
         "metric": "不良贷款率", "pct": True},
        {"key": "provision_coverage", "kind": "direct", "ratio_name": "provision_coverage",
         "metric": "拨备覆盖率", "pct": True},
        {"key": "capital_adequacy", "kind": "direct", "ratio_name": "capital_adequacy",
         "metric": "资本充足率", "pct": True},
    ],
    # 这两个维度本身不产出比率：留空键是有意为之，表示"该维度无专属口径"；
    # 风险合规的量化支撑交由下面的 RISK_BASELINE_PLAN 兜底。
    "股东回报": [],
    "风险合规": [],
}

#: 当问题只涉及风险合规、没有任何会产出比率的维度时使用的**基线指标集**。
#: 理由：如果只回答"有诉讼、有担保"而没有任何杠杆与现金流口径，
#: 核查门会判定「结论缺乏量化支撑」并把人卡在反思循环里。
#: 注意它是"兜底"而不是"叠加"：当其它维度已经算过这些口径时不会重复计算，
#: 避免同一指标在多个结论里被反复陈述。
#: 注：实际实现为 —— 这里的 ``"target": "风险合规"`` 只决定兜底指标归属到哪个维度
#: （写进 ``metrics[key]["target"]``），真正防重复靠 ``_execute`` 里按 ``key`` 的 ``seen_keys`` 去重。
RISK_BASELINE_PLAN: List[Dict[str, Any]] = [
    {"key": "debt_to_asset", "kind": "ratio", "ratio_name": "debt_to_asset",
     "numerator": "负债总额", "denominator": "资产总额", "target": "风险合规"},
    {"key": "cash_conversion", "kind": "ratio", "ratio_name": "cash_conversion",
     "numerator": "经营活动现金流净额", "denominator": "净利润", "target": "风险合规"},
    {"key": "revenue_growth", "kind": "growth", "ratio_name": "growth_rate",
     "metric": "营业收入", "label": "营业收入同比增速", "target": "风险合规"},
]

#: 收到打回意见后额外补充的计算（用于交叉验证）。
#: supplementary=True 表示这些口径是"验证材料"，不进入结论陈述本身，
#: 只在交叉验证与比率表里出现，避免把"应收增速 35%"说成"表现良好"这类误读。
REVISION_PLAN: List[Dict[str, Any]] = [
    {"key": "receivable_growth", "kind": "growth", "ratio_name": "growth_rate",
     "metric": "应收账款", "label": "应收账款同比增速", "target": "现金流质量",
     "supplementary": True},
    {"key": "total_asset_growth", "kind": "growth", "ratio_name": "growth_rate",
     "metric": "资产总额", "label": "资产总额同比增速", "target": "偿债能力",
     "supplementary": True},
]

#: 交叉验证需要的历史口径（上一年度同一比率）
#: 注：实际实现为 —— 只有 ``revision_round > 0``（即被 RiskChecker 打回过）才会去算这些上期值，
#: 结果放进 ``comparatives`` 作为对照材料，**不写进 metrics**，因此不会出现在比率表与结论引用中。
PRIOR_COMPARATIVES: List[Dict[str, Any]] = [
    {"key": "cash_conversion_prior", "ratio_name": "cash_conversion",
     "numerator": "经营活动现金流净额", "denominator": "净利润"},
    {"key": "debt_to_asset_prior", "ratio_name": "debt_to_asset",
     "numerator": "负债总额", "denominator": "资产总额"},
    {"key": "net_margin_prior", "ratio_name": "net_margin",
     "numerator": "净利润", "denominator": "营业收入"},
]


class AnalystAgent(BaseAgent):
    """财务指标计算与分析 Agent：把"证据 + 计划"变成"有量化支撑的结论"。

    职责：
        1. 依 targets 展开指标计算计划（RATIO_PLAN，必要时用 RISK_BASELINE_PLAN 兜底）；
        2. 通过 ``get_financial_metric`` 取科目原值、``calc_ratio`` 算比率——**数字全部出自工具**；
        3. 被打回时（revision_round > 0）补算交叉验证口径与上期对照；
        4. 请 LLM 基于给定数字组织结论措辞；
        5. 用真实指标键 / 证据 ID 对模型给出的引用做交集校验，剔除幻觉引用。

    关键属性（类属性）：
        name = "analyst" —— 写进 trace 与 steps。
        role = "财务指标计算与分析"。
        allowed_tools = ("get_financial_metric", "calc_ratio") —— 只有取数与计算两类工具。
        permissions = {PUBLIC_READ, COMPUTE} —— **没有 WRITE**，结构上无法落盘或分配引用编号。

    状态流转：
        入口态：plan（companies / year / period / targets）与 evidence 已就绪，
            revision_round 与 revision_requests 可能非零（被打回后的重算）。
        出口态：facts（事实缓存，供 Writer 出表格）、metrics（比率字典，RiskChecker 的
            异常判定与 Writer 的比率表都以此为准）、findings（结论列表，每条挂
            ratio_refs + evidence_ids + cross_checks）、analysis_summary。
        循环关系：本 Agent 与 RiskCheckerAgent 构成唯一的反思环
            （analyst -> risk_checker -> revise -> analyst），最多 max_revision_rounds 轮。
    """

    name = "analyst"
    role = "财务指标计算与分析"
    allowed_tools = ("get_financial_metric", "calc_ratio")
    permissions = frozenset({PermissionLevel.PUBLIC_READ, PermissionLevel.COMPUTE})

    def _execute(self, state: ResearchState) -> ResearchState:
        """按计划算指标 -> 打回时补交叉验证口径 -> LLM 出结论 -> 引用校验。

        参数：
            state: 上游共享状态。读取 ``plan``（companies[0] / year / period / targets）、
                ``evidence``、``revision_round``、``revision_requests``、``llm_stats``。

        返回：
            新 state，写入 ``facts`` / ``metrics`` / ``findings`` / ``analysis_summary`` /
            ``llm_stats["analyst"]``；并通过 ``_merge_errors`` 把「被剔除的幻觉引用」与
            「缺失指标（最多前 5 条）」追加到 ``errors``。

        副作用 / 异常：
            * 通过工具取数与计算（只读 + 纯计算，无落盘副作用）；
            * 调用一次 LLM（``task="analyze"``）；
            * 本方法不主动抛异常；单项指标算不出来只记入 ``missing``，不影响其它指标，
              这正是"部分数据缺失也能出部分结论"的降级设计。
        """
        new_state = self._copy(state)
        plan = state.get("plan") or {}
        # 与 Retriever 一致：单次运行只分析第一家公司；年份转 int（取不到时退化为 0，
        # 后续工具调用会因查不到数据而走 missing 分支，而不是抛类型错误）。
        company = (plan.get("companies") or [""])[0]
        year = int(plan.get("year") or 0)
        period = str(plan.get("period") or "年度")
        targets: List[str] = list(plan.get("targets") or [])
        revision_round = int(state.get("revision_round") or 0)
        revision_requests: List[str] = list(state.get("revision_requests") or [])

        # facts 是"科目原值"的本地缓存（键为 metric|year|period），metrics 是"算出来的比率"。
        # 两者都挂到新 state 上：facts 供 Writer 出关键财务指标表，metrics 供核查门与比率表。
        facts: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}
        missing: List[str] = []

        # ---- 1) 按维度执行指标计算计划 ----
        # "算什么"由 targets 决定，不由模型决定；这里只做计划的展开与去重。
        entries: List[Dict[str, Any]] = []
        for target in targets:
            for entry in RATIO_PLAN.get(target, []):
                item = dict(entry)
                # 把维度名回填进计划条目：Analyst 的 metrics 与 Writer 的比率表都要用到 target。
                item.setdefault("target", target)
                entries.append(item)

        # 纯风险类问题没有任何比率口径 -> 用基线指标兜底，保证结论有量化支撑
        # （否则 RiskChecker 会以 GAP-METRIC 反复打回，形成空转）。
        used_baseline = False
        if not entries:
            used_baseline = True
            entries = [dict(item) for item in RISK_BASELINE_PLAN]

        # 按 key 去重：多个维度可能算同一口径（例如"盈利能力的 net_margin"与"银行监管指标的
        # net_margin"），只算一次，避免同一指标在结论里被反复陈述。
        seen_keys = set()
        for entry in entries:
            if entry["key"] in seen_keys:
                continue
            seen_keys.add(entry["key"])
            self._compute_entry(entry, company, year, period, facts, metrics, missing)

        # ---- 2) 被打回时：补充交叉验证所需的口径 ----
        # 这一段是"反思循环真的在起作用"的证据：不是把同样的活重算一遍，
        # 而是补上第一轮没算的对照口径（应收增速、总资产增速、上期比率）。
        comparatives: Dict[str, Any] = {}
        if revision_round > 0:
            for entry in REVISION_PLAN:
                if entry["key"] in metrics:
                    continue
                item = dict(entry)
                self._compute_entry(item, company, year, period, facts, metrics, missing)

            for entry in PRIOR_COMPARATIVES:
                value = self._compute_ratio_for_year(
                    company, year - 1, period, str(entry["ratio_name"]),
                    str(entry["numerator"]), str(entry["denominator"]),
                )
                if value is not None:
                    comparatives[entry["key"]] = value

            # 收入质量交叉验证的两个关键增速：应收增速 vs 收入增速。
            # 它们已在上面的 REVISION_PLAN 中算进 metrics，这里只是把数值摘出来放进 comparatives。
            comparatives["revenue_growth"] = _value_of(metrics.get("revenue_growth"))
            comparatives["receivable_growth"] = _value_of(metrics.get("receivable_growth"))

        # ---- 3) LLM 生成结论（数字来自工具，语言来自模型） ----
        # payload 里的 ratios / comparatives / missing 都是"已由工具确定的事实"，
        # 模型只能解释它们；它无法通过 prompt 改变任何数值。
        payload = {
            "company": company,
            "year": year,
            "period": period,
            "targets": targets,
            "revision_round": revision_round,
            "revision_requests": revision_requests,
            "used_risk_baseline": used_baseline,
            "ratios": [
                {
                    "key": item.get("key"),
                    "ratio_name": item.get("ratio_name"),
                    "label": item.get("label"),
                    "target": item.get("target"),
                    "value": item.get("value"),
                    "value_pct": item.get("value_pct"),
                    "display": item.get("display"),
                    "formula": item.get("formula"),
                    "benchmark": item.get("benchmark"),
                    "source_id": item.get("source_id"),
                    # supplementary=True 的口径只是验证材料，提示模型不要拿它当业绩陈述
                    # （例如把"应收增速 35%"说成"表现良好"）。
                    "supplementary": bool(item.get("supplementary")),
                }
                for item in metrics.values()
            ],
            "evidence": [
                {
                    "child_id": e.get("child_id"),
                    "source_id": e.get("source_id"),
                    "section_title": e.get("section_title"),
                    "text": e.get("text"),
                }
                for e in state.get("evidence", [])
            ],
            "comparatives": comparatives,
            "missing": missing,
        }
        response = self.ctx.llm.chat("analyze", payload)
        data: Dict[str, Any] = dict(response.data or {})

        # ---- 4) 反幻觉校验：结论引用必须真实存在 ----
        # 合法引用的取值域 = 本次真正算出来的指标键 / 比率名 + 本次真正召回的证据 ID。
        valid_ratios = set(metrics.keys())
        ratio_names = {str(m.get("ratio_name")) for m in metrics.values()}
        valid_evidence = {str(e.get("child_id")) for e in state.get("evidence", [])}
        findings, rejected = self._validate_findings(data.get("findings") or [], valid_ratios, ratio_names, valid_evidence)

        new_state["facts"] = facts
        new_state["metrics"] = metrics
        new_state["findings"] = findings
        new_state["analysis_summary"] = str(data.get("summary") or "")
        new_state["llm_stats"] = {**(state.get("llm_stats") or {}), "analyst": self._llm_stats_fragment(response)}
        # 把幻觉引用与缺失指标显式化：既不静默吞掉（可审计），也不让整步失败（可降级）。
        # missing 只取前 5 条，避免错误列表被几十个缺科目刷屏。
        self._merge_errors(new_state, "analyst", rejected + [f"缺失指标: {m}" for m in missing[:5]])
        return new_state

    # ------------------------------------------------------------------
    def _compute_entry(
        self,
        entry: Dict[str, Any],
        company: str,
        year: int,
        period: str,
        facts: Dict[str, Any],
        metrics: Dict[str, Any],
        missing: List[str],
    ) -> None:
        """执行一条指标计算计划。

        参数：
            entry: 计划条目（见 RATIO_PLAN 的字段约定），按 ``kind`` 分派 ratio / growth / direct。
            company: 公司名（工具查询的元数据约束）。
            year: 报告年份；growth 类会自动向前取 ``year - 1`` 作上期。
            period: 报告期（"年度" / "半年度" / "一季度" / "三季度"）。
            facts: **就地修改**的科目原值缓存（键 ``metric|year|period``）。
            metrics: **就地修改**的结果字典，成功时写入 ``metrics[entry["key"]]``。
            missing: **就地修改**的缺失说明列表（唯一的失败出口）。

        返回：
            None。成功与失败都通过上面两个可变容器体现，不返回状态。

        副作用 / 异常：
            * 有副作用：修改 facts / metrics / missing 三个入参；
            * 每成功一项至少两次工具调用（取分子分母各一次 + 一次 calc_ratio）。
          本方法不抛异常；工具失败与科目缺失统一走 ``missing``，从而让"部分指标算不出"降级为
          可观测的缺口，而不是打断整步分析。
        """
        target = str(entry.get("target") or "")
        key = str(entry["key"])
        kind = str(entry["kind"])

        if kind == "ratio":
            # 分子分母任一缺失就放弃该比率：绝不"用 0 兜底"，
            # 因为把缺数据当成 0 会直接算出一个错误的财务比率（比没有指标更糟）。
            num = self._fetch(company, str(entry["numerator"]), year, facts, period)
            den = self._fetch(company, str(entry["denominator"]), year, facts, period)
            if num is None or den is None:
                missing.append(f"{key}（缺少 {entry['numerator']} 或 {entry['denominator']}）")
                return
            # 数值计算一律交给 calc_ratio 工具（precision=4），Agent 不自己除，
            # 保证口径、精度与"可审计的计算来源"都在工具层统一。
            result = self.call_tool(
                "calc_ratio",
                {
                    "ratio_name": str(entry["ratio_name"]),
                    "numerator": float(num["value"]),
                    "denominator": float(den["value"]),
                    "precision": 4,
                },
            )
            if not result.ok:
                missing.append(f"{key}（计算失败：{(result.error or {}).get('message', '')}）")
                return
            item = dict(result.data)
            # inputs_detail / unit_of_inputs 是为了让"这个比率由哪两个数、什么单位算出"可回溯，
            # source_id 取分子的出处（比率的可追溯性由此传递到报告）。
            item.update({"key": key, "target": target, "source_id": num.get("source_id"),
                         "inputs_detail": {entry["numerator"]: num["value"], entry["denominator"]: den["value"]},
                         "unit_of_inputs": num.get("unit", "")})
            metrics[key] = item

        elif kind == "growth":
            # 同比增速 = (本期 - 上期) / |上期|，同样由 calc_ratio 计算（工具内部按 growth_rate 口径）。
            cur = self._fetch(company, str(entry["metric"]), year, facts, period)
            prev = self._fetch(company, str(entry["metric"]), year - 1, facts, period)
            if cur is None or prev is None:
                missing.append(f"{key}（缺少 {entry['metric']} 的上期数据）")
                return
            result = self.call_tool(
                "calc_ratio",
                {
                    # 注：这里写死 "growth_rate"（而非 entry["ratio_name"]）；两者取值一致，
                    # 但真正生效的是这个字面量。
                    "ratio_name": "growth_rate",
                    "numerator": float(cur["value"]),
                    "denominator": float(prev["value"]),
                    "precision": 4,
                    "label": str(entry.get("label") or f"{entry['metric']}同比增速"),
                },
            )
            if not result.ok:
                missing.append(f"{key}（计算失败：{(result.error or {}).get('message', '')}）")
                return
            item = dict(result.data)
            # target 缺省归到"成长性"；supplementary 标记"仅用于交叉验证，不作业绩陈述"。
            item.update({"key": key, "target": target or "成长性", "source_id": cur.get("source_id"),
                         "supplementary": bool(entry.get("supplementary"))})
            metrics[key] = item

        elif kind == "direct":
            # 监管披露类指标：原文给的就是最终值，不参与四则运算，只做单位归一。
            fact = self._fetch(company, str(entry["metric"]), year, facts, period)
            if fact is None:
                missing.append(f"{key}（未披露 {entry['metric']}）")
                return
            raw = float(fact["value"])
            # pct=True 表示披露值本身是百分数（如"1.25"表示 1.25%），统一归一成小数口径，
            # 与 ratio 类指标（0.0125）保持同一量纲，避免下游比较时出现 100 倍误差。
            value = raw / 100.0 if entry.get("pct") else raw
            metrics[key] = {
                "key": key,
                "target": target,
                "ratio_name": str(entry["ratio_name"]),
                "label": _direct_label(str(entry["metric"])),
                "formula": "直接取披露值",
                "value": value,
                # value_pct 还原成百分数（归一化前的量纲），供展示与阈值比较；非 pct 指标为 None。
                "value_pct": round(value * 100.0, 4) if entry.get("pct") else None,
                # display 用未归一的 raw 直接拼百分号，因此与 value_pct 保持同量纲。
                "display": f"{raw:.2f}%" if entry.get("pct") else f"{raw:.4f}",
                "source_id": fact.get("source_id"),
                "benchmark": "",
            }
        # 其它 kind 一律忽略：不做任何事，也不会记 missing（保持对未知计划条目的静默兼容）。

    def _fetch(
        self,
        company: str,
        metric: str,
        year: int,
        facts: Dict[str, Any],
        period: str = "年度",
    ) -> Optional[Dict[str, Any]]:
        """取结构化指标（带本地缓存，避免同一指标重复调用工具）。

        参数：
            company: 公司名。
            metric: 科目名（如"营业收入"），须与事实库中的写法一致。
            year: 年份。
            facts: **就地修改**的缓存字典；键为 ``f"{metric}|{year}|{period}"``。
            period: 报告期，默认 "年度"。

        返回：
            工具返回的科目字典（含 value / unit / source_id 等）；未收录或工具失败时为 ``None``。

        副作用 / 异常：
            * 有副作用：把结果（含 ``None``）写进 facts 缓存。
            * **负缓存**：失败也会写入 ``None``，因此同一 (metric, year, period) 在一次
              _execute 内不会重试，避免对确定查不到的科目反复打工具。
              注：这也意味着工具一旦失败就无法在本步内恢复（对确定性事实库而言是合理取舍）。
            * 不抛异常；工具失败统一降级为 ``None``。
        """
        cache_key = f"{metric}|{year}|{period}"
        if cache_key in facts:
            # `or None` 把可能被存成空字典的失败结果也归一为 None，调用方只判断 None 即可。
            return facts[cache_key] or None
        result = self.call_tool(
            "get_financial_metric",
            {"company": company, "metric": metric, "year": year, "period": period},
        )
        if not result.ok:
            facts[cache_key] = None
            return None
        facts[cache_key] = result.data
        return result.data

    def _compute_ratio_for_year(
        self,
        company: str,
        year: int,
        period: str,
        ratio_name: str,
        numerator_metric: str,
        denominator_metric: str,
    ) -> Optional[float]:
        """计算上一年度的同一比率，用于同比对照。

        参数：
            company: 公司名。
            year: 目标年份（调用方传的是 ``year - 1``，即上一年度）。
            period: 报告期（与本期保持一致，避免"年度 vs 半年度"混比）。
            ratio_name: 传给 calc_ratio 的比率类型名。
            numerator_metric / denominator_metric: 分子 / 分母科目名。

        返回：
            比率数值（``calc_ratio`` 的 ``value``）；分子分母缺失、分母为 0 或计算失败时返回 ``None``。

        副作用 / 异常：
            * 每次调用都传**新的空字典**作为 facts 缓存，因此历史年份取数不会被
              ``_fetch`` 的负缓存影响，也不会污染本期 facts（本期 facts 要用于出表格，
              混入上期数据会让 Writer 的指标表出现重复/错年份的行）。
            * 不做除零检查以外的保护：分母为 0 时**提前返回 None**，不交给工具处理。
            * 不抛异常。
        """
        num = self._fetch(company, numerator_metric, year, {}, period)
        den = self._fetch(company, denominator_metric, year, {}, period)
        # 分母为 0 直接放弃：与 calc_ratio 的除零行为解耦，也让"上期为 0"这种无法做同比的情形
        # 表现为"没有对照值"而不是一个错误值。
        if num is None or den is None or float(den["value"]) == 0:
            return None
        result = self.call_tool(
            "calc_ratio",
            {
                "ratio_name": ratio_name,
                "numerator": float(num["value"]),
                "denominator": float(den["value"]),
                # 对照值用更高精度（6 位），避免"本期 4 位 vs 上期 4 位"的舍入差被误读为趋势。
                "precision": 6,
            },
        )
        return float(result.data["value"]) if result.ok else None

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_findings(
        raw_findings: List[Dict[str, Any]],
        valid_ratio_keys: set,
        valid_ratio_names: set,
        valid_evidence: set,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """剔除引用不存在指标 / 证据的结论（防幻觉）。

        参数：
            raw_findings: LLM 返回的结论原始列表（可能含非 dict 元素）。
            valid_ratio_keys: 本次真正算出的指标键集合（metrics 的 key）。
            valid_ratio_names: 本次真正算出的比率名集合（metrics 里的 ratio_name）。
            valid_evidence: 本次真正召回到的证据 child_id 集合。

        返回：
            ``(findings, rejected)`` 二元组：
                * findings —— 清洗后的结论列表，每条含 id / title / target / statement /
                  ratio_refs / evidence_ids / direction / cross_checks；
                * rejected —— 人类可读的剔除说明列表，由调用方写入 errors 以便审计。

        副作用 / 异常：
            无副作用（纯函数，不修改入参）。不抛异常。

        校验策略（只剔引用、不废结论）：
            幻觉引用是"措辞里的瑕疵"，不是"结论不成立"，因此这里只把非法引用从
            ``ratio_refs`` / ``evidence_ids`` 中摘掉并记 rejected，仍保留该条结论——
            但如果结论因此变成"无量化支撑/无证据"，RiskChecker 的核查门会以
            GAP-METRIC / GAP-EVIDENCE 打回，形成两级防线。
            另注：``statement`` 为空的条目会被**静默丢弃且不计入 rejected**
            （第 384-386 行），这是既有实现；空结论无信息量，丢弃不影响可审计性。
        """
        findings: List[Dict[str, Any]] = []
        rejected: List[str] = []
        for idx, raw in enumerate(raw_findings, 1):
            # 非字典元素（模型偶发返回字符串）直接跳过，避免后续 .get 崩溃。
            if not isinstance(raw, dict):
                continue
            refs = [str(r) for r in (raw.get("ratio_refs") or [])]
            # 指标引用同时接受"指标键"与"比率名"两种写法，容错但不放宽到全集。
            good_refs = [r for r in refs if r in valid_ratio_keys or r in valid_ratio_names]
            bad_refs = [r for r in refs if r not in good_refs]

            evs = [str(e) for e in (raw.get("evidence_ids") or [])]
            good_evs = [e for e in evs if e in valid_evidence]
            bad_evs = [e for e in evs if e not in good_evs]

            # 剔除动作必须留痕：告诉人"模型编了什么"，而不是悄悄改掉。
            if bad_refs:
                rejected.append(f"结论 {raw.get('id', idx)} 引用了不存在的指标 {bad_refs}，已剔除这些引用")
            if bad_evs:
                rejected.append(f"结论 {raw.get('id', idx)} 引用了不存在的证据 {bad_evs}，已剔除这些引用")

            statement = str(raw.get("statement") or "").strip()
            if not statement:
                continue

            findings.append(
                {
                    # id 缺失时按序号生成 F1/F2…，保证 findings 一定能被 Writer 的
                    # citation_map 与结论小节标题（### F1 xxx）索引到。
                    "id": str(raw.get("id") or f"F{idx}"),
                    "title": str(raw.get("title") or f"结论{idx}"),
                    "target": str(raw.get("target") or ""),
                    "statement": statement,
                    "ratio_refs": good_refs,
                    "evidence_ids": good_evs,
                    # direction 是结论方向（improving/worsening/stable 之类），
                    # 缺省 "stable"，属于措辞元数据而非数值判断，故不参与防幻觉校验。
                    "direction": str(raw.get("direction") or "stable"),
                    "cross_checks": [str(c) for c in (raw.get("cross_checks") or [])],
                }
            )
        return findings, rejected

    # ------------------------------------------------------------------
    def _trace_input(self, state: ResearchState) -> Dict[str, Any]:
        """trace 输入摘要：记录分析对象、维度与打回上下文。

        参数：state —— 进入本步时的状态，读 ``plan`` / ``revision_round`` / ``revision_requests``。
        返回：``{"company": str, "year": int|None, "targets": [...],
            "revision_round": int, "revision_requests": [...]}``。
        副作用 / 异常：无。
        """
        plan = state.get("plan") or {}
        return {
            "company": (plan.get("companies") or [""])[0],
            "year": plan.get("year"),
            "targets": plan.get("targets", []),
            "revision_round": state.get("revision_round", 0),
            "revision_requests": state.get("revision_requests", []),
        }

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        """trace 输出摘要：记录算出哪些指标、结论引用了什么。

        参数：state —— 本步执行后的状态，读 ``metrics`` / ``findings`` / ``analysis_summary``。
        返回：``{"metrics": {key: {label, display, value}}, "findings": [{id, title,
            ratio_refs, evidence_ids, cross_checks(条数)}], "summary": str}``。
            注：``cross_checks`` 在这里只记**条数**（trace 控体积），而 state 里存的是全文列表。
        副作用 / 异常：无。
        """
        return {
            "metrics": {
                k: {"label": v.get("label"), "display": v.get("display"), "value": v.get("value")}
                for k, v in (state.get("metrics") or {}).items()
            },
            "findings": [
                {
                    "id": f.get("id"),
                    "title": f.get("title"),
                    "ratio_refs": f.get("ratio_refs"),
                    "evidence_ids": f.get("evidence_ids"),
                    "cross_checks": len(f.get("cross_checks") or []),
                }
                for f in (state.get("findings") or [])
            ],
            "summary": state.get("analysis_summary", ""),
        }


def _value_of(item: Optional[Dict[str, Any]]) -> Optional[float]:
    """从一个指标字典里安全取出数值。

    参数：
        item: 指标字典（如 ``metrics["revenue_growth"]``），可为 ``None``。

    返回：
        ``float`` 数值；``item`` 为空或 ``value`` 不是 int/float 时返回 ``None``。
        注意 bool 也是 int 的子类，理论上会被转成 0.0/1.0——实际数据里不会出现布尔型指标值。

    用途：把 metrics 里的值摘进 ``comparatives``（交叉验证对照材料）。
    副作用 / 异常：无。
    """
    if not item:
        return None
    value = item.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def _direct_label(metric: str) -> str:
    """给直接披露类指标取展示名。

    参数：
        metric: 科目名（如"不良贷款率"）。

    返回：
        映射表里对应的展示名；表里没有的科目原样返回``metric``。
        注：实际实现为 —— 该映射目前四个键的「键」与「值」完全相同，
        因此当前行为等价于恒等函数；保留映射是为了将来需要改口径/改叫法时只动一处。

    副作用 / 异常：无。
    """
    return {
        "不良贷款率": "不良贷款率",
        "拨备覆盖率": "拨备覆盖率",
        "资本充足率": "资本充足率",
        "净息差": "净息差",
    }.get(metric, metric)
