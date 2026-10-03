"""内置业务工具（5 个）。

    search_filings        多路资料检索（BM25 + 向量 + 元数据重排），返回带出处的片段
    get_financial_metric  取结构化财务指标（数值 + 单位 + 出处）
    calc_ratio            财务比率 / 增长率计算（纯函数）
    check_risk_rules      风险与合规规则引擎（阈值 + 派生指标）
    cite_source           把 source_id 解析成规范引用（产生引用编号，有副作用）

每个工具都声明了 JSON Schema。schema 不只是"文档"，它同时是：
    1. 调用前的强校验（挡住 LLM 幻觉参数）；
    2. 喂给 LLM 的 function-calling 描述（registry.describe() 导出）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from .registry import PermissionLevel, ToolRegistry, ToolSpec

__all__ = ["register_all", "RATIO_DEFS", "RISK_RULES"]

# ---------------------------------------------------------------------------
# 比率定义表
# ---------------------------------------------------------------------------
RATIO_DEFS: Dict[str, Dict[str, Any]] = {
    "net_margin": {"label": "销售净利率", "formula": "净利润 / 营业收入", "pct": True},
    "gross_margin": {"label": "销售毛利率", "formula": "毛利润 / 营业收入", "pct": True},
    "debt_to_asset": {"label": "资产负债率", "formula": "负债总额 / 资产总额", "pct": True},
    "current_ratio": {"label": "流动比率", "formula": "流动资产 / 流动负债", "pct": False},
    "roe": {"label": "净资产收益率", "formula": "净利润 / 所有者权益", "pct": True},
    "roa": {"label": "总资产收益率", "formula": "净利润 / 资产总额", "pct": True},
    "asset_turnover": {"label": "总资产周转率", "formula": "营业收入 / 资产总额", "pct": False},
    "expense_ratio": {"label": "期间费用率", "formula": "期间费用合计 / 营业收入", "pct": True},
    "rnd_intensity": {"label": "研发费用率", "formula": "研发费用 / 营业收入", "pct": True},
    "cash_conversion": {"label": "净利润现金含量", "formula": "经营活动现金流净额 / 净利润", "pct": False},
    "growth_rate": {"label": "同比增长率", "formula": "(本期值 - 上期值) / |上期值|", "pct": True},
    "equity_multiplier": {"label": "权益乘数", "formula": "资产总额 / 所有者权益", "pct": False},
    "npl_ratio": {"label": "不良贷款率", "formula": "不良贷款率（直接取披露值）", "pct": True},
    "provision_coverage": {"label": "拨备覆盖率", "formula": "拨备覆盖率（直接取披露值）", "pct": True},
    "capital_adequacy": {"label": "资本充足率", "formula": "资本充足率（直接取披露值）", "pct": True},
}

# 比率的常见参考区间，用于给分析结论加一句「是否偏离常态」的判断
RATIO_BENCHMARKS: Dict[str, str] = {
    "net_margin": "制造类企业通常 5%~15%；低于 0 说明主营亏损",
    "gross_margin": "需结合行业，逐期下滑通常指向竞争加剧或成本失控",
    "debt_to_asset": "一般以 70% 为警戒线，超过则偿债压力显著",
    "current_ratio": "一般认为 2 左右较为稳健，低于 1 说明短期偿债承压",
    "roe": "长期低于 8% 通常难以覆盖股权成本",
    "cash_conversion": "健康区间为 0.8~1.2；持续低于 0.8 说明盈利质量偏弱",
    "asset_turnover": "反映资产使用效率，需与同业和历史对比",
    "growth_rate": "正增长但明显低于应收账款增速时，需警惕收入质量",
    "npl_ratio": "商业银行不良率通常以 1.5% 作为关注线",
    "provision_coverage": "监管关注线通常为 150%",
    "capital_adequacy": "监管要求通常不低于 10.5%",
}


# ---------------------------------------------------------------------------
# 风险规则表
# ---------------------------------------------------------------------------
def _rule(
    rule_id: str,
    title: str,
    level: str,
    metric: str,
    predicate: Callable[[Dict[str, float]], bool],
    threshold: str,
    detail: str,
    remediation: str,
    entities: Sequence[str] = ("non_financial",),
) -> Dict[str, Any]:
    """构造一条风险规则。

    entities 声明该规则适用的主体类型：
        non_financial —— 一般工商企业（制造业、科技公司等）
        financial     —— 银行等金融机构
    这一点很关键：银行的资产负债率天然在 90% 以上（存款是负债），
    直接套用工商企业的 70% 杠杆警戒线会产生系统性误报。
    """
    return {
        "rule_id": rule_id,
        "title": title,
        "level": level,
        "metric": metric,
        "predicate": predicate,
        "threshold": threshold,
        "detail": detail,
        "remediation": remediation,
        "entities": tuple(entities),
    }


RISK_RULES: List[Dict[str, Any]] = [
    _rule("R-DEBT-01", "资产负债率超过警戒线", "high", "debt_to_asset",
          lambda v: v.get("debt_to_asset", 0) > 0.70, "> 70%",
          "资产负债率突破 70%，长期偿债压力显著上升。",
          "补充有息负债结构与到期期限分布，评估再融资能力。"),
    _rule("R-DEBT-02", "资产负债率偏高", "medium", "debt_to_asset",
          lambda v: 0.60 < v.get("debt_to_asset", 0) <= 0.70, "60%~70%",
          "资产负债率处于偏高区间，需关注杠杆水平变化趋势。",
          "披露近三年资产负债率走势及主要负债科目变动原因。"),
    _rule("R-PROFIT-01", "主营业务亏损", "high", "net_margin",
          lambda v: "net_margin" in v and v["net_margin"] < 0, "< 0",
          "销售净利率为负，主营尚未盈利。",
          "说明亏损原因、扭亏路径与资金可支撑期限。"),
    _rule("R-CASH-01", "净利润现金含量过低", "high", "cash_conversion",
          lambda v: "cash_conversion" in v and v["cash_conversion"] < 0.5, "< 0.5 倍",
          "经营活动现金流对净利润的覆盖不足 50%，盈利的现金支撑偏弱，"
          "存在通过放宽信用政策确认收入的可能性。",
          "结合应收账款账龄、存货周转与客户结算政策做交叉验证。"),
    _rule("R-CASH-02", "净利润现金含量偏低", "medium", "cash_conversion",
          lambda v: "cash_conversion" in v and 0.5 <= v["cash_conversion"] < 0.8, "0.5~0.8 倍",
          "净利润现金含量低于健康区间，回款质量需持续跟踪。",
          "披露报告期后回款情况与主要客户的信用期安排。"),
    _rule("R-AR-01", "应收账款增速显著高于收入增速", "high", "ar_gap",
          lambda v: v.get("ar_gap", 0) > 0.10, "差值 > 10 个百分点",
          "应收账款增速大幅超过营业收入增速，收入确认质量与坏账风险需要重点核查。",
          "补充应收账款账龄结构、前五大欠款方及坏账准备计提比例。"),
    _rule("R-AR-02", "应收账款增速高于收入增速", "medium", "ar_gap",
          lambda v: 0 < v.get("ar_gap", 0) <= 0.10, "差值 0~10 个百分点",
          "应收账款增速快于收入增速，需关注结算周期变化。",
          "跟踪期后回款与信用政策调整。"),
    _rule("R-PLEDGE-01", "控股股东质押比例过高", "high", "pledge_ratio",
          lambda v: v.get("pledge_ratio", 0) >= 0.50, ">= 50%",
          "控股股东质押比例达到 50% 以上，股价波动可能引发平仓与控制权风险。",
          "披露质押融资用途、预警线与平仓线。"),
    _rule("R-PLEDGE-02", "控股股东质押比例偏高", "medium", "pledge_ratio",
          lambda v: 0.30 <= v.get("pledge_ratio", 0) < 0.50, "30%~50%",
          "控股股东质押比例偏高，需关注补充质押能力。",
          "跟踪质押比例变动与解押安排。"),
    _rule("R-GUAR-01", "对外担保占净资产比例过高", "high", "guarantee_ratio",
          lambda v: v.get("guarantee_ratio", 0) > 0.30, "> 30%",
          "对外担保余额超过净资产的 30%，存在或有负债集中风险。",
          "披露被担保方经营与偿债能力。"),
    _rule("R-GUAR-02", "存在对外担保敞口", "medium", "guarantee_ratio",
          lambda v: 0.10 < v.get("guarantee_ratio", 0) <= 0.30, "10%~30%",
          "存在一定规模的对外担保敞口，需持续跟踪被担保方状况。",
          "定期核查被担保方是否存在逾期或代偿迹象。"),
    _rule("R-LIT-01", "未决诉讼金额占净资产比例较高", "high", "litigation_ratio",
          lambda v: v.get("litigation_ratio", 0) > 0.05, "> 5%",
          "未决诉讼涉案金额超过净资产的 5%，判决结果对财务状况影响重大。",
          "披露诉讼进展、败诉可能性与预计负债计提情况。"),
    _rule("R-LIT-02", "存在未决诉讼", "medium", "litigation_ratio",
          lambda v: 0.01 < v.get("litigation_ratio", 0) <= 0.05, "1%~5%",
          "存在未决诉讼，涉案金额相对净资产可控但结果存在不确定性。",
          "在风险提示中说明诉讼性质与进展。"),
    _rule("R-CONC-01", "客户集中度偏高", "medium", "top5_customer_ratio",
          lambda v: v.get("top5_customer_ratio", 0) >= 0.50, ">= 50%",
          "前五大客户销售占比超过 50%，单一客户流失对收入冲击较大。",
          "披露主要客户合作稳定性与在手订单覆盖情况。"),
    _rule("R-GW-01", "商誉占净资产比例偏高", "medium", "goodwill_ratio",
          lambda v: v.get("goodwill_ratio", 0) > 0.20, "> 20%",
          "商誉占净资产比例偏高，存在减值风险。",
          "披露标的公司业绩承诺完成情况与减值测试关键假设。"),
    _rule("R-LIQ-01", "流动比率低于 1", "high", "current_ratio",
          lambda v: "current_ratio" in v and v["current_ratio"] < 1.0, "< 1",
          "流动比率低于 1，短期偿债能力承压。",
          "披露流动资产结构与短期借款到期安排。"),
    _rule("R-GROW-01", "营业收入负增长", "high", "revenue_growth",
          lambda v: "revenue_growth" in v and v["revenue_growth"] < 0, "< 0",
          "营业收入同比下滑，主营业务收缩。",
          "说明下滑原因、在手订单与后续经营计划。"),
    _rule("R-BANK-01", "不良贷款率上行", "medium", "npl_delta",
          lambda v: v.get("npl_delta", 0) > 0, "较上年末上升",
          "不良贷款率较上年末上升，资产质量承压。",
          "披露不良生成率、核销与重组贷款情况。",
          entities=("financial",)),
    _rule("R-BANK-02", "拨备覆盖率低于监管关注线", "high", "provision_coverage",
          lambda v: "provision_coverage" in v and v["provision_coverage"] < 1.50, "< 150%",
          "拨备覆盖率低于 150% 的监管关注线，风险抵补能力偏弱。",
          "披露拨备计提政策与未来计提压力测算。",
          entities=("financial",)),
    _rule("R-BANK-03", "净息差收窄", "medium", "nim_delta",
          lambda v: v.get("nim_delta", 0) < 0, "较上年收窄",
          "净息差较上年收窄，利息净收入增长动能减弱。",
          "披露资产端收益率与负债端成本率的变动拆解。",
          entities=("financial",)),
]

_LEVEL_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3}


def _wrap(registry: ToolRegistry, spec: ToolSpec) -> ToolSpec:
    registry.register(spec)
    return spec


def register_all(registry: ToolRegistry) -> None:
    """把所有内置工具注册到给定的注册中心。"""

    # ------------------------------------------------------------------
    # 1) search_filings —— 多路资料检索
    # ------------------------------------------------------------------
    def _search_filings(
        query: str,
        top_k: int = 5,
        company: Optional[str] = None,
        year: Optional[int] = None,
        doc_type: Optional[str] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        retriever = registry.context.get("retriever")
        if retriever is None:
            raise RuntimeError("检索器未初始化：请先装载 data/ 语料")
        snippets = retriever.search_multi(
            [query], top_k=top_k, company=company, year=year, doc_type=doc_type, strict=strict
        )
        return {
            "query": query,
            "count": len(snippets),
            "results": [s.to_dict() for s in snippets],
        }

    _wrap(
        registry,
        ToolSpec(
            name="search_filings",
            description=(
                "在本地投研资料库（公司年报摘要/季报摘要）中做混合检索，"
                "返回带出处（资料编号 + 章节标题 + 父块上下文）的片段。"
                "输入是自然语言查询串；可用 company / year / doc_type 做元数据约束。"
            ),
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 2, "maxLength": 200,
                              "description": "检索查询串，建议包含指标名与期间"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5,
                              "description": "返回片段数量"},
                    "company": {"type": ["string", "null"], "default": None,
                                "description": "限定公司名称（可省略）"},
                    "year": {"type": ["integer", "null"], "minimum": 1990, "maximum": 2100, "default": None,
                             "description": "限定年份（可省略）"},
                    "doc_type": {"type": ["string", "null"], "default": None,
                                 "description": "限定文档类型，如『年度报告摘要』『三季度报告』"},
                    "strict": {"type": "boolean", "default": False,
                               "description": "true 表示元数据硬过滤，false 表示作为加权信号"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=_search_filings,
            permission_level=PermissionLevel.PUBLIC_READ,
            timeout_s=5.0,
            idempotent=True,
            tags=["rag", "retrieval"],
        ),
    )

    # ------------------------------------------------------------------
    # 2) get_financial_metric —— 结构化指标查询
    # ------------------------------------------------------------------
    def _get_financial_metric(
        company: str,
        metric: str,
        year: Optional[int] = None,
        period: str = "年度",
    ) -> Dict[str, Any]:
        store = registry.context.get("fact_store")
        if store is None:
            raise RuntimeError("事实库未初始化：请先装载 data/ 语料")
        resolved = store.resolve_company(company) or company
        fact = store.get(company=resolved, metric=metric, year=year, period=period)
        if fact is None:
            available = store.metrics(resolved, year or 0, period) if year else []
            raise ValueError(
                f"未找到指标：company={company!r}, metric={metric!r}, year={year}, period={period!r}。"
                f"该期间可用指标：{available[:20]}"
            )
        data = fact.to_dict()
        data["available_years"] = store.years(resolved)
        return data

    _wrap(
        registry,
        ToolSpec(
            name="get_financial_metric",
            description=(
                "查询结构化财务/经营指标（营业收入、净利润、资产总额、负债总额、所有者权益、"
                "经营活动现金流净额、应收账款、对外担保余额、未决诉讼涉案金额、控股股东股权质押比例、"
                "不良贷款率、拨备覆盖率等）。返回数值、单位与出处资料编号。"
            ),
            schema={
                "type": "object",
                "properties": {
                    "company": {"type": "string", "minLength": 2, "maxLength": 60,
                                "description": "公司全称，如『示例科技股份有限公司』"},
                    "metric": {"type": "string", "minLength": 1, "maxLength": 40,
                               "description": "指标名称，如『营业收入』『净利润』『负债总额』"},
                    "year": {"type": ["integer", "null"], "minimum": 1990, "maximum": 2100, "default": None,
                             "description": "年份，省略则取最新年度"},
                    "period": {"type": "string", "default": "年度",
                               "description": "期间：年度 / 三季度 / 半年度"},
                },
                "required": ["company", "metric"],
                "additionalProperties": False,
            },
            handler=_get_financial_metric,
            permission_level=PermissionLevel.PUBLIC_READ,
            timeout_s=3.0,
            idempotent=True,
            tags=["finance", "fact"],
        ),
    )

    # ------------------------------------------------------------------
    # 3) calc_ratio —— 比率 / 增长率计算
    # ------------------------------------------------------------------
    def _calc_ratio(
        ratio_name: str,
        numerator: float,
        denominator: float,
        precision: int = 4,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec = RATIO_DEFS.get(ratio_name)
        if spec is None:
            raise ValueError(f"不支持的比率：{ratio_name}，可选：{sorted(RATIO_DEFS)}")

        if ratio_name == "growth_rate":
            if denominator == 0:
                raise ValueError("增长率计算中『上期值』不能为 0")
            value = (numerator - denominator) / abs(denominator)
        else:
            if denominator == 0:
                raise ValueError(f"{spec['label']} 的分母不能为 0")
            value = numerator / denominator

        value = round(float(value), precision)
        result: Dict[str, Any] = {
            "ratio_name": ratio_name,
            "label": label or spec["label"],
            "formula": spec["formula"],
            "value": value,
            "unit": "%" if spec["pct"] else "倍",
            "inputs": {"numerator": numerator, "denominator": denominator},
            "benchmark": RATIO_BENCHMARKS.get(ratio_name, ""),
        }
        if spec["pct"]:
            result["value_pct"] = round(value * 100.0, min(precision + 2, 6))
            result["display"] = f"{result['value_pct']}%"
        else:
            result["display"] = f"{value}"
        return result

    _wrap(
        registry,
        ToolSpec(
            name="calc_ratio",
            description=(
                "财务比率计算器。ratio_name 决定公式："
                "net_margin=净利润/营业收入，gross_margin=毛利润/营业收入，"
                "debt_to_asset=负债总额/资产总额，current_ratio=流动资产/流动负债，"
                "roe=净利润/所有者权益，roa=净利润/资产总额，asset_turnover=营业收入/资产总额，"
                "expense_ratio=期间费用/营业收入，rnd_intensity=研发费用/营业收入，"
                "cash_conversion=经营活动现金流净额/净利润，equity_multiplier=资产总额/所有者权益，"
                "growth_rate=(本期-上期)/|上期|（此时 numerator=本期值，denominator=上期值）。"
            ),
            schema={
                "type": "object",
                "properties": {
                    "ratio_name": {"type": "string", "enum": sorted(RATIO_DEFS),
                                   "description": "比率标识"},
                    "numerator": {"type": "number", "description": "分子（增长率场景为本期值）"},
                    "denominator": {"type": "number", "description": "分母（增长率场景为上期值）"},
                    "precision": {"type": "integer", "minimum": 0, "maximum": 8, "default": 4,
                                  "description": "小数位数"},
                    "label": {"type": ["string", "null"], "default": None,
                              "description": "自定义展示名"},
                },
                "required": ["ratio_name", "numerator", "denominator"],
                "additionalProperties": False,
            },
            handler=_calc_ratio,
            permission_level=PermissionLevel.COMPUTE,
            timeout_s=2.0,
            idempotent=True,
            tags=["finance", "compute"],
        ),
    )

    # ------------------------------------------------------------------
    # 4) check_risk_rules —— 风险与合规规则引擎
    # ------------------------------------------------------------------
    def _check_risk_rules(
        company: str,
        year: int,
        extra_facts: Optional[Dict[str, Any]] = None,
        min_level: str = "medium",
        period: str = "年度",
    ) -> Dict[str, Any]:
        store = registry.context.get("fact_store")
        if store is None:
            raise RuntimeError("事实库未初始化：请先装载 data/ 语料")

        resolved = store.resolve_company(company)
        if resolved is None:
            raise ValueError(f"未知公司：{company}，可用公司：{store.companies}")
        company = resolved

        current = {name: f.value for name, f in store.by_period(company, year, period).items()}
        prior = {name: f.value for name, f in store.by_period(company, year - 1, period).items()}
        if not current:
            raise ValueError(
                f"未找到 {company} {year} 年 {period} 的指标数据，无法执行风险核查"
            )

        values: Dict[str, float] = {}

        def ratio(num_key: str, den_key: str) -> Optional[float]:
            num, den = current.get(num_key), current.get(den_key)
            if num is None or den in (None, 0):
                return None
            return float(num) / float(den)

        def growth(key: str) -> Optional[float]:
            now, before = current.get(key), prior.get(key)
            if now is None or before in (None, 0):
                return None
            return (float(now) - float(before)) / abs(float(before))

        # —— 由报表原始科目派生 ——
        derived = {
            "debt_to_asset": ratio("负债总额", "资产总额"),
            "net_margin": ratio("净利润", "营业收入"),
            "gross_margin": ratio("毛利润", "营业收入"),
            "current_ratio": ratio("流动资产", "流动负债"),
            "cash_conversion": ratio("经营活动现金流净额", "净利润"),
            "revenue_growth": growth("营业收入"),
            "receivable_growth": growth("应收账款"),
        }
        for k, v in derived.items():
            if v is not None:
                values[k] = float(v)

        # 应收账款增速与收入增速的差值（收入质量的核心预警指标）
        if "receivable_growth" in values and "revenue_growth" in values:
            values["ar_gap"] = values["receivable_growth"] - values["revenue_growth"]

        # 占净资产比例类
        equity = current.get("所有者权益")
        if equity:
            for out_key, src_key in (
                ("guarantee_ratio", "对外担保余额"),
                ("litigation_ratio", "未决诉讼涉案金额"),
                ("goodwill_ratio", "商誉账面价值"),
            ):
                src = current.get(src_key)
                if src is not None:
                    values[out_key] = float(src) / float(equity)

        if current.get("控股股东股权质押比例") is not None:
            values["pledge_ratio"] = float(current["控股股东股权质押比例"]) / 100.0
        if current.get("前五大客户销售占比") is not None:
            values["top5_customer_ratio"] = float(current["前五大客户销售占比"]) / 100.0
        if current.get("不良贷款率") is not None:
            values["npl_ratio"] = float(current["不良贷款率"]) / 100.0
            if prior.get("不良贷款率") is not None:
                values["npl_delta"] = (float(current["不良贷款率"]) - float(prior["不良贷款率"])) / 100.0
        if current.get("拨备覆盖率") is not None:
            values["provision_coverage"] = float(current["拨备覆盖率"]) / 100.0
        if current.get("资本充足率") is not None:
            values["capital_adequacy"] = float(current["资本充足率"]) / 100.0
        if current.get("净息差") is not None:
            values["net_interest_margin"] = float(current["净息差"]) / 100.0
            if prior.get("净息差") is not None:
                values["nim_delta"] = (float(current["净息差"]) - float(prior["净息差"])) / 100.0

        # —— 合并调用方（AnalystAgent）通过 calc_ratio 算出来的口径 ——
        merged_from_caller: List[str] = []
        for key, val in (extra_facts or {}).items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            if key not in values:
                merged_from_caller.append(key)
            values[key] = float(val)

        # —— 逐条规则评估 ——
        # 主体类型：出现银行监管指标即视为金融机构，据此启用/停用规则集，
        # 避免把工商企业的杠杆警戒线套到银行身上（银行的资产负债率天然 >90%）
        is_financial = ("不良贷款率" in current) or ("拨备覆盖率" in current)
        entity = "financial" if is_financial else "non_financial"

        findings: List[Dict[str, Any]] = []
        evaluated = 0
        for rule in RISK_RULES:
            if entity not in rule.get("entities", ("non_financial",)):
                continue  # 该规则不适用于本主体类型
            metric = rule["metric"]
            if metric not in values:
                continue  # 该规则所需指标不适用（例如没有披露该项）
            evaluated += 1
            try:
                triggered = bool(rule["predicate"](values))
            except Exception:  # noqa: BLE001 - 单条规则异常不应中断整体核查
                continue
            if not triggered:
                continue
            findings.append(
                {
                    "rule_id": rule["rule_id"],
                    "title": rule["title"],
                    "level": rule["level"],
                    "metric": metric,
                    "metric_value": round(float(values[metric]), 6),
                    "metric_display": _format_metric(metric, values[metric]),
                    "threshold": rule["threshold"],
                    "detail": rule["detail"],
                    "remediation": rule["remediation"],
                    "source_id": _source_of(store, company, year, metric),
                }
            )

        findings.sort(key=lambda f: (-_LEVEL_RANK.get(f["level"], 0), f["rule_id"]))
        kept = [f for f in findings if _LEVEL_RANK.get(f["level"], 0) >= _LEVEL_RANK.get(min_level, 0)]
        overall = max((f["level"] for f in findings), key=lambda lv: _LEVEL_RANK.get(lv, 0), default="info")

        return {
            "company": company,
            "year": year,
            "period": period,
            "entity_type": entity,
            "overall_level": overall,
            "evaluated_rules": evaluated,
            "triggered_count": len(findings),
            "findings": kept if min_level != "info" else findings,
            "all_findings_count": len(findings),
            "facts_used": {k: round(v, 6) for k, v in sorted(values.items())},
            "facts_from_caller": sorted(merged_from_caller),
        }

    _wrap(
        registry,
        ToolSpec(
            name="check_risk_rules",
            description=(
                "风险与合规规则引擎。基于结构化指标（可从公司年报事实库自动加载，也可由调用方"
                "通过 extra_facts 传入自行计算的比率）逐条评估风险规则，返回命中项及其等级"
                "（high/medium/low）、阈值、整改建议与出处。"
            ),
            schema={
                "type": "object",
                "properties": {
                    "company": {"type": "string", "minLength": 2, "maxLength": 60},
                    "year": {"type": "integer", "minimum": 1990, "maximum": 2100},
                    "extra_facts": {
                        "type": ["object", "null"],
                        "default": None,
                        "description": "调用方补充的数值型事实，如 {debt_to_asset: 0.67, net_margin: 0.075}",
                    },
                    "min_level": {"type": "string", "enum": ["info", "low", "medium", "high"],
                                  "default": "medium", "description": "返回结果的最低等级过滤"},
                    "period": {"type": "string", "default": "年度",
                               "description": "期间：年度 / 三季度 / 半年度 / 一季度"},
                },
                "required": ["company", "year"],
                "additionalProperties": False,
            },
            handler=_check_risk_rules,
            permission_level=PermissionLevel.RESTRICTED_READ,
            timeout_s=4.0,
            idempotent=True,
            tags=["risk", "compliance"],
        ),
    )

    # ------------------------------------------------------------------
    # 5) cite_source —— 引用解析（有副作用，非幂等）
    # ------------------------------------------------------------------
    _cite_counter = {"n": 0}

    def _cite_source(source_id: str, section_keyword: Optional[str] = None) -> Dict[str, Any]:
        store = registry.context.get("document_store")
        if store is None:
            raise RuntimeError("文档库未初始化：请先装载 data/ 语料")
        doc = store.get(source_id)
        if doc is None:
            raise ValueError(f"未知资料编号：{source_id}，可用编号：{store.source_ids}")

        section_title = ""
        if section_keyword:
            section = store.locate(source_id, section_keyword)
            if section is not None:
                section_title = section.title
        if not section_title and doc.sections:
            section_title = doc.sections[0].title

        _cite_counter["n"] += 1  # 每次调用分配新的引用序号：有副作用，故声明为非幂等
        return {
            "citation_no": _cite_counter["n"],
            "source_id": doc.source_id,
            "title": doc.title,
            "company": doc.company,
            "doc_type": doc.doc_type,
            "period": doc.period,
            "year": doc.year,
            "section": section_title,
            "locator": f"{doc.source_id} · {section_title}" if section_title else doc.source_id,
            "path": doc.path,
            "publisher": doc.meta.get("publisher", ""),
            "disclaimer": doc.meta.get("disclaimer", ""),
        }

    _wrap(
        registry,
        ToolSpec(
            name="cite_source",
            description=(
                "把资料编号解析成规范引用记录（资料编号、公司、文档类型、期间、章节、文件路径），"
                "并分配一个引用序号。WriterAgent 用它保证简报中每个 [n] 都能回溯到原文。"
            ),
            schema={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string", "minLength": 2, "maxLength": 64},
                    "section_keyword": {"type": ["string", "null"], "default": None,
                                        "description": "章节标题关键词，用于定位具体章节"},
                },
                "required": ["source_id"],
                "additionalProperties": False,
            },
            handler=_cite_source,
            permission_level=PermissionLevel.WRITE,
            timeout_s=3.0,
            idempotent=False,
            tags=["citation", "provenance"],
        ),
    )


def _format_metric(metric: str, value: float) -> str:
    """风险指标的人类可读展示（内部统一用小数存储，展示时换算成百分比）。"""
    pct_metrics = {
        "debt_to_asset", "net_margin", "gross_margin", "roe", "roa",
        "revenue_growth", "receivable_growth", "ar_gap",
        "guarantee_ratio", "litigation_ratio", "goodwill_ratio",
        "pledge_ratio", "top5_customer_ratio",
        "npl_ratio", "npl_delta", "provision_coverage",
        "capital_adequacy", "net_interest_margin", "nim_delta",
    }
    if metric in pct_metrics:
        return f"{value * 100:.2f}%"
    if metric in {"current_ratio", "cash_conversion", "asset_turnover", "equity_multiplier"}:
        return f"{value:.2f} 倍"
    return f"{value:.4f}"


def _source_of(store: Any, company: str, year: int, metric: str) -> str:
    """给风险结论带上出处：优先用派生指标对应的原始科目出处。"""
    origin_metric = {
        "ar_gap": "应收账款",
        "guarantee_ratio": "对外担保余额",
        "litigation_ratio": "未决诉讼涉案金额",
        "goodwill_ratio": "商誉账面价值",
        "pledge_ratio": "控股股东股权质押比例",
        "top5_customer_ratio": "前五大客户销售占比",
        "npl_delta": "不良贷款率",
    }.get(metric, None)

    candidates = [origin_metric] if origin_metric else []
    candidates += ["资产总额", "营业收入"]
    for name in candidates:
        if not name:
            continue
        fact = store.get(company=company, metric=name, year=year, period="年度")
        if fact is not None:
            return fact.source_id
    return ""
