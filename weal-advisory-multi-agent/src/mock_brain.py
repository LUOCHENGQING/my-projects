"""确定性 mock 大脑（无 API Key 时的降级推理）。

定位
----
**它不是"假装调用模型"**，而是一组确定性的中文文案生成器：所有业务判断
（约束求解、产品筛选、权重分配、适当性判定、压力测试）都由确定性代码完成，
mock 只负责把已经算好的结构化事实组织成人话。因此：

- demo 在没有网络、没有 Key 的环境下也一定能跑通且输出有意义的内容；
- mock 输出完全可复现，适合写进单测断言；
- 接上真实模型后，语义不变，只是措辞更自然。

任务清单（key 即 `compose(task=...)` 的 task 名）
-------------------------------------------------
`client_profile_summary` / `screening_note` / `optimizer_rationale` /
`suitability_comment` / `advisor_summary` / `risk_disclosure` / `counterfactual_note`
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .utils import pct

#: 各任务要求的输出字段（真实模型也必须按此 JSON 结构返回）
TASK_SCHEMAS: dict[str, tuple[str, ...]] = {
    "client_profile_summary": ("summary", "consistency_note"),
    "screening_note": ("note",),
    "optimizer_rationale": ("rationale",),
    "suitability_comment": ("comment",),
    "advisor_summary": ("summary",),
    "risk_disclosure": ("disclosure",),
    "counterfactual_note": ("note",),
}


def _fmt_money(value: float) -> str:
    """金额千分位格式化。"""
    return f"{value:,.0f} 元"


def _client_profile_summary(ctx: Mapping[str, Any]) -> dict[str, str]:
    """客户画像摘要。"""
    name = ctx.get("display_name", "客户")
    age = ctx.get("age", 0)
    level = ctx.get("risk_level", 3)
    horizon = ctx.get("horizon_years", 0)
    amount = ctx.get("investable_amount", 0.0)
    caps = ctx.get("caps", {})
    summary = (
        f"{name}，{age} 周岁，风险承受等级 R{level}，投资期限约 {horizon:g} 年，"
        f"可投金额 {_fmt_money(amount)}；"
        f"集中度约束为单一产品 {pct(caps.get('product', 0.0))}、"
        f"单一类别 {pct(caps.get('asset_class', 0.0))}、"
        f"同一发行人 {pct(caps.get('issuer', 0.0))}，"
        f"流动性资产占比不低于 {pct(ctx.get('liquidity_floor', 0.0))}。"
    )
    prohibited = ctx.get("prohibited") or []
    if prohibited:
        summary += f"客户明确排除：{'、'.join(prohibited)}。"
    experience = ctx.get("experience") or []
    summary += f"已具备投资经验的品类：{'、'.join(experience) if experience else '无'}。"

    archived = ctx.get("archived_level")
    scored = ctx.get("questionnaire_level")
    if scored is not None and archived is not None and int(scored) < int(archived):
        consistency = (
            f"风险测评问卷折算等级为 R{scored}，低于档案登记等级 R{archived}，"
            "按从严原则取孰低者（R" + str(scored) + "）作为硬约束上限，并以书面方式提示客户。"
        )
    elif scored is not None and archived is not None and int(scored) > int(archived):
        consistency = (
            f"风险测评问卷折算等级为 R{scored}，高于档案登记等级 R{archived}，"
            "按从严原则仍以较低的档案等级 R" + str(archived) + " 作为硬约束上限，不做上调。"
        )
    else:
        consistency = "问卷折算等级与档案登记等级一致，无需按从严原则调整。"
    return {"summary": summary, "consistency_note": consistency}


def _screening_note(ctx: Mapping[str, Any]) -> dict[str, str]:
    """候选池筛选说明。"""
    universe = ctx.get("universe_size", 0)
    included = ctx.get("included_count", 0)
    excluded = universe - included
    top = ctx.get("top_exclusions") or []
    note = f"在 {universe} 只候选产品中，{included} 只落入硬约束可行域，剔除 {excluded} 只。"
    if top:
        note += "主要剔除原因：" + "；".join(str(item) for item in top[:3]) + "。"
    note += "被剔除产品不进入后续组合构建，剔除原因逐条留痕，可向客户逐项解释。"
    return {"note": note}


def _optimizer_rationale(ctx: Mapping[str, Any]) -> dict[str, str]:
    """组合权衡说明。"""
    metrics = ctx.get("metrics", {})
    binding = ctx.get("binding") or []
    targets = ctx.get("class_targets") or {}
    parts = [
        "组合按「期望收益 / 波动 / 流动性 / 集中度」四目标折中求解，"
        f"最终预期年化收益 {pct(metrics.get('expected_return', 0.0))}、"
        f"预期波动 {pct(metrics.get('expected_volatility', 0.0))}、"
        f"组合流动性资产占比 {pct(metrics.get('liquidity_ratio', 0.0))}、"
        f"最大单一持仓 {pct(metrics.get('max_single_weight', 0.0))}。"
    ]
    if targets:
        parts.append(
            "类别基准配置为 "
            + "、".join(f"{k} {pct(v)}" for k, v in sorted(targets.items()) if v > 0)
            + "。"
        )
    if binding:
        parts.append("当前起约束作用的上限：" + "；".join(binding) + "。")
    else:
        parts.append("当前组合未触及任何硬约束上限，仍有一定调整空间。")
    return {"rationale": "".join(parts)}


def _suitability_comment(ctx: Mapping[str, Any]) -> dict[str, str]:
    """适当性复核意见。"""
    directive = ctx.get("directive", "pass")
    blocks = ctx.get("block_rules") or []
    warns = ctx.get("warn_rules") or []
    round_index = int(ctx.get("round_index", 0)) + 1
    if directive == "pass":
        comment = f"第 {round_index} 轮适当性复核通过，未命中 block 级规则。"
    elif directive == "reoptimize":
        comment = (
            f"第 {round_index} 轮适当性复核不通过，命中 {'、'.join(blocks)}，"
            "已下发约束收紧指令并打回重新配置。"
        )
    elif directive == "reject":
        comment = f"第 {round_index} 轮适当性复核判定不予通过，规则命中：{'、'.join(blocks)}。"
    else:
        comment = "适当性复核完成。"
    if warns:
        comment += f"另有告警级提示需向客户揭示：{'、'.join(warns)}。"
    return {"comment": comment}


def _advisor_summary(ctx: Mapping[str, Any]) -> dict[str, str]:
    """建议书摘要。"""
    name = ctx.get("display_name", "客户")
    holdings = ctx.get("holding_count", 0)
    cash = ctx.get("cash_weight", 0.0)
    metrics = ctx.get("metrics", {})
    status = ctx.get("status", "draft")
    summary = (
        f"本建议书为 {name} 生成，共配置 {holdings} 只产品，"
        f"现金及活期留存 {pct(cash)}，"
        f"预期年化收益 {pct(metrics.get('expected_return', 0.0))}，"
        f"预期波动 {pct(metrics.get('expected_volatility', 0.0))}。"
    )
    if status == "rejected":
        summary += "因硬约束下不存在可行组合，本次不出具配置建议，仅出具拒绝对照说明。"
    else:
        summary += "全部持仓均落在客户硬约束可行域内，并通过适当性规则复核。"
    return {"summary": summary}


def _risk_disclosure(ctx: Mapping[str, Any]) -> dict[str, str]:
    """风险揭示。"""
    level = ctx.get("risk_level", 3)
    tolerance = ctx.get("drawdown_tolerance", 0.0)
    worst = ctx.get("worst_scenario_name", "压力情景")
    worst_drawdown = ctx.get("worst_drawdown", 0.0)
    disclosure = (
        f"本组合对应客户风险承受等级 R{level}，客户约定的最大回撤容忍度为 {pct(tolerance)}。"
        f"在「{worst}」情景下，组合估计最大回撤约 {pct(worst_drawdown)}。"
        "压力测试结果基于示例敏感性系数表的确定性推算，不构成对未来收益或损失的承诺；"
        "投资组合可能发生本金损失，历史业绩与情景测算均不代表未来表现。"
    )
    return {"disclosure": disclosure}


def _counterfactual_note(ctx: Mapping[str, Any]) -> dict[str, str]:
    """反事实解释导读。"""
    variants = ctx.get("variants") or []
    covered = ctx.get("non_empty", 0)
    note = (
        f"共对 {len(variants)} 条假设约束变化重新求解并做结构化差异比较，"
        f"其中 {covered} 条产生了实质差异。"
        "反事实结论由「修改约束 → 重新求解 → 差异比对」得出，可复现、可复核。"
    )
    return {"note": note}


#: 任务 -> 生成器
BUILDERS: dict[str, Callable[[Mapping[str, Any]], dict[str, str]]] = {
    "client_profile_summary": _client_profile_summary,
    "screening_note": _screening_note,
    "optimizer_rationale": _optimizer_rationale,
    "suitability_comment": _suitability_comment,
    "advisor_summary": _advisor_summary,
    "risk_disclosure": _risk_disclosure,
    "counterfactual_note": _counterfactual_note,
}


def mock_compose(task: str, context: Mapping[str, Any]) -> dict[str, str]:
    """按任务生成确定性文案（未知任务返回通用占位说明，不抛异常）。"""
    builder = BUILDERS.get(task)
    if builder is None:
        return {"note": f"（mock 大脑未登记任务 {task}，已按确定性规则跳过）"}
    result = builder(context)
    # 保证输出字段与 schema 一致
    for key in TASK_SCHEMAS.get(task, ()):
        result.setdefault(key, "")
    return result
